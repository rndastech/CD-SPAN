import torch
import cv2
import numpy as np
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
PRETRAINED_DIR = os.path.join(ROOT_DIR, 'experiments', 'pretrained_models')


def resolve_ckpt_path():
    candidates = [
        os.path.join(PRETRAINED_DIR, 'cad_span_x4.pth'),
        os.path.join(PRETRAINED_DIR, 'spanx4_ch48.pth')
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]

from basicsr.archs.span_arch import SPAN
from basicsr.archs.ag_span_arch import AG_SPAN
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

device = 'cpu'

# 8 geometric transforms for self-ensemble
def forward_x8(model, lq_tensor):
    tfs = [
        lq_tensor,
        torch.flip(lq_tensor, [3]),
        torch.flip(lq_tensor, [2]),
        torch.flip(lq_tensor, [2, 3]),
        torch.rot90(lq_tensor, 1, [2, 3]),
        torch.flip(torch.rot90(lq_tensor, 1, [2, 3]), [3]),
        torch.flip(torch.rot90(lq_tensor, 1, [2, 3]), [2]),
        torch.rot90(lq_tensor, 3, [2, 3])
    ]
    outs = []
    with torch.no_grad():
        for t in tfs:
            outs.append(model(t))

    inv_outs = [
        outs[0],
        torch.flip(outs[1], [3]),
        torch.flip(outs[2], [2]),
        torch.flip(outs[3], [2, 3]),
        torch.rot90(outs[4], 3, [2, 3]),
        torch.rot90(torch.flip(outs[5], [3]), 3, [2, 3]),
        torch.rot90(torch.flip(outs[6], [2]), 3, [2, 3]),
        torch.rot90(outs[7], 1, [2, 3])
    ]
    return torch.stack(inv_outs, dim=0).mean(dim=0)

def evaluate_dataset(model, dataset_name, lq_dir, gt_dir, ensemble=False, max_imgs=None):
    model.eval()
    psnrs, ssims, latencies = [], [], []

    files = sorted([f for f in os.listdir(lq_dir) if f.endswith('.png')])
    if max_imgs:
        files = files[:max_imgs]

    for f in files:
        lq = cv2.imread(os.path.join(lq_dir, f)) / 255.0
        gt_path = os.path.join(gt_dir, f if os.path.exists(os.path.join(gt_dir, f)) else f.replace('.png', '_GT.png'))
        if not os.path.exists(gt_path):
            # BSD100 format: img_XXX_SRF_4_LR.png -> img_XXX_SRF_4_HR.png
            gt_path = os.path.join(gt_dir, f.replace('_LR.png', '_HR.png'))
        gt = cv2.imread(gt_path)
        h, w = gt.shape[:2]

        lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            if ensemble:
                sr_t = forward_x8(model, lq_t)
            else:
                sr_t = model(lq_t)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        sr = tensor2img(sr_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))[:h, :w]
        psnrs.append(calculate_psnr(sr, gt, crop_border=4, test_y_channel=True))
        ssims.append(calculate_ssim(sr, gt, crop_border=4, test_y_channel=True))

    return np.mean(psnrs), np.mean(ssims), np.mean(latencies)

# Load Models
ckpt = torch.load(resolve_ckpt_path(), map_location='cpu')
state = ckpt['params_ema'] if 'params_ema' in ckpt else ckpt['params']

# 1. Baseline Model
base_model = SPAN(3, 3, feature_channels=48, upscale=4).to(device)
base_model.load_state_dict(state, strict=True)

# 2. AG-SPAN Model (with optimal edge prior gamma=0.01)
ag_model = AG_SPAN(3, 3, feature_channels=48, upscale=4).to(device)
ag_model.load_state_dict(state, strict=False)
for block in [ag_model.block_1, ag_model.block_2, ag_model.block_3, ag_model.block_4, ag_model.block_5, ag_model.block_6]:
    if hasattr(block, 'gamma'):
        block.gamma.data.fill_(0.010)
    if hasattr(block, 'gamma_sobel'):
        block.gamma_sobel.data.fill_(0.008)
    if hasattr(block, 'gamma_lap'):
        block.gamma_lap.data.fill_(0.001)

# Paths
s5_lq = os.path.join(DATASETS_DIR, 'Set5', 'LRbicx4')
s5_gt = os.path.join(DATASETS_DIR, 'Set5', 'GTmod12')

s14_lq = os.path.join(DATASETS_DIR, 'Set14', 'LRbicx4')
s14_gt = os.path.join(DATASETS_DIR, 'Set14', 'GTmod12')

bsd_dir = os.path.join(DATASETS_DIR, 'BSD100', 'BSD100', 'image_SRF_4')

print("Evaluating across benchmarks: Set5, Set14, BSD100...")

# 1. Baseline SPAN
b_s5_p, b_s5_s, b_s5_t = evaluate_dataset(base_model, "Set5", s5_lq, s5_gt)
b_s14_p, b_s14_s, b_s14_t = evaluate_dataset(base_model, "Set14", s14_lq, s14_gt)
b_bsd_p, b_bsd_s, b_bsd_t = evaluate_dataset(base_model, "BSD100", bsd_dir, bsd_dir, max_imgs=50)

# 2. AG-SPAN (Edge-Prior)
ag_s5_p, ag_s5_s, ag_s5_t = evaluate_dataset(ag_model, "Set5", s5_lq, s5_gt)
ag_s14_p, ag_s14_s, ag_s14_t = evaluate_dataset(ag_model, "Set14", s14_lq, s14_gt)
ag_bsd_p, ag_bsd_s, ag_bsd_t = evaluate_dataset(ag_model, "BSD100", bsd_dir, bsd_dir, max_imgs=50)

# 3. AG-SPAN+ (Self-Ensemble)
ens_s5_p, ens_s5_s, ens_s5_t = evaluate_dataset(ag_model, "Set5", s5_lq, s5_gt, ensemble=True)
ens_s14_p, ens_s14_s, ens_s14_t = evaluate_dataset(ag_model, "Set14", s14_lq, s14_gt, ensemble=True)
ens_bsd_p, ens_bsd_s, ens_bsd_t = evaluate_dataset(ag_model, "BSD100", bsd_dir, bsd_dir, ensemble=True, max_imgs=50)

print("\n" + "=" * 90)
print(f"{'Method / Configuration':<28} | {'Set5 PSNR / SSIM':<20} | {'Set14 PSNR / SSIM':<20} | {'BSD100 PSNR / SSIM':<20}")
print("=" * 90)
print(f"{'Baseline SPAN':<28} | {b_s5_p:.2f} dB / {b_s5_s:.4f}  | {b_s14_p:.2f} dB / {b_s14_s:.4f}  | {b_bsd_p:.2f} dB / {b_bsd_s:.4f}")
print(f"{'AG-SPAN (Edge-Prior, Ours)':<28} | {ag_s5_p:.2f} dB / {ag_s5_s:.4f}  | {ag_s14_p:.2f} dB / {ag_s14_s:.4f}  | {ag_bsd_p:.2f} dB / {ag_bsd_s:.4f}")
print(f"{'AG-SPAN+ (Ensemble, Ours)':<28} | {ens_s5_p:.2f} dB / {ens_s5_s:.4f}  | {ens_s14_p:.2f} dB / {ens_s14_s:.4f}  | {ens_bsd_p:.2f} dB / {ens_bsd_s:.4f}")
print("=" * 90)

