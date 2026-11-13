import torch
import cv2
import numpy as np
import os
import sys

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
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

device = 'cpu'
base_model = SPAN(3, 3, feature_channels=48, upscale=4).to(device)
ckpt = torch.load(resolve_ckpt_path(), map_location='cpu')
state = ckpt['params_ema'] if 'params_ema' in ckpt else ckpt['params']
base_model.load_state_dict(state, strict=True)
base_model.eval()

def forward_x8(model, lq_tensor):
    # Generates 8 augmented versions, feeds to model, inverse-transforms and averages
    def x8_transforms(img):
        # img: [B, C, H, W]
        tfs = [
            img,
            torch.flip(img, [3]),                          # flip horizontal
            torch.flip(img, [2]),                          # flip vertical
            torch.flip(img, [2, 3]),                       # flip both
            torch.rot90(img, 1, [2, 3]),                   # rot 90
            torch.flip(torch.rot90(img, 1, [2, 3]), [3]),  # rot 90 + flip h
            torch.flip(torch.rot90(img, 1, [2, 3]), [2]),  # rot 90 + flip v
            torch.rot90(img, 3, [2, 3])                    # rot 270
        ]
        return tfs

    def inverse_x8(outs):
        return [
            outs[0],
            torch.flip(outs[1], [3]),
            torch.flip(outs[2], [2]),
            torch.flip(outs[3], [2, 3]),
            torch.rot90(outs[4], 3, [2, 3]),
            torch.rot90(torch.flip(outs[5], [3]), 3, [2, 3]),
            torch.rot90(torch.flip(outs[6], [2]), 3, [2, 3]),
            torch.rot90(outs[7], 1, [2, 3])
        ]

    tfs = x8_transforms(lq_tensor)
    outs = []
    with torch.no_grad():
        for t in tfs:
            outs.append(model(t))
    inv_outs = inverse_x8(outs)
    avg_out = torch.stack(inv_outs, dim=0).mean(dim=0)
    return avg_out

# Test on Set5
set5_lq = os.path.join(DATASETS_DIR, 'Set5', 'LRbicx4')
set5_gt = os.path.join(DATASETS_DIR, 'Set5', 'GTmod12')

p_standard, p_ensemble = [], []
s_standard, s_ensemble = [], []

print("Running Self-Ensemble Evaluation on Set5...")
for f in sorted(os.listdir(set5_lq)):
    if not f.endswith('.png'): continue
    lq = cv2.imread(os.path.join(set5_lq, f)) / 255.0
    gt = cv2.imread(os.path.join(set5_gt, f))
    h, w = gt.shape[:2]
    lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

    # 1. Standard Forward
    with torch.no_grad():
        sr_std = model = base_model(lq_t)
    sr_std_img = tensor2img(sr_std.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))[:h, :w]
    psnr_std = calculate_psnr(sr_std_img, gt, crop_border=4, test_y_channel=True)
    ssim_std = calculate_ssim(sr_std_img, gt, crop_border=4, test_y_channel=True)
    p_standard.append(psnr_std)
    s_standard.append(ssim_std)

    # 2. Ensemble Forward
    sr_ens = forward_x8(base_model, lq_t)
    sr_ens_img = tensor2img(sr_ens.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))[:h, :w]
    psnr_ens = calculate_psnr(sr_ens_img, gt, crop_border=4, test_y_channel=True)
    ssim_ens = calculate_ssim(sr_ens_img, gt, crop_border=4, test_y_channel=True)
    p_ensemble.append(psnr_ens)
    s_ensemble.append(ssim_ens)

    print(f"  {f:<15} | Standard: {psnr_std:.2f} dB, {ssim_std:.4f}  --->  Ensemble: {psnr_ens:.2f} dB, {ssim_ens:.4f} (Delta: {psnr_ens - psnr_std:+.2f} dB)")

print(f"\n[Set5 Standard Average] PSNR: {np.mean(p_standard):.4f} dB, SSIM: {np.mean(s_standard):.4f}")
print(f"[Set5 Ensemble Average] PSNR: {np.mean(p_ensemble):.4f} dB, SSIM: {np.mean(s_ensemble):.4f}")
print(f"Set5 Overall Improvement: {np.mean(p_ensemble) - np.mean(p_standard):+.4f} dB PSNR, {np.mean(s_ensemble) - np.mean(s_standard):+.4f} SSIM!")

