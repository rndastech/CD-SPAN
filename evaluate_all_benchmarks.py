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

# 8 geometric transforms for self-ensemble (x8)
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

def load_pairs(dataset_name):
    pairs = []
    if dataset_name == "Set5":
        lq_dir = os.path.join(DATASETS_DIR, 'Set5', 'LRbicx4')
        gt_dir = os.path.join(DATASETS_DIR, 'Set5', 'GTmod12')
        for f in sorted(os.listdir(lq_dir)):
            gt_p = os.path.join(gt_dir, f if os.path.exists(os.path.join(gt_dir, f)) else f.replace('.png', '_GT.png'))
            pairs.append((f, os.path.join(lq_dir, f), gt_p))
    elif dataset_name == "Set14":
        lq_dir = os.path.join(DATASETS_DIR, 'Set14', 'LRbicx4')
        gt_dir = os.path.join(DATASETS_DIR, 'Set14', 'GTmod12')
        for f in sorted(os.listdir(lq_dir)):
            gt_p = os.path.join(gt_dir, f if os.path.exists(os.path.join(gt_dir, f)) else f.replace('.png', '_GT.png'))
            pairs.append((f, os.path.join(lq_dir, f), gt_p))
    elif dataset_name == "BSD100":
        bsd_dir = os.path.join(DATASETS_DIR, 'BSD100', 'BSD100', 'image_SRF_4')
        lr_files = sorted([f for f in os.listdir(bsd_dir) if f.endswith('_LR.png')])[:25]
        for f in lr_files:
            gt_p = os.path.join(bsd_dir, f.replace('_LR.png', '_HR.png'))
            pairs.append((f, os.path.join(bsd_dir, f), gt_p))
    return pairs

def evaluate(model, pairs, ensemble=False):
    model.eval()
    results = {}
    psnrs, ssims = [], []
    for fname, lq_p, gt_p in pairs:
        lq = cv2.imread(lq_p) / 255.0
        gt = cv2.imread(gt_p)
        h, w = gt.shape[:2]
        lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

        with torch.no_grad():
            if ensemble:
                sr_t = forward_x8(model, lq_t)
            else:
                sr_t = model(lq_t)

        sr = tensor2img(sr_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))[:h, :w]
        p = calculate_psnr(sr, gt, crop_border=4, test_y_channel=True)
        s = calculate_ssim(sr, gt, crop_border=4, test_y_channel=True)
        results[fname] = (p, s)
        psnrs.append(p)
        ssims.append(s)
    return np.mean(psnrs), np.mean(ssims), results

def main():
    print("Loading pretrained weights...", flush=True)
    ckpt = torch.load(resolve_ckpt_path(), map_location='cpu')
    state = ckpt['params_ema'] if 'params_ema' in ckpt else ckpt['params']

    # 1. Baseline SPAN
    base_model = SPAN(3, 3, feature_channels=48, upscale=4).to(device)
    base_model.load_state_dict(state, strict=True)

    # 2. AG-SPAN (Edge-Prior, gamma=0.010)
    ag_model = AG_SPAN(3, 3, feature_channels=48, upscale=4).to(device)
    ag_model.load_state_dict(state, strict=False)
    for blk in [ag_model.block_1, ag_model.block_2, ag_model.block_3, ag_model.block_4, ag_model.block_5, ag_model.block_6]:
        if hasattr(blk, 'beta'):
            blk.beta.data.fill_(1.0)
        if hasattr(blk, 'gamma'):
            blk.gamma.data.fill_(0.010)
        if hasattr(blk, 'gamma_sobel'):
            blk.gamma_sobel.data.fill_(0.008)
        if hasattr(blk, 'gamma_lap'):
            blk.gamma_lap.data.fill_(0.001)

    datasets = ["Set5", "Set14", "BSD100"]
    summary = {}

    for dname in datasets:
        print(f"\nEvaluating on {dname}...", flush=True)
        pairs = load_pairs(dname)

        # Baseline
        bp, bs, b_per = evaluate(base_model, pairs, ensemble=False)
        print(f"  [Baseline SPAN]       PSNR: {bp:.4f} dB | SSIM: {bs:.4f}", flush=True)

        # AG-SPAN (Ours)
        agp, ags, ag_per = evaluate(ag_model, pairs, ensemble=False)
        print(f"  [AG-SPAN (Edge-Prior)] PSNR: {agp:.4f} dB | SSIM: {ags:.4f} (Delta: {agp - bp:+.4f} dB)", flush=True)

        # AG-SPAN+ (Ensemble)
        ensp, enss, ens_per = evaluate(ag_model, pairs, ensemble=True)
        print(f"  [AG-SPAN+ (Ensemble)]  PSNR: {ensp:.4f} dB | SSIM: {enss:.4f} (Delta: {ensp - bp:+.4f} dB)", flush=True)

        summary[dname] = {
            'baseline': (bp, bs, b_per),
            'ag_span': (agp, ags, ag_per),
            'ag_span_plus': (ensp, enss, ens_per)
        }

    print("\n" + "=" * 95)
    print(f"{'Dataset':<10} | {'Metric':<6} | {'Baseline SPAN':<15} | {'AG-SPAN (Ours)':<18} | {'AG-SPAN+ (Ensemble)':<20} | {'Max Gain':<10}")
    print("=" * 95)
    for dname in datasets:
        bp, bs, _ = summary[dname]['baseline']
        agp, ags, _ = summary[dname]['ag_span']
        ep, es, _ = summary[dname]['ag_span_plus']
        print(f"{dname:<10} | {'PSNR':<6} | {bp:6.2f} dB        | {agp:6.2f} dB (+{agp-bp:+.3f})   | {ep:6.2f} dB (+{ep-bp:+.3f})      | {ep-bp:+.3f} dB")
        print(f"{'':<10} | {'SSIM':<6} | {bs:6.4f}           | {ags:6.4f}           | {es:6.4f}              | {es-bs:+.4f}")
        print("-" * 95)

    # Per-image breakdown for Set5
    print("\nSet5 Per-Image Breakdown (PSNR dB):")
    print(f"{'Image':<15} | {'Baseline':<10} | {'AG-SPAN':<10} | {'AG-SPAN+':<10} | {'Gain':<10}")
    print("-" * 60)
    s5_base = summary['Set5']['baseline'][2]
    s5_ag = summary['Set5']['ag_span'][2]
    s5_ens = summary['Set5']['ag_span_plus'][2]
    for img_name in sorted(s5_base.keys()):
        bp = s5_base[img_name][0]
        agp = s5_ag[img_name][0]
        ep = s5_ens[img_name][0]
        print(f"{img_name:<15} | {bp:6.2f} dB  | {agp:6.2f} dB  | {ep:6.2f} dB   | {ep-bp:+.3f} dB")

if __name__ == '__main__':
    main()

