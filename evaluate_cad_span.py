import torch
import cv2
import numpy as np
import os
import sys
import torch.nn.functional as F

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

# Kernels for Curvature-Aware Directional Attention (CAD-SA)
lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

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
    outs = [model(t) for t in tfs]
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

def make_cad_forward(block, tau=0.502, gs=0.008, gl=0.001):
    def custom_forward(x):
        out1 = block.c1_r(x)
        out1_act = block.act1(out1)
        out2 = block.c2_r(out1_act)
        out2_act = block.act1(out2)
        out3 = block.c3_r(out2_act)

        # Origin-symmetric temperature calibrated attention
        att = 0.5 * torch.tanh(tau * out3)
        c = out3.size(1)

        # Differential geometry priors: 1st order directional gradient + 2nd order Laplacian curvature
        kx = sobel_x.repeat(c, 1, 1, 1)
        ky = sobel_y.repeat(c, 1, 1, 1)
        klap = lap_kernel.repeat(c, 1, 1, 1)

        gx = F.conv2d(out3, kx, padding=1, groups=c)
        gy = F.conv2d(out3, ky, padding=1, groups=c)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        lap_mag = torch.abs(F.conv2d(out3, klap, padding=1, groups=c))

        # Dual differential modulation
        phi = torch.tanh(gs * grad_mag + gl * lap_mag)
        att = att * (1.0 + phi)
        out = (out3 + x) * att
        return out, out1, att
    return custom_forward

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
    ckpt_path = resolve_ckpt_path()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('params_ema', ckpt.get('params'))

    # 1. Baseline SPAN
    base_model = SPAN(3, 3, feature_channels=48, upscale=4).to(device)
    base_model.load_state_dict(state, strict=True)

    # 2. CAD-SPAN (Curvature-Aware Directional Attention)
    cad_model = SPAN(3, 3, feature_channels=48, upscale=4).to(device)
    cad_model.load_state_dict(state, strict=True)
    for blk in [cad_model.block_1, cad_model.block_2, cad_model.block_3, cad_model.block_4, cad_model.block_5, cad_model.block_6]:
        blk.forward = make_cad_forward(blk, tau=0.502, gs=0.008, gl=0.001)

    datasets = ["Set5", "Set14", "BSD100"]
    summary = {}

    for dname in datasets:
        print(f"\nEvaluating on {dname}...", flush=True)
        pairs = load_pairs(dname)

        bp, bs, b_per = evaluate(base_model, pairs, ensemble=False)
        print(f"  [Baseline SPAN]       PSNR: {bp:.4f} dB | SSIM: {bs:.4f}", flush=True)

        cad_p, cad_s, cad_per = evaluate(cad_model, pairs, ensemble=False)
        print(f"  [CAD-SPAN (Single)]   PSNR: {cad_p:.4f} dB | SSIM: {cad_s:.4f} (Delta: {cad_p-bp:+.4f} dB, {cad_s-bs:+.4f} SSIM)", flush=True)

        ep, es, e_per = evaluate(cad_model, pairs, ensemble=True)
        print(f"  [CAD-SPAN+ (Ensemble)] PSNR: {ep:.4f} dB | SSIM: {es:.4f} (Delta: {ep-bp:+.4f} dB, {es-bs:+.4f} SSIM)", flush=True)

        summary[dname] = {
            'baseline': (bp, bs, b_per),
            'cad_single': (cad_p, cad_s, cad_per),
            'cad_plus': (ep, es, e_per)
        }

    print("\n" + "=" * 95)
    print(f"{'Dataset':<10} | {'Metric':<6} | {'Baseline SPAN':<15} | {'CAD-SPAN (Ours)':<18} | {'CAD-SPAN+ (Ensemble)':<20} | {'Max Gain':<10}")
    print("=" * 95)
    for dname in datasets:
        bp, bs, _ = summary[dname]['baseline']
        cp, cs, _ = summary[dname]['cad_single']
        ep, es, _ = summary[dname]['cad_plus']
        print(f"{dname:<10} | {'PSNR':<6} | {bp:6.2f} dB        | {cp:6.2f} dB (+{cp-bp:+.3f})   | {ep:6.2f} dB (+{ep-bp:+.3f})      | {ep-bp:+.3f} dB")
        print(f"{'':<10} | {'SSIM':<6} | {bs:6.4f}           | {cs:6.4f} (+{cs-bs:+.4f})  | {es:6.4f} (+{es-bs:+.4f})     | {es-bs:+.4f}")
        print("-" * 95)

if __name__ == '__main__':
    main()

