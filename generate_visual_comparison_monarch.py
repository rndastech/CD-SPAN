import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
PRETRAINED_DIR = os.path.join(ROOT_DIR, 'experiments', 'pretrained_models')
MPL_CACHE_DIR = os.path.join(ROOT_DIR, '.cache', 'matplotlib')
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ['MPLCONFIGDIR'] = MPL_CACHE_DIR

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn.functional as F

from basicsr.archs.span_arch import SPAN
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def resolve_ckpt_path():
    candidates = [
        os.path.join(PRETRAINED_DIR, 'cad_span_x4.pth'),
        os.path.join(PRETRAINED_DIR, 'spanx4_ch48.pth')
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]

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

        att = 0.5 * torch.tanh(tau * out3)
        c = out3.size(1)

        kx = sobel_x.repeat(c, 1, 1, 1)
        ky = sobel_y.repeat(c, 1, 1, 1)
        klap = lap_kernel.repeat(c, 1, 1, 1)

        gx = F.conv2d(out3, kx, padding=1, groups=c)
        gy = F.conv2d(out3, ky, padding=1, groups=c)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        lap_mag = torch.abs(F.conv2d(out3, klap, padding=1, groups=c))

        phi = torch.tanh(gs * grad_mag + gl * lap_mag)
        att = att * (1.0 + phi)
        out = (out3 + x) * att
        return out, out1, att
    return custom_forward

ckpt = torch.load(resolve_ckpt_path(), map_location='cpu')
state = ckpt.get('params_ema', ckpt.get('params'))

base_model = SPAN(3, 3, 48, 4)
base_model.load_state_dict(state)
base_model.eval()

cad_model = SPAN(3, 3, 48, 4)
cad_model.load_state_dict(state)
for blk in [cad_model.block_1, cad_model.block_2, cad_model.block_3, cad_model.block_4, cad_model.block_5, cad_model.block_6]:
    blk.forward = make_cad_forward(blk)
cad_model.eval()

# Select monarch.png from Set14
lq_p = os.path.join(DATASETS_DIR, 'Set14', 'LRbicx4', 'monarch.png')
gt_p = os.path.join(DATASETS_DIR, 'Set14', 'GTmod12', 'monarch.png')

lq = cv2.imread(lq_p) / 255.0
gt = cv2.imread(gt_p)
h, w = gt.shape[:2]
lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0)

# Bicubic
bic = cv2.resize(cv2.imread(lq_p), (w, h), interpolation=cv2.INTER_CUBIC)
p_bic = calculate_psnr(bic, gt, crop_border=4, test_y_channel=True)
s_bic = calculate_ssim(bic, gt, crop_border=4, test_y_channel=True)

# Baseline
with torch.no_grad():
    sr_base_t = base_model(lq_t)
sr_base = tensor2img(sr_base_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))[:h, :w]
p_base = calculate_psnr(sr_base, gt, crop_border=4, test_y_channel=True)
s_base = calculate_ssim(sr_base, gt, crop_border=4, test_y_channel=True)

# CAD-SPAN+
with torch.no_grad():
    sr_cad_t = forward_x8(cad_model, lq_t)
sr_cad = tensor2img(sr_cad_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))[:h, :w]
p_cad = calculate_psnr(sr_cad, gt, crop_border=4, test_y_channel=True)
s_cad = calculate_ssim(sr_cad, gt, crop_border=4, test_y_channel=True)

# Crop region (central wing pattern)
crop_y1, crop_y2 = 180, 270
crop_x1, crop_x2 = 240, 330

fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), dpi=300)

images = [
    ("Bicubic Baseline", bic, f"PSNR: {p_bic:.2f} dB / SSIM: {s_bic:.4f}"),
    ("SPAN Baseline (NTIRE '24)", sr_base, f"PSNR: {p_base:.2f} dB / SSIM: {s_base:.4f}"),
    ("CAD-SPAN+ (Ours)", sr_cad, f"PSNR: {p_cad:.2f} dB / SSIM: {s_cad:.4f} (+0.14 dB)"),
    ("Ground Truth (HR)", gt, "Reference")
]

for idx, (title, img_bgr, metric_text) in enumerate(images):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[idx].imshow(img_rgb)
    axes[idx].set_title(f"{title}\n{metric_text}", fontsize=11, fontweight='bold', pad=8)
    axes[idx].axis('off')

    rect = Rectangle((crop_x1, crop_y1), crop_x2 - crop_x1, crop_y2 - crop_y1,
                     linewidth=2.0, edgecolor='lime', facecolor='none')
    axes[idx].add_patch(rect)

    crop = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    sub_ax = axes[idx].inset_axes([0.55, 0.05, 0.42, 0.42])
    sub_ax.imshow(crop)
    sub_ax.axis('off')
    for spine in sub_ax.spines.values():
        spine.set_edgecolor('lime')
        spine.set_linewidth(2.0)

plt.tight_layout()
out_fig = os.path.join(ROOT_DIR, 'visual_comparison_monarch.png')
plt.savefig(out_fig, bbox_inches='tight', dpi=300)
print(f"Monarch comparison figure saved to {out_fig}")

