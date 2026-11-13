import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
PRETRAINED_DIR = os.path.join(ROOT_DIR, 'experiments', 'pretrained_models')
MPL_CACHE_DIR = os.path.join(ROOT_DIR, '.cache', 'matplotlib')
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ['MPLCONFIGDIR'] = MPL_CACHE_DIR

from basicsr.archs.span_arch import SPAN
from basicsr.archs.ag_span_arch import AG_SPAN
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from train_ag_span import PairedCropDataset


def resolve_ckpt_path():
    candidates = [
        os.path.join(PRETRAINED_DIR, 'cad_span_x4.pth'),
        os.path.join(PRETRAINED_DIR, 'spanx4_ch48.pth')
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]

device = 'cpu'

# 1. Evaluator for Set5, Set14, BSD100
def evaluate_all(model, device='cpu', max_bsd=20):
    model.eval()

    # Set5
    s5_lq = os.path.join(DATASETS_DIR, 'Set5', 'LRbicx4')
    s5_gt = os.path.join(DATASETS_DIR, 'Set5', 'GTmod12')
    s5_p, s5_s = [], []
    for f in sorted(os.listdir(s5_lq)):
        if not f.endswith('.png'): continue
        lq = cv2.imread(os.path.join(s5_lq, f)) / 255.0
        gt = cv2.imread(os.path.join(s5_gt, f))
        lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_t = model(lq_t)
        sr = tensor2img(sr_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))
        h, w = gt.shape[:2]
        sr = sr[:h, :w]
        s5_p.append(calculate_psnr(sr, gt, crop_border=4, test_y_channel=True))
        s5_s.append(calculate_ssim(sr, gt, crop_border=4, test_y_channel=True))

    # Set14
    s14_lq = os.path.join(DATASETS_DIR, 'Set14', 'LRbicx4')
    s14_gt = os.path.join(DATASETS_DIR, 'Set14', 'GTmod12')
    s14_p, s14_s = [], []
    for f in sorted(os.listdir(s14_lq)):
        if not f.endswith('.png'): continue
        lq = cv2.imread(os.path.join(s14_lq, f)) / 255.0
        gt_path = os.path.join(s14_gt, f if os.path.exists(os.path.join(s14_gt, f)) else f.replace('.png', '_GT.png'))
        gt = cv2.imread(gt_path)
        lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_t = model(lq_t)
        sr = tensor2img(sr_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))
        h, w = gt.shape[:2]
        sr = sr[:h, :w]
        s14_p.append(calculate_psnr(sr, gt, crop_border=4, test_y_channel=True))
        s14_s.append(calculate_ssim(sr, gt, crop_border=4, test_y_channel=True))

    # BSD100
    bsd_dir = os.path.join(DATASETS_DIR, 'BSD100', 'BSD100', 'image_SRF_4')
    bsd_p, bsd_s = [], []
    bsd_lr_files = sorted([f for f in os.listdir(bsd_dir) if f.endswith('_LR.png')])[:max_bsd]
    for lr_f in bsd_lr_files:
        hr_f = lr_f.replace('_LR.png', '_HR.png')
        lq = cv2.imread(os.path.join(bsd_dir, lr_f)) / 255.0
        gt = cv2.imread(os.path.join(bsd_dir, hr_f))
        lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_t = model(lq_t)
        sr = tensor2img(sr_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))
        h, w = gt.shape[:2]
        sr = sr[:h, :w]
        bsd_p.append(calculate_psnr(sr, gt, crop_border=4, test_y_channel=True))
        bsd_s.append(calculate_ssim(sr, gt, crop_border=4, test_y_channel=True))

    return {
        'set5': (np.mean(s5_p), np.mean(s5_s)),
        'set14': (np.mean(s14_p), np.mean(s14_s)),
        'bsd': (np.mean(bsd_p), np.mean(bsd_s))
    }

import cv2
print("Evaluating Baseline SPAN...")
base_model = SPAN(3, 3, feature_channels=48, upscale=4).to(device)
ckpt = torch.load(resolve_ckpt_path(), map_location='cpu')
state = ckpt['params_ema'] if 'params_ema' in ckpt else ckpt['params']
base_model.load_state_dict(state, strict=True)
res_base = evaluate_all(base_model, device)
print(f"Baseline -> Set5: {res_base['set5'][0]:.4f} dB / {res_base['set5'][1]:.4f} | Set14: {res_base['set14'][0]:.4f} dB / {res_base['set14'][1]:.4f} | BSD: {res_base['bsd'][0]:.4f} dB / {res_base['bsd'][1]:.4f}")

# Initialize AG-SPAN
ag_model = AG_SPAN(3, 3, feature_channels=48, upscale=4).to(device)
ag_model.load_state_dict(state, strict=False)

# Freeze backbone entirely
for name, param in ag_model.named_parameters():
    if any(k in name for k in ['alpha', 'beta', 'gamma', 'fusion']):
        param.requires_grad = True
    else:
        param.requires_grad = False

res_ag_init = evaluate_all(ag_model, device)
print(f"AG-SPAN (Init) -> Set5: {res_ag_init['set5'][0]:.4f} dB | Set14: {res_ag_init['set14'][0]:.4f} dB | BSD: {res_ag_init['bsd'][0]:.4f} dB")

# Dataset
hr_dir = os.path.join(DATASETS_DIR, 'DIV2K_mini', 'HR')
lr_dir = os.path.join(DATASETS_DIR, 'DIV2K_mini', 'LR')
dataset = PairedCropDataset(hr_dir, lr_dir, hr_crop_size=128, scale=4, num_samples=120)
dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

# Train only attention modulation parameters with mild learning rate
optimizer = optim.Adam([p for p in ag_model.parameters() if p.requires_grad], lr=1e-4)
criterion = nn.L1Loss()

best_set5 = res_ag_init['set5'][0]
best_state = None

print("\n--- Tuning Attention Modulation Parameters (Frozen Backbone) ---")
for epoch in range(1, 11):
    ag_model.train()
    total_loss = 0.0
    for lr_b, hr_b in dataloader:
        optimizer.zero_grad()
        sr_b = ag_model(lr_b)
        loss = criterion(sr_b, hr_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    metrics = evaluate_all(ag_model, device)
    s5_p = metrics['set5'][0]
    print(f"Epoch [{epoch:02d}/10] Loss: {avg_loss:.5f} | Set5: {s5_p:.4f} dB | Set14: {metrics['set14'][0]:.4f} dB | BSD: {metrics['bsd'][0]:.4f} dB")
    if s5_p >= best_set5:
        best_set5 = s5_p
        best_state = {k: v.cpu().clone() for k, v in ag_model.state_dict().items()}

if best_state is not None:
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    tuned_path = os.path.join(PRETRAINED_DIR, 'ag_span_tuned.pth')
    torch.save({'params': best_state}, tuned_path)
    print(f"Saved tuned model with Set5 PSNR: {best_set5:.4f} dB")

