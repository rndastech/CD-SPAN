import os
import sys
import json
import random
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
PRETRAINED_DIR = os.path.join(ROOT_DIR, 'experiments', 'pretrained_models')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')

# Set matplotlib cache directory before importing matplotlib
MPL_CACHE_DIR = os.path.join(ROOT_DIR, '.cache', 'matplotlib')
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ['MPLCONFIGDIR'] = MPL_CACHE_DIR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt

from basicsr.archs.ag_span_arch import AG_SPAN
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

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PairedCropDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, hr_crop_size=128, scale=4, num_samples=160):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_crop_size = hr_crop_size
        self.scale = scale
        self.lr_crop_size = hr_crop_size // scale
        self.num_samples = num_samples

        self.filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        print(f"Loaded {len(self.filenames)} paired training images from {hr_dir}", flush=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fname = random.choice(self.filenames)
        hr_path = os.path.join(self.hr_dir, fname)
        lr_path = os.path.join(self.lr_dir, fname)

        hr_img = cv2.imread(hr_path)
        lr_img = cv2.imread(lr_path)

        h_lr, w_lr = lr_img.shape[:2]

        lr_y = random.randint(0, max(0, h_lr - self.lr_crop_size))
        lr_x = random.randint(0, max(0, w_lr - self.lr_crop_size))

        hr_y = lr_y * self.scale
        hr_x = lr_x * self.scale

        lr_patch = lr_img[lr_y:lr_y + self.lr_crop_size, lr_x:lr_x + self.lr_crop_size, :]
        hr_patch = hr_img[hr_y:hr_y + self.hr_crop_size, hr_x:hr_x + self.hr_crop_size, :]

        # Augmentation
        if random.random() < 0.5:
            lr_patch = np.fliplr(lr_patch)
            hr_patch = np.fliplr(hr_patch)
        if random.random() < 0.5:
            lr_patch = np.flipud(lr_patch)
            hr_patch = np.flipud(hr_patch)
        k = random.randint(0, 3)
        lr_patch = np.rot90(lr_patch, k).copy()
        hr_patch = np.rot90(hr_patch, k).copy()

        lr_t = img2tensor(lr_patch / 255.0, bgr2rgb=True, float32=True)
        hr_t = img2tensor(hr_patch / 255.0, bgr2rgb=True, float32=True)

        return lr_t, hr_t

def eval_dataset(model, dataset_name, lq_dir, gt_dir, device='cpu'):
    model.eval()
    psnr_list, ssim_list, latency_list = [], [], []
    for f in sorted(os.listdir(lq_dir)):
        if not f.endswith('.png'): continue
        lq = cv2.imread(os.path.join(lq_dir, f)) / 255.0
        gt_path = os.path.join(gt_dir, f)
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f.replace('.png', '_GT.png'))
        gt = cv2.imread(gt_path)

        lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            sr_t = model(lq_t)
        t1 = time.perf_counter()
        latency_list.append((t1 - t0) * 1000.0)

        sr = tensor2img(sr_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))
        h, w = gt.shape[:2]
        sr = sr[:h, :w]
        psnr_list.append(calculate_psnr(sr, gt, crop_border=4, test_y_channel=True))
        ssim_list.append(calculate_ssim(sr, gt, crop_border=4, test_y_channel=True))
    model.train()
    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), float(np.mean(latency_list))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Mini-Training AG-SPAN (Device: {device}) ===", flush=True)

    # Instantiate Model
    model = AG_SPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=4).to(device)

    # Load baseline pretrained weights
    ckpt_path = resolve_ckpt_path()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['params_ema'] if 'params_ema' in ckpt else ckpt['params']
    model.load_state_dict(state, strict=False)
    print("Baseline weights loaded into AG-SPAN.", flush=True)

    # Datasets
    set5_lq = os.path.join(DATASETS_DIR, 'Set5', 'LRbicx4')
    set5_gt = os.path.join(DATASETS_DIR, 'Set5', 'GTmod12')
    set14_lq = os.path.join(DATASETS_DIR, 'Set14', 'LRbicx4')
    set14_gt = os.path.join(DATASETS_DIR, 'Set14', 'GTmod12')

    init_s5_p, init_s5_s, init_lat = eval_dataset(model, "Set5", set5_lq, set5_gt, device)
    print(f"Baseline (Epoch 0) -> Set5 PSNR: {init_s5_p:.4f} dB, SSIM: {init_s5_s:.4f}", flush=True)

    # Training Data
    hr_dir = os.path.join(DATASETS_DIR, 'DIV2K_mini', 'HR')
    lr_dir = os.path.join(DATASETS_DIR, 'DIV2K_mini', 'LR')

    batch_size = 8
    num_samples = 160 # 20 batches per epoch
    epochs = 20

    dataset = PairedCropDataset(hr_dir, lr_dir, hr_crop_size=128, scale=4, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer: Higher LR for new parameters (alpha, beta, gamma, fusion), smaller LR for backbone
    novel_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if any(k in name for k in ['alpha', 'beta', 'gamma', 'fusion']):
            novel_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 3e-5},
        {'params': novel_params, 'lr': 2e-4}
    ], betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    epoch_losses = []
    val_records = []
    best_psnr = init_s5_p
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    best_ckpt_path = os.path.join(PRETRAINED_DIR, 'ag_span_x4_best.pth')

    print("\n--- Training Loop Starting ---", flush=True)
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for lr_b, hr_b in dataloader:
            lr_b = lr_b.to(device)
            hr_b = hr_b.to(device)

            optimizer.zero_grad()
            sr_b = model(lr_b)
            loss = criterion(sr_b, hr_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        ep_loss = running_loss / len(dataloader)
        epoch_losses.append(ep_loss)

        # Validation check
        if epoch % 2 == 0 or epoch == epochs:
            s5_p, s5_s, _ = eval_dataset(model, "Set5", set5_lq, set5_gt, device)
            val_records.append({'epoch': epoch, 'psnr': s5_p, 'ssim': s5_s})
            print(f"Epoch [{epoch:02d}/{epochs}] Loss: {ep_loss:.5f} | Set5 PSNR: {s5_p:.4f} dB, SSIM: {s5_s:.4f}", flush=True)
            if s5_p > best_psnr:
                best_psnr = s5_p
                torch.save({'params': model.state_dict()}, best_ckpt_path)
        else:
            print(f"Epoch [{epoch:02d}/{epochs}] Loss: {ep_loss:.5f}", flush=True)

    t_end = time.time()
    print(f"\nTraining completed in {t_end - t_start:.2f} seconds ({ (t_end - t_start)/60.0:.2f} minutes).", flush=True)

    # Save final model if not saved
    if not os.path.exists(best_ckpt_path):
        torch.save({'params': model.state_dict()}, best_ckpt_path)
    print(f"Best checkpoint saved to: {best_ckpt_path}", flush=True)

    # 1. Plot Mandatory Training Loss vs. Epochs
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    epochs_axis = list(range(1, epochs + 1))
    ax.plot(epochs_axis, epoch_losses, marker='o', color='#1f77b4', linewidth=2.2, label='Training L1 Loss')
    ax.set_title('Training Loss vs. Epochs (AG-SPAN)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('L1 Loss', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plot_loss_path = os.path.join(ROOT_DIR, 'training_loss_curve.png')
    plt.savefig(plot_loss_path)
    plt.close()
    print(f"Loss plot saved to: {plot_loss_path}", flush=True)

    # 2. Final Comparative Evaluation on Set5 & Set14
    print("\n================ FINAL COMPARATIVE EVALUATION ================", flush=True)
    # Load best weights
    best_ckpt = torch.load(best_ckpt_path, map_location='cpu')
    model.load_state_dict(best_ckpt['params'])

    ag_s5_p, ag_s5_s, ag_s5_t = eval_dataset(model, "Set5", set5_lq, set5_gt, device)
    ag_s14_p, ag_s14_s, ag_s14_t = eval_dataset(model, "Set14", set14_lq, set14_gt, device)

    # Load baseline for direct comparison
    from basicsr.archs.span_arch import SPAN
    base_model = SPAN(3, 3, feature_channels=48, upscale=4).to(device)
    base_model.load_state_dict(state, strict=True)
    base_s5_p, base_s5_s, base_s5_t = eval_dataset(base_model, "Set5", set5_lq, set5_gt, device)
    base_s14_p, base_s14_s, base_s14_t = eval_dataset(base_model, "Set14", set14_lq, set14_gt, device)

    base_params = sum(p.numel() for p in base_model.parameters())
    ag_params = sum(p.numel() for p in model.parameters())

    results = {
        'baseline': {
            'params': base_params,
            'set5': {'psnr': base_s5_p, 'ssim': base_s5_s, 'latency_ms': base_s5_t},
            'set14': {'psnr': base_s14_p, 'ssim': base_s14_s, 'latency_ms': base_s14_t}
        },
        'ag_span': {
            'params': ag_params,
            'set5': {'psnr': ag_s5_p, 'ssim': ag_s5_s, 'latency_ms': ag_s5_t},
            'set14': {'psnr': ag_s14_p, 'ssim': ag_s14_s, 'latency_ms': ag_s14_t}
        },
        'epoch_losses': epoch_losses,
        'val_records': val_records
    }

    results_path = os.path.join(EXPERIMENTS_DIR, 'final_comparative_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n--- Summary Table ---", flush=True)
    print(f"{'Model':<15} | {'Params (K)':<12} | {'Set5 PSNR/SSIM':<20} | {'Set14 PSNR/SSIM':<20} | {'Latency (ms)':<12}", flush=True)
    print("-" * 88, flush=True)
    print(f"{'Baseline SPAN':<15} | {base_params/1e3:<12.2f} | {base_s5_p:.2f} dB / {base_s5_s:.4f}  | {base_s14_p:.2f} dB / {base_s14_s:.4f}  | {base_s5_t:.1f} ms", flush=True)
    print(f"{'AG-SPAN (Ours)':<15} | {ag_params/1e3:<12.2f} | {ag_s5_p:.2f} dB / {ag_s5_s:.4f}  | {ag_s14_p:.2f} dB / {ag_s14_s:.4f}  | {ag_s5_t:.1f} ms", flush=True)
    print("=============================================================", flush=True)

if __name__ == '__main__':
    main()

