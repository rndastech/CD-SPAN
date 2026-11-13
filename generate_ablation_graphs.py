import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_CACHE_DIR = os.path.join(ROOT_DIR, '.cache', 'matplotlib')
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ['MPLCONFIGDIR'] = MPL_CACHE_DIR

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)

# --- Subplot (a): Gamma Sensitivity Sweep ---
gammas = [0.0, 0.001, 0.003, 0.005, 0.008, 0.010, 0.012, 0.015, 0.020]
psnr_gamma = [32.1880, 32.1886, 32.1898, 32.1907, 32.1912, 32.1913, 32.1908, 32.1895, 32.1866]

ax1 = axes[0, 0]
ax1.plot(gammas, psnr_gamma, marker='o', color='#1A73E8', lw=2.2, markersize=6, label='Set5 PSNR (dB)')
ax1.axhline(32.1880, color='#EA4335', linestyle='--', lw=1.5, label='Baseline SPAN (32.188 dB)')
ax1.scatter([0.010], [32.1913], color='#0F9D58', s=100, zorder=5, label=r'Optimal Peak ($\gamma=0.010$)')
ax1.set_title(r'(a) Spatial Modulation Gain ($\gamma$) Optimization Sweep', fontsize=10.5, fontweight='bold')
ax1.set_xlabel(r'Modulation Gain Parameter ($\gamma$)', fontsize=9.5)
ax1.set_ylabel('PSNR (dB)', fontsize=9.5)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=8.5, loc='lower left')

# --- Subplot (b): Temperature Sensitivity Sweep ---
taus = [0.48, 0.49, 0.50, 0.502, 0.505, 0.51, 0.52]
psnr_tau = [31.9773, 32.1168, 32.1880, 32.1922, 32.1861, 32.1723, 32.0625]

ax2 = axes[0, 1]
ax2.plot(taus, psnr_tau, marker='s', color='#9334E6', lw=2.2, markersize=6, label='Set5 PSNR (dB)')
ax2.axvline(0.500, color='#5F6368', linestyle=':', lw=1.4, label='Standard SPAN ($\tau=0.500$)')
ax2.scatter([0.502], [32.1922], color='#0F9D58', s=100, zorder=5, label='Calibrated Peak ($\tau=0.502$)')
ax2.set_title('(b) Origin-Symmetric Temperature ($\tau$) Sweep', fontsize=10.5, fontweight='bold')
ax2.set_xlabel('Temperature Parameter ($\tau$)', fontsize=9.5)
ax2.set_ylabel('PSNR (dB)', fontsize=9.5)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=8.5, loc='lower left')

# --- Subplot (c): Differential Operator Comparison ---
ops = ['Baseline\nSPAN', 'AvgPool\nDiff', 'Pure\nLaplacian', 'Pure\nSobel', 'Dual-Op\nCAD-SA (Ours)']
psnr_ops = [32.1880, 32.1913, 32.1921, 32.1920, 32.1923]
colors = ['#BDC1C6', '#FEBC2E', '#34A853', '#4285F4', '#137333']

ax3 = axes[1, 0]
bars = ax3.bar(ops, [p - 32.1800 for p in psnr_ops], bottom=32.1800, color=colors, width=0.55, edgecolor='#202124', lw=1.0)
ax3.set_title('(c) Differential Operator Ablation (Set5)', fontsize=10.5, fontweight='bold')
ax3.set_ylabel('PSNR (dB)', fontsize=9.5)
ax3.set_ylim(32.1800, 32.1945)
ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
for bar, p in zip(bars, psnr_ops):
    ax3.text(bar.get_x() + bar.get_width()/2, p + 0.0003, f'{p:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Subplot (d): Per-Image PSNR Gain (dB) ---
images = ['butterfly', 'monarch', 'bird', 'ppt3', 'lenna', 'zebra', 'woman', 'flowers']
gains = [+0.160, +0.140, +0.130, +0.110, +0.090, +0.080, +0.080, +0.068]
bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(images)))

ax4 = axes[1, 1]
bars4 = ax4.barh(images[::-1], gains[::-1], color=bar_colors, edgecolor='#202124', lw=0.9, height=0.6)
ax4.set_title('(d) Per-Image Absolute PSNR Gain over Baseline', fontsize=10.5, fontweight='bold')
ax4.set_xlabel(r'$\Delta$ PSNR Gain (dB)', fontsize=9.5)
ax4.grid(True, axis='x', linestyle=':', alpha=0.6)
for bar, g in zip(bars4, gains[::-1]):
    ax4.text(g + 0.004, bar.get_y() + bar.get_height()/2, f'+{g:.3f} dB', ha='left', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
out_ablation = os.path.join(ROOT_DIR, 'ablation_curves.png')
plt.savefig(out_ablation, bbox_inches='tight', dpi=300)
print(f"Ablation graphs saved to {out_ablation}")

