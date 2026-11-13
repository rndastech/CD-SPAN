import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_CACHE_DIR = os.path.join(ROOT_DIR, '.cache', 'matplotlib')
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ['MPLCONFIGDIR'] = MPL_CACHE_DIR

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# Set font family and styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize=(16, 9.4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.4)
ax.axis('off')

# Modern Academic Color Palette
C_BORDER_PANEL = '#CBD5E1' # Slate 300
C_TEXT_MAIN = '#0F172A'    # Slate 900
C_TEXT_MUTED = '#475569'   # Slate 600

C_CONV = '#EEF2FF'         # Indigo 50
C_CONV_BORDER = '#6366F1'  # Indigo 500
C_ATTN = '#FFF1F2'         # Rose 50
C_ATTN_BORDER = '#F43F5E'  # Rose 500
C_KERNEL = '#FFFBEB'       # Amber 50
C_KERNEL_BORDER = '#F59E0B'# Amber 500
C_TENSOR = '#F8FAFC'       # Slate 50
C_TENSOR_BORDER = '#64748B'# Slate 500
C_OP = '#E0F2FE'           # Sky 100
C_OP_BORDER = '#0284C7'    # Sky 600

# Utility: Draw rounded panel
def draw_panel(ax, x, y, w, h, title, subtitle=''):
    panel = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                           linewidth=1.4, edgecolor=C_BORDER_PANEL, facecolor='#F8FAFC', zorder=1)
    ax.add_patch(panel)
    # Header bar
    hbar = FancyBboxPatch((x, y + h - 0.48), w, 0.48, boxstyle="round,pad=0.02,rounding_size=0.12",
                          linewidth=0, facecolor='#1E293B', zorder=2)
    ax.add_patch(hbar)
    ax.text(x + 0.25, y + h - 0.24, title, ha='left', va='center',
            fontsize=10.5, fontweight='bold', color='#FFFFFF', zorder=3)
    if subtitle:
        ax.text(x + w - 0.25, y + h - 0.24, subtitle, ha='right', va='center',
                fontsize=8, fontfamily='monospace', color='#94A3B8', zorder=3)

# Utility: Draw functional block
def draw_block(ax, x, y, w, h, title, sub='', dim='', color='#FFFFFF', border='#94A3B8', lw=1.3, title_size=8.5, sub_size=7):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                         linewidth=lw, edgecolor=border, facecolor=color, zorder=3)
    ax.add_patch(box)
    cy = y + h/2
    if dim and sub:
        ax.text(x + w/2, cy + 0.22, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy + 0.02, sub, ha='center', va='center', fontsize=sub_size, color=C_TEXT_MUTED, zorder=4)
        ax.text(x + w/2, cy - 0.22, dim, ha='center', va='center', fontsize=6.6, fontfamily='monospace', color='#2563EB', zorder=4)
    elif sub:
        ax.text(x + w/2, cy + 0.12, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy - 0.14, sub, ha='center', va='center', fontsize=sub_size, color=C_TEXT_MUTED, zorder=4)
    elif dim:
        ax.text(x + w/2, cy + 0.12, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy - 0.14, dim, ha='center', va='center', fontsize=6.8, fontfamily='monospace', color='#2563EB', zorder=4)
    else:
        ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)

# Utility: Draw mathematical operator node
def draw_op(ax, x, y, symbol, r=0.18, color=C_OP, border=C_OP_BORDER, text_color='#0369A1', size=11):
    circle = Circle((x, y), r, facecolor=color, edgecolor=border, linewidth=1.4, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=size, fontweight='bold', color=text_color, zorder=6)

# Utility: Orthogonal arrow
def draw_conn(ax, x1, y1, x2, y2, color='#475569', lw=1.3, text='', text_pos='top', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1, shrinkB=1), zorder=4)
    if text:
        tx = (x1 + x2)/2
        ty = (y1 + y2)/2 + (0.12 if text_pos=='top' else -0.14)
        ax.text(tx, ty, text, ha='center', va='center', fontsize=6.8, color='#334155', fontweight='bold', zorder=5)

# Utility: Draw a 3x3 kernel matrix grid
def draw_kernel_grid(ax, x, y, matrix, label, cell_size=0.18, font_size=5.8):
    ax.text(x + 1.5 * cell_size, y + 3 * cell_size + 0.08, label, ha='center', va='bottom',
            fontsize=6.5, fontweight='bold', color='#B45309')
    for r in range(3):
        for c in range(3):
            val = matrix[r][c]
            bg = '#FEF3C7' if val != 0 else '#FFFFFF'
            rect = patches.Rectangle((x + c * cell_size, y + (2 - r) * cell_size), cell_size, cell_size,
                                     facecolor=bg, edgecolor='#F59E0B', lw=0.6, zorder=4)
            ax.add_patch(rect)
            ax.text(x + (c + 0.5) * cell_size, y + (2.5 - r) * cell_size, str(val),
                    ha='center', va='center', fontsize=font_size, fontweight='bold', color='#78350F', zorder=5)

# =========================================================================
# PANEL (A): MACRO-ARCHITECTURE OF CAD-SPAN
# =========================================================================
draw_panel(ax, 0.2, 5.65, 15.6, 3.55,
           '(a) Macro-Architecture of Proposed CAD-SPAN Framework',
           'Cascade: 6 CAD-SPAB Blocks · Multi-Scale Tap Fusion · PixelShuffle x4')

# 1. Input Tensor
draw_block(ax, 0.45, 6.9, 1.25, 1.3, 'Low-Res Input', r'$I_{LR} \in \mathbb{R}^{3 \times H \times W}$',
           '[B, 3, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER)

# 2. Shallow Feature Extractor
draw_conn(ax, 1.7, 7.55, 2.15, 7.55)
draw_block(ax, 2.15, 6.9, 1.35, 1.3, 'Shallow Conv', 'Conv3XC (3 to 48)\nReparameterizable',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER)

# Arrow to Backbone
draw_conn(ax, 3.5, 7.55, 3.95, 7.55, text='$F_0$')

# 3. Cascaded CAD-SPAB Blocks (6 blocks)
block_xs = []
for i in range(1, 7):
    bx = 3.95 + (i - 1) * 1.15
    block_xs.append(bx)
    draw_block(ax, bx, 7.0, 0.95, 1.1, f'CAD-SPAB\n#{i}', f'$F_{{b{i}}}$',
               '[B, 48, H, W]', color=C_ATTN, border=C_ATTN_BORDER, title_size=7.5, sub_size=7)
    if i < 6:
        draw_conn(ax, bx + 0.95, 7.55, bx + 1.15, 7.55)

# 4. Multi-Stage Hierarchical Aggregation (Taps from F0, Fb1, Fb5_2, Fb6)
# F0 Tap
ax.plot([3.7, 3.7, 11.2], [7.55, 8.55, 8.55], color='#4F46E5', lw=1.3, linestyle='--', zorder=3)
ax.plot([11.2, 11.2], [8.55, 7.9], color='#4F46E5', lw=1.3, linestyle='--', zorder=3)
ax.text(5.5, 8.65, r'Shallow Tap $F_0$', fontsize=7, color='#4F46E5', fontweight='bold')

# Fb1 Tap
ax.plot([block_xs[0] + 0.47, block_xs[0] + 0.47, 11.0], [7.0, 6.1, 6.1], color='#059669', lw=1.2, linestyle=':', zorder=3)
ax.plot([11.0, 11.0], [6.1, 7.1], color='#059669', lw=1.2, linestyle=':', zorder=3)
ax.text(6.2, 5.95, r'Hierarchical Stage Tap $F_{b1}$', fontsize=7, color='#059669', fontweight='bold')

# Fb5_2 Tap
ax.plot([block_xs[4] + 0.47, block_xs[4] + 0.47, 11.4], [7.0, 6.4, 6.4], color='#D97706', lw=1.2, linestyle=':', zorder=3)
ax.plot([11.4, 11.4], [6.4, 7.1], color='#D97706', lw=1.2, linestyle=':', zorder=3)
ax.text(9.2, 6.5, r'Intermediate Tap $F_{b5\_2}$', fontsize=7, color='#D97706', fontweight='bold')

# Fb6 connection
draw_conn(ax, block_xs[5] + 0.95, 7.55, 11.0, 7.55, text='$F_{b6}$')

# Concat Box
draw_block(ax, 11.0, 7.05, 1.45, 1.0, 'Hierarchical Concat', r'$[F_0, F_{b6}, F_{b1}, F_{b5\_2}]$',
           '[B, 192, H, W]', color='#FEE2E2', border='#EF4444', title_size=7.5, sub_size=6.8)

# 5. Fusion Conv (1x1)
draw_conn(ax, 12.45, 7.55, 12.75, 7.55)
draw_block(ax, 12.75, 7.05, 1.15, 1.0, r'Fusion $1\times 1$ Conv', 'Conv + SiLU',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=7.5, sub_size=7)

# 6. Global Residual Addition
draw_op(ax, 14.15, 7.55, '+', r=0.18)
draw_conn(ax, 13.9, 7.55, 13.97, 7.55)
# Long skip from F0 to +
ax.plot([3.7, 3.7, 14.15, 14.15], [7.55, 8.85, 8.85, 7.73], color='#6366F1', lw=1.4, linestyle='-.', zorder=3)
ax.text(8.5, 8.95, r'Global Feature Residual Skip ($+ F_0$)', fontsize=7.5, color='#4338CA', fontweight='bold', ha='center')

# 7. PixelShuffle Upsampler
draw_conn(ax, 14.33, 7.55, 14.65, 7.55)
draw_block(ax, 14.65, 6.9, 1.05, 1.3, 'PixelShuffle\nUpsampler', r'Conv3x3 ($48\to 48$)' + '\nShuffle (r=4)',
           '[B, 3, 4H, 4W]', color=C_OP, border=C_OP_BORDER, title_size=7.2, sub_size=6.5)

# Global Bicubic Skip Connection
draw_op(ax, 15.35, 6.15, '+', r=0.16)
ax.plot([1.07, 1.07, 15.35, 15.35], [6.9, 5.85, 5.85, 5.99], color='#0D9488', lw=1.3, linestyle='--', zorder=3)
ax.text(8.0, 5.72, r'Global Bicubic Residual Interpolation Skip: $+ \mathrm{Bicubic}(I_{LR})$', fontsize=7.5, color='#0F766E', fontweight='bold', ha='center')
ax.plot([15.17, 15.35], [6.9, 6.31], color='#475569', lw=1.2, zorder=3)

# Output Tensor
draw_conn(ax, 15.35, 6.15, 15.35, 6.35, style='<-')
ax.plot([15.35, 15.35], [6.15, 6.0], color='#475569', lw=1.3)
ax.text(15.35, 6.55, r'$I_{SR} \in \mathbb{R}^{3 \times 4H \times 4W}$', fontsize=8, fontweight='bold', color='#15803D', ha='center')

# =========================================================================
# PANEL (B): MICRO-ARCHITECTURE OF CAD-SPAB
# =========================================================================
draw_panel(ax, 0.2, 0.2, 15.6, 5.25,
           '(b) Micro-Architecture: Curvature-Aware Directional Swift Parameter-free Attention Block (CAD-SPAB)',
           'Dual-Differential Geometric Prior · Temperature Calibration · Exact Origin-Symmetry')

# 1. Block Input
draw_block(ax, 0.45, 2.7, 1.15, 1.1, 'Block Input', r'$X_{in}$',
           '[B, 48, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER)

# 2. Three Consecutive Conv3XC Layers with SiLU
draw_conn(ax, 1.6, 3.25, 1.95, 3.25)
draw_block(ax, 1.95, 2.6, 2.05, 1.3, '3x Conv3XC Cascade', 'Conv3XC + SiLU (#1)\nConv3XC + SiLU (#2)\nConv3XC + SiLU (#3)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=8, sub_size=6.8)

# Output of 3x convs is H
draw_conn(ax, 4.0, 3.25, 4.4, 3.25)
draw_block(ax, 4.4, 2.85, 0.85, 0.8, r'Feature $H$', r'$H \in \mathbb{R}^{C \times H \times W}$',
           '[B, 48, H, W]', color='#FEF3C7', border='#D97706', title_size=8, sub_size=6.5)

# Forking Point from H into 3 Branches
ax.plot([5.25, 5.6], [3.25, 3.25], color='#475569', lw=1.4)
ax.plot([5.6, 5.6], [1.15, 4.35], color='#475569', lw=1.4)

# -------------------------------------------------------------------------
# BRANCH 1: Temperature-Calibrated Origin-Symmetric Gating (Top)
# -------------------------------------------------------------------------
draw_conn(ax, 5.6, 4.35, 6.05, 4.35)
draw_block(ax, 6.05, 3.85, 3.75, 1.0,
           '1. Calibrated Origin-Symmetric Gate',
           r'$A_{sym}(H) = \frac{1}{2}\tanh(\tau \cdot H)$  with  $\tau^* = 0.502$',
           r'Analytically equivalent: $\sigma(x)-0.5 \equiv \frac{1}{2}\tanh(x/2)$',
           color=C_ATTN, border=C_ATTN_BORDER, title_size=8, sub_size=7.2)

# -------------------------------------------------------------------------
# BRANCH 2: First-Order Directional Sobel Gradient Flux (Middle)
# -------------------------------------------------------------------------
draw_conn(ax, 5.6, 2.75, 6.05, 2.75)
draw_block(ax, 6.05, 2.15, 3.75, 1.2,
           '2. Directional Gradient Flux Operator',
           r'$G_x = K_x * H, \quad G_y = K_y * H, \quad \|\nabla H\|_2 = \sqrt{G_x^2 + G_y^2 + \epsilon}$',
           'Captures Horizontal & Vertical Flux Boundaries',
           color=C_KERNEL, border=C_KERNEL_BORDER, title_size=8, sub_size=7.0)

# Sobel Kernel Visual Grids
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
draw_kernel_grid(ax, 8.4, 2.22, sobel_x, r'$K_x$', cell_size=0.15, font_size=5.2)
draw_kernel_grid(ax, 9.15, 2.22, sobel_y, r'$K_y$', cell_size=0.15, font_size=5.2)

# -------------------------------------------------------------------------
# BRANCH 3: Second-Order Isotropic Laplacian Curvature (Bottom)
# -------------------------------------------------------------------------
draw_conn(ax, 5.6, 1.15, 6.05, 1.15)
draw_block(ax, 6.05, 0.65, 3.75, 1.05,
           '3. Isotropic Curvature Operator',
           r'$\Delta H = |\nabla^2 H| = |K_L * H|$   (Inflection & Corner Prior)',
           'Discrete 8-Neighbor Isotropic Stencil',
           color='#F0FDF4', border='#16A34A', title_size=8, sub_size=7.0)

# Laplacian Kernel Visual Grid
laplacian_k = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
draw_kernel_grid(ax, 8.75, 0.72, laplacian_k, r'$K_L$', cell_size=0.15, font_size=5.2)

# -------------------------------------------------------------------------
# DUAL-DIFFERENTIAL SPATIAL MODULATOR FUSION
# -------------------------------------------------------------------------
# Merge Branch 2 & Branch 3 into Dual Differential Prior Phi(H)
ax.plot([9.8, 10.15], [2.75, 1.95], color='#475569', lw=1.3)
ax.plot([9.8, 10.15], [1.15, 1.95], color='#475569', lw=1.3)
draw_conn(ax, 10.15, 1.95, 10.45, 1.95)

draw_block(ax, 10.45, 1.3, 2.5, 1.3,
           r'Dual-Differential Prior $\Phi(H)$',
           r'$\Phi(H) = \tanh(\gamma_s \|\nabla H\|_2 + \gamma_l \Delta H)$' + '\n'
           r'$\gamma_s^* = 0.008$ (Sobel Flux Gain)' + '\n'
           r'$\gamma_l^* = 0.001$ (Laplacian Gain)',
           '[B, 48, H, W]', color='#F1F5F9', border='#475569', title_size=8, sub_size=6.8)

# -------------------------------------------------------------------------
# CURVATURE-MODULATED ATTENTION GATE SYNTHESIS
# -------------------------------------------------------------------------
# Shift by 1: (1 + Phi(H))
draw_op(ax, 13.35, 1.95, '+', r=0.18)
draw_conn(ax, 12.95, 1.95, 13.17, 1.95)
ax.text(13.35, 1.55, r'$\mathbf{1}$', fontsize=9, fontweight='bold', color='#2563EB', ha='center')
ax.plot([13.35, 13.35], [1.72, 1.77], color='#2563EB', lw=1.3)

# Hadamard Modulation with Asym: A_tilde = Asym * (1 + Phi)
draw_op(ax, 13.35, 4.35, r'$\odot$', r=0.22, color='#FCE7F3', border='#EC4899', text_color='#BE185D', size=12)
draw_conn(ax, 9.8, 4.35, 13.13, 4.35, text=r'$A_{sym}$')
draw_conn(ax, 13.35, 2.13, 13.35, 4.13, text=r'$1 + \Phi(H)$', text_pos='right')

# Attention Gate Box
draw_block(ax, 13.9, 3.85, 1.75, 1.0,
           r'Modulated Gate $\widetilde{A}$',
           r'$\widetilde{A} = A_{sym} \odot (\mathbf{1} + \Phi(H))$' + '\n'
           'Zero-parameter Edge Prior',
           '[B, 48, H, W]', color='#FDF2F8', border='#DB2777', title_size=7.8, sub_size=6.8)
draw_conn(ax, 13.57, 4.35, 13.9, 4.35)

# -------------------------------------------------------------------------
# GATED RESIDUAL OUTPUT: O = (H + Xin) * A_tilde
# -------------------------------------------------------------------------
# Local Residual Addition: (H + Xin)
draw_op(ax, 5.0, 0.45, '+', r=0.18)
# Xin skip to local +
ax.plot([1.02, 1.02, 5.0, 5.0], [2.7, 0.45, 0.45, 0.27], color='#2563EB', lw=1.3, linestyle='--', zorder=3)
ax.text(3.0, 0.3, r'Local Input Residual $X_{in}$', fontsize=7, color='#1D4ED8', fontweight='bold')
# H to local +
ax.plot([4.82, 4.82, 5.0], [2.85, 1.8, 0.63], color='#D97706', lw=1.2, linestyle=':', zorder=3)

# Final Hadamard Product O = (H + Xin) * A_tilde
draw_op(ax, 14.75, 2.6, r'$\odot$', r=0.24, color='#DCFCE7', border='#16A34A', text_color='#15803D', size=13)
# (H + Xin) path to final Hadamard
ax.plot([5.18, 14.75], [0.45, 0.45], color='#2563EB', lw=1.3, zorder=3)
draw_conn(ax, 14.75, 0.45, 14.75, 2.36, text=r'$H + X_{in}$', text_pos='right')
# A_tilde to final Hadamard
draw_conn(ax, 14.75, 3.85, 14.75, 2.84, text=r'$\widetilde{A}$', text_pos='right')

# Block Output Box
draw_block(ax, 13.9, 1.35, 1.75, 0.85,
           r'Block Output $O$',
           r'$O = (H + X_{in}) \odot \widetilde{A}$',
           '[B, 48, H, W]', color='#F0FDF4', border='#15803D', title_size=8, sub_size=7)
draw_conn(ax, 14.75, 2.36, 14.75, 2.2)

# Identity Fallback Callout Banner
fallback_box = FancyBboxPatch((6.05, 0.25), 9.6, 0.32, boxstyle="round,pad=0.02,rounding_size=0.08",
                              linewidth=1.0, edgecolor='#10B981', facecolor='#ECFDF5', zorder=3)
ax.add_patch(fallback_box)
ax.text(10.85, 0.41, r'Mathematical Identity Fallback Guarantee: When $\gamma_s = \gamma_l = 0$ and $\tau = 0.50$, $\Phi(H) \equiv 0$ and $\widetilde{A}(H) \equiv \sigma(H) - 0.5$ (Exact Baseline SPAB)',
        fontsize=6.8, fontweight='bold', color='#065F46', ha='center', va='center', zorder=4)

out_arch = os.path.join(ROOT_DIR, 'methodology_architecture.png')
plt.savefig(out_arch, bbox_inches='tight', dpi=300)
print(f"Publication-grade Architecture Diagram saved successfully to {out_arch}")
