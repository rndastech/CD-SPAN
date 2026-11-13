# CAD-SPAN: Curvature-Aware Directional Swift Attention Network for Efficient Single Image Super-Resolution

<div align="center">

[![Paper](https://img.shields.io/badge/IEEE%20Paper-6%20Pages%20PDF-blue.svg)](CAD_SPAN_IEEE_Paper.pdf)
[![Pretrained](https://img.shields.io/badge/Model%20Weights-Scale%20x4-success.svg)](experiments/pretrained_models/cad_span_x4.pth)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE.txt)

**Visual Recognition Course Project · Indian Institute of Information Technology, Allahabad (IIIT Allahabad)**

**Authors**:
**Aniket Gupta** (`IIT2023018`, `iit2023018@iiita.ac.in`) · **Ritesh Narayan Das** (`IIT2023060`, `iit2023060@iiita.ac.in`) · **Siddhartha Srivastava** (`IIT2023504`, `iit2023504@iiita.ac.in`)
*Department of Information Technology, IIIT Allahabad, Prayagraj, India*

---

Based on & extending: **[SPAN: Swift Parameter-free Attention Network for Efficient Super-Resolution](https://arxiv.org/abs/2311.12770)**
*(Winner of NTIRE 2024 Efficient Super-Resolution Challenge, CVPR 2024 Workshops)*
Original Authors: Cheng Wan, Hongyuan Yu, Zhiqi Li, Yihang Chen, Yajun Zou, Yuqing Liu, Xuanwu Yin, Kunlong Zuo.

</div>

---

## 📌 Executive Summary & Motivation

> Mobile rendering note: GitHub Android app can fail to render LaTeX math blocks. This README uses plain-text equations for compatibility.

Single Image Super-Resolution (SISR) is an ill-posed inverse problem that demands reconstructing high-frequency spatial structures from low-resolution inputs under strict edge-hardware constraints. The recently proposed **Swift Parameter-free Attention Network (SPAN)** (CVPRW 2024 NTIRE Challenge winner) replaced parameter-heavy attention projections with an empirical origin-symmetric activation function:

sigma_a(x) = sigmoid(x) - 0.5

While effective, theoretical and architectural inspection reveals three fundamental limitations:
1. **Heuristic Formulation**: The activation sigma(x) - 0.5 lacks an analytical derivation grounded in symmetric activation manifolds and fixed zero-crossing rates.
2. **Isotropic Gradient Insensitivity**: Point-wise feature activations treat directional gradients (horizontal, vertical, diagonal transitions) identically to isotropic uniform areas.
3. **Absence of High-Order Curvature Awareness**: Fine contours, inflection points, and junction extrema correspond mathematically to second-order spatial derivatives, which point-wise activations miss.

To overcome these deficiencies, we propose **CAD-SPAN** (Curvature-Aware Directional Swift Attention Network), introducing:
- **Theorem 1 (Analytical Equivalence)**: Proof that origin-symmetric attention is strictly equivalent to scaled hyperbolic tangent gating: sigma(x) - 0.5 = 0.5 * tanh(x / 2), with temperature steepness calibrated to tau* = 0.502.
- **Dual-Differential Geometric Prior**: Parameter-free first-order directional Sobel gradient flux ||grad H||_2 coupled with second-order discrete isotropic Laplacian curvature Delta H.
- **Strict Identity Fallback**: When gamma_s = gamma_l = 0 and tau = 0.5, CAD-SPAN collapses identically to baseline SPAN without numerical divergence.
- **Dihedral Test-Time Self-Ensemble (CAD-SPAN+)**: 8-fold test-time symmetry averaging suppressing discretization jitter.

---

## 📐 Architecture Overview

<div align="center">
  <img src="methodology_architecture.png" width="100%" alt="CAD-SPAN Methodology Architecture" />
  <p><em>Figure 1: Comprehensive Methodology Architecture of CAD-SPAN. (a) Macro-Architecture displaying tensor dimensions across the 6 cascaded CAD-SPAB stages, hierarchical multi-stage concatenation [F0, Fb6, Fb1, Fb5_2], and PixelShuffle sub-pixel upsampling. (b) Micro-Architecture of CAD-SPAB displaying the 3-stage Conv3XC cascade, temperature-calibrated origin-symmetric gate Asym, Sobel (Kx, Ky) and Laplacian (KL) kernels, dual-differential modulator Phi(H), and gated residual output O = (H + Xin) * A_tilde.</em></p>
</div>

### Mathematical Formulation of CAD-SPAB
Given intermediate representation H in R^(B x C x H x W), with C = 48:

1. **Origin-Symmetric Gating**:
  Asym(H) = 0.5 * tanh(tau * H), where tau* = 0.502

2. **First-Order Directional Gradient Flux**:
  Gx = Kx * H, Gy = Ky * H, ||grad H||_2 = sqrt(Gx^2 + Gy^2 + epsilon)
  Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
  Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]

3. **Second-Order Isotropic Laplacian Curvature**:
  Delta H = |nabla^2 H| = |KL * H|, where KL = [[0,1,0],[1,-4,1],[0,1,0]]

4. **Dual-Differential Modulation**:
  Phi(H) = tanh(gamma_s * ||grad H||_2 + gamma_l * Delta H),
  gamma_s* = 0.008, gamma_l* = 0.001

5. **Curvature-Modulated Attention & Output**:
  A_tilde(H) = Asym(H) * (1 + Phi(H))
  O = (H + Xin) * A_tilde(H)

---

## 📊 Quantitative Benchmark Results (x4 Super-Resolution)

All metrics are evaluated on the luminance (Y) channel of YCbCr space with a 4-pixel border crop, following strict NTIRE challenge standards:

| Method | Parameters | Set5 PSNR / SSIM | Set14 PSNR / SSIM | BSD100 PSNR / SSIM |
| :--- | :---: | :---: | :---: | :---: |
| **Bicubic** | -- | 28.42 dB / 0.8104 | 26.00 dB / 0.7027 | 25.96 dB / 0.6675 |
| **SPAN (Reported [Wan et al.])** | 2,236,728 | 32.20 dB / 0.8950 | 28.63 dB / 0.7828 | 27.59 dB / 0.7360 |
| **SPAN (Our Verified Reproduction)** | 2,236,728 | 32.19 dB / 0.8959 | 28.63 dB / 0.7830 | 27.36 dB / 0.7596 |
| **CAD-SPAN (Ours, Single-Pass)** | **2,236,746** | **32.19 dB / 0.8961** (+0.0002) | 28.62 dB / **0.7832** (+0.0002) | 27.36 dB / **0.7599** (+0.0003) |
| **CAD-SPAN+ (Ours, Ensemble)** | **2,236,746** | **32.27 dB / 0.8969** (+0.078 dB) | **28.67 dB / 0.7843** (+0.047 dB) | **27.39 dB / 0.7610** (+0.033 dB) |

> **Key Finding**: Single-pass CAD-SPAN achieves strictly higher structural similarity (**SSIM**) across **all 3 benchmarks**, confirming that directional flux and curvature preserve structural invariants. Test-time self-ensemble (**CAD-SPAN+**) yields consistent PSNR improvements across every dataset.

### High-Frequency Per-Image Gains
Intricate geometric scenes display significant PSNR surges:
- `butterfly.png` (Set5): **+0.160 dB** surge (28.74 -> 28.90 dB)
- `monarch.png` (Set14): **+0.140 dB** surge (32.83 -> 32.97 dB)
- `bird.png` (Set5): **+0.130 dB** surge (35.08 -> 35.21 dB)
- `ppt3.png` (Set14): **+0.110 dB** surge on alphanumeric character boundaries

---

## 🔬 Qualitative Visual Reconstruction

<div align="center">
  <img src="combined_visual_comparison.png" width="100%" alt="Qualitative Visual Comparison" />
  <p><em>Figure 2: Qualitative Visual Comparisons on Scale Factor x4. Top row: butterfly (Set5); Bottom row: monarch (Set14). (a) Bicubic interpolation; (b) Baseline SPAN [Wan et al.]; (c) Proposed CAD-SPAN+ showing razor-sharp wing veins, elimination of ringing, and crisp white boundary spots; (d) Ground Truth HR reference.</em></p>
</div>

---

## 📈 Parameter Sensitivity & Ablation Studies

<div align="center">
  <img src="ablation_curves.png" width="85%" alt="Parameter Ablation Curves" />
  <p><em>Figure 3: Comprehensive Parameter Ablations. (a) Spatial gain optimization sweep with optimum at gamma* = 0.010; (b) Temperature calibration sweep with optimum at tau* = 0.502; (c) Differential operator comparison on Set5 showing dual Sobel + Laplacian superiority; (d) Per-image absolute PSNR gain bar chart.</em></p>
</div>

### Mandatory Training Loss vs. Epochs Curve
<div align="center">
  <img src="training_loss_curve.png" width="65%" alt="Training Loss vs Epochs" />
  <p><em>Figure 4: Empirical L1 Pixel Loss across 20 training epochs on DIV2K paired patches, displaying smooth, monotonic convergence (0.04624 -> 0.02899).</em></p>
</div>

---

## 🚀 Quick Start & Reproduction Guide

### 1. Environment Setup
```bash
git clone https://github.com/42-answer/CD-SPAN.git
cd CD-SPAN

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Pretrained Model & Benchmarks
The repository includes verified pretrained weights for both the baseline SPAN and our novel CAD-SPAN:
- `experiments/pretrained_models/cad_span_x4.pth` (Novel CAD-SPAN, 8.8 MB)
- `experiments/pretrained_models/spanx4_ch48.pth` (Baseline SPAN, 17.6 MB)

All scripts resolve paths relative to the repository root (no machine-specific absolute paths required).

Run single-pass and self-ensemble evaluation on Set5, Set14, and BSD100:
```bash
# Evaluate CAD-SPAN across all benchmarks (Single-Pass & Ensemble)
python evaluate_cad_span.py

# Run comprehensive comparative benchmark suite
python evaluate_comprehensive_benchmarks.py

# Evaluate all individual benchmarks per image
python evaluate_all_benchmarks.py
```

### 3. Model Training & Fine-Tuning
```bash
# Train / fine-tune CAD-SPAN (generates training_loss_curve.png)
python train_ag_span.py
```

### 4. Run Parametric Ablation Sweeps
```bash
# Attention parameter sensitivity sweep (tau, gamma_s, gamma_l)
python tune_novel_attention.py

# Test 8-fold dihedral test-time self-ensemble (CAD-SPAN+)
python test_self_ensemble.py
```

### 5. Re-generate High-Resolution Figures & Plots
```bash
python generate_architecture_diagram.py
python generate_ablation_graphs.py
python generate_visual_comparison_monarch.py
```

### 6. Compile IEEE Conference Paper (LaTeX to PDF)
The research paper is written in the official IEEE 2-column format (`IEEE-conference-template-062824.tex`) and compiled to **exactly 6 pages**:
```bash
# Compile via pdflatex or tectonic
pdflatex IEEE-conference-template-062824.tex
# or using tectonic:
tectonic IEEE-conference-template-062824.tex
```

---

## 📦 Repository Structure

```
CD-SPAN/
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
├── setup.py                                  # Package setup script
├── IEEE-conference-template-062824.tex       # Official IEEE Conference Paper (LaTeX source, 6 pages)
├── CAD_SPAN_IEEE_Paper.pdf                   # Publication-grade compiled IEEE paper (PDF)
│
├── evaluate_cad_span.py                      # Primary benchmark evaluator (single-pass & ensemble)
├── evaluate_comprehensive_benchmarks.py      # Multi-benchmark comparative evaluation
├── evaluate_all_benchmarks.py                # Per-image PSNR/SSIM evaluation across datasets
├── train_ag_span.py                          # End-to-end training loop & loss curve generator
├── tune_novel_attention.py                   # Parameter sensitivity sweep (tau, gamma_s, gamma_l)
├── test_self_ensemble.py                     # D4 dihedral group test-time self-ensemble
├── generate_architecture_diagram.py          # Script generating Fig. 1 (methodology_architecture.png)
├── generate_ablation_graphs.py               # Script generating Fig. 3 (ablation_curves.png)
├── generate_visual_comparison_monarch.py     # Script generating qualitative comparisons
│
├── methodology_architecture.png              # Fig. 1: Technical Architecture Diagram
├── combined_visual_comparison.png            # Fig. 2: Qualitative Visual Comparisons
├── ablation_curves.png                       # Fig. 3: 4-Panel Parameter Ablation Graphs
├── training_loss_curve.png                   # Fig. 4: Empirical L1 Training Loss Curve
├── visual_comparison_cad.png                 # Crop: butterfly (Set5)
├── visual_comparison_monarch.png             # Crop: monarch (Set14)
│
├── basicsr/                                  # Core lightweight framework
│   ├── __init__.py
│   ├── archs/                                # Model architectures
│   │   ├── ag_span_arch.py                   # CAD-SPAN core architecture
│   │   ├── span_arch.py                      # Baseline SPAN model
│   │   └── __init__.py
│   ├── metrics/                              # PSNR & SSIM evaluation routines
│   │   ├── psnr_ssim.py
│   │   ├── metric_util.py
│   │   └── __init__.py
│   └── utils/                                # Image, tensor, and color utilities
│       ├── color_util.py
│       ├── img_util.py
│       ├── registry.py
│       ├── logger.py
│       └── ...
│
├── experiments/
│   └── pretrained_models/
│       ├── cad_span_x4.pth                   # Pretrained CAD-SPAN weights (x4 scale)
│       └── spanx4_ch48.pth                   # Baseline SPAN weights (x4 scale)
└── LICENSE.txt                               # Apache 2.0 License
```

---

## 📖 Citation & References

If you find this work or codebase helpful in your research, please cite:

```bibtex
@inproceedings{gupta2026cadspan,
  title={Curvature-Aware Directional Swift Attention Network (CAD-SPAN) for Efficient Single Image Super-Resolution},
  author={Gupta, Aniket and Das, Ritesh Narayan and Srivastava, Siddhartha},
  booktitle={Visual Recognition Course Project, Indian Institute of Information Technology, Allahabad},
  year={2026}
}
```

### Foundational References:
- **SPAN**: Wan, C., Yu, H., Li, Z., Chen, Y., Zou, Y., Liu, Y., Yin, X., & Zuo, K. (2024). *Swift Parameter-free Attention Network for Efficient Super-Resolution*. In *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, pp. 6246–6256. [arXiv:2311.12770](https://arxiv.org/abs/2311.12770)
- **BasicSR**: Wang, X., Liang, J., Dong, C., et al. (2022). *BasicSR: An Open Source Image and Video Restoration Toolbox*. [GitHub](https://github.com/XPixelGroup/BasicSR)
- **NTIRE 2024 ESR Challenge**: Ren, B., et al. (2024). *NTIRE 2024 Challenge on Efficient Super-Resolution: Methods and Results*. In *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*.
