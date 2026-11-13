import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.span_arch import Conv3XC, conv_layer, pixelshuffle_block, activation

class CAD_SPAB(nn.Module):
    """
    Curvature-Aware Directional Symmetric Attention Block (CAD-SPAB)
    Novel Differential-Geometric Attention Module:
    1. Origin-Symmetric Hyperbolic Tangent Gating with Temperature Scaling:
       A_sym(H) = 0.5 * tanh(tau * H)   [Generalization of sigmoid(H) - 0.5]
    2. First-Order Directional Gradient Flux (Sobel Operator):
       ||grad H|| = sqrt((K_x * H)^2 + (K_y * H)^2 + eps)
    3. Second-Order Isotropic Gaussian-Laplacian Curvature:
       Delta H = |K_L * H|
    4. Dual-Differential Geometric Modulation:
       Phi(H) = tanh(gamma_sobel * ||grad H|| + gamma_lap * Delta H)
       A_mod(H) = A_sym(H) * (1.0 + Phi(H))
       out = (H + x) * A_mod(H)
    """
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False,
                 tau=0.502, gamma_sobel=0.008, gamma_lap=0.001):
        super(CAD_SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = nn.SiLU(inplace=True)
        self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)

        self.tau = nn.Parameter(torch.full((1, 1, 1, 1), float(tau)))
        self.gamma_sobel = nn.Parameter(torch.full((1, 1, 1, 1), float(gamma_sobel)))
        self.gamma_lap = nn.Parameter(torch.full((1, 1, 1, 1), float(gamma_lap)))

        # Fixed parameter-free differential kernels
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('lap_kernel', lap)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)

        c = out3.size(1)

        # 1. Temperature-scaled origin-symmetric gating
        att_sym = 0.5 * torch.tanh(self.tau * out3)

        # 2. First-order directional gradient flux
        kx = self.sobel_x.repeat(c, 1, 1, 1)
        ky = self.sobel_y.repeat(c, 1, 1, 1)
        gx = F.conv2d(out3, kx, padding=1, groups=c)
        gy = F.conv2d(out3, ky, padding=1, groups=c)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

        # 3. Second-order isotropic curvature
        klap = self.lap_kernel.repeat(c, 1, 1, 1)
        lap_mag = torch.abs(F.conv2d(out3, klap, padding=1, groups=c))

        # 4. Modulated dual-differential attention
        phi = torch.tanh(self.gamma_sobel * grad_mag + self.gamma_lap * lap_mag)
        att_mod = att_sym * (1.0 + phi)

        # 5. Content gating
        out = (out3 + x) * att_mod
        return out, out1, att_mod


@ARCH_REGISTRY.register()
class CAD_SPAN(nn.Module):
    """
    Curvature-Aware Directional Swift Parameter-free Attention Network (CAD-SPAN)
    NTIRE-Winner backbone upgraded with Differential-Geometric Attention Gating.
    """
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=48,
                 upscale=4,
                 bias=True,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 tau=0.502,
                 gamma_sobel=0.008,
                 gamma_lap=0.001):
        super(CAD_SPAN, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = CAD_SPAB(feature_channels, bias=bias, tau=tau, gamma_sobel=gamma_sobel, gamma_lap=gamma_lap)
        self.block_2 = CAD_SPAB(feature_channels, bias=bias, tau=tau, gamma_sobel=gamma_sobel, gamma_lap=gamma_lap)
        self.block_3 = CAD_SPAB(feature_channels, bias=bias, tau=tau, gamma_sobel=gamma_sobel, gamma_lap=gamma_lap)
        self.block_4 = CAD_SPAB(feature_channels, bias=bias, tau=tau, gamma_sobel=gamma_sobel, gamma_lap=gamma_lap)
        self.block_5 = CAD_SPAB(feature_channels, bias=bias, tau=tau, gamma_sobel=gamma_sobel, gamma_lap=gamma_lap)
        self.block_6 = CAD_SPAB(feature_channels, bias=bias, tau=tau, gamma_sobel=gamma_sobel, gamma_lap=gamma_lap)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, _, att1 = self.block_1(out_feature)
        out_b2, _, att2 = self.block_2(out_b1)
        out_b3, _, att3 = self.block_3(out_b2)

        out_b4, _, att4 = self.block_4(out_b3)
        out_b5, _, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)
        return output

    def forward_ensemble(self, x):
        tfs = [
            x,
            torch.flip(x, [3]),
            torch.flip(x, [2]),
            torch.flip(x, [2, 3]),
            torch.rot90(x, 1, [2, 3]),
            torch.flip(torch.rot90(x, 1, [2, 3]), [3]),
            torch.flip(torch.rot90(x, 1, [2, 3]), [2]),
            torch.rot90(x, 3, [2, 3])
        ]
        outs = [self.forward(t) for t in tfs]
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


# Aliases for backward compatibility
AG_SPAN = CAD_SPAN
AG_SPAB = CAD_SPAB
