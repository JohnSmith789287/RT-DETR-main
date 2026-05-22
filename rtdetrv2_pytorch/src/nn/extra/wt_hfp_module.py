"""Wavelet high/low-frequency perception module for FPN lateral features."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _wavelet_util as wavelet

__all__ = ['WaveletHighLowFrequencyPerception']


def _valid_groups(channels: int, preferred: int = 32) -> int:
    groups = math.gcd(channels, preferred)
    return groups if groups > 0 else 1


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class WaveletHighLowFrequencyPerception(nn.Module):
    """DWT-based frequency perception inspired by HFP and WTConv.

    Input and output shape are both ``[B, C, H, W]``. The module decomposes a
    feature map into low-frequency LL and high-frequency LH/HL/HH responses,
    learns spatial and channel gates for both parts, and injects the result
    through a small residual branch.
    """

    def __init__(
        self,
        in_channels: int,
        wt_type: str = 'db1',
        kernel_size: int = 3,
        init_alpha: float = 0.01,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f'kernel_size must be odd, got {kernel_size}')

        self.in_channels = in_channels
        self.wt_type = wt_type
        self.kernel_size = kernel_size

        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.subband_conv = nn.Conv2d(
            in_channels * 4,
            in_channels * 4,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels * 4,
            bias=False,
        )
        self.subband_scale = _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)

        self.spatial_gate = nn.Conv2d(in_channels * 3, 3, kernel_size=1, bias=True)

        channel_groups = _valid_groups(in_channels * 2, norm_groups)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1,
                      groups=channel_groups, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1,
                      groups=channel_groups, bias=True),
        )

        out_groups = _valid_groups(in_channels, norm_groups)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_groups, in_channels),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def _pad_to_even(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def _idwt_crop(self, coeffs: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        x = wavelet.inverse_2d_wavelet_transform(coeffs, self.iwt_filter)
        h, w = shape
        return x[:, :, :h, :w]

    def _frequency_responses(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_pad, original_shape = self._pad_to_even(x)
        coeffs = wavelet.wavelet_2d_transform(x_pad, self.wt_filter)
        b, c, bands, h, w = coeffs.shape

        coeffs_tag = coeffs.reshape(b, c * bands, h, w)
        coeffs_tag = self.subband_scale(self.subband_conv(coeffs_tag))
        coeffs_tag = coeffs_tag.reshape(b, c, bands, h, w)

        low_coeffs = torch.zeros_like(coeffs_tag)
        high_coeffs = torch.zeros_like(coeffs_tag)
        low_coeffs[:, :, 0, :, :] = coeffs_tag[:, :, 0, :, :]
        high_coeffs[:, :, 1:4, :, :] = coeffs_tag[:, :, 1:4, :, :]

        x_low = self._idwt_crop(low_coeffs, original_shape)
        x_high = self._idwt_crop(high_coeffs, original_shape)
        return x_low, x_high

    def _channel_weights(self, x_low: torch.Tensor, x_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low_desc = F.adaptive_avg_pool2d(x_low, 1) + F.adaptive_max_pool2d(x_low, 1)
        high_desc = F.adaptive_avg_pool2d(x_high, 1) + F.adaptive_max_pool2d(x_high, 1)
        logits = self.channel_gate(torch.cat([low_desc, high_desc], dim=1))
        b, _, _, _ = logits.shape
        logits = logits.view(b, 2, self.in_channels, 1, 1)
        weights = torch.softmax(logits, dim=1)
        return weights[:, 0], weights[:, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low, x_high = self._frequency_responses(x)

        spatial = self.spatial_gate(torch.cat([x, x_low, x_high], dim=1))
        spatial = torch.softmax(spatial, dim=1)
        low_spatial = spatial[:, 1:2]
        high_spatial = spatial[:, 2:3]

        low_channel, high_channel = self._channel_weights(x_low, x_high)
        x_phy = low_spatial * low_channel * x_low + high_spatial * high_channel * x_high
        return x + self.alpha * self.out(x_phy)
