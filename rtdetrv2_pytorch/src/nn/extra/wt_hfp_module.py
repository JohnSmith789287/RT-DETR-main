"""Wavelet high-frequency perception module for FPN lateral features."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _wavelet_util as wavelet

__all__ = [
    'WaveletHighFrequencyPerception',
    'WaveletHighLowFrequencyPerception',
]


def _valid_groups(channels: int, preferred: int = 32) -> int:
    groups = math.gcd(channels, preferred)
    return groups if groups > 0 else 1


class WaveletHighFrequencyPerception(nn.Module):
    """HFP-style module with DWT replacing the DCT high-pass generator.

    Input and output shape are both ``[B, C, H, W]``. The module decomposes a
    feature map with DWT, refines LH/HL/HH using depthwise convolution,
    reconstructs the learnable high-frequency response, then follows HFP's
    spatial branch, channel branch, and output projection structure.
    """

    def __init__(
        self,
        in_channels: int,
        wt_type: str = 'db1',
        kernel_size: int = 3,
        patch: tuple[int, int] = (8, 8),
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f'kernel_size must be odd, got {kernel_size}')

        self.in_channels = in_channels
        self.wt_type = wt_type
        self.kernel_size = kernel_size
        self.patch = patch

        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.high_subband_conv = nn.Conv2d(
            in_channels * 3,
            in_channels * 3,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels * 3,
            bias=False,
        )

        channel_groups = _valid_groups(in_channels, norm_groups)
        self.channel1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, groups=channel_groups, bias=True)
        self.channel2x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, groups=channel_groups, bias=True)
        self.relu = nn.ReLU(inplace=True)

        out_groups = _valid_groups(in_channels, norm_groups)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_groups, in_channels),
        )

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

    def _high_frequency_response(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, original_shape = self._pad_to_even(x)
        coeffs = wavelet.wavelet_2d_transform(x_pad, self.wt_filter)
        b, c, _, h, w = coeffs.shape

        high = coeffs[:, :, 1:4, :, :].reshape(b, c * 3, h, w)
        high = self.high_subband_conv(high)
        high = high.reshape(b, c, 3, h, w)

        high_coeffs = torch.zeros_like(coeffs)
        high_coeffs[:, :, 1:4, :, :] = high
        return self._idwt_crop(high_coeffs, original_shape)

    def _channel_weight(self, high_response: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = high_response.shape
        max_pool = F.adaptive_max_pool2d(high_response, output_size=self.patch)
        avg_pool = F.adaptive_avg_pool2d(high_response, output_size=self.patch)
        max_pool = torch.sum(self.relu(max_pool), dim=[2, 3]).view(n, c, 1, 1)
        avg_pool = torch.sum(self.relu(avg_pool), dim=[2, 3]).view(n, c, 1, 1)
        channel = self.channel1x1(max_pool) + self.channel1x1(avg_pool)
        return torch.sigmoid(self.channel2x1(channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high_response = self._high_frequency_response(x)
        spatial = x * high_response
        channel = x * self._channel_weight(high_response)
        return self.out(spatial + channel)


WaveletHighLowFrequencyPerception = WaveletHighFrequencyPerception
