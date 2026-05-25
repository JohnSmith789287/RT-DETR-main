"""WTConv2d — Wavelet Transform Convolution (ECCV 2024).

This is a near-verbatim port of the official implementation:
    https://github.com/BGU-CS-VIL/WTConv/blob/main/wtconv/wtconv2d.py
(MIT License, (c) the WTConv authors)

We intentionally preserve the upstream API surface --- including the
per-channel ``_ScaleModule`` fusion, multi-level LL-accumulation, and
pywavelets-based filter construction --- so that **numerical behaviour
matches the paper**.

Additions on top of the vendored file (minimal, non-behavioural):
    * A ``load_spatial_from_dwconv`` helper for the SF-Backbone weight
      hot-swap (copies a pretrained depthwise kernel into ``base_conv``).
    * Adjusted padding of ``base_conv`` to `padding='same'` same as upstream;
      ``kernel_size`` defaults to 5, matching the official README example.

Reference:
    Shahaf E. Finder, Roy Amoyal, Eran Treister, Oren Freifeld.
    "Wavelet Convolutions for Large Receptive Fields", ECCV 2024.
    arXiv:2407.05848
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _wavelet_util as wavelet


__all__ = ['WTConv2d']


class _ScaleModule(nn.Module):
    """Element-wise multiplication by a learnable scale (per-channel).

    Matches the upstream definition (no bias, init_scale is a float).
    """

    def __init__(self, dims, init_scale: float = 1.0, init_bias: float = 0) -> None:
        super().__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None  # kept for API parity with the upstream repo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    """Depthwise wavelet-augmented 2D convolution.

    Parameters
    ----------
    in_channels : int
        Number of input / output channels (depthwise, ``in_channels == out_channels``).
    out_channels : int
        Ignored except for the asserted equality to ``in_channels``.
    kernel_size : int
        Size of the depthwise kernel used for both the spatial base branch
        and every per-subband convolution.
    stride : int
        If ``stride > 1`` an ``AvgPool2d`` is applied at the end (same as the
        upstream implementation).
    bias : bool
        Whether ``base_conv`` has a bias term. Default True (upstream default).
    wt_levels : int
        Number of wavelet decomposition levels (1, 2, or 3 are typical).
    wt_type : str
        A PyWavelets wavelet name. ``'db1'`` is Haar; ``'db2'`` is Daubechies-4.

    Shape
    -----
    - Input : ``(B, C, H, W)``
    - Output: ``(B, C, H / stride, W / stride)``

    Notes
    -----
    * This module is **depthwise only**. In ConvNeXt-style blocks, follow it
      with a 1x1 conv for channel mixing (same as the upstream WTConvNeXt).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 5,
        stride: int = 1,
        bias: bool = True,
        wt_levels: int = 1,
        wt_type: str = 'db1',
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels, (
            'WTConv2d is strictly depthwise: in_channels must equal out_channels '
            f'(got {in_channels} vs {out_channels}).')

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        # Spatial branch (plain depthwise conv). Uses padding='same' so any
        # odd kernel size keeps H, W unchanged.
        self.base_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding='same', stride=1, dilation=1, groups=in_channels, bias=bias,
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # Per-level depthwise convs over the 4-subband packed representation.
        # We stack the 4 subbands on the channel axis (× 4) and use groups=4C
        # so each subband gets its own depthwise kernel.
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels * 4, in_channels * 4, kernel_size,
                padding='same', stride=1, dilation=1,
                groups=in_channels * 4, bias=False,
            )
            for _ in range(self.wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    # ------------------------------------------------------------------
    # weight hot-swap helper (SF-Backbone bridge from ConvNeXt-V2 pretrain)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_spatial_from_dwconv(self, source_weight: torch.Tensor,
                                 source_bias: torch.Tensor | None = None) -> None:
        """Copy a pretrained depthwise-conv kernel into ``base_conv``.

        Expected shape: ``(C, 1, k, k)``. This is the bridge used by the
        SF-Backbone weight hot-swap script (see STATUS.md §12).
        """
        assert source_weight.shape == self.base_conv.weight.shape, (
            f'shape mismatch: got {source_weight.shape}, '
            f'expected {self.base_conv.weight.shape}')
        self.base_conv.weight.copy_(source_weight)
        if source_bias is not None and self.base_conv.bias is not None:
            self.base_conv.bias.copy_(source_bias)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            # Pad to even spatial size (DWT requires H, W % 2 == 0).
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(
                shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll  # ← multi-level accumulation

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)
        return x

    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, wt_levels={self.wt_levels}, '
                f'stride={self.stride}')
