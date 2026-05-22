"""Wavelet filter-bank helpers used by WT-HFP.

The DWT/IDWT filters are fixed analysis/synthesis filters from PyWavelets.
"""
from __future__ import annotations

import pywt
import torch
import torch.nn.functional as F


def create_2d_wavelet_filter(wave: str, in_size: int, out_size: int, type=torch.float):
    """Build 2D separable filters for LL, LH, HL, and HH subbands."""
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_2d_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """Single-level 2D DWT. Input [B, C, H, W] -> [B, C, 4, H/2, W/2]."""
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_2d_wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """Single-level 2D IDWT. Input [B, C, 4, H/2, W/2] -> [B, C, H, W]."""
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
