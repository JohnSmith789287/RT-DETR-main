"""PhyGuideNet backbone -- DIRECT PORT of PHDL-RTDETR's backbone.

!!! IMPORTANT !!!
This file is a near-verbatim port of the FasterNet+physics-regularization
backbone published in:
    Guan Z., Zhang R., Lin M. (2026)
    "Research on injection molded defects detection algorithm based on
     physics-guided regularization constraints"
    Expert Systems with Applications, vol. 296, 129212.
    doi:10.1016/j.eswa.2025.129212
    code: https://github.com/<phdl-rtdetr-repo>

This is NOT an original contribution of our work.  It exists in this
codebase ONLY as a competitive baseline to be reported in the comparison
table, so reviewers can see that our actual method (TBD: smoke-diffusion /
wavelet-sparsity) outperforms a published physics-guided detector on the
D-FIRE dataset.

The Laplacian + bias-normalization regularization is reproduced as-is,
including the original 0.01 / 0.005 coefficients.  DO NOT claim this as
our novelty.

FasterNet itself is from:
    Chen J. et al. (CVPR 2023) "Run, Don't Walk: Chasing Higher FLOPS for
    Faster Neural Networks"  -- https://github.com/JierunChen/FasterNet
"""

from __future__ import annotations

import os
from functools import partial
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core import register

try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath


# ---------------------------------------------------------------------------
# Core building blocks (from FasterNet + PHDL physics constraint)
# ---------------------------------------------------------------------------


class HaarDWT(nn.Module):
    """Single-level Haar wavelet decomposition.

    The transform is implemented with fixed tensor slicing so it has no
    learnable parameters. It returns LL/LH/HL/HH subbands with half spatial
    resolution.
    """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class HaarIDWT(nn.Module):
    """Inverse of :class:`HaarDWT`."""

    def forward(self, ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor,
                out_size: tuple[int, int]) -> Tensor:
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        b, c, h, w = ll.shape
        x = ll.new_zeros(b, c, h * 2, w * 2)
        x[..., 0::2, 0::2] = x00
        x[..., 0::2, 1::2] = x01
        x[..., 1::2, 0::2] = x10
        x[..., 1::2, 1::2] = x11
        return x[..., :out_size[0], :out_size[1]]


class FireSmokeDualPrior(nn.Module):
    """Fire-smoke dual physical prior.

    LL is treated as a smoke-like low-frequency diffusion candidate, while
    LH/HL/HH are treated as fire-like high-frequency variation candidates. A
    lightweight semantic gate from the original feature chooses how much of each
    prior to inject. The learnable residual strength starts small so the module
    does not disrupt pretrained features at initialization.
    """

    def __init__(self, channels: int, norm_layer=nn.BatchNorm2d,
                 alpha_init: float = 0.01):
        super().__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        self.smoke_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.smoke_pw = nn.Conv2d(channels, channels, 1, bias=False)

        high_channels = channels * 3
        self.fire_dw = nn.Conv2d(high_channels, high_channels, 3, padding=1,
                                 groups=high_channels, bias=False)
        self.fire_pw = nn.Conv2d(high_channels, high_channels, 1, bias=False)

        self.gate = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            norm_layer(channels),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: Tensor) -> Tensor:
        out_size = x.shape[-2:]
        ll, lh, hl, hh = self.dwt(x)

        smoke_ll = self.smoke_pw(self.smoke_dw(ll))
        zero_ll = torch.zeros_like(smoke_ll)
        f_smoke = self.idwt(smoke_ll, torch.zeros_like(lh),
                            torch.zeros_like(hl), torch.zeros_like(hh),
                            out_size)

        high = torch.cat([lh, hl, hh], dim=1)
        high = self.fire_pw(self.fire_dw(high))
        fire_lh, fire_hl, fire_hh = torch.chunk(high, 3, dim=1)
        f_fire = self.idwt(zero_ll, fire_lh, fire_hl, fire_hh, out_size)

        gate = self.gate(x.mean(dim=1, keepdim=True))
        f_phy = gate * f_fire + (1.0 - gate) * f_smoke
        return x + self.alpha * self.proj(f_phy)

class Partial_conv3(nn.Module):
    """PConv: only convolve dim//n_div channels, pass the rest through."""

    def __init__(self, dim: int, n_div: int = 4, forward: str = 'split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    """FasterNet block with optional physics-guided regularization."""

    def __init__(self, dim, n_div, mlp_ratio, drop_path, layer_scale_init_value,
                 act_layer, norm_layer, pconv_fw_type, use_physics: bool = True):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        self.use_physics = use_physics

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward_plain

    def forward_plain(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        if self.use_physics:
            x = x + self._physics_constraint(x)
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        if self.use_physics:
            x = x + self._physics_constraint(x)
        return x

    def _physics_constraint(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        diff = x - mean
        constraint = torch.norm(diff, dim=1, keepdim=True) / (torch.std(x) + 1e-5)
        laplacian = self._laplacian_smoothing(x)
        return 0.01 * constraint + 0.005 * laplacian

    @staticmethod
    def _laplacian_smoothing(x: Tensor) -> Tensor:
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                              dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(x.size(1), 1, 3, 3)
        lap = F.conv2d(x, kernel, padding=1, groups=x.size(1))
        return torch.mean(torch.abs(lap), dim=(2, 3), keepdim=True)


class BasicStage(nn.Module):
    def __init__(self, dim, depth, n_div, mlp_ratio, drop_path,
                 layer_scale_init_value, norm_layer, act_layer,
                 pconv_fw_type, use_physics: bool = True):
        super().__init__()
        self.blocks = nn.Sequential(*[
            MLPBlock(dim=dim, n_div=n_div, mlp_ratio=mlp_ratio,
                     drop_path=drop_path[i],
                     layer_scale_init_value=layer_scale_init_value,
                     norm_layer=norm_layer, act_layer=act_layer,
                     pconv_fw_type=pconv_fw_type,
                     use_physics=use_physics)
            for i in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_stride, bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class PatchMerging(nn.Module):
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2,
                                   stride=patch_stride2, bias=False)
        self.norm = norm_layer(2 * dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.reduction(x))


class FasterNet(nn.Module):
    """FasterNet backbone with multi-scale feature output."""

    def __init__(self, in_chans=3, embed_dim=40, depths=(1, 2, 8, 2),
                 mlp_ratio=2., n_div=4,
                 patch_size=4, patch_stride=4,
                 patch_size2=2, patch_stride2=2,
                 patch_norm=True, drop_path_rate=0.,
                 layer_scale_init_value=0,
                 norm_layer='BN', act_layer='GELU',
                 pconv_fw_type='split_cat',
                 use_physics: bool = True,
                 use_fsdp: bool = False,
                 fsdp_stage_idx: Sequence[int] = (),
                 fsdp_alpha_init: float = 0.01,
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError(f'norm_layer={norm_layer}')

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError(f'act_layer={act_layer}')

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.depths = depths
        self.fsdp_stage_idx = set(fsdp_stage_idx or [])

        # stem
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, patch_stride=patch_stride,
            in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages interleaved with patch merging
        stages_list = []
        for i_stage in range(self.num_stages):
            dim_i = int(embed_dim * 2 ** i_stage)
            stage = BasicStage(
                dim=dim_i, depth=depths[i_stage], n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer, act_layer=act_layer,
                pconv_fw_type=pconv_fw_type,
                use_physics=use_physics)
            stages_list.append(stage)

            if i_stage < self.num_stages - 1:
                stages_list.append(PatchMerging(
                    patch_size2=patch_size2, patch_stride2=patch_stride2,
                    dim=dim_i, norm_layer=norm_layer))

        self.stages = nn.Sequential(*stages_list)

        self.fsdp_layers = nn.ModuleDict()
        if use_fsdp:
            for stage_i in self.fsdp_stage_idx:
                if stage_i < 0 or stage_i >= self.num_stages:
                    raise ValueError(f'fsdp_stage_idx contains invalid stage {stage_i}')
                self.fsdp_layers[str(2 * stage_i)] = FireSmokeDualPrior(
                    channels=int(embed_dim * 2 ** stage_i),
                    norm_layer=norm_layer,
                    alpha_init=fsdp_alpha_init)

        # stage output indices: [0, 2, 4, 6] → stage0, stage1, stage2, stage3
        self.out_indices = [2 * i for i in range(self.num_stages)]

        # norm layers for each output
        for i_emb, i_layer in enumerate(self.out_indices):
            layer = norm_layer(int(embed_dim * 2 ** i_emb))
            self.add_module(f'norm{i_layer}', layer)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if str(idx) in self.fsdp_layers:
                x = self.fsdp_layers[str(idx)](x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                outs.append(norm_layer(x))
        return outs


# ---------------------------------------------------------------------------
# RT-DETR v2 backbone wrapper
# ---------------------------------------------------------------------------

def _load_pretrained(model: FasterNet, weights_path: str):
    """Load pretrained FasterNet weights with shape-matched partial loading."""
    state = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    elif 'model' in state:
        state = state['model']

    model_dict = model.state_dict()
    matched, skipped = 0, 0
    new_dict = {}
    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_dict[k] = v
            matched += 1
        else:
            skipped += 1
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    print(f'[PhyGuideNet] loaded pretrained: {matched} matched, {skipped} skipped '
          f'(total model keys: {len(model_dict)})')


@register()
class PhyGuideNet(nn.Module):
    """PhyGuideNet backbone for RT-DETR v2.

    Parameters
    ----------
    embed_dim : int
        Base channel dimension.  40 for T0 (→ channels [40, 80, 160, 320]).
    depths : list[int]
        Number of blocks per stage.  [1, 2, 8, 2] for T0.
    return_idx : list[int]
        Which stages to return (0-indexed).  Default [1, 2, 3] gives
        strides [8, 16, 32] and channels [80, 160, 320].
    pretrained : str or bool
        Path to FasterNet pretrained weights (.pth), or False to skip.
    use_physics : bool
        If True, include physics-guided regularization in each block.
        Set False for ablation.
    """

    def __init__(self,
                 embed_dim: int = 40,
                 depths: list = [1, 2, 8, 2],
                 mlp_ratio: float = 2.,
                 n_div: int = 4,
                 drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 0.,
                 norm_layer: str = 'BN',
                 act_layer: str = 'GELU',
                 return_idx: list = [1, 2, 3],
                 pretrained: str = '',
                 use_physics: bool = True):
        super().__init__()

        self.backbone = FasterNet(
            embed_dim=embed_dim, depths=depths,
            mlp_ratio=mlp_ratio, n_div=n_div,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer, act_layer=act_layer,
            use_physics=use_physics)

        self.return_idx = return_idx

        # compute strides and channels
        # stage i has stride = 4 * 2^i, channels = embed_dim * 2^i
        self.strides = [4 * (2 ** i) for i in return_idx]
        self.channels = [embed_dim * (2 ** i) for i in return_idx]

        # load pretrained weights
        if pretrained and isinstance(pretrained, str) and os.path.isfile(pretrained):
            _load_pretrained(self.backbone, pretrained)
        elif pretrained:
            print(f'[PhyGuideNet] pretrained path not found: {pretrained}, '
                  f'training from scratch')

    def forward(self, x: Tensor) -> List[Tensor]:
        all_outs = self.backbone(x)  # 4 scale features
        return [all_outs[i] for i in self.return_idx]


@register()
class FireSmokeGuideNet(nn.Module):
    """FasterNet backbone with Fire-Smoke Dual Prior stage plugins.

    This is the D-Fire-oriented variant: unlike PHDL's Laplacian-only
    regularization, it injects a wavelet low/high-frequency dual prior after
    selected FasterNet stages. By default, it uses stage1 and stage2, i.e.
    stride-8 and stride-16 features (C3/C4).
    """

    def __init__(self,
                 embed_dim: int = 40,
                 depths: list = [1, 2, 8, 2],
                 mlp_ratio: float = 2.,
                 n_div: int = 4,
                 drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 0.,
                 norm_layer: str = 'BN',
                 act_layer: str = 'GELU',
                 return_idx: list = [1, 2, 3],
                 pretrained: str = '',
                 fsdp_stage_idx: list = [1, 2],
                 fsdp_alpha_init: float = 0.01,
                 use_physics: bool = False):
        super().__init__()

        self.backbone = FasterNet(
            embed_dim=embed_dim, depths=depths,
            mlp_ratio=mlp_ratio, n_div=n_div,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer, act_layer=act_layer,
            use_physics=use_physics,
            use_fsdp=True,
            fsdp_stage_idx=fsdp_stage_idx,
            fsdp_alpha_init=fsdp_alpha_init)

        self.return_idx = return_idx
        self.strides = [4 * (2 ** i) for i in return_idx]
        self.channels = [embed_dim * (2 ** i) for i in return_idx]

        if pretrained and isinstance(pretrained, str) and os.path.isfile(pretrained):
            _load_pretrained(self.backbone, pretrained)
        elif pretrained:
            print(f'[FireSmokeGuideNet] pretrained path not found: {pretrained}, '
                  f'training from scratch')

    def forward(self, x: Tensor) -> List[Tensor]:
        all_outs = self.backbone(x)
        return [all_outs[i] for i in self.return_idx]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    m = PhyGuideNet(embed_dim=40, depths=[1, 2, 8, 2],
                    return_idx=[1, 2, 3], pretrained='', use_physics=True)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'params: {n_params:.2f}M')
    print(f'strides:  {m.strides}')
    print(f'channels: {m.channels}')
    x = torch.randn(1, 3, 640, 640)
    outs = m(x)
    for i, o in enumerate(outs):
        print(f'  stage {m.return_idx[i]}: {tuple(o.shape)}')
