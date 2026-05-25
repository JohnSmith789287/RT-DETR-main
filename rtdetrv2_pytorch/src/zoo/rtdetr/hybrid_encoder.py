"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from ...core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class WaveletCSPRepLayer(nn.Module):
    """CSPRepLayer variant that replaces RepVgg blocks with WT-HFP blocks."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu",
                 wt_type='db1',
                 kernel_size=3,
                 init_alpha=0.01):
        super(WaveletCSPRepLayer, self).__init__()
        from ...nn.extra.wt_hfp_module import WaveletHighLowFrequencyPerception

        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            WaveletHighLowFrequencyPerception(
                hidden_channels,
                wt_type=wt_type,
                kernel_size=kernel_size,
                init_alpha=init_alpha,
            )
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class WTConvBlock(nn.Module):
    """WTConv block used as a RepVggBlock replacement in neck fusion."""

    def __init__(self,
                 channels,
                 act="silu",
                 wt_type='db1',
                 kernel_size=5,
                 wt_levels=1):
        super(WTConvBlock, self).__init__()
        from ...nn.extra import WTConv2d

        self.wtconv = WTConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            wt_levels=wt_levels,
            wt_type=wt_type,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.proj = ConvNormLayer(channels, channels, 1, 1, act=act)

    def forward(self, x):
        x = self.act(self.norm(self.wtconv(x)))
        return self.proj(x)


class WTConvCSPRepLayer(nn.Module):
    """CSPRepLayer variant that replaces RepVgg blocks with WTConv blocks."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu",
                 wt_type='db1',
                 kernel_size=5,
                 wt_levels=1):
        super(WTConvCSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            WTConvBlock(
                hidden_channels,
                act=act,
                wt_type=wt_type,
                kernel_size=kernel_size,
                wt_levels=wt_levels,
            )
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class HFPFusionLayer(nn.Module):
    """PHDL LW-Fusion-position replacement using high-frequency perception."""

    def __init__(self,
                 channels,
                 act="silu",
                 wt_type='db1',
                 kernel_size=3,
                 init_alpha=0.01):
        super(HFPFusionLayer, self).__init__()
        from ...nn.extra.wt_hfp_module import WaveletHighFrequencyPerception

        self.fuse = ConvNormLayer(channels * 2, channels, 1, 1, act=act)
        self.hfp = WaveletHighFrequencyPerception(
            channels,
            wt_type=wt_type,
            kernel_size=kernel_size,
        )
        self.out = ConvNormLayer(channels, channels, 1, 1, act=act)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, c_feat, neck_feat):
        fused = self.fuse(torch.concat([c_feat, neck_feat], dim=1))
        high = self.out(self.hfp(fused))
        return neck_feat + self.alpha * high


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2',
                 use_wt_hfp=False,
                 wt_hfp_apply_idx=None,
                 wt_hfp_wt_type='db1',
                 wt_hfp_kernel_size=3,
                 use_wt_hfp_csp=False,
                 wt_hfp_csp_wt_type='db1',
                 wt_hfp_csp_kernel_size=3,
                 wt_hfp_csp_init_alpha=0.01,
                 use_wtconv_csp=False,
                 wtconv_csp_wt_type='db1',
                 wtconv_csp_kernel_size=5,
                 wtconv_csp_levels=1,
                 use_hfp_fusion=False,
                 hfp_fusion_wt_type='db1',
                 hfp_fusion_kernel_size=3,
                 hfp_fusion_init_alpha=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # WT-HFP follows the HS-FPN lateral-enhancement idea: enhance projected
        # C3/C4/C5 features before AIFI and top-down fusion.
        if use_wt_hfp:
            from ...nn.extra.wt_hfp_module import WaveletHighFrequencyPerception
            if wt_hfp_apply_idx is None:
                wt_hfp_apply_idx = list(range(len(in_channels)))
            self.wt_hfp_apply_idx = list(wt_hfp_apply_idx)
            invalid_idx = [idx for idx in self.wt_hfp_apply_idx if idx < 0 or idx >= len(in_channels)]
            if invalid_idx:
                raise ValueError(f'Invalid wt_hfp_apply_idx values: {invalid_idx}')
            self.wt_hfp_modules = nn.ModuleList([
                WaveletHighFrequencyPerception(
                    hidden_dim,
                    wt_type=wt_hfp_wt_type,
                    kernel_size=wt_hfp_kernel_size,
                )
                for _ in in_channels
            ])

        if use_wt_hfp_csp and use_wtconv_csp:
            raise ValueError('use_wt_hfp_csp and use_wtconv_csp are mutually exclusive')

        fpn_pan_block = CSPRepLayer
        fpn_pan_block_kwargs = {}
        if use_wt_hfp_csp:
            fpn_pan_block = WaveletCSPRepLayer
            fpn_pan_block_kwargs = {
                'wt_type': wt_hfp_csp_wt_type,
                'kernel_size': wt_hfp_csp_kernel_size,
                'init_alpha': wt_hfp_csp_init_alpha,
            }
        elif use_wtconv_csp:
            fpn_pan_block = WTConvCSPRepLayer
            fpn_pan_block_kwargs = {
                'wt_type': wtconv_csp_wt_type,
                'kernel_size': wtconv_csp_kernel_size,
                'wt_levels': wtconv_csp_levels,
            }

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                fpn_pan_block(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                    **fpn_pan_block_kwargs,
                )
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                fpn_pan_block(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                    **fpn_pan_block_kwargs,
                )
            )

        if use_hfp_fusion:
            self.hfp_fusion_layers = nn.ModuleList([
                HFPFusionLayer(
                    hidden_dim,
                    act=act,
                    wt_type=hfp_fusion_wt_type,
                    kernel_size=hfp_fusion_kernel_size,
                    init_alpha=hfp_fusion_init_alpha,
                )
                for _ in in_channels
            ])

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if hasattr(self, 'wt_hfp_modules'):
            for idx in self.wt_hfp_apply_idx:
                proj_feats[idx] = self.wt_hfp_modules[idx](proj_feats[idx])
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        if hasattr(self, 'hfp_fusion_layers'):
            outs = [
                self.hfp_fusion_layers[i](proj_feats[i], outs[i])
                for i in range(len(outs))
            ]

        return outs
