"""Smoke tests for WT-HFP and HybridEncoder integration."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rtdetrv2_pytorch'))

import torch

from src.nn.extra.wt_hfp_module import WaveletHighFrequencyPerception
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder


def test_standalone_wt_hfp():
    print("=" * 60)
    print("Test 1: standalone WT-HFP with HFP-style branches")
    module = WaveletHighFrequencyPerception(256, wt_type='db1')
    assert not hasattr(module, 'spatial_gate')
    assert not hasattr(module, 'channel_gate')
    assert not hasattr(module, 'subband_conv')
    assert not hasattr(module, 'alpha')

    constant = torch.ones(1, 256, 16, 16)
    high_response = module._high_frequency_response(constant)
    assert high_response.shape == constant.shape
    assert high_response.abs().max() < 1e-5

    for h, w in [(80, 80), (41, 39), (20, 20)]:
        x = torch.randn(2, 256, h, w, requires_grad=True)
        y = module(x)
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
        y.mean().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
        print(f"  input={(h, w)} output={tuple(y.shape)} => OK")


def test_hybrid_encoder_wt_hfp():
    print("=" * 60)
    print("Test 2: HybridEncoder + WT-HFP on C3/C4/C5")
    enc = HybridEncoder(
        in_channels=[128, 256, 512],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        expansion=0.5,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        use_wt_hfp=True,
        wt_hfp_apply_idx=[0, 1, 2],
    )
    feats = [
        torch.randn(1, 128, 80, 80),
        torch.randn(1, 256, 40, 40),
        torch.randn(1, 512, 20, 20),
    ]
    outs = enc(feats)
    assert len(enc.wt_hfp_modules) == 3
    assert enc.wt_hfp_apply_idx == [0, 1, 2]
    assert [tuple(o.shape) for o in outs] == [
        (1, 256, 80, 80),
        (1, 256, 40, 40),
        (1, 256, 20, 20),
    ]
    print(f"  outputs={[tuple(o.shape) for o in outs]} => OK")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_standalone_wt_hfp()
    test_hybrid_encoder_wt_hfp()
    print("=" * 60)
    print("ALL WT-HFP TESTS PASSED")
