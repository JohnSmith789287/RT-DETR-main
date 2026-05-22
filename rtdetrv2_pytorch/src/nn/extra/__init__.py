"""Extra neural network modules for RT-DETRv2 experiments."""

from .wt_hfp_module import WaveletHighFrequencyPerception, WaveletHighLowFrequencyPerception

__all__ = [
    'WaveletHighFrequencyPerception',
    'WaveletHighLowFrequencyPerception',
]
