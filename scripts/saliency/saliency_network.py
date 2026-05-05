"""Lightweight saliency-network helpers for SaliencySampler."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


def saliency_network_mobilenetv3_small(pretrained: bool = True) -> nn.Module:
    """Truncated MobileNetV3-Small returning ``(B, 48, 14, 14)`` for a 224^2 input.

    ~190 K parameters. Block-by-block check shows ``features[:9]`` ends at
    48 channels and 14x14 spatial resolution; ``features[9]`` would already
    drop to 7x7.
    """
    weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = tvm.mobilenet_v3_small(weights=weights)
    return nn.Sequential(*backbone.features[:9])
