"""Pretrained-capable ResNet semantic path + low-resolution contextual blocks.

Pretraining applies ONLY to torchvision's ResNet encoder, not to Mamba,
fusion, detail path, or decoder.
"""
from __future__ import annotations
from torch import nn
import torch
from ..blocks.context import build_context


class SemanticEncoder(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = False,
                 context_dim: int = 160, context_depth: int = 2,
                 context_type: str = "conv"):
        super().__init__()
        try:
            from torchvision.models import (
                resnet18, resnet34, ResNet18_Weights, ResNet34_Weights,
            )
        except (ImportError, RuntimeError) as e:
            raise ImportError("Please install a matching torch + torchvision pair") from e
        constructors = {
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
            "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
        }
        if encoder_name not in constructors:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        make, weights = constructors[encoder_name]
        # When pretrained=True torchvision may download weights; enable Kaggle
        # internet or pre-cache the matching checkpoint.
        net = make(weights=weights if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3  # 1/16, 256 channels, skip layer4 for T4
        self.project = nn.Sequential(
            nn.Conv2d(256, context_dim, 1, bias=False),
            nn.GroupNorm(8 if context_dim % 8 == 0 else 1, context_dim),
        )
        self.context = build_context(context_dim, context_type, context_depth)
        self.out_channels = context_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(self.stem(x))
        x = self.layer2(x)
        x = self.layer3(x)
        return self.context(self.project(x))