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
                 context_type: str = "conv", weights_path: str | None = None):
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
        net = make(weights=weights if pretrained and weights_path is None else None)
        if weights_path is not None:
            from pathlib import Path
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"ResNet checkpoint not found: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError("Expected torchvision ResNet state_dict checkpoint")
            state = {k.removeprefix("module."): v for k, v in state.items()}
            # Accept classification checkpoint with full ResNet weights.
            required = ("conv1.weight", "layer1.0.conv1.weight", "layer3.0.conv1.weight")
            if not all(k in state for k in required):
                raise ValueError(
                    "Local checkpoint does not contain torchvision ResNet encoder weights: "
                    + ", ".join(required)
                )
            incompatible = net.load_state_dict(state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(f"[ResNet] Local checkpoint key differences: {incompatible}", flush=True)
            print(f"[ResNet] Loaded weights from {path}", flush=True)
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