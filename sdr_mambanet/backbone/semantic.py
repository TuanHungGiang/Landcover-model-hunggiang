"""Full torchvision ResNet34 plus contextual blocks at layer4 (1/32).

layer2: 1/8, 128 channels; layer3: 1/16, 256 channels;
layer4: 1/32, 512 channels. Only ResNet layers use ImageNet pretraining.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from ..blocks.context import build_context


class SemanticEncoder(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = False,
                 context_dim: int = 160, context_depth: int = 2,
                 context_type: str = "conv", weights_path: str | None = None):
        super().__init__()
        try:
            from torchvision.models import resnet34, ResNet34_Weights
        except (ImportError, RuntimeError) as exc:
            raise ImportError("Install compatible torch and torchvision versions") from exc
        if encoder_name != "resnet34":
            raise ValueError("This full-layer4 bidirectional architecture requires resnet34")
        net = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained and not weights_path else None)
        if weights_path:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"ResNet34 checkpoint not found: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError("Expected a torchvision ResNet34 state_dict")
            state = {key.removeprefix("module."): value for key, value in state.items()}
            required = ("conv1.weight", "layer1.0.conv1.weight",
                        "layer3.0.conv1.weight", "layer4.0.conv1.weight")
            if any(key not in state for key in required):
                raise ValueError(f"Local checkpoint must include all ResNet34 layers: {required}")
            incompatible = net.load_state_dict(state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(f"[ResNet] Checkpoint differences: {incompatible}", flush=True)
            print(f"[ResNet] Loaded ImageNet weights: {path}", flush=True)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.project32 = nn.Sequential(
            nn.Conv2d(512, context_dim, 1, bias=False),
            nn.GroupNorm(8 if context_dim % 8 == 0 else 1, context_dim),
        )
        self.context = build_context(context_dim, context_type, context_depth)
        self.out_channels = context_dim

    def encode_s4(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(self.stem(x))

    def encode_s8(self, s4: torch.Tensor) -> torch.Tensor:
        return self.layer2(s4)

    def encode_s16(self, s8: torch.Tensor) -> torch.Tensor:
        return self.layer3(s8)

    def encode_s32(self, s16: torch.Tensor) -> torch.Tensor:
        return self.context(self.project32(self.layer4(s16)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s4 = self.encode_s4(x)
        s8 = self.encode_s8(s4)
        s16 = self.encode_s16(s8)
        s32 = self.encode_s32(s16)
        return {"s4": s4, "s8": s8, "s16": s16, "s32": s32}
