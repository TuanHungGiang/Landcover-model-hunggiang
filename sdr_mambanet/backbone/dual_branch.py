"""Full ResNet34 and RSF-Rep detail: staged bidirectional exchange.

At 1/8: detail <-> ResNet layer2; at 1/16: ResNet layer3
receives contextual layer4 information and 1/8 detail feedback.
The top-level model adds a final semantic->detail gate before the decoder.
"""
from __future__ import annotations

import torch
from torch import nn
from .detail import DetailEncoder
from .semantic import SemanticEncoder
from ..blocks.fusion import BidirectionalStageFusion, SemanticCrossScaleFusion


class DualBranchBackbone(nn.Module):
    def __init__(self, *, detail4: int = 48, detail8: int = 96,
                 depth4: int = 2, depth8: int = 2, rep_kernel: int = 7,
                 semantic_encoder: str = "resnet34", pretrained_semantic: bool = False,
                 context_dim: int = 160, context_depth: int = 2,
                 context_type: str = "conv", semantic_weights_path: str | None = None):
        super().__init__()
        self.detail = DetailEncoder(detail4, detail8, depth4, depth8, rep_kernel)
        self.semantic = SemanticEncoder(
            encoder_name=semantic_encoder, pretrained=pretrained_semantic,
            context_dim=context_dim, context_depth=context_depth,
            context_type=context_type, weights_path=semantic_weights_path,
        )
        self.bi8 = BidirectionalStageFusion(detail8, 128)
        self.cross16 = SemanticCrossScaleFusion(detail8, context_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        details = self.detail(x)
        d4, d8 = details["d4"], details["d8"]
        s4 = self.semantic.encode_s4(x)
        s8 = self.semantic.encode_s8(s4)
        d8, s8 = self.bi8(d8, s8)
        s16_raw = self.semantic.encode_s16(s8)
        s32 = self.semantic.encode_s32(s16_raw)
        s16 = self.cross16(s16_raw, s32, d8)
        return {"d4": d4, "d8": d8, "s8": s8, "s16": s16, "s32": s32}
