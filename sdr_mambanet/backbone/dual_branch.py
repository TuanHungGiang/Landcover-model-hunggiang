"""Asymmetric dual-branch backbone: detail CNN and semantic context encoder."""
from torch import nn
import torch
from .detail import DetailEncoder
from .semantic import SemanticEncoder


class DualBranchBackbone(nn.Module):
    def __init__(self, *, detail4: int = 48, detail8: int = 96,
                 depth4: int = 2, depth8: int = 2, rep_kernel: int = 7,
                 semantic_encoder: str = "resnet18", pretrained_semantic: bool = False,
                 context_dim: int = 160, context_depth: int = 2,
                 context_type: str = "conv", semantic_weights_path: str | None = None):
        super().__init__()
        self.detail = DetailEncoder(detail4, detail8, depth4, depth8, rep_kernel)
        self.semantic = SemanticEncoder(
            encoder_name=semantic_encoder, pretrained=pretrained_semantic,
            context_dim=context_dim, context_depth=context_depth,
            context_type=context_type, weights_path=semantic_weights_path,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        maps = self.detail(x)
        maps["s16"] = self.semantic(x)
        return maps