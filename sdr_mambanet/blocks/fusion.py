"""Semantic-to-detail residual fusion inspired by source PAPER MOTIVATIONS,
not an exact reproduction of CRNet's SSFRM or DUSA-UNet's Dual-SA.
"""
from torch import nn
from torch.nn import functional as F
import torch


class SemanticGuidedFusion(nn.Module):
    def __init__(self, detail_channels: int, semantic_channels: int, out_channels: int = 96):
        super().__init__()
        self.detail_proj = nn.Conv2d(detail_channels, out_channels, 1, bias=False)
        self.semantic_proj = nn.Conv2d(semantic_channels, out_channels, 1, bias=False)
        self.gate = nn.Conv2d(out_channels * 2, out_channels, 1)
        self.semantic_scale = nn.Parameter(torch.tensor(0.1))
        self.post = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.GroupNorm(8 if out_channels % 8 == 0 else 1, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        sem = F.interpolate(semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        d, s = self.detail_proj(detail), self.semantic_proj(sem)
        gate = torch.sigmoid(self.gate(torch.cat([d, s], dim=1)))
        mixed = d + self.semantic_scale * gate * s
        return mixed + self.post(mixed)