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

class BidirectionalStageFusion(nn.Module):
    """Bidirectional gated residual exchange between d8 and layer2 s8."""

    def __init__(self, detail_channels: int = 96, semantic_channels: int = 128):
        super().__init__()
        self.to_detail = nn.Conv2d(semantic_channels, detail_channels, 1, bias=False)
        self.to_semantic = nn.Conv2d(detail_channels, semantic_channels, 1, bias=False)
        self.detail_gate = nn.Conv2d(2 * detail_channels, detail_channels, 1)
        self.semantic_gate = nn.Conv2d(2 * semantic_channels, semantic_channels, 1)
        self.s2d_scale = nn.Parameter(torch.tensor(0.10))
        self.d2s_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, d8: torch.Tensor, s8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if d8.shape[-2:] != s8.shape[-2:]:
            s8 = F.interpolate(s8, size=d8.shape[-2:], mode="bilinear", align_corners=False)
        s_to_d = self.to_detail(s8)
        d_to_s = self.to_semantic(d8)
        dg = torch.sigmoid(self.detail_gate(torch.cat((d8, s_to_d), dim=1)))
        sg = torch.sigmoid(self.semantic_gate(torch.cat((s8, d_to_s), dim=1)))
        return d8 + self.s2d_scale * dg * s_to_d, s8 + self.d2s_scale * sg * d_to_s


class SemanticCrossScaleFusion(nn.Module):
    """At 1/16, combine layer3, top-down contextual layer4, and detail d8."""

    def __init__(self, detail_channels: int = 96, context_dim: int = 160):
        super().__init__()
        self.from_layer3 = nn.Sequential(
            nn.Conv2d(256, context_dim, 1, bias=False),
            nn.GroupNorm(8 if context_dim % 8 == 0 else 1, context_dim),
        )
        self.from_detail = nn.Conv2d(detail_channels, context_dim, 1, bias=False)
        self.deep_gate = nn.Conv2d(context_dim * 2, context_dim, 1)
        self.detail_gate = nn.Conv2d(context_dim * 2, context_dim, 1)
        self.deep_scale = nn.Parameter(torch.tensor(0.10))
        self.detail_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, s16: torch.Tensor, s32: torch.Tensor,
                d8: torch.Tensor) -> torch.Tensor:
        high = self.from_layer3(s16)
        deep = F.interpolate(s32, size=high.shape[-2:], mode="bilinear", align_corners=False)
        deep_g = torch.sigmoid(self.deep_gate(torch.cat((high, deep), dim=1)))
        mixed = high + self.deep_scale * deep_g * deep
        detail = self.from_detail(F.interpolate(
            d8, size=high.shape[-2:], mode="bilinear", align_corners=False))
        detail_g = torch.sigmoid(self.detail_gate(torch.cat((mixed, detail), dim=1)))
        return mixed + self.detail_scale * detail_g * detail
