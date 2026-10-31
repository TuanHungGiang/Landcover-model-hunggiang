"""Global-context alternatives for controlled ablation.

'conv': pure PyTorch large-kernel CNN baseline (NOT Mamba).
'mamba': genuine 1-D Mamba from mamba_ssm, applied to rasterized 2-D tokens.
Alternate HW and WH scan orders for directional diversity; no claim of
implementing VMamba's SS2D or transferring pretrained VMamba weights.
"""
from __future__ import annotations

from torch import nn
import torch


class ConvContextBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


class RasterMambaBlock(nn.Module):
    def __init__(self, channels: int, scan_order: str = "hw", d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        if scan_order not in ("hw", "wh"):
            raise ValueError("scan_order must be 'hw' or 'wh'")
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as e:
            raise ImportError(
                "context_type='mamba' requires a working CUDA build of mamba_ssm. "
                "Use context_type='conv' for a dependency-free CNN baseline."
            ) from e
        self.order = scan_order
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gamma = nn.Parameter(torch.full((channels,), 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.order == "hw":
            seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            out = self.mamba(self.norm(seq))
            y = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        else:
            seq = x.permute(0, 3, 2, 1).reshape(b, h * w, c)
            out = self.mamba(self.norm(seq))
            y = out.reshape(b, w, h, c).permute(0, 3, 2, 1)
        return x + y * self.gamma[None, :, None, None]


def build_context(channels: int, context_type: str, depth: int) -> nn.Sequential:
    if context_type not in ("conv", "mamba"):
        raise ValueError("context_type must be 'conv' or 'mamba'")
    blocks = []
    for i in range(depth):
        if context_type == "mamba":
            blocks.append(RasterMambaBlock(channels, scan_order="hw" if i % 2 == 0 else "wh"))
        else:
            blocks.append(ConvContextBlock(channels))
    return nn.Sequential(*blocks)