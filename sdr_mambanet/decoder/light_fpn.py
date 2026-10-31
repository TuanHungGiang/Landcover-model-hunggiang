"""Lightweight 1/8->1/4 FPN decoder, logits resized to original dimensions."""
from torch import nn
from torch.nn import functional as F
import torch
from ..blocks.rsf_rep import RSFRepBlock


class LightFPNDecoder(nn.Module):
    def __init__(self, detail4_channels: int = 48, fused8_channels: int = 96,
                 decoder_channels: int = 64, num_classes: int = 7,
                 rep_kernel: int = 7, boundary_aux: bool = False):
        super().__init__()
        self.lat8 = nn.Conv2d(fused8_channels, decoder_channels, 1)
        self.lat4 = nn.Conv2d(detail4_channels, decoder_channels, 1)
        self.refine = RSFRepBlock(decoder_channels, kernel_size=rep_kernel)
        self.head = nn.Conv2d(decoder_channels, num_classes, 1)
        self.boundary_head = nn.Conv2d(decoder_channels, 1, 1) if boundary_aux else None

    def forward(self, d4: torch.Tensor, f8: torch.Tensor,
                out_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        x = F.interpolate(self.lat8(f8), size=d4.shape[-2:],
                          mode="bilinear", align_corners=False) + self.lat4(d4)
        x = self.refine(x)
        logits = self.head(x)
        result = {"logits": F.interpolate(logits, size=out_size, mode="bilinear",
                                           align_corners=False)}
        if self.training and self.boundary_head is not None:
            edge = self.boundary_head(x)
            result["boundary_logits"] = F.interpolate(edge, size=out_size,
                                                         mode="bilinear", align_corners=False)
        return result

    def remove_training_only_heads(self) -> None:
        self.boundary_head = None