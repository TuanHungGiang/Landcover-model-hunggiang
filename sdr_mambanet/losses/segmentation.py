"""Multi-class segmentation losses (LoveDA/Potsdam compatible).

No topology loss here: connectivity-specific losses require a task-specific
road mask/centerline target and should NOT be applied blindly to all classes.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   ignore_index: int = 255, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft Dice loss over valid classes with at least one GT pixel."""
    classes = logits.shape[1]
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    probs = logits.float().softmax(dim=1)
    hot = F.one_hot(safe_target.long(), num_classes=classes).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1).float()
    probs = probs * mask
    hot = hot * mask
    inter = (probs * hot).sum(dim=(0, 2, 3))
    union = (probs + hot).sum(dim=(0, 2, 3))
    keep = hot.sum(dim=(0, 2, 3)) > 0
    if not torch.any(keep):
        return logits.sum() * 0
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice[keep].mean()


def get_multiclass_boundary(target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Binary class-transition map, excluding transitions involving ignored pixels."""
    valid = target != ignore_index
    edge = torch.zeros_like(valid)
    hor = (target[:, :, 1:] != target[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    ver = (target[:, 1:, :] != target[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, :, 1:] |= hor
    edge[:, :, :-1] |= hor
    edge[:, 1:, :] |= ver
    edge[:, :-1, :] |= ver
    return edge.unsqueeze(1).float()


class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, dice_weight: float = 1.0,
                 boundary_weight: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, out: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = out["logits"]
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(f"target BHW mismatch: {target.shape} vs {logits.shape}")
        valid = (target != self.ignore_index)
        if valid.any():
            ce = F.cross_entropy(logits.float(), target.long(), ignore_index=self.ignore_index)
        else:
            ce = logits.sum() * 0.0
        dice = soft_dice_loss(logits, target, ignore_index=self.ignore_index)
        total = ce + self.dice_weight * dice
        values = {"ce": ce, "dice": dice}
        if "boundary_logits" in out and self.boundary_weight > 0:
            edges = get_multiclass_boundary(target, ignore_index=self.ignore_index)
            per_pixel = F.binary_cross_entropy_with_logits(
                out["boundary_logits"].float(), edges, reduction="none"
            )
            boundary = (per_pixel * valid.unsqueeze(1)).sum() / valid.sum().clamp(min=1)
            values["boundary"] = boundary
            total = total + self.boundary_weight * boundary
        values["total"] = total
        return values