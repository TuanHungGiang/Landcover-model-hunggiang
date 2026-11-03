"""Training objective for HCR, with risk targets from Train ground truth only."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .segmentation import SegmentationLoss


class HCRSegmentationLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, hierarchy_weight: float = 0.2,
                 flat_weight: float = 0.2, risk_weight: float = 0.15):
        super().__init__()
        self.ignore_index = ignore_index
        self.hierarchy_weight = hierarchy_weight
        self.flat_weight = flat_weight
        self.risk_weight = risk_weight
        self.base = SegmentationLoss(ignore_index=ignore_index, dice_weight=1.0)

    def forward(self, out: dict[str, torch.Tensor], target: torch.Tensor):
        parts = self.base(out, target)
        valid = target != self.ignore_index
        if not bool(valid.any()):
            zero = out["logits"].sum() * 0.
            return {**parts, "hierarchy": zero, "flat": zero,
                    "risk": zero, "total": parts["total"] + zero}
        flat = out["flat_logits"]
        flat_ce = F.cross_entropy(flat.float(), target.long(), ignore_index=self.ignore_index)
        h, w = out["hier_probs"].shape[-2:]
        # Group/fine prediction supervised at the graph resolution.
        target_small = F.interpolate(target.unsqueeze(1).float(), size=(h, w),
                                     mode="nearest").squeeze(1).long()
        if bool((target_small != self.ignore_index).any()):
            hier_loss = F.nll_loss(out["hier_probs"].float().clamp_min(1e-6).log(),
                                   target_small, ignore_index=self.ignore_index)
        else:
            hier_loss = out["hier_probs"].sum() * 0.
        with torch.no_grad():
            # Train risk on the exact pre-graph prediction it will refine,
            # rather than on the flat head which may disagree with hierarchy.
            wrong = (out["mixed_logits"].detach().argmax(1) != target) & valid
            size = out["risk_logits"].shape[-2:]
            count = F.adaptive_avg_pool2d(valid[:, None].float(), size)
            errors = F.adaptive_avg_pool2d(wrong[:, None].float(), size)
            risk_target = (errors / count.clamp_min(1e-6)).clamp(0, 1)
            risk_valid = count > 0
        risk_pixels = F.binary_cross_entropy_with_logits(
            out["risk_logits"].float(), risk_target, reduction="none")
        risk_loss = (risk_pixels * risk_valid).sum() / risk_valid.sum().clamp_min(1)
        parts.update(hierarchy=hier_loss, flat=flat_ce, risk=risk_loss)
        parts["total"] = (parts["total"] + self.hierarchy_weight * hier_loss
                          + self.flat_weight * flat_ce + self.risk_weight * risk_loss)
        return parts
