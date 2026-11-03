"""HCR: train-confusion hierarchy + reliability-anchored local graph refinement.

Research prototype. No validation/test labels enter the hierarchy construction.
Both spatial paths from SDRMambaNet are retained; the HCR module acts on logits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import torch
from torch import nn
from torch.nn import functional as F
from .utils.checkpointing import checkpoint_module

from .models import SDRConfig, SDRMambaNet


@dataclass
class HCRConfig(SDRConfig):
    architecture_version: str = "r34_layer4_bi_hcr_v1"
    hierarchy_mix: float = 0.25
    graph_steps: int = 3
    graph_lambda: float = 0.5
    graph_eta: float = 0.5
    graph_channels: int = 24
    memory_saver: bool = False


def build_confusion_groups(confusion: torch.Tensor, min_score: float = 0.02):
    """Disjoint pairs chosen from *training* confusion, remaining classes singleton.

    Symmetric per-class normalization prevents frequent classes dominating.
    No group selection is allowed from validation/holdout statistics.
    """
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Expected square confusion matrix")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be in [0,1]")
    n = confusion.shape[0]
    counts = confusion.double().sum(1).clamp_min(1.0)
    rates = confusion.double() / counts[:, None]
    symmetric = 0.5 * (rates + rates.T)
    symmetric.fill_diagonal_(0)
    options = sorted(
        ((float(symmetric[i, j]), i, j) for i in range(n) for j in range(i + 1, n)),
        reverse=True,
    )
    used, groups = set(), []
    for score, i, j in options:
        if score < min_score:
            break
        if i not in used and j not in used:
            groups.append([i, j])
            used.update((i, j))
    for i in range(n):
        if i not in used:
            groups.append([i])
    groups.sort(key=lambda g: min(g))
    mapping = torch.empty(n, dtype=torch.long)
    for g, members in enumerate(groups):
        for cls in members:
            mapping[cls] = g
    return mapping, groups


class ReliabilityGraphRefiner(nn.Module):
    """Sparse 4-neighbor, symmetric, confidence-anchored Jacobi smoothing.

    The graph is built from semantic + RSF detail features, the coarse hierarchy,
    and an independently supervised error-risk predictor.
    """
    OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, detail_channels: int, semantic_channels: int,
                 fused_channels: int, classes: int, embed: int = 24,
                 steps: int = 3, smooth_lambda: float = 0.5, eta: float = 0.5):
        super().__init__()
        if steps < 1 or smooth_lambda < 0 or not (0 <= eta <= 1):
            raise ValueError("Graph steps>=1, lambda>=0, eta in [0,1] required")
        self.steps, self.smooth_lambda, self.eta = steps, smooth_lambda, eta
        self.detail_proj = nn.Conv2d(detail_channels, embed, 1, bias=False)
        self.semantic_proj = nn.Conv2d(semantic_channels, embed, 1, bias=False)
        self.risk_head = nn.Sequential(
            nn.Conv2d(fused_channels + classes * 2 + 1, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 1, 1),
        )
        nn.init.constant_(self.risk_head[-1].bias, -1.0)

    @staticmethod
    def _shift_valid(x: torch.Tensor, dy: int, dx: int):
        # roll corresponds to neighbor at (row - dy, col - dx).
        y = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        mask = torch.ones((1, 1, x.shape[-2], x.shape[-1]),
                          device=x.device, dtype=x.dtype)
        if dy > 0:
            mask[..., :dy, :] = 0
        if dy < 0:
            mask[..., dy:, :] = 0
        if dx > 0:
            mask[..., :, :dx] = 0
        if dx < 0:
            mask[..., :, dx:] = 0
        return y, mask

    def forward(self, detail, semantic, fused, flat_probs, hier_probs,
                group_probs, mixed_probs):
        h, w = fused.shape[-2:]
        detail = F.interpolate(detail, (h, w), mode="bilinear", align_corners=False)
        semantic = F.interpolate(semantic, (h, w), mode="bilinear", align_corners=False)
        sd = F.normalize(self.detail_proj(detail).float(), dim=1, eps=1e-6)
        ss = F.normalize(self.semantic_proj(semantic).float(), dim=1, eps=1e-6)

        flat_probs = flat_probs.float().clamp_min(1e-6)
        hier_probs = hier_probs.float().clamp_min(1e-6)
        mean = 0.5 * (flat_probs + hier_probs)
        js = 0.5 * (
            (flat_probs * (flat_probs / mean).log()).sum(1, keepdim=True)
            + (hier_probs * (hier_probs / mean).log()).sum(1, keepdim=True)
        )
        # Detached predictions prevent the error-risk supervision from
        # manufacturing disagreement through the classifiers.
        risk_input = torch.cat((fused.float(), flat_probs.detach(),
                                hier_probs.detach(), js.detach()), dim=1)
        risk_logits = self.risk_head(risk_input).float()
        risk = risk_logits.sigmoid()
        anchor = 0.2 + (1.0 - risk)  # strictly positive, avoids singular solve
        z0 = mixed_probs.float().clamp_min(1e-6).log()
        z = z0
        weights = []
        degree = torch.zeros_like(risk)
        for dy, dx in self.OFFSETS:
            sd_n, mask = self._shift_valid(sd, dy, dx)
            ss_n, _ = self._shift_valid(ss, dy, dx)
            group_n, _ = self._shift_valid(group_probs.float(), dy, dx)
            risk_n, _ = self._shift_valid(risk, dy, dx)
            sim_d = (sd * sd_n).sum(1, keepdim=True).clamp(-1, 1)
            sim_s = (ss * ss_n).sum(1, keepdim=True).clamp(-1, 1)
            visual = torch.exp(-0.7 * ((1 - sim_d) + (1 - sim_s)))
            same_group = (group_probs.float() * group_n).sum(1, keepdim=True)
            loosen = torch.maximum(risk, risk_n)
            agreement = 0.10 + (1 - loosen) * same_group + loosen
            wij = visual * agreement * mask
            weights.append((dy, dx, wij))
            degree = degree + wij
        for _ in range(self.steps):
            neighbor_sum = torch.zeros_like(z)
            for dy, dx, wij in weights:
                z_n, _ = self._shift_valid(z, dy, dx)
                neighbor_sum = neighbor_sum + wij * z_n
            z = (anchor * z0 + self.smooth_lambda * neighbor_sum) / (
                anchor + self.smooth_lambda * degree).clamp_min(1e-6)
        delta = self.eta * risk * (z - z0)
        return delta, risk_logits


class HCRSDRNet(SDRMambaNet):
    """Preserve SDR backbone / RSF-Rep / three fusion stages; append HCR head."""

    def __init__(self, cfg: HCRConfig | None = None):
        cfg = HCRConfig() if cfg is None else cfg
        if cfg.architecture_version != "r34_layer4_bi_hcr_v1":
            raise ValueError("Wrong HCR architecture version")
        if not 0 <= cfg.hierarchy_mix <= 1:
            raise ValueError("hierarchy_mix must be between 0 and 1")
        # The parent builds the exact original feature extractor and decoder.
        base_names = {f.name for f in fields(SDRConfig)}
        base_args = {k: v for k, v in asdict(cfg).items() if k in base_names}
        base_args["architecture_version"] = "r34_layer4_bi_v2"
        super().__init__(SDRConfig(**base_args))
        self.cfg = cfg
        self.backbone.enable_gradient_checkpointing(cfg.memory_saver)
        self.decoder.activation_checkpointing = cfg.memory_saver
        k = cfg.num_classes
        self.group_head = nn.Conv2d(cfg.detail8, k, 1)
        self.fine_head = nn.Sequential(
            nn.Conv2d(cfg.detail8, cfg.detail8, 3, padding=1, groups=cfg.detail8),
            nn.GroupNorm(8 if cfg.detail8 % 8 == 0 else 1, cfg.detail8),
            nn.SiLU(), nn.Conv2d(cfg.detail8, k, 1),
        )
        self.graph_refiner = ReliabilityGraphRefiner(
            cfg.detail8, cfg.context_dim, cfg.detail8, k,
            embed=cfg.graph_channels, steps=cfg.graph_steps,
            smooth_lambda=cfg.graph_lambda, eta=cfg.graph_eta,
        )
        self.register_buffer("class_groups", torch.arange(k, dtype=torch.long))
        self.register_buffer("class_train_confusion", torch.zeros((k, k), dtype=torch.long))

    @torch.no_grad()
    def configure_groups_from_training(self, confusion: torch.Tensor, min_score: float = 0.02):
        mapping, groups = build_confusion_groups(confusion.detach().cpu(), min_score)
        self.class_groups.copy_(mapping.to(self.class_groups.device))
        self.class_train_confusion.copy_(confusion.to(self.class_train_confusion.device))
        return groups

    def _hierarchy(self, fused: torch.Tensor):
        # Both heads always produce K channels. Grouping changes only
        # deterministic probability aggregation; heads remain trainable.
        pg = self.group_head(fused).float().softmax(dim=1)
        pf = self.fine_head(fused).float().softmax(dim=1)
        ids = self.class_groups
        group_mass = torch.zeros_like(pg).index_add(1, ids, pg)
        fine_mass = torch.zeros_like(pf).index_add(1, ids, pf)
        hierarchical = group_mass.index_select(1, ids) * (
            pf / fine_mass.index_select(1, ids).clamp_min(1e-6))
        hierarchical = hierarchical / hierarchical.sum(1, keepdim=True).clamp_min(1e-6)
        return hierarchical, group_mass

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        maps = self.backbone(x)
        fused = (checkpoint_module(self.fusion, maps["d8"], maps["s16"])
                 if self.cfg.memory_saver else self.fusion(maps["d8"], maps["s16"]))
        decoder_result = self.decoder(maps["d4"], fused, x.shape[-2:])
        flat_logits = decoder_result["logits"]
        flat_probs = flat_logits.float().softmax(dim=1)
        ph8, pg8 = self._hierarchy(fused)
        pf8 = F.interpolate(flat_probs, size=fused.shape[-2:],
                            mode="bilinear", align_corners=False)
        p08 = (1 - self.cfg.hierarchy_mix) * pf8 + self.cfg.hierarchy_mix * ph8
        ph_full = F.interpolate(ph8, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        p0full = (1 - self.cfg.hierarchy_mix) * flat_probs + self.cfg.hierarchy_mix * ph_full
        graph_inputs = (maps["d8"], maps["s16"], fused, pf8, ph8, pg8, p08)
        delta, risk_logits = (checkpoint_module(self.graph_refiner, *graph_inputs)
                              if self.cfg.memory_saver else self.graph_refiner(*graph_inputs))
        correction = F.interpolate(delta, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        logits = p0full.clamp_min(1e-6).log() + correction
        result = {"logits": logits, "flat_logits": flat_logits,
                  "hier_probs": ph8, "risk_logits": risk_logits}
        if "boundary_logits" in decoder_result:
            result["boundary_logits"] = decoder_result["boundary_logits"]
        return result
