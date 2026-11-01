"""Experimental training-only augmentation for LoveDA (no validation changes).

Indexed masks are resized with nearest neighbor; 255 remains ignore. All new
augmentations are opt-in through the v3 trainer (v2 is left unchanged).
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from scripts.loveda_fast import FastLoveDA, MEAN, STD


def jitter_rgb(img: np.ndarray, strength: float) -> np.ndarray:
    """Modest brightness/contrast/saturation jitter without external dependencies."""
    if strength <= 0:
        return img
    x = img.astype(np.float32)
    b = 1.0 + random.uniform(-strength, strength)
    c = 1.0 + random.uniform(-strength, strength)
    s = 1.0 + random.uniform(-0.75 * strength, 0.75 * strength)
    x = (x - 127.5) * c + 127.5
    gray = np.mean(x, axis=2, keepdims=True)
    x = gray + (x - gray) * s
    x *= b
    return np.clip(x, 0, 255).astype(np.uint8)


class RegularizedLoveDA(FastLoveDA):
    """FastLoveDA with optional random zoom, target-aware crop, and color jitter.

    Only used for Train. The original FastLoveDA is used for all validation,
    calibration, and held-out evaluation. Focus sampling is non-uniform training
    sampling, NOT a modification of ground-truth semantics or evaluation protocol.
    """
    def __init__(self, pairs, size, train=True, cache_size=2, *,
                 augment: str = "strong", focus_probability: float = 0.50,
                 focus_classes: Sequence[int] = (4, 6),
                 zoom_min: float = 1.0, zoom_max: float = 1.35):
        super().__init__(pairs, size, train, cache_size)
        if not train:
            raise ValueError("RegularizedLoveDA must only be used for training")
        if augment not in {"none", "light", "strong"}:
            raise ValueError("augment must be none/light/strong")
        if not 0 <= focus_probability <= 1:
            raise ValueError("focus_probability must be in [0,1]")
        if not 0.5 <= zoom_min <= zoom_max <= 2.0:
            raise ValueError("zoom factors must satisfy 0.5<=min<=max<=2")
        if any(int(c) not in range(7) for c in focus_classes):
            raise ValueError("focus classes must be training labels 0..6")
        self.augment = augment
        self.focus_probability = focus_probability
        self.focus_classes = tuple(int(c) for c in focus_classes)
        self.zoom_min, self.zoom_max = zoom_min, zoom_max

    def __getitem__(self, index):
        if self.augment == "none":
            # Preserve the exact original training dataset behavior when disabled.
            return super().__getitem__(index)
        img, mask = self._pair(index)
        h, w = mask.shape
        size = self.size
        if self.augment == "strong":
            zoom = random.uniform(self.zoom_min, self.zoom_max)
        else:
            zoom = random.uniform(1.0, min(self.zoom_max, 1.12))
        if abs(zoom - 1) > 1e-6:
            scaled_w = max(1, int(round(w * zoom)))
            scaled_h = max(1, int(round(h * zoom)))
            img = np.asarray(Image.fromarray(img).resize((scaled_w, scaled_h), Image.Resampling.BILINEAR))
            mask = np.asarray(Image.fromarray(mask).resize((scaled_w, scaled_h), Image.Resampling.NEAREST))
            h, w = mask.shape
        # Crop with target-aware offset when target class is available.
        chosen = None
        if self.focus_classes and random.random() < self.focus_probability:
            candidates = [c for c in self.focus_classes if np.any(mask == c)]
            if candidates:
                c = random.choice(candidates)
                coords = np.flatnonzero(mask.reshape(-1) == c)
                idx = int(coords[random.randrange(len(coords))])
                cy, cx = divmod(idx, w)
                cx_off = random.randint(size // 4, max(size // 4, 3 * size // 4))
                cy_off = random.randint(size // 4, max(size // 4, 3 * size // 4))
                chosen = (max(0, min(w - size, cx - cx_off)),
                          max(0, min(h - size, cy - cy_off)))
        if chosen is None:
            x = random.randint(0, max(0, w - size))
            y = random.randint(0, max(0, h - size))
        else:
            x, y = chosen
        # Pad to fixed shape if input is smaller than the crop. Padded mask ignored.
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        target = np.full((size, size), 255, dtype=np.uint8)
        hh, ww = min(size, max(0, h-y)), min(size, max(0, w-x))
        crop[:hh, :ww] = img[y:y+hh, x:x+ww]
        target[:hh, :ww] = mask[y:y+hh, x:x+ww]
        if random.random() < 0.5:
            crop, target = np.flip(crop, axis=0), np.flip(target, axis=0)
        if random.random() < 0.5:
            crop, target = np.flip(crop, axis=1), np.flip(target, axis=1)
        k = random.randrange(4)
        crop, target = np.rot90(crop, k), np.rot90(target, k)
        crop = np.ascontiguousarray(crop)
        target = np.ascontiguousarray(target)
        if random.random() < 0.80:
            crop = jitter_rgb(crop, strength=0.08 if self.augment == "light" else 0.15)
        normalized = (crop.astype(np.float32)/255.0 - MEAN)/STD
        return (torch.from_numpy(np.ascontiguousarray(normalized.transpose(2,0,1))),
                torch.from_numpy(target.astype(np.int64)))


def top_confusions(confusion, names, top_k=5):
    """Highest conditional P(pred | ground-truth), excluding correct classes."""
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion must be square")
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    row_sum = cm.sum(axis=1)
    normalized = np.divide(errors, row_sum[:, None],
                           out=np.zeros_like(errors), where=row_sum[:, None]>0)
    pairs=[]
    for gt in range(cm.shape[0]):
        for pred in range(cm.shape[1]):
            if gt != pred and errors[gt,pred] > 0:
                pairs.append({"gt":names[gt],"pred":names[pred],
                              "percent_of_gt":round(float(normalized[gt,pred]*100),2),
                              "pixels":int(errors[gt,pred])})
    pairs.sort(key=lambda row: row["percent_of_gt"], reverse=True)
    return pairs[:top_k]


def lr_multiplier(epoch: int, epochs: int, warmup_epochs: int, min_ratio: float = 0.1) -> float:
    """Epoch-based linear warmup then cosine decay (epoch=0 is first epoch)."""
    if epochs < 1 or not 0 <= warmup_epochs < epochs or not 0 < min_ratio <= 1:
        raise ValueError("Invalid lr schedule")
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)))
    return min_ratio + (1.0-min_ratio)*0.5*(1+math.cos(math.pi*progress))


def build_discriminative_groups(model: torch.nn.Module, base_lr: float,
                                backbone_lr: float):
    """Pretrained ResNet layers receive backbone_lr; new modules receive base_lr."""
    resnet = model.backbone.semantic
    old = []
    for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
        old.extend(getattr(resnet,name).parameters())
    old_ids = {id(p) for p in old}
    new = [p for p in model.parameters() if p.requires_grad and id(p) not in old_ids]
    old = [p for p in old if p.requires_grad]
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if not old or not new or (old_ids & {id(p) for p in new}) or {id(p) for p in old+new} != all_ids:
        raise RuntimeError("Invalid optimizer groups; some parameters missing/duplicated")
    return [dict(params=old, lr=backbone_lr, name="pretrained_resnet34"),
            dict(params=new, lr=base_lr, name="new_detail_fusion_context_decoder")]