"""Efficient LoveDA inputs, pixel metrics, sharded evaluation, calibration.

Original LoveDA: raw 0/255 are ignored; raw labels 1..7 => model 0..6.
Uses bounded per-worker decoded-image LRU cache. Does not cache entire holdout.
No segmentation-model parameters or module behavior are altered here.
"""
from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, Sampler

ROOT = "/kaggle/input/datasets/alienxc137/loveda-satellite-images-semantic-segmentation"
NAMES = ("background", "building", "road", "water", "barren", "forest", "agriculture")
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MEAN = np.array((.485, .456, .406), dtype=np.float32).reshape(1, 1, 3)
STD = np.array((.229, .224, .225), dtype=np.float32).reshape(1, 1, 3)


def find_pairs(root: Path, split: str):
    roots = ("Train", "train") if split == "train" else ("Val", "val", "Validation")
    candidates = [root / n for n in roots] + [root / n / m for n in roots for m in roots]
    for parent in candidates:
        pairs = []
        for domain in ("Rural", "Urban"):
            base = parent / domain
            img_dir = next((base / p for p in ("images_png", "images") if (base / p).is_dir()), None)
            mask_dir = next((base / p for p in ("masks_png", "masks", "labels") if (base / p).is_dir()), None)
            if img_dir is None or mask_dir is None:
                continue
            masks = {p.stem.lower(): p for p in mask_dir.iterdir() if p.suffix.lower() in EXTS}
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in EXTS:
                    continue
                if p.stem.lower() not in masks:
                    raise FileNotFoundError(f"Missing mask for {p}")
                pairs.append((p, masks[p.stem.lower()], domain))
        if pairs:
            print(f"{split}: {len(pairs)} images (Rural={sum(p[2]=='Rural' for p in pairs)}, "
                  f"Urban={sum(p[2]=='Urban' for p in pairs)})", flush=True)
            return pairs
    raise FileNotFoundError(f"Cannot locate LoveDA {split} images and masks under {root}")


class FastLoveDA(Dataset):
    """Whole-image decoding once per worker per LRU hit, tiling afterward.

    Each validation example is still exactly the original non-overlapping crop,
    including ignore padding on non-divisible image dimensions. Training still
    uses one random crop/image per epoch with the original simple augmentation.
    """
    def __init__(self, pairs, size: int, train: bool, cache_size: int = 8):
        self.pairs = pairs
        self.size = int(size)
        self.train = bool(train)
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()
        self.tiles = []
        if self.size < 32 or self.cache_size < 0:
            raise ValueError("Invalid size or cache size")
        if not self.train:
            for i, (path, _, _) in enumerate(self.pairs):
                with Image.open(path) as image:
                    w, h = image.size
                for y in range(0, h, self.size):
                    for x in range(0, w, self.size):
                        self.tiles.append((i, x, y))

    def __len__(self):
        return len(self.pairs) if self.train else len(self.tiles)

    def _decode(self, i):
        path, mask_path, _ = self.pairs[i]
        with Image.open(path) as f:
            img = np.asarray(f.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as f:
            mask = np.asarray(f).copy()
        if mask.ndim != 2:
            raise ValueError(f"Expected indexed/grayscale LoveDA mask: {mask_path}")
        if mask.shape != img.shape[:2]:
            raise ValueError(f"Image and mask shape mismatch: {path} vs {mask_path}")
        if not np.isin(mask, (0, 1, 2, 3, 4, 5, 6, 7, 255)).all():
            raise ValueError(f"Unexpected LoveDA mask IDs in {mask_path}: {np.unique(mask)}")
        target = np.full(mask.shape, 255, dtype=np.uint8)
        valid = (mask >= 1) & (mask <= 7)
        target[valid] = mask[valid] - 1
        return img, target

    def _pair(self, i):
        if self.cache_size and i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        pair = self._decode(i)
        if self.cache_size:
            self._cache[i] = pair
            self._cache.move_to_end(i)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return pair

    def __getitem__(self, j):
        if self.train:
            i = j
        else:
            i, x, y = self.tiles[j]
        img, target = self._pair(i)
        h, w = target.shape
        s = self.size
        if self.train:
            x = random.randint(0, max(0, w - s))
            y = random.randint(0, max(0, h - s))
        end_x = min(w, x + s)
        end_y = min(h, y + s)
        # Fixed-size crop, padded with raw black image and ignore mask.
        crop = np.zeros((s, s, 3), dtype=np.uint8)
        mask = np.full((s, s), 255, dtype=np.uint8)
        ph, pw = max(0, end_y - y), max(0, end_x - x)
        crop[:ph, :pw] = img[y:end_y, x:end_x]
        mask[:ph, :pw] = target[y:end_y, x:end_x]
        if self.train:
            if random.random() < .5:
                crop, mask = np.flip(crop, 0), np.flip(mask, 0)
            if random.random() < .5:
                crop, mask = np.flip(crop, 1), np.flip(mask, 1)
            rotation = random.randrange(4)
            crop, mask = np.rot90(crop, rotation), np.rot90(mask, rotation)
        normalized = (np.ascontiguousarray(crop).astype(np.float32) / 255 - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64))))


class ImageShardedEvalSampler(Sampler[int]):
    """No padding/duplication, each original image assigned to only one rank.

    All tiles from an image stay on the same rank so splitting 200 LoveDA
    1024x1024 images produces exactly 400 tiles per rank at 512 resolution.
    """
    def __init__(self, dataset: FastLoveDA, rank: int, world_size: int):
        if dataset.train:
            raise ValueError("Only valid for evaluation datasets")
        self.indices = [j for j, (image_id, _, _) in enumerate(dataset.tiles)
                        if image_id % world_size == rank]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def accumulate_confusion(matrix: torch.Tensor, predicted: torch.Tensor,
                         target: torch.Tensor) -> None:
    """GPU-side 7x7 confusion; invalid GT stored as extra bincount bin.

    Avoids copying full predicted masks to CPU or masking into dynamic GPU sizes.
    """
    if predicted.shape != target.shape:
        raise ValueError("Predicted/target shape mismatch")
    labels = target.to(torch.int64)
    codes = torch.where(labels == 255, 49, labels * 7 + predicted.to(torch.int64))
    matrix.add_(torch.bincount(codes.reshape(-1), minlength=50)[:49].reshape(7, 7))


def confusion_metrics(matrix: torch.Tensor) -> dict:
    cm = matrix.cpu().numpy().astype(np.int64, copy=False)
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros(7, dtype=np.float64), where=union != 0)
    # Match train_loveda.validate: absent classes excluded from mIoU.
    result = {"miou": float(iou[union > 0].mean()) if (union > 0).any() else 0.,
              "oa": float(tp.sum() / max(1, cm.sum())),
              "iou_per_class": {name: float(v) for name, v in zip(NAMES, iou)},
              "confusion": cm.tolist()}
    return result


@torch.no_grad()
def synchronize_eval_buffers(model) -> None:
    """Synchronize rank-0 BN running stats and other buffers before DDP evaluation.

    DDP broadcasts buffers *before* training forwards, but BatchNorm then
    updates them locally on each rank. Since validation calls ddp.module
    directly (not DDP), broadcasts are needed to evaluate one consistent model.
    All ranks must call this helper in identical collective order.
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    for buffer in model.buffers():
        if buffer.numel() > 0:
            dist.broadcast(buffer, src=0)


@torch.no_grad()
def eval_distributed(model, loader, device: torch.device, thresholds=None, sync: bool = True,
                     channels_last: bool = False):
    """Both ranks evaluate separate full images; only TWO allreduces at end."""
    if sync:
        synchronize_eval_buffers(model)
    model.eval()
    raw = torch.zeros((7, 7), device=device, dtype=torch.int64)
    adjusted = torch.zeros_like(raw) if thresholds is not None else None
    if thresholds is not None:
        bias = torch.as_tensor(thresholds, device=device, dtype=torch.float32).log().view(1, 7, 1, 1)
    for image, target in loader:
        image = image.to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(image)["logits"]
        scores = logits.float()
        accumulate_confusion(raw, scores.argmax(1), target)
        if adjusted is not None:
            accumulate_confusion(adjusted, (scores - bias).argmax(1), target)
    if sync and dist.is_available() and dist.is_initialized():
        dist.all_reduce(raw, op=dist.ReduceOp.SUM)
        if adjusted is not None:
            dist.all_reduce(adjusted, op=dist.ReduceOp.SUM)
    if adjusted is None:
        return confusion_metrics(raw)
    return {"argmax": confusion_metrics(raw), "calibrated": confusion_metrics(adjusted)}


@torch.no_grad()
def calibration_samples(model, loader, device: torch.device, per_tile: int = 2048,
                        seed: int = 42, channels_last: bool = False):
    """Transfer only sampled logits, NOT complete float probability maps.

    Sample locations from CPU GT first, gather logits at selected spatial sites
    using an efficient GPU gather; return small arrays for cross-rank collection.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    logit_batches = []
    label_batches = []
    for image, target in loader:
        y = target[0].numpy().reshape(-1)
        valid = np.flatnonzero(y != 255)
        # In reference pipeline validation batch is always one tile.
        if not valid.size:
            continue
        inds = rng.choice(valid, size=min(per_tile, valid.size), replace=False)
        image = image.to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(image)["logits"]
        index = torch.as_tensor(inds, device=device, dtype=torch.long)
        sampled = logits[0].reshape(7, -1).index_select(1, index).T
        logit_batches.append(sampled.to(torch.float16).cpu().numpy())
        label_batches.append(y[inds].astype(np.uint8))
    if not logit_batches:
        return np.empty((0, 7), np.float16), np.empty((0,), np.uint8)
    return np.concatenate(logit_batches, 0), np.concatenate(label_batches, 0)


@torch.no_grad()
def tune_thresholds_logits(logits: np.ndarray, targets: np.ndarray,
                           device: torch.device, passes: int = 2):
    """Tune classwise bias using GPU sampled logits; no per-candidate softmax.

    argmax(softmax(logits)/threshold) == argmax(logits-log(threshold)).
    Fixed grid anchored to class 0 matches previous decision-rule tuning.
    """
    if logits.shape != (len(targets), 7) or len(targets) == 0:
        raise ValueError("Invalid calibration sample shapes")
    logits_t = torch.as_tensor(logits, device=device, dtype=torch.float32)
    y = torch.as_tensor(targets, device=device, dtype=torch.long)
    grid = (.55, .65, .75, .85, .95, 1., 1.05, 1.15, 1.30, 1.50)
    threshold = torch.ones(7, device=device)

    def score(values):
        pred = (logits_t - values.log().view(1, 7)).argmax(dim=1)
        cm = torch.bincount(y * 7 + pred, minlength=49).reshape(7, 7)
        tp = cm.diag().float()
        union = cm.sum(0) + cm.sum(1) - tp
        iou = tp / union.clamp_min(1)
        # Keep exact 7-class objective of existing tune_thresholds(): absent=0.
        return float(iou.mean().item())

    baseline = current = score(threshold)
    for _ in range(passes):
        changed = False
        for c in range(1, 7):
            chosen = float(threshold[c])
            for v in grid:
                candidate = threshold.clone()
                candidate[c] = v
                s = score(candidate)
                if s > current + 1e-10:
                    current, chosen, changed = s, v, True
            threshold[c] = chosen
        if not changed:
            break
    return (threshold.cpu().numpy(), {"sampled_pixels": len(targets),
            "raw_sample_miou": baseline, "tuned_sample_miou": current,
            "grid": list(grid), "rule": "argmax(logits-log(class_thresholds))"})
