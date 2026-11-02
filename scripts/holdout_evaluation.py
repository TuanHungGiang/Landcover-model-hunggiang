"""Post-training LoveDA evaluation without holdout leakage into model selection.

All ranks enter this routine collectively. 200 selected validation images are
used for threshold tuning; untouched images are evaluated with fixed weights
and thresholds only. Model selection, training, and optimizer are never updated.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from scripts.loveda_fast import (
    NAMES, FastLoveDA, ImageShardedEvalSampler, calibration_samples,
    eval_distributed, synchronize_eval_buffers, tune_thresholds_logits,
    accumulate_confusion, confusion_metrics,
)


class DomainTaggedVal(Dataset):
    """Keep original images assigned to one rank while returning domain ID."""
    DOMAINS = ("Rural", "Urban")

    def __init__(self, dataset):
        if dataset.train:
            raise ValueError("DomainTaggedVal is for evaluation only")
        self.dataset = dataset
        self.train = False
        self.tiles = dataset.tiles
        self.pairs = dataset.pairs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        image_id = self.tiles[index][0]
        domain = self.pairs[image_id][2]
        if domain not in self.DOMAINS:
            raise ValueError(f"Unexpected domain {domain!r}")
        return image, mask, self.DOMAINS.index(domain)


def _holdout_loader(pairs, args, rank, world):
    base = FastLoveDA(pairs, args.size, train=False, cache_size=args.val_cache_images)
    ds = DomainTaggedVal(base)
    sampler = ImageShardedEvalSampler(ds, rank, world)
    kw = dict(batch_size=1, sampler=sampler, shuffle=False,
              pin_memory=True, num_workers=args.workers)
    if args.workers:
        kw.update(persistent_workers=True, prefetch_factor=2,
                  timeout=args.loader_timeout_seconds)
    return DataLoader(ds, **kw), len(ds), len(sampler)


@torch.no_grad()
def evaluate_holdout_with_domains(model, loader, device, rank,
                                  thresholds=None, channels_last=False,
                                  progress_every=100):
    """One inference pass gives global, Rural and Urban confusion matrices.

    Per-domain matrices are additive so the global result is their sum.
    Only two allreduces (raw + calibrated) for the entire distributed holdout.
    """
    synchronize_eval_buffers(model)
    model.eval()
    raw = torch.zeros((2, 7, 7), dtype=torch.int64, device=device)
    calibrated = torch.zeros_like(raw) if thresholds is not None else None
    bias = (torch.as_tensor(thresholds, dtype=torch.float32, device=device)
            .log().view(1, 7, 1, 1)) if thresholds is not None else None

    total = len(loader)
    for tile_no, (image, target, domain) in enumerate(loader, start=1):
        if domain.numel() != 1:
            raise ValueError("Holdout loader requires batch_size=1")
        domain_id = int(domain.item())
        if domain_id not in (0, 1):
            raise ValueError(f"Invalid holdout domain ID: {domain_id}")
        image = image.to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            scores = model(image)["logits"].float()
        accumulate_confusion(raw[domain_id], scores.argmax(dim=1), target)
        if calibrated is not None:
            accumulate_confusion(calibrated[domain_id],
                                 (scores - bias).argmax(dim=1), target)
        if tile_no == 1 or tile_no % progress_every == 0 or tile_no == total:
            print(f"[HOLDOUT] rank={rank} tiles={tile_no}/{total}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(raw, op=dist.ReduceOp.SUM)
        if calibrated is not None:
            dist.all_reduce(calibrated, op=dist.ReduceOp.SUM)
    result = {
        "argmax": confusion_metrics(raw.sum(dim=0)),
        "by_domain": {
            name: {"raw": confusion_metrics(raw[index])}
            for index, name in enumerate(DomainTaggedVal.DOMAINS)
        },
    }
    if calibrated is not None:
        result["calibrated"] = confusion_metrics(calibrated.sum(dim=0))
        for index, name in enumerate(DomainTaggedVal.DOMAINS):
            result["by_domain"][name]["calibrated"] = confusion_metrics(calibrated[index])
    return result


def run_final_evaluation(*, model, args, rank, world, device, output,
                         checkpoint, selected, remaining, val_loader, diagnostics=None):
    """Evaluate best checkpoint, optionally calibrate, and independently score holdout.

    The fixed holdout is NEVER read by tune_thresholds_logits().
    All per-rank collective calls are issued in the same order.
    """
    output = Path(output)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")
    started = time.perf_counter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError("Evaluation checkpoint is missing model weights")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    synchronize_eval_buffers(model)
    if rank == 0:
        source_epoch = state.get("epoch")
        print(f"[EVAL] checkpoint={checkpoint} "
              f"source_epoch={source_epoch + 1 if isinstance(source_epoch, int) else 'unknown'}",
              flush=True)

    thresholds = None
    fit = None
    if not args.no_calibrate:
        local_logits, local_targets = calibration_samples(
            model, val_loader, device,
            per_tile=args.calibration_samples_per_tile,
            seed=args.seed + rank, channels_last=args.channels_last,
        )
        gathered = [None] * world if rank == 0 else None
        dist.gather_object((local_logits, local_targets), gathered, dst=0)
        thresholds = torch.ones(7, device=device, dtype=torch.float32)
        if rank == 0:
            logits = np.concatenate([item[0] for item in gathered], axis=0)
            targets = np.concatenate([item[1] for item in gathered], axis=0)
            tuned, fit = tune_thresholds_logits(logits, targets, device=device)
            thresholds.copy_(torch.as_tensor(tuned, device=device))
        dist.broadcast(thresholds, src=0)

    val_eval = eval_distributed(model, val_loader, device, thresholds=thresholds,
                                channels_last=args.channels_last)
    if rank == 0 and not args.no_calibrate:
        threshold_array = [float(t) for t in thresholds.tolist()]
        calibration_record = {
            "selected_images": len(selected),
            "untouched_holdout_images": len(remaining),
            "checkpoint": str(checkpoint),
            "threshold_array": threshold_array,
            "thresholds": dict(zip(NAMES, threshold_array)),
            "fit": fit,
            "selected_200": val_eval,
            "rule": "argmax(logits-log(class_thresholds))",
            "note": "Only selected 200 validation images used for threshold fitting; not independent test performance.",
        }
        (output / "calibration.json").write_text(
            json.dumps(calibration_record, indent=2), encoding="utf-8")

    if args.evaluate_holdout and remaining:
        holdout_loader, tile_count, rank_tiles = _holdout_loader(
            remaining, args, rank, world)
        if rank == 0:
            print(f"[HOLDOUT] Evaluating {len(remaining)} untouched images "
                  f"({tile_count} tiles across {world} GPUs), calibration={not args.no_calibrate}",
                  flush=True)
        t0 = time.perf_counter()
        holdout_eval = evaluate_holdout_with_domains(
            model, holdout_loader, device, rank, thresholds=thresholds,
            channels_last=args.channels_last)
        duration = time.perf_counter() - t0
        if rank == 0:
            output_record = {
                "checkpoint": str(checkpoint),
                "images": len(remaining),
                "tiles_total": tile_count,
                "domain_images": {
                    name: sum(p[2] == name for p in remaining)
                    for name in DomainTaggedVal.DOMAINS
                },
                "domain_metrics": holdout_eval["by_domain"],
                "duration_seconds_rank0": round(duration, 2),
                "thresholds_from_selected_200": ([float(t) for t in thresholds.tolist()]
                                                 if thresholds is not None else None),
                "raw": holdout_eval["argmax"] if thresholds is not None else holdout_eval,
                "calibrated": (holdout_eval["calibrated"] if thresholds is not None else None),
                "note": "Held-out validation; never used for checkpoint selection or threshold tuning.",
            }
            (output / "holdout_metrics.json").write_text(
                json.dumps(output_record, indent=2), encoding="utf-8")
            for domain, metrics in output_record["domain_metrics"].items():
                msg = f"[HOLDOUT][{domain}] raw_mIoU={metrics['raw']['miou']:.4f}"
                if "calibrated" in metrics:
                    msg += f" calibrated_mIoU={metrics['calibrated']['miou']:.4f}"
                print(msg, flush=True)
            print(f"[HOLDOUT] raw_mIoU={output_record['raw']['miou']:.4f} "
                  + (f"calibrated_mIoU={output_record['calibrated']['miou']:.4f} "
                     if output_record['calibrated'] is not None else "")
                  + f"elapsed={duration:.1f}s saved={output / 'holdout_metrics.json'}",
                  flush=True)
            if diagnostics is not None:
                diagnostics.write("holdout_complete", images=len(remaining),
                                  miou_raw=output_record["raw"]["miou"],
                                  miou_calibrated=(output_record["calibrated"]["miou"]
                                                   if output_record["calibrated"] is not None else None))
    if rank == 0:
        print(f"[EVAL] complete elapsed={time.perf_counter()-started:.1f}s", flush=True)
    dist.barrier()
