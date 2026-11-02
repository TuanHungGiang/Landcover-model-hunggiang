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
from torch.utils.data import DataLoader

from scripts.loveda_fast import (
    NAMES, FastLoveDA, ImageShardedEvalSampler, calibration_samples,
    eval_distributed, synchronize_eval_buffers, tune_thresholds_logits,
)


def _holdout_loader(pairs, args, rank, world):
    ds = FastLoveDA(pairs, args.size, train=False, cache_size=args.val_cache_images)
    sampler = ImageShardedEvalSampler(ds, rank, world)
    kw = dict(batch_size=1, sampler=sampler, shuffle=False,
              pin_memory=True, num_workers=args.workers)
    if args.workers:
        kw.update(persistent_workers=True, prefetch_factor=2,
                  timeout=args.loader_timeout_seconds)
    return DataLoader(ds, **kw), len(ds), len(sampler)


def _report_eval(model, loader, device, rank, name, thresholds, channels_last,
                 progress_every=100):
    """Report forward progress without altering the distributed metric computation."""
    def logging_batches():
        count = 0
        total = len(loader)
        for batch in loader:
            count += 1
            if count == 1 or count % progress_every == 0 or count == total:
                print(f"[{name}] rank={rank} tiles={count}/{total}", flush=True)
            yield batch
    return eval_distributed(model, logging_batches(), device,
                            thresholds=thresholds, channels_last=channels_last)


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
        holdout_eval = _report_eval(model, holdout_loader, device, rank,
                                    "HOLDOUT", thresholds, args.channels_last)
        duration = time.perf_counter() - t0
        if rank == 0:
            output_record = {
                "checkpoint": str(checkpoint),
                "images": len(remaining),
                "tiles_total": tile_count,
                "duration_seconds_rank0": round(duration, 2),
                "thresholds_from_selected_200": ([float(t) for t in thresholds.tolist()]
                                                 if thresholds is not None else None),
                "raw": holdout_eval["argmax"] if thresholds is not None else holdout_eval,
                "calibrated": (holdout_eval["calibrated"] if thresholds is not None else None),
                "note": "Held-out validation; never used for checkpoint selection or threshold tuning.",
            }
            (output / "holdout_metrics.json").write_text(
                json.dumps(output_record, indent=2), encoding="utf-8")
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
