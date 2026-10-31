#!/usr/bin/env python3
"""Efficient two-T4 training: DDP train AND sharded evaluation/calibration.

No change to SDR-MambaNet architecture, loss, labels, optimizer defaults or
200-image validation membership relative to the previous DDP trainer.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.loveda_fast import (ROOT, NAMES, FastLoveDA, ImageShardedEvalSampler,
                                 find_pairs, eval_distributed, calibration_samples,
                                 tune_thresholds_logits)
from scripts.val200_calibration import select_validation_200
from sdr_mambanet import SDRConfig, SDRMambaNet
from sdr_mambanet.losses.segmentation import SegmentationLoss


def arguments():
    p = argparse.ArgumentParser(description="Optimized 2x T4 DDP LoveDA trainer")
    p.add_argument("--root", default=ROOT)
    p.add_argument("--size", type=int, choices=(512, 1024), default=512)
    p.add_argument("--context", choices=("conv", "mamba"), default="conv")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=1, help="Per-GPU batch size")
    p.add_argument("--accum", type=int, default=2, help="Accumulation per GPU")
    p.add_argument("--workers", type=int, default=2, help="Dataloader workers PER GPU")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output", default="/kaggle/working/sdr_loveda_r34_bi_v2_ddp")
    p.add_argument("--resnet-weights-path", default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-count", type=int, default=200)
    p.add_argument("--calibration-samples-per-tile", type=int, default=2048)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--evaluate-holdout", action="store_true")
    p.add_argument("--val-cache-images", type=int, default=8,
                   help="Decoded original images cached PER dataloader worker")
    p.add_argument("--train-cache-images", type=int, default=2)
    p.add_argument("--channels-last", action="store_true",
                   help="Optional NHWC format; benchmark first on your T4")
    p.add_argument("--fused-adamw", action="store_true",
                   help="Optional CUDA fused AdamW if available")
    return p.parse_args()


def make_loader(ds, batch, workers, sampler=None, *, shuffle=False):
    kwargs = dict(batch_size=batch, num_workers=workers, pin_memory=True,
                  sampler=sampler, shuffle=shuffle and sampler is None)
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **kwargs)


def main():
    a = arguments()
    if a.batch < 1 or a.accum < 1 or a.workers < 0:
        raise ValueError("--batch/--accum must be positive, --workers nonnegative")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle accelerator 2 x Tesla T4")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        dist.destroy_process_group()
        raise RuntimeError("Launch with torchrun --nproc_per_node=2")
    try:
        random.seed(a.seed + rank)
        np.random.seed(a.seed + rank)
        torch.manual_seed(a.seed + rank)
        torch.backends.cudnn.benchmark = True  # fixed 512/1024 resolution

        root, output = Path(a.root), Path(a.output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            selected, remaining = select_validation_200(
                find_pairs(root, "val"), root, output,
                seed=a.seed, count=a.val_count, resume=bool(a.resume))
            (output / "args.json").write_text(json.dumps(vars(a), indent=2))
        dist.barrier()
        if rank != 0:
            selected, remaining = select_validation_200(
                find_pairs(root, "val"), root, output,
                seed=a.seed, count=a.val_count, resume=True)

        train_ds = FastLoveDA(find_pairs(root, "train"), a.size, True,
                              cache_size=a.train_cache_images)
        train_sampler = DistributedSampler(train_ds, num_replicas=world,
                                           rank=rank, shuffle=True, seed=a.seed)
        train_dl = make_loader(train_ds, a.batch, a.workers, train_sampler)
        val_ds = FastLoveDA(selected, a.size, False, cache_size=a.val_cache_images)
        val_sampler = ImageShardedEvalSampler(val_ds, rank, world)
        val_dl = make_loader(val_ds, 1, a.workers, val_sampler)

        cfg = SDRConfig(
            num_classes=7, semantic_encoder="resnet34",
            pretrained_semantic=(rank == 0 and not a.no_pretrained and not a.resume),
            semantic_weights_path=a.resnet_weights_path if rank == 0 and not a.resume else None,
            context_type=a.context)
        # Rank 0 downloads pretrained weights; the DDP constructor broadcasts its state.
        if rank == 0:
            raw = SDRMambaNet(cfg)
        dist.barrier()
        if rank != 0:
            raw = SDRMambaNet(cfg)
        raw = raw.to(device)
        if a.channels_last:
            raw = raw.to(memory_format=torch.channels_last)
        ddp = DDP(raw, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=False, gradient_as_bucket_view=True)
        loss_fn = SegmentationLoss(ignore_index=255, dice_weight=1.).to(device)
        if a.fused_adamw:
            try:
                optimizer = torch.optim.AdamW(ddp.parameters(), lr=a.lr,
                                              weight_decay=.01, fused=True)
            except (TypeError, RuntimeError) as exc:
                raise RuntimeError("Your PyTorch/CUDA does not support --fused-adamw") from exc
        else:
            optimizer = torch.optim.AdamW(ddp.parameters(), lr=a.lr, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=a.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        start, best = 0, -1.
        if a.resume:
            ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
            previous = ckpt.get("model_config", {})
            if previous.get("architecture_version") != "r34_layer4_bi_v2":
                raise ValueError("Cannot resume layer3 model from v2 checkpoint")
            if previous.get("context_type") != a.context:
                raise ValueError("--context must match resume checkpoint")
            ddp.module.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start, best = ckpt["epoch"] + 1, ckpt["best_miou"]
        if rank == 0:
            print(f"DDP: {world} GPUs | batch/GPU={a.batch} | accumulation={a.accum} "
                  f"| effective_batch={world * a.batch * a.accum} "
                  f"| val_images={len(selected)} | val_tiles/rank={[len(val_sampler), len(val_ds)-len(val_sampler)]} "
                  f"| params={sum(p.numel() for p in ddp.module.parameters())/1e6:.2f}M",
                  flush=True)
        for epoch in range(start, a.epochs):
            train_sampler.set_epoch(epoch)
            ddp.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device)
            nsteps = len(train_dl)
            start_train = time.perf_counter()
            for step, (image, mask) in enumerate(train_dl):
                image = image.to(device, non_blocking=True)
                if a.channels_last:
                    image = image.contiguous(memory_format=torch.channels_last)
                mask = mask.to(device, non_blocking=True)
                first = (step // a.accum) * a.accum
                group_size = min(a.accum, nsteps - first)
                sync = (step + 1) % a.accum == 0 or step + 1 == nsteps
                with (nullcontext() if sync else ddp.no_sync()):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = loss_fn(ddp(image), mask)["total"]
                    loss_total.add_(loss.detach())
                    scaler.scale(loss / group_size).backward()
                if sync:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
            train_time = time.perf_counter() - start_train
            # All ranks evaluate disjoint original images; confusion allreduce ONCE.
            start_val = time.perf_counter()
            stats = eval_distributed(ddp.module, val_dl, device,
                                     channels_last=a.channels_last)
            val_time = time.perf_counter() - start_val
            if rank == 0:
                stats.update(epoch=epoch + 1,
                             train_loss=float((loss_total / (world * nsteps)).item()),
                             train_seconds=round(train_time, 2),
                             val_seconds=round(val_time, 2),
                             train_images_per_second=round((len(train_ds) / max(train_time, 1e-9)), 2))
                print(f"Epoch {epoch+1}/{a.epochs}: loss={stats['train_loss']:.4f} "
                      f"mIoU={stats['miou']:.4f} OA={stats['oa']:.4f} "
                      f"train={train_time:.1f}s val={val_time:.1f}s", flush=True)
                with (output / "history.jsonl").open("a") as fp:
                    fp.write(json.dumps(stats) + "\n")
                state = dict(model=ddp.module.state_dict(), epoch=epoch,
                             best_miou=best, model_config=asdict(ddp.module.cfg),
                             optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
                if stats["miou"] > best:
                    best = stats["miou"]
                    state["best_miou"] = best
                    torch.save({"model": state["model"], "epoch": epoch,
                                "best_miou": best, "model_config": state["model_config"]},
                               output / "best.pt")
                    (output / "best_metrics.json").write_text(json.dumps(stats, indent=2))
                state["best_miou"] = best
                torch.save(state, output / "last.pt")
            dist.barrier()  # Do not let the next epoch start while rank0 saves.

        if not a.no_calibrate and (output / "best.pt").exists():
            best_state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            ddp.module.load_state_dict(best_state["model"])
            ddp.module.eval()
            local_logits, local_labels = calibration_samples(
                ddp.module, val_dl, device,
                per_tile=a.calibration_samples_per_tile, seed=a.seed + rank,
                channels_last=a.channels_last)
            gathered = [None] * world if rank == 0 else None
            # Gather samples once, not dense probability maps after every tile.
            dist.gather_object((local_logits, local_labels), gathered, dst=0)
            thresholds = torch.ones(7, device=device)
            if rank == 0:
                logits = np.concatenate([v[0] for v in gathered], axis=0)
                targets = np.concatenate([v[1] for v in gathered], axis=0)
                tuned, fit = tune_thresholds_logits(logits, targets, device=device)
                thresholds.copy_(torch.as_tensor(tuned, dtype=torch.float32, device=device))
            dist.broadcast(thresholds, src=0)
            evaluated = eval_distributed(ddp.module, val_dl, device,
                                          thresholds=thresholds,
                                          channels_last=a.channels_last)
            result = None
            if rank == 0:
                result = {
                    "selected_images": len(selected),
                    "untouched_holdout_images": len(remaining),
                    "thresholds": dict(zip(NAMES, [float(v) for v in thresholds.cpu().tolist()])),
                    "threshold_array": [float(v) for v in thresholds.cpu().tolist()],
                    "fit": fit, "selected_200": evaluated,
                    "rule": "argmax(logits-log(class_thresholds))",
                    "note": "Decision thresholds fitted on selected 200 Val images; not test performance.",
                }
            if a.evaluate_holdout and remaining:
                holdout_ds = FastLoveDA(remaining, a.size, False,
                                         cache_size=a.val_cache_images)
                holdout_sampler = ImageShardedEvalSampler(holdout_ds, rank, world)
                holdout_dl = make_loader(holdout_ds, 1, a.workers, holdout_sampler)
                holdout_stats = eval_distributed(ddp.module, holdout_dl, device,
                                                  thresholds=thresholds,
                                                  channels_last=a.channels_last)
                if rank == 0:
                    result["remaining_val_holdout"] = holdout_stats
            if rank == 0:
                (output / "calibration.json").write_text(json.dumps(result, indent=2))
                print("Saved calibration.json (multi-GPU Val + holdout evaluation)", flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
