#!/usr/bin/env python3
"""Train LoveDA on two Kaggle T4 GPUs using torchrun DistributedDataParallel.

torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp.py ...
Single-GPU scripts/train_loveda.py remains unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_loveda import ROOT, CLASSES, LoveDA, find_pairs, validate
from scripts.val200_calibration import (
    select_validation_200, sample_pixels, tune_thresholds, evaluate_thresholds
)
from sdr_mambanet import SDRConfig, SDRMambaNet
from sdr_mambanet.losses.segmentation import SegmentationLoss


def arguments():
    p = argparse.ArgumentParser(description="2x T4 DDP LoveDA trainer")
    p.add_argument("--root", default=ROOT)
    p.add_argument("--size", type=int, choices=(512, 1024), default=512)
    p.add_argument("--context", choices=("conv", "mamba"), default="conv")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=1, help="Per-GPU batch")
    p.add_argument("--accum", type=int, default=2, help="Gradient accumulation per GPU")
    p.add_argument("--workers", type=int, default=2, help="Workers per GPU")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output", default="/kaggle/working/sdr_loveda_r34_bi_v2_ddp")
    p.add_argument("--resnet-weights-path", default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume", default=None, help="Resume matching DDP last.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-count", type=int, default=200)
    p.add_argument("--calibration-samples-per-tile", type=int, default=2048)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--evaluate-holdout", action="store_true")
    return p.parse_args()


def make_loader(ds, batch, workers, sampler=None):
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=True,
              sampler=sampler, shuffle=False)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **kw)


def main():
    a = arguments()
    if a.batch < 1 or a.accum < 1 or a.workers < 0:
        raise ValueError("--batch, --accum must be positive; --workers >= 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle accelerator GPU T4 x2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        dist.destroy_process_group()
        raise RuntimeError("Expected --nproc_per_node=2")
    try:
        torch.manual_seed(a.seed + rank)
        np.random.seed(a.seed + rank)
        random.seed(a.seed + rank)
        root, output = Path(a.root), Path(a.output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            selected, remaining = select_validation_200(
                find_pairs(root, "val"), root, output,
                seed=a.seed, count=a.val_count, resume=bool(a.resume)
            )
            (output / "args.json").write_text(json.dumps(vars(a), indent=2))
        dist.barrier()  # rank1 must not race val_split_200.json creation

        train_ds = LoveDA(find_pairs(root, "train"), size=a.size, train=True)
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True,
            seed=a.seed, drop_last=False
        )
        train_dl = make_loader(train_ds, a.batch, a.workers, train_sampler)
        if rank == 0:
            val_dl = make_loader(LoveDA(selected, a.size, False), 1, a.workers)

        config = SDRConfig(
            num_classes=7, semantic_encoder="resnet34",
            pretrained_semantic=(rank == 0 and not a.no_pretrained and not a.resume),
            semantic_weights_path=a.resnet_weights_path if rank == 0 and not a.resume else None,
            context_type=a.context
        )
        # Only rank0 downloads ImageNet pretrained weights; DDP will broadcast parameters.
        if rank == 0:
            raw = SDRMambaNet(config)
        dist.barrier()
        if rank != 0:
            raw = SDRMambaNet(config)
        model = DDP(raw.to(device), device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=False,
                    gradient_as_bucket_view=True)
        loss_fn = SegmentationLoss(ignore_index=255, dice_weight=1.).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=a.epochs, eta_min=1e-6
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        start, best = 0, -1.
        if a.resume:
            checkpoint = torch.load(a.resume, map_location="cpu", weights_only=False)
            saved_cfg = checkpoint.get("model_config", {})
            if saved_cfg.get("architecture_version") != "r34_layer4_bi_v2":
                raise ValueError("Legacy layer3 checkpoint incompatible with v2")
            if saved_cfg.get("context_type") != a.context:
                raise ValueError("Resume --context does not match checkpoint")
            model.module.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start, best = checkpoint["epoch"]+1, checkpoint["best_miou"]

        if rank == 0:
            print(f"DDP: {world} GPUs, per_gpu_batch={a.batch}, accum={a.accum}, "
                  f"effective_batch={world*a.batch*a.accum}, val_images={len(selected)}, "
                  f"params={sum(p.numel() for p in model.module.parameters())/1e6:.2f}M",
                  flush=True)

        for epoch in range(start, a.epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device)
            nsteps = len(train_dl)
            for step, (image, mask) in enumerate(train_dl):
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                group_start = (step // a.accum) * a.accum
                group_size = min(a.accum, nsteps-group_start)
                sync = (step+1) % a.accum == 0 or step+1 == nsteps
                with (nullcontext() if sync else model.no_sync()):
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = loss_fn(model(image), mask)["total"]
                    loss_total += loss.detach()
                    scaler.scale(loss / group_size).backward()
                if sync:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
            if rank == 0:
                # Evaluate the *unwrapped* module; other ranks wait, no collectives.
                stats = validate(model.module, val_dl, device)
                stats.update(epoch=epoch+1, train_loss=float(
                    (loss_total / (world*nsteps)).item()
                ))
                print(f"Epoch {epoch+1}/{a.epochs}: loss={stats['train_loss']:.4f} "
                      f"val_mIoU={stats['miou']:.4f} OA={stats['oa']:.4f}", flush=True)
                with (output/"history.jsonl").open("a") as f:
                    f.write(json.dumps(stats)+"\n")
                state = dict(model=model.module.state_dict(), epoch=epoch,
                             best_miou=best, model_config=asdict(model.module.cfg),
                             optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
                if stats["miou"] > best:
                    best = stats["miou"]
                    state["best_miou"] = best
                    torch.save(dict(model=state["model"], epoch=epoch,
                                    best_miou=best, model_config=state["model_config"]),
                               output/"best.pt")
                    (output/"best_metrics.json").write_text(json.dumps(stats, indent=2))
                state["best_miou"] = best
                torch.save(state, output/"last.pt")
            dist.barrier()  # rank1 must wait for rank0 validation/checkpoint

        if rank == 0 and not a.no_calibrate and (output/"best.pt").is_file():
            saved = torch.load(output/"best.pt", map_location="cpu", weights_only=False)
            model.module.load_state_dict(saved["model"])
            model.module.eval()
            probs, targets = sample_pixels(
                model.module, val_dl, device,
                per_tile=a.calibration_samples_per_tile, seed=a.seed
            )
            thresholds, fit = tune_thresholds(probs, targets)
            val_scores = evaluate_thresholds(model.module, val_dl, device, thresholds)
            result = dict(selected_images=len(selected), untouched_holdout_images=len(remaining),
                          thresholds=dict(zip(CLASSES, map(float, thresholds))),
                          threshold_array=[float(v) for v in thresholds],
                          fit=fit, selected_200=val_scores,
                          note="Calibrated on selected Val; not independent test performance.")
            if a.evaluate_holdout and remaining:
                holdout_dl = make_loader(LoveDA(remaining, a.size, False), 1, a.workers)
                result["remaining_val_holdout"] = evaluate_thresholds(
                    model.module, holdout_dl, device, thresholds
                )
            (output/"calibration.json").write_text(json.dumps(result, indent=2))
            print("Saved calibration.json", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
