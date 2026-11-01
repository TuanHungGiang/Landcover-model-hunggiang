#!/usr/bin/env python3
"""Efficient two-T4 training: DDP train AND sharded evaluation/calibration.

Generalization-focused v3 trainer, preserving SDR-MambaNet v2 architecture,
LoveDA labels, and the fixed 200-image validation and holdout protocol.
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
from scripts.loveda_regularization import (
    RegularizedLoveDA, top_confusions, lr_multiplier, build_discriminative_groups,
)
from scripts.training_diagnostics import (
    StepCudaTimer, DiagnosticsWriter, class_health, gate_scales,
    selected_grad_norms,
)
from sdr_mambanet import SDRConfig, SDRMambaNet
from sdr_mambanet.losses.segmentation import SegmentationLoss


def arguments():
    p = argparse.ArgumentParser(description="V3 generalization: 2x T4 DDP LoveDA trainer")
    p.add_argument("--root", default=ROOT)
    p.add_argument("--size", type=int, choices=(512, 1024), default=512)
    p.add_argument("--context", choices=("conv", "mamba"), default="conv")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8, help="Per-GPU batch size")
    p.add_argument("--accum", type=int, default=2, help="Accumulation per GPU")
    p.add_argument("--workers", type=int, default=2, help="Dataloader workers PER GPU")
    p.add_argument("--lr", type=float, default=2e-4, help="LR for newly initialized modules")
    p.add_argument("--backbone-lr", type=float, default=5e-5,
                   help="Lower LR for ImageNet-pretrained ResNet34")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--early-stop-patience", type=int, default=20,
                   help="Stop after N epochs without mIoU improvement; 0 disables")
    p.add_argument("--early-stop-min-delta", type=float, default=0.001,
                   help="Minimum mIoU gain to reset early-stop patience")
    p.add_argument("--augment", choices=("none", "light", "strong"), default="strong")
    p.add_argument("--focus-prob", type=float, default=0.5,
                   help="Chance to center training crop on Barren or Agriculture")
    p.add_argument("--focus-classes", default="4,6",
                   help="Training IDs of classes to favor, 4=Barren 6=Agriculture")
    p.add_argument("--zoom-min", type=float, default=1.0)
    p.add_argument("--zoom-max", type=float, default=1.35)
    p.add_argument("--init-from", default=None,
                   help="Fine-tune weights from the v2 best.pt; reset optimizer and LR")
    p.add_argument("--output", default="/kaggle/working/sdr_loveda_r34_bi_v2_ddp_v3")
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
    p.add_argument("--log-every", type=int, default=100,
                   help="Report progress and loss components every N local GPU iterations")
    p.add_argument("--profile-every", type=int, default=50,
                   help="Sample CUDA event/module timings every N synchronized steps or at epoch end; 0 disables")
    p.add_argument("--no-diagnostics", action="store_true",
                   help="Disable extra JSONL diagnostics, module timers and printed per-class health")
    p.add_argument("--warn-iou-below", type=float, default=0.10,
                   help="Warn when a present class remains below this IoU after epoch 5")
    return p.parse_args()


def make_loader(ds, batch, workers, sampler=None, *, shuffle=False):
    kwargs = dict(batch_size=batch, num_workers=workers, pin_memory=True,
                  sampler=sampler, shuffle=shuffle and sampler is None)
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **kwargs)


def main():
    a = arguments()
    if a.log_every < 1 or a.profile_every < 0 or not 0 <= a.warn_iou_below <= 1:
        raise ValueError("--log-every >=1, --profile-every >=0, --warn-iou-below in [0,1]")
    if a.resume and a.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    if a.epochs < 1 or a.warmup_epochs < 0 or a.warmup_epochs >= a.epochs:
        raise ValueError("0 <= warmup-epochs < epochs required")
    if a.backbone_lr <= 0 or a.lr <= 0 or a.early_stop_patience < 0 or a.early_stop_min_delta < 0:
        raise ValueError("Learning rates and early stop arguments must be positive")
    focus_classes = tuple(int(part) for part in a.focus_classes.split(",") if part.strip())
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

        diag = DiagnosticsWriter(output, rank, enabled=not a.no_diagnostics)
        train_ds = RegularizedLoveDA(
            find_pairs(root, "train"), a.size, True,
            cache_size=a.train_cache_images, augment=a.augment,
            focus_probability=a.focus_prob, focus_classes=focus_classes,
            zoom_min=a.zoom_min, zoom_max=a.zoom_max,
        )
        train_sampler = DistributedSampler(train_ds, num_replicas=world,
                                           rank=rank, shuffle=True, seed=a.seed)
        train_dl = make_loader(train_ds, a.batch, a.workers, train_sampler)
        val_ds = FastLoveDA(selected, a.size, False, cache_size=a.val_cache_images)
        val_sampler = ImageShardedEvalSampler(val_ds, rank, world)
        val_dl = make_loader(val_ds, 1, a.workers, val_sampler)

        cfg = SDRConfig(
            num_classes=7, semantic_encoder="resnet34",
            pretrained_semantic=(rank == 0 and not a.no_pretrained and not a.resume and not a.init_from),
            semantic_weights_path=a.resnet_weights_path if rank == 0 and not a.resume and not a.init_from else None,
            context_type=a.context)
        # Rank 0 downloads pretrained weights; the DDP constructor broadcasts its state.
        if rank == 0:
            raw = SDRMambaNet(cfg)
            if a.init_from:
                incoming = torch.load(a.init_from, map_location="cpu", weights_only=False)
                if incoming.get("model_config", {}).get("architecture_version") != "r34_layer4_bi_v2":
                    raise ValueError("--init-from must be a training checkpoint for layer4 bidirectional v2")
                if incoming.get("model_config", {}).get("context_type") != a.context:
                    raise ValueError("--init-from context_type must match --context")
                raw.load_state_dict(incoming["model"], strict=True)
                print(f"[FINETUNE] initialized from {a.init_from} "
                      f"(source epoch={incoming.get('epoch', 'unknown')}); "
                      "optimizer/scheduler reset", flush=True)
        dist.barrier()
        if rank != 0:
            raw = SDRMambaNet(cfg)
        raw = raw.to(device)
        if a.channels_last:
            raw = raw.to(memory_format=torch.channels_last)
        ddp = DDP(raw, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=False, gradient_as_bucket_view=True)
        profiler = StepCudaTimer(ddp.module) if rank == 0 and not a.no_diagnostics and a.profile_every else None
        loss_fn = SegmentationLoss(ignore_index=255, dice_weight=1.).to(device)
        if a.fused_adamw:
            try:
                optimizer = torch.optim.AdamW(build_discriminative_groups(ddp.module, a.lr, a.backbone_lr),
                                              weight_decay=.01, fused=True)
            except (TypeError, RuntimeError) as exc:
                raise RuntimeError("Your PyTorch/CUDA does not support --fused-adamw") from exc
        else:
            optimizer = torch.optim.AdamW(build_discriminative_groups(ddp.module, a.lr, a.backbone_lr),
                                           weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: lr_multiplier(e, a.epochs,
                a.warmup_epochs, a.min_lr_ratio))
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        start, best, stale = 0, -1., 0
        if a.resume:
            ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
            previous = ckpt.get("model_config", {})
            if ckpt.get("training_recipe_version") != "generalization_v3":
                raise ValueError("Use --init-from for older v2 checkpoints; only resume v3 last.pt")
            prev_args = ckpt.get("training_args", {})
            for key in ("epochs", "seed", "size", "batch", "accum", "augment", "focus_prob", "focus_classes", "zoom_min", "zoom_max", "warmup_epochs", "min_lr_ratio", "lr", "backbone_lr"):
                if key in prev_args and prev_args[key] != getattr(a,key):
                    raise ValueError(f"Resume option changed: {key} ({prev_args[key]} -> {getattr(a,key)})")
            if previous.get("architecture_version") != "r34_layer4_bi_v2":
                raise ValueError("Cannot resume layer3 model from v2 checkpoint")
            if previous.get("context_type") != a.context:
                raise ValueError("--context must match resume checkpoint")
            ddp.module.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start, best, stale = ckpt["epoch"] + 1, ckpt["best_miou"], ckpt.get("early_stop_stale", 0)
        if rank == 0:
            print(f"DDP: {world} GPUs | batch/GPU={a.batch} | accumulation={a.accum} "
                  f"| effective_batch={world * a.batch * a.accum} "
                  f"| val_images={len(selected)} | val_tiles/rank={[len(val_sampler), len(val_ds)-len(val_sampler)]} "
                  f"| params={sum(p.numel() for p in ddp.module.parameters())/1e6:.2f}M "
                  f"| augmentation={a.augment} | focus={a.focus_prob:.2f} "
                  f"| backbone_lr={a.backbone_lr:g} | new_lr={a.lr:g}",
                  flush=True)
        if a.init_from:
            initial = eval_distributed(ddp.module, val_dl, device,
                                       channels_last=a.channels_last)
            if rank == 0:
                best = initial["miou"]
                print(f"[FINETUNE] initial raw mIoU={best:.4f}; baseline weights retained as fallback", flush=True)
                torch.save({"model":ddp.module.state_dict(), "epoch": -1,
                            "best_miou":best, "model_config":asdict(ddp.module.cfg)},
                           output / "best.pt")
                (output / "best_metrics.json").write_text(json.dumps(initial, indent=2))
        for epoch in range(start, a.epochs):
            train_sampler.set_epoch(epoch)
            epoch_lrs = [group["lr"] for group in optimizer.param_groups]
            ddp.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device)
            ce_total = torch.zeros_like(loss_total)
            dice_total = torch.zeros_like(loss_total)
            nsteps = len(train_dl)
            if not a.no_diagnostics:
                torch.cuda.reset_peak_memory_stats(device)
            start_train = time.perf_counter()
            fetch_start = time.perf_counter()
            loader_wait_sum, loader_wait_max = 0.0, 0.0
            for step, (image, mask) in enumerate(train_dl):
                loader_wait = time.perf_counter() - fetch_start
                loader_wait_sum += loader_wait
                loader_wait_max = max(loader_wait_max, loader_wait)
                first = (step // a.accum) * a.accum
                group_size = min(a.accum, nsteps - first)
                sync = (step + 1) % a.accum == 0 or step + 1 == nsteps
                profile_now = bool(profiler and step > 0 and sync and
                                   ((step + 1) % a.profile_every == 0 or step + 1 == nsteps))
                if profile_now:
                    profiler.begin()
                    profiler.mark("copy_begin")
                image = image.to(device, non_blocking=True)
                if a.channels_last:
                    image = image.contiguous(memory_format=torch.channels_last)
                mask = mask.to(device, non_blocking=True)
                if profile_now:
                    profiler.mark("copy_end")
                    profiler.mark("forward_begin")
                with (nullcontext() if sync else ddp.no_sync()):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss_parts = loss_fn(ddp(image), mask)
                        loss = loss_parts["total"]
                    loss_total.add_(loss.detach())
                    ce_total.add_(loss_parts["ce"].detach())
                    dice_total.add_(loss_parts["dice"].detach())
                    if profile_now:
                        profiler.mark("forward_end")
                        profiler.mark("backward_begin")
                    scaler.scale(loss / group_size).backward()
                if profile_now:
                    profiler.mark("backward_end")
                grad_norm = None
                grad_parts = None
                if sync:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.)
                    if profile_now:
                        grad_parts = selected_grad_norms(ddp.module)
                        profiler.mark("optimizer_begin")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if profile_now:
                        profiler.mark("optimizer_end")
                if profile_now:
                    gpu_times = profiler.finish(device)  # sampled GPU sync, NOT each iteration
                    gpu_times["stage_forward_ms"] = {
                        k: gpu_times.pop(k) for k, _ in (
                            ("detail",0),("resnet_stem",0),("resnet_layer1",0),
                            ("resnet_layer2",0),("resnet_layer3",0),("resnet_layer4",0),
                            ("context_1_32",0),("fusion_bi8",0),
                            ("fusion_cross16",0),("fusion_final",0),("decoder",0)
                        ) if k in gpu_times
                    }
                    gpu_times["grad_norm_by_module"] = {
                        k: round(float(v.item()), 5) for k, v in (grad_parts or {}).items()
                    }
                    gpu_times["grad_norm_before_clipping"] = float(grad_norm.item())
                    gpu_times["data_wait_ms"] = round(loader_wait * 1000, 2)
                    gpu_times["gpu_allocated_gib"] = round(torch.cuda.memory_allocated(device)/2**30, 3)
                    gpu_times["gpu_reserved_gib"] = round(torch.cuda.memory_reserved(device)/2**30, 3)
                    diag.write("gpu_profile", epoch=epoch+1, local_step=step+1, **gpu_times)
                    if rank == 0:
                        top_stages = sorted(gpu_times["stage_forward_ms"].items(),
                                            key=lambda item: item[1], reverse=True)[:4]
                        print("[PROFILE] GPU0 " + " | ".join(
                            f"{k}={v:.1f}ms" for k,v in top_stages) +
                            f" | fwd={gpu_times.get('forward_ms',0):.1f}ms"
                            f" bwd={gpu_times.get('backward_ms',0):.1f}ms"
                            f" opt={gpu_times.get('optimizer_ms',0):.1f}ms",flush=True)
                if not a.no_diagnostics and ((step+1) % a.log_every == 0 or step+1 == nsteps):
                    # Host sync only every log interval, never at all other steps.
                    progress = dict(epoch=epoch+1, local_step=step+1, local_steps=nsteps,
                                    loss=float((loss_total/(step+1)).item()),
                                    ce=float((ce_total/(step+1)).item()),
                                    dice=float((dice_total/(step+1)).item()),
                                    avg_loader_wait_ms=round(loader_wait_sum/(step+1)*1000,2),
                                    max_loader_wait_ms=round(loader_wait_max*1000,2),
                                    elapsed_s=round(time.perf_counter()-start_train,2),
                                    peak_allocated_gib=round(torch.cuda.max_memory_allocated(device)/2**30,3),
                                    peak_reserved_gib=round(torch.cuda.max_memory_reserved(device)/2**30,3),
                                    lr=optimizer.param_groups[0]["lr"],
                                    amp_scale=scaler.get_scale())
                    diag.write("train_step", **progress)
                    if rank == 0:
                        print(f"[TRAIN] epoch={epoch+1} step={step+1}/{nsteps} "
                              f"loss={progress['loss']:.4f} CE={progress['ce']:.4f} "
                              f"Dice={progress['dice']:.4f} "
                              f"loader_wait={progress['avg_loader_wait_ms']:.1f}ms "
                              f"peakVRAM={progress['peak_allocated_gib']:.2f}GiB "
                              f"AMP={progress['amp_scale']:.0f}", flush=True)
                fetch_start = time.perf_counter()
            dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
            train_time = time.perf_counter() - start_train
            # All ranks evaluate disjoint original images; confusion allreduce ONCE.
            start_val = time.perf_counter()
            stats = eval_distributed(ddp.module, val_dl, device,
                                     channels_last=a.channels_last)
            val_time = time.perf_counter() - start_val
            if not a.no_diagnostics:
                diag.write("epoch_rank", epoch=epoch+1,
                           train_seconds=round(train_time,2), val_seconds=round(val_time,2),
                           avg_loader_wait_ms=round(loader_wait_sum/max(nsteps,1)*1000,2),
                           max_loader_wait_ms=round(loader_wait_max*1000,2),
                           peak_allocated_gib=round(torch.cuda.max_memory_allocated(device)/2**30,3),
                           peak_reserved_gib=round(torch.cuda.max_memory_reserved(device)/2**30,3))
            scheduler.step()
            if rank == 0:
                if not a.no_diagnostics:
                    per_class, health_alerts = class_health(
                        stats["confusion"], epoch=epoch+1, low_iou=a.warn_iou_below)
                    scales = gate_scales(ddp.module)
                    stats["class_health"] = per_class
                    stats["gate_scales"] = scales
                    diag.write("validation_health", epoch=epoch+1, miou=stats["miou"],
                               per_class=per_class, gate_scales=scales,
                               warnings=health_alerts)
                    print("[CLASSES] " + " | ".join(
                        f"{row['class']} IoU={row['iou']:.3f} "
                        f"GT={row['gt_fraction']:.3f} PRED={row['pred_fraction']:.3f}"
                        for row in per_class), flush=True)
                    print("[GATES] " + ", ".join(
                        f"{k}={v:+.4f}" for k, v in scales.items()), flush=True)
                    for warning in health_alerts:
                        print("[WARNING] " + warning, flush=True)
                stats["top_confusions"] = top_confusions(stats["confusion"], NAMES, top_k=5)
                print("[CONFUSIONS] " + " | ".join(
                    f"{pair['gt']}→{pair['pred']}:{pair['percent_of_gt']:.1f}%"
                    for pair in stats["top_confusions"]), flush=True)
                stats.update(epoch=epoch + 1,
                             train_loss=float((loss_total / (world * nsteps)).item()),
                             train_seconds=round(train_time, 2),
                             val_seconds=round(val_time, 2),
                             train_images_per_second=round((len(train_ds) / max(train_time, 1e-9)), 2))
                print(f"Epoch {epoch+1}/{a.epochs}: loss={stats['train_loss']:.4f} "
                      f"mIoU={stats['miou']:.4f} OA={stats['oa']:.4f} "
                      f"train={train_time:.1f}s val={val_time:.1f}s "
                      f"lr_resnet={stats['lr_pretrained']:.2g} lr_new={stats['lr_new']:.2g}", flush=True)
                stats["lr_pretrained"] = epoch_lrs[0]
                stats["lr_new"] = epoch_lrs[1]
                stats["early_stop_stale_before_update"] = stale
                with (output / "history.jsonl").open("a") as fp:
                    fp.write(json.dumps(stats) + "\n")
                state = dict(model=ddp.module.state_dict(), epoch=epoch,
                             best_miou=best, model_config=asdict(ddp.module.cfg),
                             optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
                previous_best = best
                if stats["miou"] > best:
                    best = stats["miou"]
                    state["best_miou"] = best
                    torch.save({"model": state["model"], "epoch": epoch,
                                "best_miou": best, "model_config": state["model_config"]},
                               output / "best.pt")
                    (output / "best_metrics.json").write_text(json.dumps(stats, indent=2))
                stale = 0 if stats["miou"] > previous_best + a.early_stop_min_delta else stale + 1
                state["best_miou"] = best
                state["early_stop_stale"] = stale
                state["training_recipe_version"] = "generalization_v3"
                state["training_args"] = vars(a)
                torch.save(state, output / "last.pt")
                print(f"[EARLY_STOP] stale={stale}/{a.early_stop_patience} best_mIoU={best:.4f}",flush=True)
            # Rank 0 sends early-stop decision to every GPU. No DDP worker can exit alone.
            stop = torch.zeros(1, dtype=torch.int32, device=device)
            if rank == 0 and a.early_stop_patience > 0 and stale >= a.early_stop_patience:
                stop.fill_(1)
            dist.broadcast(stop, src=0)
            if int(stop.item()):
                if rank == 0:
                    print(f"[EARLY_STOP] stopped after epoch {epoch+1}; best_mIoU={best:.4f}", flush=True)
                break
            dist.barrier()  # Do not let next epoch start while rank0 saves.

        if not a.no_calibrate and (output / "best.pt").exists():
            calibration_started = time.perf_counter()
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
            if rank == 0 and not a.no_diagnostics:
                diag.write("calibration_fit", samples=int(fit["sampled_pixels"]),
                           raw_sample_miou=float(fit["raw_sample_miou"]),
                           tuned_sample_miou=float(fit["tuned_sample_miou"]),
                           thresholds=[round(float(v),4) for v in thresholds.cpu().tolist()])
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
                if not a.no_diagnostics:
                    elapsed = round(time.perf_counter()-calibration_started,2)
                    diag.write("calibration_complete", elapsed_s=elapsed,
                               calibrated_miou=evaluated["calibrated"]["miou"],
                               raw_miou=evaluated["argmax"]["miou"])
                    print(f"[CALIB] elapsed={elapsed:.1f}s, "
                          f"raw_mIoU={evaluated['argmax']['miou']:.4f}, "
                          f"tuned_mIoU={evaluated['calibrated']['miou']:.4f}", flush=True)
            dist.barrier()
    finally:
        if "profiler" in locals() and profiler is not None:
            profiler.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()