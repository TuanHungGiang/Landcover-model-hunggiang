#!/usr/bin/env python3
"""HCR primary method: DDP training, train-only class graph, and spatial refinement.

Retains V3 LoveDA data/validation/DDP protocol without modifying v3 trainer.
Builds class confusion groups once from training batches after warm-up.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import random
import sys
import time
import traceback
from datetime import timedelta
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
                                 tune_thresholds_logits, accumulate_confusion,
                                 confusion_metrics, synchronize_eval_buffers)
from scripts.val200_calibration import select_validation_200
from scripts.holdout_evaluation import run_final_evaluation
from scripts.checkpoint_io import snapshot_cpu, atomic_torch_save, stage as ckpt_stage, should_save_last
from scripts.loveda_regularization import (
    RegularizedLoveDA, top_confusions, lr_multiplier, build_discriminative_groups,
)
from scripts.training_diagnostics import (
    StepCudaTimer, DiagnosticsWriter, class_health, gate_scales,
    selected_grad_norms,
)
from sdr_mambanet.hcr_model import HCRConfig, HCRSDRNet
from sdr_mambanet.losses.hcr import HCRSegmentationLoss


def arguments():
    p = argparse.ArgumentParser(description="HCR-Graph primary run: 2x T4 DDP LoveDA trainer")
    p.add_argument("--root", default=ROOT)
    p.add_argument("--size", type=int, choices=(512, 1024), default=512)
    p.add_argument("--context", choices=("conv", "mamba"), default="conv")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--group-warmup-epochs", type=int, default=3,
                   help="Freeze class groups after this training epoch (Train labels only)")
    p.add_argument("--confusion-min-reciprocal", type=float, default=0.015)
    p.add_argument("--hierarchy-ramp-epochs", type=int, default=6)
    p.add_argument("--confusion-min-score", type=float, default=0.02,
                   help="Minimum symmetric conditional confusion to join a class pair")
    p.add_argument("--hierarchy-mix", type=float, default=0.25)
    p.add_argument("--graph-steps", type=int, default=3)
    p.add_argument("--graph-lambda", type=float, default=0.5)
    p.add_argument("--graph-eta", type=float, default=0.5)
    p.add_argument("--hierarchy-weight", type=float, default=0.2)
    p.add_argument("--flat-weight", type=float, default=0.2)
    p.add_argument("--risk-weight", type=float, default=0.15)
    p.add_argument("--batch", type=int, default=1, help="Per-GPU batch size (default 1 for 16 GB T4)")
    p.add_argument("--accum", type=int, default=8, help="Accumulation per GPU (effective batch 16 on 2 GPUs)")
    p.add_argument("--workers", type=int, default=2, help="Dataloader workers PER GPU")
    p.add_argument("--memory-saver", action="store_true",
                   help="Checkpoint ResNet, RSF detail/decoder, fusion and graph; preserve train BatchNorm stats")
    p.add_argument("--loader-timeout-seconds", type=int, default=120,
                   help="Per-batch timeout when workers>0 (0 disables)")
    p.add_argument("--hang-dump-seconds", type=int, default=180,
                   help="Dump rank's Python stacks periodically if process stalls (0 disables)")
    p.add_argument("--heartbeat-every", type=int, default=25,
                   help="Log both ranks before data fetch and after step every N iterations (0 disables)")
    p.add_argument("--ddp-timeout-seconds", type=int, default=600,
                   help="NCCL process-group collective timeout; use 300 for faster debugging")
    p.add_argument("--checkpoint-timeout-seconds", type=int, default=900,
                   help="Separate CPU/Gloo control timeout while rank0 saves checkpoint")
    p.add_argument("--save-last-every", type=int, default=1,
                   help="Save optimizer/scaler resume checkpoint every N epochs (1 = original behavior)")
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
                   help="Initialize shared SDR weights from a v2/v3 best.pt; HCR heads start new")
    p.add_argument("--output", default="/kaggle/working/rsf_hcrnet_loveda_first_run")
    p.add_argument("--resnet-weights-path", default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval-only", default=None,
                   help="Evaluate an existing best.pt using the saved validation split; skip training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-count", type=int, default=200)
    p.add_argument("--calibration-samples-per-tile", type=int, default=2048)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--evaluate-holdout", dest="evaluate_holdout", action="store_true", default=False,
                   help="Evaluate untouched remaining Val ONCE on final run (disabled by default)")
    p.add_argument("--no-evaluate-holdout", dest="evaluate_holdout", action="store_false",
                   help="Skip remaining-Val evaluation (e.g. for hyperparameter search)")
    p.add_argument("--val-cache-images", type=int, default=8,
                   help="Decoded original images cached PER dataloader worker")
    p.add_argument("--train-cache-images", type=int, default=2)
    p.add_argument("--channels-last", action="store_true",
                   help="Optional NHWC format; benchmark first on your T4")
    p.add_argument("--fused-adamw", action="store_true",
                   help="Optional CUDA fused AdamW if available")
    p.add_argument("--log-every", type=int, default=100,
                   help="Report progress and loss components every N local GPU iterations")
    p.add_argument("--profile-every", type=int, default=0,
                   help="Sample CUDA event/module timings every N synchronized steps or at epoch end; 0 disables")
    p.add_argument("--no-diagnostics", action="store_true",
                   help="Disable extra JSONL diagnostics, module timers and printed per-class health")
    p.add_argument("--warn-iou-below", type=float, default=0.10,
                   help="Warn when a present class remains below this IoU after epoch 5")
    return p.parse_args()


def make_loader(ds, batch, workers, sampler=None, *, shuffle=False, timeout_seconds=120):
    kwargs = dict(batch_size=batch, num_workers=workers, pin_memory=True,
                  sampler=sampler, shuffle=shuffle and sampler is None)
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2,
                      timeout=timeout_seconds)
    return DataLoader(ds, **kwargs)


def heartbeat(rank: int, epoch: int, step: int, nsteps: int, phase: str) -> None:
    """Rank-specific progress marker; intentionally never calls a DDP collective."""
    print(f"[HEARTBEAT] rank={rank} epoch={epoch} step={step}/{nsteps} "
          f"phase={phase} unix={time.time():.3f}", flush=True)


def load_init_weights(path: str, context: str, rank: int):
    """Preflight baseline or HCR checkpoint on CPU before NCCL initialization."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fine-tuning checkpoint not found: {checkpoint_path}")
    print(f"[rank{rank}][INIT] reading {checkpoint_path} "
          f"({checkpoint_path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    incoming = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(incoming, dict) or not isinstance(incoming.get("model"), dict):
        raise ValueError("--init-from must contain a model state_dict")
    previous_config = incoming.get("model_config", {})
    if previous_config.get("architecture_version") not in (
            "r34_layer4_bi_v2", "r34_layer4_bi_hcr_v1"):
        raise ValueError("Checkpoint must be full layer4 baseline or HCR v1")
    if previous_config.get("context_type") != context:
        raise ValueError("--init-from context_type must match --context")
    print(f"[rank{rank}][INIT] checkpoint validated "
          f"(source epoch={incoming.get('epoch', 'unknown')})", flush=True)
    return incoming["model"]



@torch.no_grad()
def validate_hcr_heads(model, loader, device, *, channels_last=False):
    """Single-pass DDP validation: Flat, Hierarchy, Mixed, Final and pixel edits."""
    synchronize_eval_buffers(model)
    model.eval()
    cm=torch.zeros((4,7,7),dtype=torch.int64,device=device)
    changes=torch.zeros(5,dtype=torch.int64,device=device)
    risk_sum=torch.zeros(4,dtype=torch.float64,device=device)
    for x,y in loader:
        x=x.to(device,non_blocking=True)
        if channels_last: x=x.contiguous(memory_format=torch.channels_last)
        y=y.to(device,non_blocking=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"):
            o=model(x)
        pred_flat=o["flat_logits"].argmax(1)
        pred_hier=torch.nn.functional.interpolate(
            o["hier_probs"].float(),size=y.shape[-2:],mode="bilinear",
            align_corners=False).argmax(1)
        pred_mix=o["mixed_logits"].argmax(1)
        pred_final=o["logits"].argmax(1)
        for i,p in enumerate((pred_flat,pred_hier,pred_mix,pred_final)):
            accumulate_confusion(cm[i],p,y)
        valid=y!=255
        was=pred_mix==y
        now=pred_final==y
        changed=valid & (pred_mix!=pred_final)
        changes[0]+=valid.sum()
        changes[1]+=changed.sum()
        changes[2]+=(valid & ~was & now).sum()
        changes[3]+=(valid & was & ~now).sum()
        changes[4]+=(changed & ~was & ~now).sum()
        cover=torch.nn.functional.adaptive_avg_pool2d(
            valid[:,None].float(),o["risk_logits"].shape[-2:])
        wrong=torch.nn.functional.adaptive_avg_pool2d(
            (valid & ~was)[:,None].float(),o["risk_logits"].shape[-2:])
        risk=o["risk_logits"].float().sigmoid()
        target=wrong/cover.clamp_min(1e-6)
        selected=cover>0
        risk_sum[0]+=risk[selected].double().sum()
        risk_sum[1]+=target[selected].double().sum()
        risk_sum[2]+=(risk[selected].double()-target[selected].double()).square().sum()
        risk_sum[3]+=selected.sum()
    dist.all_reduce(cm,op=dist.ReduceOp.SUM)
    dist.all_reduce(changes,op=dist.ReduceOp.SUM)
    dist.all_reduce(risk_sum,op=dist.ReduceOp.SUM)
    names=("flat","hierarchy","mixed","final")
    heads={name:confusion_metrics(cm[i]) for i,name in enumerate(names)}
    graph=dict(valid_pixels=int(changes[0]),changed_pixels=int(changes[1]),
               corrected_pixels=int(changes[2]),damaged_pixels=int(changes[3]),
               wrong_to_wrong=int(changes[4]),
               net_correct=int(changes[2]-changes[3]),
               risk_mean=float(risk_sum[0]/risk_sum[3].clamp_min(1)),
               actual_error=float(risk_sum[1]/risk_sum[3].clamp_min(1)),
               risk_brier=float(risk_sum[2]/risk_sum[3].clamp_min(1)))
    return heads["final"],heads,graph


def main():
    a = arguments()
    if a.log_every < 1 or a.profile_every < 0 or not 0 <= a.warn_iou_below <= 1:
        raise ValueError("--log-every >=1, --profile-every >=0, --warn-iou-below in [0,1]")
    if sum(bool(x) for x in (a.resume, a.init_from, a.eval_only)) > 1:
        raise ValueError("Use only one of --resume, --init-from, --eval-only")
    if a.eval_only:
        manifest = (Path(a.output) / ("val_split_200.json" if a.val_count == 200
                                    else f"val_split_{a.val_count}.json"))
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Evaluation requires original validation split manifest: {manifest}")
    if a.epochs < 1 or a.warmup_epochs < 0 or a.warmup_epochs >= a.epochs:
        raise ValueError("0 <= warmup-epochs < epochs required")
    if not (1 <= a.group_warmup_epochs <= a.epochs):
        raise ValueError("1 <= group-warmup-epochs <= epochs required")
    if not (0 <= a.confusion_min_reciprocal <= 1) or a.hierarchy_ramp_epochs < 1:
        raise ValueError("Invalid reciprocal confusion / hierarchy ramp")
    if not (0 <= a.confusion_min_score <= 1):
        raise ValueError("--confusion-min-score must be in [0,1]")
    if not (0 <= a.hierarchy_mix <= 1 and 0 <= a.graph_eta <= 1):
        raise ValueError("hierarchy-mix and graph-eta must be in [0,1]")
    if a.graph_steps < 1 or a.graph_lambda < 0 or min(a.hierarchy_weight, a.flat_weight, a.risk_weight) < 0:
        raise ValueError("Invalid HCR steps/lambda/loss weights")
    if a.backbone_lr <= 0 or a.lr <= 0 or a.early_stop_patience < 0 or a.early_stop_min_delta < 0:
        raise ValueError("Learning rates and early stop arguments must be positive")
    focus_classes = tuple(int(part) for part in a.focus_classes.split(",") if part.strip())
    if a.batch < 1 or a.accum < 1 or a.workers < 0:
        raise ValueError("--batch/--accum must be positive, --workers nonnegative")
    if a.save_last_every < 1 or a.checkpoint_timeout_seconds < 60:
        raise ValueError("save-last-every>=1 and checkpoint-timeout-seconds>=60 required")
    if min(a.loader_timeout_seconds, a.hang_dump_seconds,
           a.heartbeat_every) < 0 or a.ddp_timeout_seconds < 60:
        raise ValueError("Timeouts/heartbeat must be >=0 and DDP timeout >=60 seconds")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle accelerator 2 x Tesla T4")
    local_rank = int(os.environ["LOCAL_RANK"])
    # Resolve/validate --init-from BEFORE NCCL initializes. This prevents
    # a bad checkpoint from leaving the second rank waiting in a collective.
    weight_source = a.init_from or a.eval_only
    initial_weights = (load_init_weights(weight_source, a.context, local_rank)
                       if weight_source else None)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Explicit mapping avoids NCCL guessing the device for barrier/all-reduce.
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device,
                            timeout=timedelta(seconds=a.ddp_timeout_seconds))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        dist.destroy_process_group()
        raise RuntimeError("Launch with torchrun --nproc_per_node=2")
    try:
        # CPU/Gloo control channel is separate from NCCL data collectives:
        # rank1 can safely wait while rank0 serializes a potentially large
        # checkpoint, without enqueueing a timed NCCL GPU broadcast.
        control_group = dist.new_group(
            backend="gloo", timeout=timedelta(seconds=a.checkpoint_timeout_seconds))
        print(f"[CHECKPOINT-CONTROL] rank={rank} backend=gloo "
              f"timeout={a.checkpoint_timeout_seconds}s", flush=True)
        random.seed(a.seed + rank)
        np.random.seed(a.seed + rank)
        torch.manual_seed(a.seed + rank)
        # cudnn.benchmark may select large convolution workspaces on 16 GB T4.
        torch.backends.cudnn.benchmark = not a.memory_saver

        root, output = Path(a.root), Path(a.output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            selected, remaining = select_validation_200(
                find_pairs(root, "val"), root, output,
                seed=a.seed, count=a.val_count, resume=bool(a.resume or a.eval_only))
            (output / ("eval_args.json" if a.eval_only else "args.json")).write_text(
                json.dumps(vars(a), indent=2))
        dist.barrier()
        if rank != 0:
            selected, remaining = select_validation_200(
                find_pairs(root, "val"), root, output,
                seed=a.seed, count=a.val_count, resume=True)

        # Both ranks record their own blocking Python stacks, even if torchrun
        # later terminates the process due to a NCCL watchdog timeout.
        hang_log = None
        if a.hang_dump_seconds:
            hang_log = (output / f"hang_stacks_rank{rank}.log").open("a", encoding="utf-8")
            faulthandler.dump_traceback_later(a.hang_dump_seconds,
                                             repeat=True, file=hang_log)
        diag = DiagnosticsWriter(output, rank, enabled=not a.no_diagnostics)
        train_ds = RegularizedLoveDA(
            find_pairs(root, "train"), a.size, True,
            cache_size=a.train_cache_images, augment=a.augment,
            focus_probability=a.focus_prob, focus_classes=focus_classes,
            zoom_min=a.zoom_min, zoom_max=a.zoom_max,
        )
        train_sampler = DistributedSampler(train_ds, num_replicas=world,
                                           rank=rank, shuffle=True, seed=a.seed)
        train_dl = make_loader(train_ds, a.batch, a.workers, train_sampler,
                               timeout_seconds=a.loader_timeout_seconds)
        val_ds = FastLoveDA(selected, a.size, False, cache_size=a.val_cache_images)
        val_sampler = ImageShardedEvalSampler(val_ds, rank, world)
        val_dl = make_loader(val_ds, 1, a.workers, val_sampler,
                             timeout_seconds=a.loader_timeout_seconds)

        cfg = HCRConfig(
            num_classes=7, semantic_encoder="resnet34",
            pretrained_semantic=(rank == 0 and not a.no_pretrained and not a.resume and not weight_source),
            semantic_weights_path=a.resnet_weights_path if rank == 0 and not a.resume and not weight_source else None,
            context_type=a.context, hierarchy_mix=a.hierarchy_mix,
            graph_steps=a.graph_steps, graph_lambda=a.graph_lambda,
            graph_eta=a.graph_eta, memory_saver=a.memory_saver,
             hierarchy_ramp_epochs=a.hierarchy_ramp_epochs)
        # Every rank constructs its own model. For --init-from all ranks have
        # already loaded the same checkpoint before NCCL initialization.
        # DDP still broadcasts rank0 weights/buffers on construction.
        print(f"[rank{rank}][INIT] constructing RSF-HCRNet", flush=True)
        raw = HCRSDRNet(cfg)
        if initial_weights is not None:
            incompat = raw.load_state_dict(initial_weights, strict=False)
            acceptable = ("group_head.", "fine_head.", "graph_refiner.")
            allowed_buffers = {"class_groups", "class_train_confusion"}
            if incompat.unexpected_keys or any(
                not (key.startswith(acceptable) or key in allowed_buffers)
                for key in incompat.missing_keys
            ):
                raise ValueError(f"Incompatible HCR checkpoint: {incompat}")
            if a.eval_only and incompat.missing_keys:
                raise ValueError("--eval-only requires an HCR checkpoint, not baseline")
            del initial_weights
            if rank == 0:
                print(f"[FINETUNE] initialized from {a.init_from}; "
                      "optimizer/scheduler reset", flush=True)
        print(f"[rank{rank}][INIT] moving model to {device}", flush=True)
        raw = raw.to(device)
        if a.channels_last:
            raw = raw.to(memory_format=torch.channels_last)
        ddp = DDP(raw, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=False, gradient_as_bucket_view=True)
        profiler = StepCudaTimer(ddp.module) if rank == 0 and not a.no_diagnostics and a.profile_every else None
        loss_fn = HCRSegmentationLoss(
            ignore_index=255, hierarchy_weight=a.hierarchy_weight,
            flat_weight=a.flat_weight, risk_weight=a.risk_weight).to(device)
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
            if ckpt.get("training_recipe_version") != "hcr_v2":
                raise ValueError("Only resume HCR last.pt; use --init-from for baseline checkpoints")
            prev_args = ckpt.get("training_args", {})
            for key in ("epochs", "seed", "size", "batch", "accum", "memory_saver", "confusion_min_reciprocal", "hierarchy_ramp_epochs", "augment", "focus_prob", "focus_classes", "zoom_min", "zoom_max", "warmup_epochs", "min_lr_ratio", "lr", "backbone_lr", "group_warmup_epochs", "confusion_min_score", "hierarchy_mix", "graph_steps", "graph_lambda", "graph_eta", "hierarchy_weight", "flat_weight", "risk_weight"):
                if key in prev_args and prev_args[key] != getattr(a,key):
                    raise ValueError(f"Resume option changed: {key} ({prev_args[key]} -> {getattr(a,key)})")
            if previous.get("architecture_version") != "r34_layer4_bi_hcr_v2":
                raise ValueError("Cannot resume a non-HCR checkpoint")
            if previous.get("context_type") != a.context:
                raise ValueError("--context must match resume checkpoint")
            ddp.module.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start, best, stale = ckpt["epoch"] + 1, ckpt["best_miou"], ckpt.get("early_stop_stale", 0)
        if a.eval_only:
            start = a.epochs  # Skip training entirely; final evaluation uses supplied checkpoint.
            if rank == 0:
                print(f"[EVAL-ONLY] checkpoint={a.eval_only}; training skipped", flush=True)
        if rank == 0:
            if a.memory_saver:
                print("[MEMORY] activation checkpoint: ResNet + RSF detail/decoder + fusion + graph; "
                      "BN training statistics preserved; cudnn benchmark OFF", flush=True)
            print(f"DDP: {world} GPUs | batch/GPU={a.batch} | accumulation={a.accum} "
                  f"| effective_batch={world * a.batch * a.accum} "
                  f"| val_images={len(selected)} | val_total_tiles={len(val_ds)} "
                  f"| val_tiles/rank={[len(val_sampler), len(val_ds)-len(val_sampler)]} "
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
            strength = min(1.0, max(0.0, (epoch + 1 - a.group_warmup_epochs) / a.hierarchy_ramp_epochs))
            ddp.module.set_hierarchy_strength(strength)
            if rank == 0:
                print(f"[HCR-SCHEDULE] epoch={epoch+1} strength={strength:.4f} "
                      f"effective_mix={strength * a.hierarchy_mix:.4f}",flush=True)
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
            train_iterator = iter(train_dl)
            # Train-only predictions. After warm-up, freeze disjoint confusion groups.
            train_confusion = (torch.zeros((7, 7), dtype=torch.int64, device=device)
                               if epoch + 1 == a.group_warmup_epochs else None)
            for step in range(nsteps):
                trace_now = (a.heartbeat_every > 0 and
                             (step % a.heartbeat_every == 0 or step == nsteps - 1))
                if trace_now:
                    heartbeat(rank, epoch + 1, step + 1, nsteps, "fetch_start")
                try:
                    image, mask = next(train_iterator)
                except Exception as exc:
                    print(f"[RANK_FAILURE] rank={rank} epoch={epoch+1} "
                          f"step={step+1}/{nsteps} fetch_error={exc!r}", flush=True)
                    raise
                if trace_now:
                    heartbeat(rank, epoch + 1, step + 1, nsteps, "fetch_done")
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
                        out = ddp(image)
                        loss_parts = loss_fn(out, mask)
                        loss = loss_parts["total"]
                    if train_confusion is not None:
                        with torch.no_grad():
                            pred_train = out["flat_logits"].detach().argmax(dim=1)
                            valid_train = mask != 255
                            bins = torch.bincount(
                                (mask[valid_train].long() * 7 + pred_train[valid_train]).long(),
                                minlength=49).reshape(7, 7)
                            train_confusion.add_(bins)
                    loss_total.add_(loss.detach())
                    ce_total.add_(loss_parts["ce"].detach())
                    dice_total.add_(loss_parts["dice"].detach())
                    if not a.no_diagnostics and ((step+1) % a.log_every == 0 or step+1 == nsteps):
                        with torch.no_grad():
                            mask_ok = mask != 255
                            change_pct = 100 * (((out["mixed_logits"].argmax(1) != out["logits"].argmax(1)) & mask_ok).sum().float() / mask_ok.sum().clamp_min(1))
                            hcr_diag = dict(hierarchy_ce=float(loss_parts["hierarchy"].detach()),
                                            flat_ce=float(loss_parts["flat"].detach()),
                                            risk_loss=float(loss_parts["risk"].detach()),
                                            risk_mean=float(out["risk_logits"].float().sigmoid().mean()),
                                            graph_delta_abs=float(out["graph_delta"].float().abs().mean()),
                                            graph_changes_pct=float(change_pct),
                                            hierarchy_conf=float(out["hier_probs"].float().amax(1).mean()))
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
                        print(f"[HCR-STEP] epoch={epoch+1} step={step+1}/{nsteps} " +
                              " ".join(f"{key}={value:.5f}" for key,value in hcr_diag.items()),flush=True)
                        diag.write("hcr_step", epoch=epoch+1, local_step=step+1, **hcr_diag)
                if trace_now:
                    heartbeat(rank, epoch + 1, step + 1, nsteps, "step_complete")
                fetch_start = time.perf_counter()
            heartbeat(rank, epoch + 1, nsteps, nsteps, "epoch_loss_allreduce_enter")
            dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
            heartbeat(rank, epoch + 1, nsteps, nsteps, "epoch_loss_allreduce_exit")
            train_time = time.perf_counter() - start_train
            # No validation labels or predictions are used to design class groups.
            if train_confusion is not None:
                dist.all_reduce(train_confusion, op=dist.ReduceOp.SUM)
                groups = ddp.module.configure_groups_from_training(
                    train_confusion, min_score=a.confusion_min_score,
                    min_reciprocal=a.confusion_min_reciprocal)
                if rank == 0:
                    (output / "train_confusion_groups.json").write_text(json.dumps({
                        "epoch": epoch + 1, "groups": groups,
                        "training_confusion": train_confusion.cpu().tolist(),
                        "min_score": a.confusion_min_score,
                        "min_reciprocal": a.confusion_min_reciprocal,
                        "note": "Calculated from training predictions and training GT only."
                    }, indent=2))
                    print(f"[HCR] class groups frozen from Train: {groups}", flush=True)
            # All ranks evaluate disjoint original images; confusion allreduce ONCE.
            start_val = time.perf_counter()
            stats, head_metrics, graph_metrics = validate_hcr_heads(
                ddp.module, val_dl, device, channels_last=a.channels_last)
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
                stats["head_metrics"] = head_metrics
                stats["graph_metrics"] = graph_metrics
                print(f"[VAL-HCR] epoch={epoch+1} " + " ".join(
                    f"{name}={head_metrics[name]['miou']:.4f}"
                    for name in ("flat","hierarchy","mixed","final")) +
                    f" graph_fixed={graph_metrics['corrected_pixels']}"+
                    f" graph_broken={graph_metrics['damaged_pixels']}"+
                    f" graph_net={graph_metrics['net_correct']:+d}"+
                    f" risk_mean={graph_metrics['risk_mean']:.4f}"+
                    f" risk_brier={graph_metrics['risk_brier']:.4f}",flush=True)
                for name in ("flat","hierarchy","mixed","final"):
                    print(f"[VAL-HCR-CLASS] epoch={epoch+1} stage={name} " +
                          " | ".join(f"{cls}={head_metrics[name]['iou_per_class'][cls]:.3f}"
                                     for cls in NAMES),flush=True)
                stats["hierarchy_strength"] = strength
                stats["effective_hierarchy_mix"] = a.hierarchy_mix * strength
                stats["lr_pretrained"] = epoch_lrs[0]
                stats["lr_new"] = epoch_lrs[1]
                print(f"Epoch {epoch+1}/{a.epochs}: loss={stats['train_loss']:.4f} "
                      f"mIoU={stats['miou']:.4f} OA={stats['oa']:.4f} "
                      f"train={train_time:.1f}s val={val_time:.1f}s "
                      f"lr_resnet={stats['lr_pretrained']:.2g} lr_new={stats['lr_new']:.2g}", flush=True)
                save_error = None
                try:
                    ckpt_stage(rank, epoch+1, "history_begin")
                    stats["early_stop_stale_before_update"] = stale
                    with (output / "history.jsonl").open("a") as fp:
                        fp.write(json.dumps(stats, allow_nan=False) + "\n")
                    ckpt_stage(rank, epoch+1, "history_done")
                    prior_best = best
                    is_best = stats["miou"] > prior_best
                    if is_best:
                        best = stats["miou"]
                    stale = (0 if stats["miou"] > prior_best + a.early_stop_min_delta
                             else stale+1)
                    stopping = bool(a.early_stop_patience>0 and stale>=a.early_stop_patience)
                    need_last = should_save_last(epoch+1, a.save_last_every,
                                                 is_final=(epoch+1==a.epochs or stopping))
                    if is_best or need_last:
                        ckpt_stage(rank, epoch+1, "gpu_sync_begin")
                        torch.cuda.synchronize(device)
                        ckpt_stage(rank, epoch+1, "gpu_sync_done")
                        ckpt_stage(rank, epoch+1, "model_cpu_snapshot_begin")
                        model_cpu = snapshot_cpu(ddp.module.state_dict())
                        ckpt_stage(rank, epoch+1, "model_cpu_snapshot_done")
                    else:
                        model_cpu = None
                    if is_best:
                        ckpt_stage(rank, epoch+1, "best_pt_write_begin")
                        bytes_saved = atomic_torch_save(
                            {"model": model_cpu, "epoch": epoch,
                             "best_miou": best, "model_config": asdict(ddp.module.cfg)},
                            output/"best.pt")
                        ckpt_stage(rank, epoch+1, "best_pt_write_done",
                                   detail=f"bytes={bytes_saved}")
                        (output/"best_metrics.json").write_text(json.dumps(stats,indent=2))
                    if need_last:
                        ckpt_stage(rank, epoch+1, "optimizer_cpu_snapshot_begin")
                        optimizer_cpu = snapshot_cpu(optimizer.state_dict())
                        ckpt_stage(rank, epoch+1, "optimizer_cpu_snapshot_done")
                        ckpt_stage(rank, epoch+1, "last_pt_write_begin")
                        bytes_saved = atomic_torch_save(
                            {"model":model_cpu, "epoch":epoch, "best_miou":best,
                             "model_config":asdict(ddp.module.cfg),
                             "optimizer":optimizer_cpu,
                             "scheduler":scheduler.state_dict(),
                             "scaler":scaler.state_dict(),
                             "early_stop_stale":stale,
                             "training_recipe_version":"hcr_v2",
                             "training_args":vars(a)},
                            output/"last.pt")
                        ckpt_stage(rank, epoch+1, "last_pt_write_done",
                                   detail=f"bytes={bytes_saved}")
                    else:
                        ckpt_stage(rank,epoch+1,"last_pt_skipped",
                                   detail=f"save_last_every={a.save_last_every}")
                    print(f"[EARLY_STOP] stale={stale}/{a.early_stop_patience} "
                          f"best_mIoU={best:.4f} last_saved={need_last}",flush=True)
                    ckpt_stage(rank, epoch+1, "checkpoint_complete")
                except Exception as exc:
                    save_error = f"{type(exc).__name__}: {exc}"
                    print(f"[CHECKPOINT-ERROR] rank={rank} epoch={epoch+1} "
                          f"{save_error}",flush=True)
                    traceback.print_exc()
            # Rank1 waits on the CPU control group, NEVER an in-flight NCCL
            # broadcast during rank0 CPU serialization / filesystem writes.
            ckpt_stage(rank, epoch+1, "control_gloo_enter")
            # Fields: [should_stop, checkpoint_failed].
            status = torch.zeros(2, dtype=torch.int32)  # CPU for gloo
            if rank == 0:
                status[0] = int(a.early_stop_patience>0 and
                                stale>=a.early_stop_patience)
                status[1] = int(save_error is not None)
            dist.broadcast(status, src=0, group=control_group)
            ckpt_stage(rank, epoch+1, "control_gloo_exit",
                       detail=f"stop={int(status[0])} failed={int(status[1])}")
            if int(status[1]):
                raise RuntimeError("Rank0 checkpoint saving failed. See [CHECKPOINT-ERROR] and "
                                   "checkpoint stage logs on rank0.")
            if int(status[0]):
                if rank == 0:
                    print(f"[EARLY_STOP] stopped after epoch {epoch+1}; "
                          f"best_mIoU={best:.4f}", flush=True)
                break
            # No NCCL collectives occur while rank0 is writing checkpoints.
            ckpt_stage(rank, epoch+1, "next_epoch_barrier_enter")
            dist.barrier()
            ckpt_stage(rank, epoch+1, "next_epoch_barrier_exit")

        # This path is entered by BOTH ranks, including after early stopping.
        # Keep model/threshold selection on selected 200 images; holdout is scored once.
        eval_checkpoint = Path(a.eval_only) if a.eval_only else output / "best.pt"
        if not eval_checkpoint.is_file():
            raise FileNotFoundError(f"Expected best checkpoint for final evaluation: {eval_checkpoint}")
        if not a.no_calibrate or a.evaluate_holdout:
            run_final_evaluation(
                model=ddp.module, args=a, rank=rank, world=world, device=device,
                output=output, checkpoint=eval_checkpoint,
                selected=selected, remaining=remaining, val_loader=val_dl,
                diagnostics=diag if rank == 0 and not a.no_diagnostics else None,
            )

    finally:
        if "hang_log" in locals() and hang_log is not None:
            faulthandler.cancel_dump_traceback_later()
            hang_log.close()
        if "profiler" in locals() and profiler is not None:
            profiler.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()