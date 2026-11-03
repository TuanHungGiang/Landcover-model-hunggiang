# RSF-HCRNet v2 checkpoint / Gloo wait fix

## What the crash log reveals

At epoch 1 training and validation completed normally, then rank1 timed out
in the 1-element NCCL broadcast for early-stopping decision at the old line
671. The absence of [EARLY_STOP] means rank0 did not finish history/checkpoint
saving. This is **evidence of a checkpoint-stage stall**, but the old log
does not identify whether it was GPU synchronization, best.pt, last.pt, or
filesystem I/O. Do not simply raise the NCCL watchdog timeout.

## New checkpoint protocol

1. Rank0 logs every stage: history, CUDA sync, CPU model snapshot, best.pt
   write, CPU optimizer snapshot, last.pt write, checkpoint completion.
2. Snapshot state on CPU; write via atomic temporary-file rename to avoid
   exposing truncated best.pt/last.pt.
3. During expensive CPU/disk checkpointing, rank1 waits in an independent
   **CPU/Gloo control broadcast**, not a GPU/NCCL collective. That controls
   early-stop and propagates rank0 checkpoint exceptions to rank1.
4. Optional `--save-last-every 5` saves optimizer checkpoint every 5 epochs
   and at the scheduled/final early-stop epoch. best.pt is still saved
   whenever mIoU improves. Default remains 1 for backward-compatible saves.
5. `--checkpoint-timeout-seconds 900` gives the CPU control channel a
   separate timeout, while the usual NCCL GPU collective timeout is unchanged.

The diagnostic is conservative: network/disk failure could still hang a
checkpoint stage, but stage logs plus rank0 stack traces should now show where.

## Kaggle Notebook (live logs + persistent log)

In one cell:

```python
%cd /kaggle/working/Landcover-model-hunggiang
!git pull origin main
!git rev-parse --short HEAD
!python -m pytest -q tests/test_hcr_model.py tests/test_memory_checkpoint.py tests/test_checkpoint_io.py
!mkdir -p /kaggle/working/hcr_v2_logs
```

In a new cell:

```python
!PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp_hcr.py \
  --context conv --size 1024 --batch 8 --accum 2 --workers 2 \
  --memory-saver --epochs 100 --warmup-epochs 5 \
  --group-warmup-epochs 10 --hierarchy-ramp-epochs 6 \
  --confusion-min-score 0.02 --confusion-min-reciprocal 0.015 \
  --hierarchy-mix 0.25 --graph-steps 3 --graph-lambda 0.5 --graph-eta 0.5 \
  --hierarchy-weight 0.2 --flat-weight 0.2 --risk-weight 0.15 \
  --lr 2e-4 --backbone-lr 5e-5 --min-lr-ratio 0.1 \
  --augment strong --focus-prob 0.5 --focus-classes 4,6 \
  --zoom-min 1.0 --zoom-max 1.35 --early-stop-patience 20 \
  --early-stop-min-delta 0.001 --val-count 200 --seed 42 \
  --log-every 1 --heartbeat-every 25 --profile-every 0 \
  --save-last-every 5 --checkpoint-timeout-seconds 900 \
  --no-evaluate-holdout \
  --output /kaggle/working/rsf_hcrnet_v2_1024_batch8_safe_ckpt \
  2>&1 | tee /kaggle/working/hcr_v2_logs/train_safe_ckpt.log
```

Run directory MUST be new for a fresh training run. Old best.pt may still be
valid; inspect before deleting any directory:

```python
!ls -lh /kaggle/working/rsf_hcrnet_v2_1024_batch8/{best.pt,last.pt} 2>/dev/null
```

If a valid last.pt is present, it may be resumable with --resume and original
hyperparameters, though changes in save cadence are permissible. Use a
separate output directory if you want a new training experiment. A best.pt
contains model weights only, and is NOT a full optimizer resume checkpoint.

To locate problems, grep for [CHECKPOINT], [CHECKPOINT-ERROR], [CHECKPOINT-CONTROL],
[VAL-HCR], [HCR-STEP] and [RANK_FAILURE]. Avoid using --evaluate-holdout on
every research iteration.
