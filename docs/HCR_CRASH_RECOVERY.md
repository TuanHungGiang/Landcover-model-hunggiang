# RSF-HCRNet: native SIGSEGV safe mode and auto-resume

## What this does and does NOT do

CUDA/PyTorch native SIGSEGV cannot be completely eliminated by trainer Python code.
After a SIGSEGV on one rank, torchrun shuts down the other rank. Our robust
launcher runs torchrun as a child and retries only if its output shows SIGSEGV.
Native failures can recur; max_restarts stops an infinite crash loop.

Safety changes:
- Training DataLoader workers default to 0, validation 0; no CUDA-initialized
  rank is forked. If nonzero workers are requested, use spawn, not Linux fork.
- pin_memory disabled in single-process loader, reducing ancillary CUDA pinning.
- All existing training weights, weighted CE, schedule and HCR/Graph retained.
- Full atomic best.pt/last.pt checkpoints remain; launcher FORCES
  --save-last-every 1 so at most an unfinished epoch is repeated on a new run.
- Launcher restores FULL last.pt model/optimizer/scheduler/GradScaler, not best.pt.
- Logs saved per attempt, and trainer fatal_stacks_rankN.log preserved.

## Recover the user's Weighted V2 run at epoch 30

The training stdout reported epoch 31 completed with best mIoU .5371, and
--save-last-every 5 skipped saving last.pt at epoch 31. Expected latest
resume state is therefore epoch 30; verify rather than assume.

Kaggle notebook, cell 1:

\`\`\`python
%cd /kaggle/working/Landcover-model-hunggiang
!git pull origin main
!python -m pytest -q tests/test_hcr_robust_launcher.py tests/test_checkpoint_io.py
!ls -lh /kaggle/working/rsf_hcrnet_v2_weighted_alpha05/{last.pt,best.pt}
!tail -n 100 /kaggle/working/rsf_hcrnet_v2_weighted_alpha05/fatal_stacks_rank1.log
\`\`\`

Cell 2: dry-run verifies which hyperparameters are restored and shows the
command, without allocating GPU:

\`\`\`python
!python scripts/run_hcr_robust.py \\
  --from-run /kaggle/working/rsf_hcrnet_v2_weighted_alpha05 \\
  --max-restarts 2 --dry-run
\`\`\`

Cell 3: actually resume with bounded crash recovery:

\`\`\`python
!mkdir -p /kaggle/working/hcr_weighted_logs
!PYTHONUNBUFFERED=1 python scripts/run_hcr_robust.py \\
  --from-run /kaggle/working/rsf_hcrnet_v2_weighted_alpha05 \\
  --max-restarts 2 \\
  --log-dir /kaggle/working/hcr_weighted_logs/robust_attempts \\
  2>&1 | tee /kaggle/working/hcr_weighted_logs/robust_master.log
\`\`\`

Note: This launcher is for an **existing** run with a complete optimizer
last.pt. If only best.pt is available, no exact training resume is possible.

The launcher obtains the original training configuration from
last.pt["training_args"]. It forces workers=0, val_workers=0,
save_last_every=1, heartbeat_every=5, log_every=25,
no_calibrate=True, evaluate_holdout=False. These are infrastructure changes
or eval-policy changes, not model/loss/scheduler hyperparameter changes.

If SIGSEGV recurs, consult:
- output/robust_attempt_logs/attempt_NN.log (or chosen log-dir)
- output/fatal_stacks_rank1.log and fatal_stacks_rank0.log
- Kaggle GPU driver/system logs if available.
Repeated native SIGSEGV should be investigated via CUDA/cuDNN/torch version,
activation checkpoint implementation and hardware. Do not just increase NCCL
timeout; a native crash is not a collective timeout.

Caveats:
- Recovery is from epoch boundary only. A crash within an epoch loses that
  partial epoch; stochastic crop RNG is not precisely restored.
- Zero workers can reduce input pipeline throughput on Kaggle. After the
  run becomes stable, try --workers 2 explicitly with spawn and benchmark
  time per epoch; do not change training hyperparameters on resume.
- An entire Kaggle session reset may remove /kaggle/working, so back up
  both best.pt and last.pt outside the ephemeral session.
- The 12-hour GPU session time limit is NOT bypassed by a restart loop.
- CPU tests test orchestration and serialization. Full CUDA stability requires
  real execution on 2xT4 and cannot be proven by CI.
