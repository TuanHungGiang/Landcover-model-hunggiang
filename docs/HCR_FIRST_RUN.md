# RSF-HCRNet first training run (LoveDA, 2×T4)

This is the **main method**, not an ablation suite. It retains the complete
ResNet34 semantic branch, RSF-Rep detail branch, bi8 / cross16 / final gated
fusion and the V3 data/validation protocol. HCR adds:
1. Flat K-class segmentation + a train-confusion-derived **soft hierarchy**.
2. Independently supervised **error-risk estimation**, learned from Train GT.
3. A **sparse 4-neighbor graph** using semantic + RSF detail features, class
   group consistency and reliability-anchored Jacobi refinement.

The hierarchy uses disjoint confused class pairs, not fixed "background versus
foreground". The same HCR model supports K=6 for Vaihingen/Potsdam (dataset
loaders/trainers for ISPRS are NOT part of this first LoveDA run).

## Start on Kaggle

1. Enable **GPU T4 ×2**.
2. Make sure the LoveDA dataset is attached at the path shown by --root.
3. If your V3 best checkpoint is available, use the recommended warm-start.
   All original backbone/decoder/fusion weights are loaded; HCR-specific heads
   are random-initialized. The initial validation does **not** represent a
   fully trained HCR model.
4. Run in the project directory:

\`\`\`bash
!git clone https://github.com/TuanHungGiang/Landcover-model-hunggiang.git
%cd /kaggle/working/Landcover-model-hunggiang
!python -m pytest -q tests/test_hcr_model.py
\`\`\`

### First full main-method training (warm-start from V3 best)

Update the checkpoint path below to your own location:

\`\`\`bash
!torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp_hcr.py \
  --context conv --size 512 --batch 6 --accum 2 --workers 2 \
  --epochs 60 --warmup-epochs 3 \
  --group-warmup-epochs 3 --confusion-min-score 0.02 \
  --hierarchy-mix 0.25 --graph-steps 3 --graph-lambda 0.5 --graph-eta 0.5 \
  --lr 8e-5 --backbone-lr 2e-5 --augment strong \
  --focus-prob 0.5 --focus-classes 4,6 \
  --early-stop-patience 15 --val-count 200 --seed 42 \
  --no-evaluate-holdout \
  --init-from /kaggle/input/YOUR_BASELINE_CHECKPOINT/best.pt
\`\`\`

The batch size of 6 per T4 is a **conservative starting value, not measured**
for HCR. Reduce to 4 or 2 if CUDA OOM. Effective batch size =
2 GPUs × per-GPU batch × accumulation.

### If you do not have a baseline best.pt

Remove --init-from and use ImageNet pretrained ResNet34:

\`\`\`bash
!torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp_hcr.py \
  --context conv --size 512 --batch 6 --accum 2 --workers 2 \
  --epochs 60 --warmup-epochs 3 --group-warmup-epochs 3 \
  --lr 2e-4 --backbone-lr 5e-5 --augment strong \
  --early-stop-patience 15 --val-count 200 --seed 42 \
  --no-evaluate-holdout
\`\`\`

If Kaggle Internet is OFF and ImageNet weights are not cached, supply
--resnet-weights-path /kaggle/input/.../resnet34.pth. --no-pretrained trains
the encoder from scratch and is not an equivalent baseline.

### Resume without restarting group discovery

\`\`\`bash
!torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp_hcr.py \
  --context conv --size 512 --batch 6 --accum 2 --workers 2 \
  --epochs 60 --warmup-epochs 3 --group-warmup-epochs 3 \
  --lr 8e-5 --backbone-lr 2e-5 --augment strong \
  --focus-prob 0.5 --focus-classes 4,6 \
  --early-stop-patience 15 --val-count 200 --seed 42 \
  --resume /kaggle/working/rsf_hcrnet_loveda_first_run/last.pt \
  --no-evaluate-holdout
\`\`\`

For resume, **all options that affect training must match** the initial run,
including the HCR options. Do not combine --resume with --init-from.

## Files generated

- \`best.pt\`, \`last.pt\`: full model state including frozen class-group mapping
- \`train_confusion_groups.json\`: grouping from TRAIN predictions/labels at warm-up
- \`best_metrics.json\`, \`history.jsonl\`: selected-200 metrics and train record
- \`val_split_200.json\`: fixed balanced Rural/Urban selection
- \`calibration.json\`: calibrated on the selected 200 only if enabled

**Scientific protection:** The remaining 1,469 validation images are not
consulted during training/group discovery/threshold fitting. Run
--evaluate-holdout **only once**, after choosing the final model. For this
first exploratory run, --no-evaluate-holdout is deliberately the default.

## Limitations before first GPU test

- Local CPU tests verify shape/gradients/checkpoint/reparameterization, not
  actual Kaggle T4 throughput, distributed scaling or final mIoU.
- Four-neighbor graph/Jacobi steps approximate a quadratic Laplacian solver;
  they can oversmooth narrow roads and tiny cars.
- Confusion groups are learned once on Train after warm-up and fixed afterward.
  Graph grouping/affinity is not guaranteed to correct erroneous predictions.
- This is an experimental combination of established ideas, not a validated
  novel/SOTA method. Claims require literature comparison and results.


## OOM: memory-saver mode (2 × T4, 16 GB each)

When a prior run used --batch 4, restart the Kaggle kernel (terminate all old
torchrun GPU processes) and run the same main method with --batch 1 --accum 8
--memory-saver. The effective batch remains 16 images. Example without a
previous SDR checkpoint (ResNet34 pretrained ImageNet):

\`\`\`bash
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp_hcr.py \
  --context conv --size 512 --batch 1 --accum 8 --workers 2 \
  --memory-saver --epochs 100 --warmup-epochs 5 --group-warmup-epochs 10 \
  --lr 2e-4 --backbone-lr 5e-5 --augment strong \
  --focus-prob 0.5 --focus-classes 4,6 \
  --early-stop-patience 20 --val-count 200 --seed 42 --no-evaluate-holdout \
  --output /kaggle/working/rsf_hcrnet_loveda_memory_saver
\`\`\`

The allocator environment setting is optional and may help fragmentation; it
does **not** fix genuinely insufficient GPU memory. Note: memory-saver
recomputes encoder stages + graph during backward. It intentionally freezes
pretrained semantic BatchNorm running statistics to avoid updating them twice
and can increase training time. Semantic BN affine parameters still learn.
If training ResNet34 fully from scratch (--no-pretrained), freezing randomly
initialized BN running statistics is not recommended. Prefer the normal mode
with --batch 1 or supply pretrained weights.

If still OOM at batch 1, capture the **full CUDA OOM traceback**, especially
the free/allocated/reserved VRAM numbers and which operation triggers OOM.
Also run !nvidia-smi to check other processes. The absence of a T4 benchmark
means VRAM sufficiency is not guaranteed.

For a truly fresh run, do not reuse an old output directory; resume only if
all training options (including --memory-saver) match the saved checkpoint.
