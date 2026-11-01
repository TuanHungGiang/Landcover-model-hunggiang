# LoveDA v3 generalization experiments (ResNet34 full layer4 + Bidirectional Fusion)

This trainer is an experimental alternative to `scripts/train_loveda_ddp.py`. **It does not replace or modify the v2 architecture, loss, label mapping, selected 200 images, calibration or held-out evaluation**. Compare against the existing v2 `best.pt` (raw validation mIoU ≈ 0.5317; calibrated ≈ 0.549 on its selected Val images). These are *not* independent test metrics.

## Fine-tune the existing best checkpoint (recommended first)

The v2 `best.pt` **must be accessible in the current Kaggle session** or attached as a Kaggle input dataset. Change the path if you have stored it elsewhere. Weights are loaded strictly; old optimizer is **not** resumed. Rank 0 reads v2 best weights before DDP initialization, rank 1 receives weights through DDP broadcast. The initial mIoU is evaluated on the exact selected 200 images and the original `best.pt` weights are preserved as fallback in the new output folder.

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp_v3.py \
  --context conv --size 512 --batch 8 --accum 2 --workers 2 \
  --epochs 60 --warmup-epochs 2 \
  --lr 8e-5 --backbone-lr 2e-5 --min-lr-ratio 0.1 \
  --augment strong --focus-prob 0.5 --focus-classes 4,6 \
  --early-stop-patience 15 --early-stop-min-delta 0.001 \
  --val-count 200 --seed 42 --evaluate-holdout \
  --init-from /kaggle/working/sdr_loveda_r34_bi_v2_ddp/best.pt
```

**Do not** pass `--resume` together with `--init-from`. Resume a v3 `last.pt` only with the same training recipe and total number of epochs:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp_v3.py \
  --context conv --size 512 --batch 8 --accum 2 --workers 2 \
  --epochs 60 --warmup-epochs 2 --lr 8e-5 --backbone-lr 2e-5 \
  --augment strong --focus-prob 0.5 --focus-classes 4,6 \
  --early-stop-patience 15 --seed 42 --val-count 200 \
  --resume /kaggle/working/sdr_loveda_r34_bi_v2_ddp_v3/last.pt
```

If the original best checkpoint was lost, run new v3 training from ImageNet-pretrained ResNet34 by **omitting `--init-from`**. Without internet, provide local ImageNet weights via `--resnet-weights-path` or explicitly use `--no-pretrained` for a different from-scratch experiment.

## Experimental options

- `--augment none` uses the original training augmentation; `light` adds gentle zoom and brightness/color; `strong` adds up to 1.35x zoom, crop jitter and moderate photometric variations. **Validation images are untouched**.
- `--focus-prob 0.5 --focus-classes 4,6` draws zoomed crops near rare/problematic training labels: raw LoveDA IDs 5 (`Barren`) and 7 (`Agriculture`), mapped to training IDs 4 and 6. When the source image and crop have the same size and zoom is disabled, class-aware cropping cannot change the visible image.
- `--lr` applies to detail, fusion, context and decoder; `--backbone-lr` to pretrained ResNet34 stem + layers 1–4. Both use the same warmup + cosine factor.
- `--early-stop-patience 15` synchronizes stop decision across two DDP ranks, so both exit together. Best is retained even when improvement is less than the early-stop minimum delta.
- `--init-from` loads v2 **model weights only** and evaluates the starting point before updates; `--resume` requires a v3 `last.pt` with optimizer/scheduler/scaler state.
- Per-epoch `[CONFUSIONS]` prints the most common conditional GT→prediction confusions and stores the full confusion table in `history.jsonl`.

## Test

```bash
python -m pytest -q tests/test_regularization.py
```

## Important scientific caveats

Do **not** change the fixed 200 validation selection or use the remaining 1469 official Val images to pick hyperparameters. Repeatedly trying recipes against the same 200 images still risks selection bias. Reserve holdout for a final comparison; `--evaluate-holdout` should be enabled on the final experiment, not every exploratory run. Benchmark augmentation overhead and train seconds/epoch on actual T4 hardware before concluding v3 is faster or more accurate than v2. The new recipe is a hypothesis, not a guaranteed improvement.