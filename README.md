# SDR-MambaNet v2 — ResNet34 layer4 + Bidirectional Cross-Scale Fusion

**Research prototype:** Full ResNet34 ImageNet-pretrained encoder + RSF-Rep detail encoder + gated cross-scale bidirectional feature exchange. Optional Mamba at 1/32.

## Architecture

```text
RGB
├─ Detail: conv 1/2 -> RSF-Rep 1/4 (D4:48) -> RSF-Rep 1/8 (D8:96)
└─ Semantic: pretrained ResNet34 layer1 (S4:64) -> layer2 (S8:128)
     D8 <-> S8 : BidirectionalStageFusion (spatial-channel gates)
     S8' -> layer3 (S16:256) -> layer4 (S32:512)
     S32 -> 1x1 (160ch) -> [Mamba x2 OR CNN context x2] at 1/32
     upsample(S32) + S16 + downsample(D8) -> SemanticCrossScaleFusion at 1/16
     final SemanticGuidedFusion(S16 -> D8) at 1/8
     D8 fused + D4 skip -> FPN/RSF-Rep decoder -> 7-class logits
```

The original one-way fusion is retained **after** the new gated exchanges. Gates remain dynamic at inference; only RSF-Rep spatial depthwise branches support exact kernel folding. The architecture is an experimental proposal, not validated SOTA.

## Kaggle LoveDA dataset

Supported folders:
```text
Train/Train/Rural/images_png + masks_png
Train/Train/Urban/images_png + masks_png
Val/Val/Rural/images_png + masks_png
Val/Val/Urban/images_png + masks_png
```

```python
!git clone https://github.com/TuanHungGiang/Landcover-model-hunggiang.git
%cd /kaggle/working/Landcover-model-hunggiang
!python smoke_test.py
```

**CNN-context baseline (no Mamba), ResNet34 to layer4:**
```python
!python scripts/train_loveda.py --context conv --size 512 --batch 1 --accum 4 --epochs 100 --val-count 200 --seed 42 --evaluate-holdout
```

**Real Mamba at 1/32 (requires compatible `mamba_ssm` CUDA installation):**
```python
!python scripts/train_loveda.py --context mamba --size 512 --batch 1 --accum 4 --epochs 100 --val-count 200 --seed 42 --evaluate-holdout
```

For 1024 crops set `--size 1024`, keep batch 1 and profile T4 VRAM before long runs. With no cached ImageNet weights and no Kaggle Internet, pass `--resnet-weights-path /kaggle/input/.../resnet34.pth` or use `--no-pretrained` **only as a from-scratch baseline**.

**Checkpoints:** default output changed to `/kaggle/working/sdr_loveda_r34_bi_v2` to protect earlier training. Do NOT resume old layer3 checkpoints with v2; new layer4 and fusion weights change parameter shapes and names. Old code lives in the `baseline-resnet34-layer3` branch.

## Validation and threshold selection

- Train 2522 images (official Train).
- Pick fixed 100 Rural + 100 Urban from official Val with seed 42; save `val_split_200.json`.
- Validate raw mIoU after every epoch on these 200 whole images (non-overlapping 512 tiles => 800 tiles).
- After selecting best checkpoint, optimize multiclass *decision thresholds* `argmax(softmax(logits)/class_threshold)` on the same 200; it is NOT softmax probability calibration.
- Optionally evaluate the remaining 1469 Val images once via `--evaluate-holdout`.
- Original LoveDA mask 0 -> ignore 255, raw IDs 1..7 -> class indices 0..6.

Outputs: `best.pt`, `last.pt`, `best_metrics.json`, `history.jsonl`, `val_split_200.json`, `calibration.json`.

## Tests

```bash
python smoke_test.py
python -m pytest -q tests/test_bidirectional.py
```

A no-pretrained CPU test of v2 (CNN context) checks forward/backward gradients, intermediate shapes and deploy numerical equivalence. The Mamba CUDA extension and actual T4 memory/performance have **not** been verified here.

Reference configs in `configs/` are informational; the runnable trainer uses CLI flags.
