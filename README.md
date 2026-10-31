# SDR-MambaNet — ResNet34 + RSF-Rep (research prototype)

PyTorch semantic segmentation model with a ResNet34 semantic encoder, RSF-Rep detail branch, optional Mamba, gated fusion, and a lightweight decoder.

## LoveDA on Kaggle

Dataset: alienxc137/loveda-satellite-images-semantic-segmentation

The trainer supports the following exact layout:
```text
Train/Train/Rural/images_png + masks_png
Train/Train/Urban/images_png + masks_png
Val/Val/Rural/images_png + masks_png
Val/Val/Urban/images_png + masks_png
```

Clone from a Kaggle notebook:

```python
!git clone https://github.com/TuanHungGiang/Landcover-model-hunggiang.git
%cd /kaggle/working/Landcover-model-hunggiang
!python smoke_test.py
```

Train **CNN baseline**, 512×512, single T4:
```python
!python scripts/train_loveda.py --context conv --size 512 --batch 1 --accum 4 --epochs 100
```

Train 1024×1024 (peak memory is higher; batch 1 is only a starting point):
```python
!python scripts/train_loveda.py --context conv --size 1024 --batch 1 --accum 4 --epochs 100
```

Train **real Mamba** (requires working compatible `mamba_ssm` CUDA installation):
```python
!python scripts/train_loveda.py --context mamba --size 512 --batch 1 --accum 4 --epochs 100
```

When Kaggle Internet is disabled and no ImageNet weights are cached, add `--no-pretrained` for a **from-scratch baseline**, or attach a local ResNet34 weights file and use `--resnet-weights-path /kaggle/input/your-weights/resnet34.pth`.

Dataset root defaults to:
`/kaggle/input/datasets/alienxc137/loveda-satellite-images-semantic-segmentation`.

Outputs: `/kaggle/working/sdr_loveda/best.pt`, `last.pt`, `history.jsonl`, `best_metrics.json`.

### Data protocol
Original LoveDA IDs: `0` means no-data; `1..7` represent the seven classes. The trainer remaps `1..7 → 0..6`, while `0/255 → 255` (ignore). Validation uses all nonoverlapping patches with ignored padding and global confusion-matrix mIoU. This is **validation-set performance**, not an independent test-set result. Current trainer monitors official Val each epoch and selects best checkpoint there; for rigorous scientific reporting, consider an internal validation split within Train and reserve official Val for final evaluation.

### Limitations
- Research prototype only, not verified SOTA or a peer-reviewed novel block.
- `--context conv` **does not use Mamba**.
- `--context mamba` requires compiling/installing compatible `mamba_ssm`; not validated on Kaggle T4.
- Training with 1024×1024 on T4 is not guaranteed to fit in VRAM; profile first.
- Trainer uses a single GPU (does not automatically distribute training over both Kaggle T4s).
- `configs/t4_*.yaml` are configuration references; the runnable trainer accepts CLI flags and currently does not parse YAML.
- `Kaggle_Quickstart.ipynb`, data, outputs, and checkpoints are intentionally not committed.


## 200-image validation and multiclass threshold tuning

Training now selects a fixed **100 Rural + 100 Urban** images from official LoveDA `Val/Val` using seed 42. These 200 images are used for checkpoint selection by raw mIoU, then **after training** for multiclass classwise decision thresholds. Remaining official Val images are excluded from tuning and optionally evaluated once with `--evaluate-holdout`.

```bash
python scripts/train_loveda.py --context conv --size 512 --batch 1 --accum 4 --epochs 100 --val-count 200 --seed 42 --evaluate-holdout
```

The selected file list is saved in `val_split_200.json` and must be retained for reproducible resumed training. Final `calibration.json` stores the seven classwise thresholds and raw/tuned metrics. Prediction is `argmax(softmax(logits)/class_threshold)`; this is **decision-threshold tuning**, not temperature-based probability calibration and not binary sigmoid thresholding. Calibrated mIoU on the selected 200 images is **not independent test performance**. The remaining held-out Val subset is a more suitable estimate if only evaluated once. The trainer reads command line arguments, not the YAML settings.
