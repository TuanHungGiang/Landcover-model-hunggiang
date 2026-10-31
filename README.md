# SDR-MambaNet | ResNet34 + RSF-Rep + optional Mamba | Kaggle T4

> **Research prototype, NOT a published or validated SOTA model.** It is not an exact reproduction of CRNet, DUSA-UNet, CGCNet, WeavingUnet or SFMNet. RSF-Rep is an experimental constrained multi-branch depthwise kernel parameterization, NOT established novel prior art. Prove novelty via literature review and controlled ablations.

## Architecture

```
RGB input
├── Detail encoder (new): Conv 1/2 -> 1/4 RSF-Rep x2 -> 1/8 RSF-Rep x2
└── Semantic encoder: torchvision ResNet34 up to layer3 (1/16, 256 channels)
    └── 1x1 projection 160 -> 2 raster Mamba blocks (or pure-CNN baseline)
             ↓
      Semantic-guided residual fusion @ 1/8
             ↓
      lightweight FPN + RSF-Rep @ 1/4
             ↓
      logits upsampled to input resolution
```

RSF-Rep trains seven BN-normalized spatial branches: region basis (spatial softmax), structure basis (zero-mean kernel), free K×K, local 3×3, 1×K, K×1, identity. It folds these **linear** branches into one K×K depthwise convolution in `model.eval()` using `switch_to_deploy()`. The pointwise layers, activations, Mamba, and fusion gates stay in the deploy network. The novel training regularization/parameterization must be evaluated against classical reparam schemes; conversion alone is *not* new.

### Files

| Path | Function |
|---|---|
| `sdr_mambanet/backbone/semantic.py` | ResNet34 torchvision ImageNet weights, local checkpoint option |
| `sdr_mambanet/backbone/detail.py` | RSF-Rep high-resolution detail CNN |
| `sdr_mambanet/backbone/dual_branch.py` | Connect branches |
| `sdr_mambanet/blocks/rsf_rep.py` | Region/structure/free bases; exact branch folding |
| `sdr_mambanet/blocks/context.py` | REAL raster Mamba or named CNN baseline |
| `sdr_mambanet/blocks/fusion.py` | Semantic-to-detail gated residual fusion |
| `sdr_mambanet/decoder/light_fpn.py` | Decoder and optional train-only boundary head |
| `sdr_mambanet/models.py` | SDRConfig / SDRMambaNet / deploy API |
| `sdr_mambanet/losses/segmentation.py` | CE + multi-class Dice + optional boundary BCE |
| `sdr_mambanet/data/seg_dataset.py` | LoveDA/Potsdam paired data + class labels/augmentation |
| `sdr_mambanet/engine/metrics.py` | Per-class IoU, mean IoU, F1, precision, recall, OA |
| `scripts/train.py` | AMP, AdamW, cosine LR, grad accumulation, DDP, checkpoints |
| `scripts/prepare_potsdam_tiles.py` | Prepare scene-disjoint Potsdam 512² RGB/mask tiles |
| `scripts/infer.py` | Sliding-window full-image inference with overlap |
| `scripts/export_deploy.py` | Verify conversion then save folded checkpoint |
| `scripts/profile_t4.py` | Inference latency, Params and VRAM profiling |
| `tests/test_complete.py` | Synthetic LoveDA/Potsdam tests + backward + deploy roundtrip |
| `configs/*.yaml` | ResNet34 conv/mamba baselines; Potsdam mamba |

## 1. Kaggle setup

Kaggle GPU settings: **T4** (or **T4×2**). A single T4 has 16 GiB physical VRAM. Two T4s do not share a 32 GiB allocation for one image. For first run, use **1 GPU** and `batch_size_per_gpu: 1` and `grad_accum_steps: 4`.

In a Kaggle notebook cell (after uploading this ZIP and extracting to `/kaggle/working/SDR_MambaNet_ResNet34_Kaggle`):

```bash
%cd /kaggle/working/SDR_MambaNet_ResNet34_Kaggle
!python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name(0))"
!pip install -q -r requirements.txt
!python -m tests.test_complete
```

The `torchvision` and `torch` versions must be compatible. **Do not reinstall torch just to install Mamba**. Mamba install on Kaggle T4 can require matching CUDA extensions; setup may fail depending on runtime.

Optional REAL Mamba (in Kaggle cell, after verifying the CUDA toolkit):

```bash
!pip install causal-conv1d --no-build-isolation
!pip install mamba-ssm --no-build-isolation
!python -c 'from mamba_ssm import Mamba; print("Mamba installed")'
```

If those commands fail, use the **explicit CNN baseline** config; `context_type: conv` does not pretend to be Mamba. The raster-Mamba implementation here is **not** pretrained VMamba/SS2D and has no pretrained Mamba weights; only ResNet34 can load ImageNet weights.

## 2. Load ResNet34 ImageNet weights

`pretrained_semantic: true` downloads official torchvision ImageNet weights when not cached; Kaggle internet access may be disabled.

For offline runs: attach the official torchvision checkpoint as a Kaggle input, then edit YAML:

```yaml
  pretrained_semantic: false
  semantic_weights_path: /kaggle/input/my-resnet34/resnet34-b627a593.pth
```

Alternatively, when Kaggle internet access works, keep `pretrained_semantic: true` and `semantic_weights_path: null`. The model uses only ResNet `conv1`, `bn1`, `layer1–3` (1/16, 256 channels), not `layer4` or `fc`.

## 3. Dataset layouts and labels

### LoveDA

This is the original layout after extracting official Train / Val archives:

```
/kaggle/input/loveda/
├── Train/
│   ├── Urban/images_png/*.png
│   ├── Urban/masks_png/*.png
│   ├── Rural/images_png/*.png
│   └── Rural/masks_png/*.png
└── Val/
    ├── Urban/images_png/*.png
    ├── Urban/masks_png/*.png
    ├── Rural/images_png/*.png
    └── Rural/masks_png/*.png
```

**Critical label mapping**: official LoveDA IDs `0=no-data`, `1=background`, `2=building`, `3=road`, `4=water`, `5=barren`, `6=forest`, `7=agriculture`. The loader maps raw `1..7` → train IDs `0..6`, raw `0` → `255` (ignored by loss/metrics). We use `num_classes=7` and `ignore_index=255`. Inference exports IDs back to `1..7`.

Validation covers entire images with non-overlapping tiles, padded pixels ignored, so you are not evaluating a single center crop. The `best.pt` checkpoint is chosen by **validation mIoU**. If this Validation also serves as your published evaluation set, declare checkpoint selection on it; for a more independent protocol, split part of *Train* for checkpoint selection and reserve official Validation for final evaluation. The script currently uses the configured `val` folder for checkpoint selection.

### ISPRS Potsdam

This loader expects pre-tiled RGB images and colored label masks. Potsdam classes are:

`impervious` = white, `building` = blue, `low_vegetation` = cyan, `tree` = green, `car` = yellow, `clutter` = red.

Prepare **scene-level** train/val partition (replace IDs with your verified split):

```bash
python scripts/prepare_potsdam_tiles.py \
  --images-dir /kaggle/input/potsdam/2_Ortho_RGB \
  --masks-dir /kaggle/input/potsdam/5_Labels_all \
  --output-dir /kaggle/working/potsdam_tiles \
  --train-ids 2_10,2_11,3_10 \
  --val-ids 3_11,3_12 \
  --tile 512 --stride 512
```

The IDs above are **examples only**, not an official or recommended benchmark partition. Avoid scene leakage; never randomly split tiles from the same original Potsdam scene between train/val.

Result:

```
potsdam_tiles/{train,val}/{images,masks}/*.png
```

Use 6 output classes. This project currently processes **RGB only**, not IRRG or DSM. For an apples-to-apples Potsdam paper comparison, match training scene lists, color mapping, ignore policy and metric convention.

## 4. Train LoveDA

Edit `data.root` and `train.output_dir` inside the YAML as necessary. For safe first run with no Mamba:

```bash
python scripts/train.py --config configs/loveda_resnet34_conv.yaml --epochs 2
```

For REAL Mamba **after successful Mamba CUDA installation**:

```bash
python scripts/train.py --config configs/loveda_resnet34_mamba.yaml
```

Two GPUs via DDP:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/loveda_resnet34_mamba.yaml
```

The trainer creates `best.pt`, `last.pt`, `history.jsonl`, `best_metrics.json`, and `config_used.yaml` in the output folder. Validation metrics are **global pixel-count metrics** from all non-overlapping 512² validation tiles. No test-time augmentation is implemented in the trainer.

Resume from an interrupted run:

```bash
python scripts/train.py --config configs/loveda_resnet34_mamba.yaml \
  --resume /kaggle/working/sdr_loveda_mamba/last.pt
```

## 5. Test deploy and full-scene prediction

```bash
python scripts/export_deploy.py \
  --checkpoint /kaggle/working/sdr_loveda_mamba/best.pt \
  --output /kaggle/working/sdr_loveda_mamba/deploy.pt

python scripts/infer.py \
  --checkpoint /kaggle/working/sdr_loveda_mamba/deploy.pt \
  --dataset loveda --input /kaggle/input/example/example.png \
  --output-dir /kaggle/working/predictions --tile 512 --overlap 128
```

The inference function combines **logits** across overlapping tiles, then takes argmax. LoveDA output pixels are raw IDs 1..7; this is not a confidence-calibrated or test-server submission implementation.

## 6. Limitations, scientific claims and T4 caution

- **Mamba CUDA compatibility** is not verified on a Kaggle T4 in this delivered environment. The pure-PyTorch CNN path is tested on CPU; Mamba code requires a compiled package. `mamba-ssm` does not automatically give you VMamba pretrained weights.
- **Weights are not bundled**. Setting `pretrained_semantic=True` alone is not evidence that ImageNet weights loaded successfully; a real run must verify its console output / load status.
- **RSF-Rep mathematical conversion** is evaluated in FP32 in smoke tests. Do a separate CUDA FP16 speed/accuracy test; combining branches does not guarantee an FPS improvement on every kernel size/hardware.
- **Parameters** in deploy are lower because branches are removed, but the encoder, context, decoder and gates remain.
- **Actual mIoU, Boundary IoU and GPU memory** must be measured after real training. Nothing in this package establishes SOTA or performance targets.
- If your objective changes to **binary road extraction**, adapt `num_classes`, labels and add connectivity evaluation such as clDice/APLS. Do not blindly apply road-specific topology loss to all LoveDA or Potsdam categories.

## Source inspiration (not exact implementation)

- CRNet (IEEE TGRS, 2026): semantic–spatial skip feature refinement and global–local context.
- DUSA-UNet (IEEE TGRS, 2025): local connectivity / holistic structural context.
- CGCNet (IEEE TGRS, 2025): compact global context.
- WeavingUnet (JAG, 2026): directional feature extraction.
- SFMNet (JAG, 2026): spatial–frequency modeling.