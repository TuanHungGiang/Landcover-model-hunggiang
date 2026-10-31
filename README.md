# SDR-MambaNet — ResNet34 backbone (research prototype)

A modular PyTorch model for high-resolution remote-sensing semantic segmentation, designed around a **ResNet34 semantic branch**, an experimental **RSF-Rep detail branch**, optional **Mamba context**, gated fusion and a lightweight decoder.

> **Status:** This repository currently contains the **model components and configuration reference only**, not the full dataset loaders, training/validation scripts, or inference pipeline. The complete local archive is available in the original ChatGPT conversation. The architecture and novel re-parameterization claims have **not been validated on LoveDA or Potsdam**.

## Clone and test in Kaggle

```bash
!git clone https://github.com/TuanHungGiang/Landcover-model-hunggiang.git
%cd /kaggle/working/Landcover-model-hunggiang
!python smoke_test.py
```

Kaggle normally already includes PyTorch and torchvision; install packages from `requirements.txt` if necessary. The smoke test uses ResNet34 without pretrained weights and uses `context_type="conv"` (a **CNN baseline, not Mamba**), so the smoke test avoids network downloads and custom CUDA dependencies.

## Construct ResNet34 + optional Mamba

```python
from sdr_mambanet import SDRConfig, SDRMambaNet

cfg = SDRConfig(
    num_classes=7,
    semantic_encoder="resnet34",
    pretrained_semantic=True,  # needs cached weights or Internet
    context_type="conv",       # change to "mamba" after installing compatible mamba-ssm
    context_dim=160,
    context_depth=2,
)
model = SDRMambaNet(cfg).cuda()
```

Only convolutional RSF-Rep submodules support structural reparameterization; Mamba and input-dependent gates remain dynamic. `pretrained_semantic=True` initializes the ResNet backbone with ImageNet weights, not the rest of the network.

## Current modules

- `sdr_mambanet/models.py` — model assembly and configuration
- `sdr_mambanet/backbone/` — detail, ResNet34 semantic, and dual-branch backbone
- `sdr_mambanet/blocks/rsf_rep.py` — experimental constrained spatial-kernel reparameterization
- `sdr_mambanet/blocks/context.py` — optional Mamba and explicit convolution baseline
- `sdr_mambanet/blocks/fusion.py` — semantic-guided fusion
- `sdr_mambanet/decoder/light_fpn.py` — decoder
- `sdr_mambanet/losses/segmentation.py` — segmentation loss
- `sdr_mambanet/utils/reparameterize.py` — deploy conversion
- `configs/` — reference training configs (not executable training entry points yet)

## Important data protocol

For original LoveDA labels, ignore raw `0` (no-data) and map raw classes `1..7` to training IDs `0..6`. Potsdam has a different six-class taxonomy, so use a separate class mapping and segmentation head.

This code is a **prototype**, not a published SOTA implementation or reproduction of CGCNet, CRNet, DUSA-UNet, SFMNet or WeavingUnet.
