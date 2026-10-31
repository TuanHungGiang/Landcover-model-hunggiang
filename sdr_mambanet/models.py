"""SDR-MambaNet: original modular PROTOTYPE for experimentation.

Not a reproduced implementation nor validated SOTA.
"""
from __future__ import annotations
from dataclasses import dataclass
from torch import nn
import torch
from .backbone.dual_branch import DualBranchBackbone
from .blocks.fusion import SemanticGuidedFusion
from .blocks.rsf_rep import RSFRepDW
from .decoder.light_fpn import LightFPNDecoder


@dataclass
class SDRConfig:
    num_classes: int = 7    # LoveDA: adapt label mapping to your dataset
    semantic_encoder: str = "resnet34"  # or 'resnet34'
    pretrained_semantic: bool = True   # True may download ImageNet weights
    semantic_weights_path: str | None = None  # Optional local torchvision ResNet checkpoint
    context_type: str = "conv"          # 'conv' baseline, or genuine 'mamba'
    context_dim: int = 160
    context_depth: int = 2
    detail4: int = 48
    detail8: int = 96
    detail_depth4: int = 2
    detail_depth8: int = 2
    rep_kernel: int = 7
    decoder_channels: int = 64
    boundary_aux: bool = False


class SDRMambaNet(nn.Module):
    def __init__(self, cfg: SDRConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else SDRConfig()
        c = self.cfg
        self.backbone = DualBranchBackbone(
            detail4=c.detail4, detail8=c.detail8,
            depth4=c.detail_depth4, depth8=c.detail_depth8,
            rep_kernel=c.rep_kernel,
            semantic_encoder=c.semantic_encoder,
            pretrained_semantic=c.pretrained_semantic,
            semantic_weights_path=c.semantic_weights_path,
            context_dim=c.context_dim, context_depth=c.context_depth,
            context_type=c.context_type,
        )
        self.fusion = SemanticGuidedFusion(
            c.detail8, c.context_dim, out_channels=c.detail8,
        )
        self.decoder = LightFPNDecoder(
            detail4_channels=c.detail4,
            fused8_channels=c.detail8, decoder_channels=c.decoder_channels,
            num_classes=c.num_classes, rep_kernel=c.rep_kernel,
            boundary_aux=c.boundary_aux,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        maps = self.backbone(x)
        f8 = self.fusion(maps["d8"], maps["s16"])
        return self.decoder(maps["d4"], f8, x.shape[-2:])

    def switch_to_deploy(self, remove_aux: bool = False) -> "SDRMambaNet":
        """Fold RSFRepDW branches. Must be called in evaluation mode.

        Exact equivalence applies to re-parameterized convolutions. Other
        contextual blocks/gates/decoder ops remain unchanged.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before switch_to_deploy()")
        for module in self.modules():
            if isinstance(module, RSFRepDW):
                module.switch_to_deploy()
        if remove_aux:
            self.decoder.remove_training_only_heads()
        return self