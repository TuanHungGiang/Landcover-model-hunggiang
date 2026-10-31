"""Region–Structure–Free (RSF) depthwise structural re-parameterization.

ORIGINAL RESEARCH PROTOTYPE, not an implementation copied from a paper.
The following linear branches are folded exactly into ONE depthwise KxK
convolution at evaluation time:
  constrained region low-pass, zero-DC structural kernel, unconstrained
  KxK, local 3x3, horizontal 1xK, vertical Kx1, identity.
Only the linear branch sum is folded; SiLU, pointwise convs, and GroupNorm
remain as ordinary inference operations.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BasisDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, kind: str):
        super().__init__()
        if kind not in ("region", "structure"):
            raise ValueError(f"Unsupported kernel basis: {kind}")
        self.channels, self.kernel_size, self.kind = channels, kernel_size, kind
        self.raw_weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.normal_(self.raw_weight, std=0.02)

    def effective_weight(self) -> torch.Tensor:
        if self.kind == "region":
            return torch.softmax(self.raw_weight.flatten(2), dim=-1).view_as(self.raw_weight)
        return self.raw_weight - self.raw_weight.mean(dim=(2, 3), keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.effective_weight(), padding=self.kernel_size // 2,
            groups=self.channels,
        )


class RSFRepDW(nn.Module):
    """All branches act on the SAME tensor and have linear outputs before sum.

    BatchNorm statistics must be frozen via eval() before conversion.
    `switch_to_deploy()` removes all training branches and creates one DWConv.
    """
    def __init__(self, channels: int, kernel_size: int = 7, deploy: bool = False):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        self.channels, self.kernel_size = channels, kernel_size
        self.deploy = deploy
        if deploy:
            self.reparam = nn.Conv2d(
                channels, channels, kernel_size, padding=kernel_size // 2,
                groups=channels, bias=True,
            )
            return
        branch_specs = {
            "region": BasisDepthwiseConv(channels, kernel_size, "region"),
            "structure": BasisDepthwiseConv(channels, kernel_size, "structure"),
            "free": nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                              groups=channels, bias=False),
            "local": nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            "horizontal": nn.Conv2d(channels, channels, (1, kernel_size),
                                    padding=(0, kernel_size // 2), groups=channels, bias=False),
            "vertical": nn.Conv2d(channels, channels, (kernel_size, 1),
                                  padding=(kernel_size // 2, 0), groups=channels, bias=False),
        }
        self.branches = nn.ModuleDict({
            name: nn.Sequential(conv, nn.BatchNorm2d(channels))
            for name, conv in branch_specs.items()
        })
        self.identity_bn = nn.BatchNorm2d(channels)
        # Average the initial branch amplitudes; learned gamma remains unconstrained.
        for br in self.branches.values():
            nn.init.constant_(br[1].weight, 1 / 7)
            nn.init.zeros_(br[1].bias)
        nn.init.constant_(self.identity_bn.weight, 1 / 7)
        nn.init.zeros_(self.identity_bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)
        y = self.identity_bn(x)
        for branch in self.branches.values():
            y = y + branch(x)
        return y

    @staticmethod
    def _fold_bn(weight: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        if bn.running_mean is None or bn.running_var is None:
            raise RuntimeError("BN must track running statistics to be folded")
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_kernel = weight * scale.reshape(-1, 1, 1, 1)
        fused_bias = bn.bias - bn.running_mean * scale
        return fused_kernel, fused_bias

    def _pad_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        kh, kw = kernel.shape[-2:]
        pad_h = self.kernel_size - kh
        pad_w = self.kernel_size - kw
        if pad_h < 0 or pad_w < 0 or pad_h % 2 or pad_w % 2:
            raise RuntimeError("Cannot center-align this kernel")
        return F.pad(kernel, (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2))

    @torch.no_grad()
    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.reparam.weight.detach().clone(), self.reparam.bias.detach().clone()
        kernel = torch.zeros(
            self.channels, 1, self.kernel_size, self.kernel_size,
            device=self.identity_bn.weight.device, dtype=self.identity_bn.weight.dtype,
        )
        bias = torch.zeros(self.channels, device=kernel.device, dtype=kernel.dtype)
        for branch in self.branches.values():
            conv, bn = branch[0], branch[1]
            w = conv.effective_weight() if isinstance(conv, BasisDepthwiseConv) else conv.weight
            fused_w, fused_b = self._fold_bn(w, bn)
            kernel += self._pad_kernel(fused_w)
            bias += fused_b
        eye = torch.zeros_like(kernel)
        eye[:, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        eye_w, eye_b = self._fold_bn(eye, self.identity_bn)
        kernel += eye_w
        bias += eye_b
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self) -> "RSFRepDW":
        if self.deploy:
            return self
        if self.training:
            raise RuntimeError("Call model.eval() before switch_to_deploy()")
        w, b = self.equivalent_kernel_bias()
        conv = nn.Conv2d(
            self.channels, self.channels, self.kernel_size,
            padding=self.kernel_size // 2, groups=self.channels, bias=True,
        ).to(device=w.device, dtype=w.dtype)
        conv.weight.copy_(w)
        conv.bias.copy_(b)
        self.reparam = conv
        del self.branches
        del self.identity_bn
        self.deploy = True
        return self


class RSFRepBlock(nn.Module):
    """Residual detail block with deployable spatial submodule."""
    def __init__(self, channels: int, kernel_size: int = 7, expansion: float = 2.0,
                 deploy: bool = False):
        super().__init__()
        hidden = max(channels, int(channels * expansion))
        self.spatial = RSFRepDW(channels, kernel_size, deploy=deploy)
        self.act1 = nn.SiLU(inplace=False)
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, hidden) if hidden % min(8, hidden) == 0 else 1, hidden)
        self.act2 = nn.SiLU(inplace=False)
        self.project = nn.Conv2d(hidden, channels, 1, bias=True)
        # Modest residual initial contribution while keeping branch trainable.
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spatial(x)
        y = self.project(self.act2(self.norm(self.expand(self.act1(y)))))
        return x + y * self.layer_scale[None, :, None, None]