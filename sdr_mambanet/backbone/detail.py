"""Independent detail path, retaining 1/4 and 1/8 spatial feature maps."""
from torch import nn
import torch
from ..blocks.rsf_rep import RSFRepBlock
from ..utils.checkpointing import checkpoint_module


def _stem(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8 if out_channels % 8 == 0 else 1, out_channels),
        nn.SiLU(inplace=True),
    )


class DetailEncoder(nn.Module):
    def __init__(self, channels4: int = 48, channels8: int = 96,
                 depth4: int = 2, depth8: int = 2, rep_kernel: int = 7):
        super().__init__()
        self.stem2 = _stem(3, 32, stride=2)
        self.stem4 = _stem(32, channels4, stride=2)
        self.stage4 = nn.Sequential(*[
            RSFRepBlock(channels4, kernel_size=rep_kernel) for _ in range(depth4)
        ])
        self.down8 = _stem(channels4, channels8, stride=2)
        self.stage8 = nn.Sequential(*[
            RSFRepBlock(channels8, kernel_size=rep_kernel) for _ in range(depth8)
        ])
        self.activation_checkpointing = False

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.activation_checkpointing = bool(enabled)

    def _run(self, module, x):
        return checkpoint_module(module, x) if self.activation_checkpointing else module(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._run(self.stem2, x)
        d4 = self._run(self.stage4, self._run(self.stem4, x))
        d8 = self._run(self.stage8, self._run(self.down8, d4))
        return {"d4": d4, "d8": d8}