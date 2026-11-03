"""Activation checkpointing without double-updating BatchNorm running statistics.

PyTorch's non-reentrant checkpoint recomputes the forward during backward.
For BatchNorm in train mode, that recomputation would otherwise count as an
additional batch and change running statistics. We restore ONLY running
buffers after recompute, without changing train-mode forward/gradients.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@contextmanager
def _restore_bn_stats_after_recompute(module: nn.Module):
    tracked = []
    # These buffers are small (O(number of channels)), even in ResNet.
    with torch.no_grad():
        for layer in module.modules():
            if (isinstance(layer, nn.modules.batchnorm._BatchNorm)
                    and layer.training and layer.track_running_stats
                    and layer.running_mean is not None):
                tracked.append((
                    layer,
                    layer.running_mean.detach().clone(),
                    layer.running_var.detach().clone(),
                    layer.num_batches_tracked.detach().clone(),
                ))
    try:
        yield
    finally:
        with torch.no_grad():
            for layer, mean, var, seen in tracked:
                layer.running_mean.copy_(mean)
                layer.running_var.copy_(var)
                layer.num_batches_tracked.copy_(seen)


def checkpoint_module(module: nn.Module, *inputs: torch.Tensor):
    """Drop intermediate activations, recompute on backward preserving train BN.

    The checkpoint context is entered only at recomputation. The initial
    forward updates BN exactly once, as without checkpointing. Non-reentrant
    checkpoint supports frozen input tensors and DDP gradient accumulation.
    """
    if not (module.training and torch.is_grad_enabled()):
        return module(*inputs)
    return checkpoint(
        module, *inputs,
        use_reentrant=False,
        preserve_rng_state=True,
        context_fn=lambda: (nullcontext(), _restore_bn_stats_after_recompute(module)),
    )
