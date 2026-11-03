"""Make sure checkpointing does NOT change train-mode BN outputs/gradients."""
import copy

import torch
from torch import nn
from sdr_mambanet.utils.checkpointing import checkpoint_module


def test_checkpoint_preserves_bn_training_gradients_and_stats():
    torch.manual_seed(131)
    normal = nn.Sequential(
        nn.Conv2d(8, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.SiLU(),
    ).train()
    saved = copy.deepcopy(normal)
    a = torch.randn(2, 8, 12, 12, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    out_a = normal(a)
    out_b = checkpoint_module(saved, b)
    (out_a.square().mean()).backward()
    (out_b.square().mean()).backward()
    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-6)
    for x, y in zip(normal.parameters(), saved.parameters()):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-6)
    for (key, val), (key_b, val_b) in zip(normal.state_dict().items(),
                                           saved.state_dict().items()):
        assert key == key_b
        torch.testing.assert_close(val, val_b, atol=1e-6, rtol=1e-6)
    assert int(saved[1].num_batches_tracked.item()) == 1
