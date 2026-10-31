import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.loveda_fast import (FastLoveDA, ImageShardedEvalSampler,
                                  eval_distributed, accumulate_confusion,
                                  calibration_samples, tune_thresholds_logits)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Seven deterministic class logits, spatially-varying via RGB.
        return {"logits": torch.stack([x[:, i % 3] * (i + 1) / 7 + i / 10
                                      for i in range(7)], dim=1)}


def fixture_pairs(tmp_path, n=6):
    items = []
    for i in range(n):
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, :, 0] = i * 12
        img[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :]
        mask = np.tile(np.arange(64, dtype=np.uint8) % 8, (64, 1))
        # ensure original raw IDs 0..7 are present
        imp, mp = tmp_path / f"image_{i}.png", tmp_path / f"mask_{i}.png"
        Image.fromarray(img).save(imp)
        Image.fromarray(mask).save(mp)
        items.append((imp, mp, "Rural" if i % 2 else "Urban"))
    return items


def test_original_masks_caching_and_sharding(tmp_path):
    ds = FastLoveDA(fixture_pairs(tmp_path), size=32, train=False, cache_size=2)
    assert len(ds) == 6 * 4
    a, mask = ds[0]
    assert a.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert int(mask[0, 0]) == 255  # raw zero ignored
    assert int(mask[0, 1]) == 0    # raw class one => 0
    assert int(mask[0, 7]) == 6    # raw class seven => 6
    for i in range(4):
        ds[i]
    assert len(ds._cache) == 1, "All tiles of first image should reuse decoded original"
    r0 = list(ImageShardedEvalSampler(ds, 0, 2))
    r1 = list(ImageShardedEvalSampler(ds, 1, 2))
    assert len(r0) == len(r1) == 12
    assert set(r0).isdisjoint(r1) and sorted(r0+r1) == list(range(len(ds)))
    assert {ds.tiles[i][0] for i in r0}.isdisjoint({ds.tiles[i][0] for i in r1})


def test_gpu_confusion_equivalence_cpu(tmp_path):
    ds = FastLoveDA(fixture_pairs(tmp_path), size=32, train=False, cache_size=2)
    model = DummyModel()
    full_loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
    full = eval_distributed(model, full_loader, torch.device("cpu"), sync=False)
    matrices = []
    for rank in (0, 1):
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=0,
            sampler=ImageShardedEvalSampler(ds, rank, 2))
        matrices.append(eval_distributed(model, loader, torch.device("cpu"), sync=False))
    actual = (np.array(matrices[0]["confusion"]) + np.array(matrices[1]["confusion"]))
    np.testing.assert_array_equal(actual, np.array(full["confusion"]))
    assert actual.sum() == 6*64*64*(7/8)


def test_calibration_logits_matches_probability_rule(tmp_path):
    rng = np.random.default_rng(99)
    logits = rng.normal(size=(160, 7)).astype(np.float16)
    target = rng.integers(0, 7, size=(160,), dtype=np.uint8)
    t, fit = tune_thresholds_logits(logits, target, torch.device("cpu"))
    probs = torch.tensor(logits).float().softmax(1)
    a = (probs/torch.tensor(t).reshape(1,7)).argmax(1)
    b = (torch.tensor(logits).float()-torch.tensor(t).log().reshape(1,7)).argmax(1)
    assert torch.equal(a, b)
    assert fit["tuned_sample_miou"] >= fit["raw_sample_miou"]


def test_fast_sampling_returns_only_sampled_pixels(tmp_path):
    ds = FastLoveDA(fixture_pairs(tmp_path), size=32, train=False, cache_size=1)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
    x, y = calibration_samples(DummyModel(), loader, torch.device("cpu"), per_tile=12)
    assert x.shape == (len(ds)*12, 7)
    assert y.shape == (len(ds)*12,)
    assert (y < 7).all()
