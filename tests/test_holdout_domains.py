"""Regression tests for domain-stratified holdout under CPU/Gloo."""
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from scripts.holdout_evaluation import (
    DomainTaggedVal, _holdout_loader, evaluate_holdout_with_domains,
)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        a = x[:, 0:1]
        return {"logits": torch.cat([a, -a, *[torch.full_like(a, -99.) for _ in range(5)]], dim=1)}


def _fixture(tmp_path):
    pairs = []
    for idx, domain in enumerate(("Rural", "Urban", "Rural", "Urban")):
        red = 240 if domain == "Rural" else 10
        x = np.zeros((32, 32, 3), np.uint8)
        x[..., 0] = red
        label = np.ones((32, 32), np.uint8) if domain == "Rural" else np.full((32, 32), 2, np.uint8)
        img = tmp_path / f"{idx}.png"
        mask = tmp_path / f"m{idx}.png"
        Image.fromarray(x).save(img)
        Image.fromarray(label).save(mask)
        pairs.append((img, mask, domain))
    return pairs


def test_domain_eval_single_process(tmp_path):
    loader, total_tiles, local_tiles = _holdout_loader(
        _fixture(tmp_path),
        SimpleNamespace(size=32, val_cache_images=0, workers=0,
                        loader_timeout_seconds=12), 0, 1)
    assert total_tiles == local_tiles == 4
    result = evaluate_holdout_with_domains(DummyModel(), loader, torch.device("cpu"), 0,
        thresholds=torch.ones(7))
    raw = result["argmax"]["confusion"]
    rural = result["by_domain"]["Rural"]["raw"]["confusion"]
    urban = result["by_domain"]["Urban"]["raw"]["confusion"]
    np.testing.assert_array_equal(np.asarray(raw), np.asarray(rural) + np.asarray(urban))
    assert sum(map(sum, raw)) == 4 * 32 * 32
    assert result["by_domain"]["Rural"]["calibrated"]["miou"] >= 0


def _ddp_worker(rank, initfile, root, resultfile):
    dist.init_process_group("gloo", init_method=f"file://{initfile}", rank=rank, world_size=2)
    try:
        pairs = _fixture(Path(root))
        loader, _, _ = _holdout_loader(
            pairs, SimpleNamespace(size=32, val_cache_images=0, workers=0,
                                   loader_timeout_seconds=12), rank, 2)
        result = evaluate_holdout_with_domains(DummyModel(), loader, torch.device("cpu"), rank,
                                                thresholds=torch.ones(7))
        if rank == 0:
            import json
            Path(resultfile).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_domain_eval_distributed(tmp_path):
    initfile = tmp_path / "ddp_init"
    output = tmp_path / "ddp_result.json"
    torch.multiprocessing.spawn(_ddp_worker, args=(str(initfile), str(tmp_path), str(output)),
                                nprocs=2, join=True)
    import json
    result = json.loads(output.read_text())
    raw = np.array(result["argmax"]["confusion"])
    rural = np.array(result["by_domain"]["Rural"]["raw"]["confusion"])
    urban = np.array(result["by_domain"]["Urban"]["raw"]["confusion"])
    np.testing.assert_array_equal(raw, rural + urban)
    assert raw.sum() == 4096
