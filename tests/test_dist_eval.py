"""Two-process CPU/Gloo smoke test for distributed all-reduce validation."""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from scripts.loveda_fast import FastLoveDA, ImageShardedEvalSampler, eval_distributed


class Dummy(torch.nn.Module):
    def forward(self, x):
        return {"logits": torch.stack([x[:, i % 3] + i*.1 for i in range(7)], dim=1)}


def _worker(rank, world, init_file, base, result_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        pairs = [(Path(base) / f"image_{i}.png", Path(base)/f"mask_{i}.png", "Rural") for i in range(5)]
        data = FastLoveDA(pairs, size=32, train=False, cache_size=2)
        sampler = ImageShardedEvalSampler(data, rank, world)
        loader = torch.utils.data.DataLoader(data, batch_size=1, sampler=sampler, num_workers=0)
        results = eval_distributed(Dummy(), loader, torch.device("cpu"),
                                   thresholds=[1., 1.1, .85, 1., 1., 1., 1.])
        if rank == 0:
            Path(result_file).write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_multi_process_equal_single_process(tmp_path):
    for i in range(5):
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, :, 1] = i * 20
        mask = np.tile(np.arange(64, dtype=np.uint8) % 8, (64, 1))
        Image.fromarray(img).save(tmp_path/f"image_{i}.png")
        Image.fromarray(mask).save(tmp_path/f"mask_{i}.png")
    pairs = [(tmp_path/f"image_{i}.png", tmp_path/f"mask_{i}.png", "Rural") for i in range(5)]
    full = FastLoveDA(pairs, size=32, train=False)
    loader = torch.utils.data.DataLoader(full, batch_size=1, num_workers=0)
    expected = eval_distributed(Dummy(), loader, torch.device("cpu"),
                                thresholds=[1., 1.1, .85, 1., 1., 1., 1.], sync=False)
    init_path = str(tmp_path/"init")
    result_path = str(tmp_path/"result.json")
    mp.spawn(_worker, args=(2, init_path, str(tmp_path), result_path),
             nprocs=2, join=True)
    actual = json.loads(Path(result_path).read_text())
    assert actual["argmax"]["confusion"] == expected["argmax"]["confusion"]
    assert actual["calibrated"]["confusion"] == expected["calibrated"]["confusion"]
