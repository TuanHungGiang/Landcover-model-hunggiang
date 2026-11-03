"""Safe model and optimizer checkpointing: CPU snapshots and atomic writes."""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import torch


def snapshot_cpu(value):
    """Copy tensors to CPU without retaining aliases to live optimizer/model."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, OrderedDict):
        copied = OrderedDict((k, snapshot_cpu(v)) for k, v in value.items())
        if hasattr(value, "_metadata"):
            copied._metadata = snapshot_cpu(value._metadata)
        return copied
    if isinstance(value, dict):
        return {key: snapshot_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_cpu(item) for item in value)
    return value


def atomic_torch_save(value, path: Path):
    """Write a temporary file, atomically rename only after successful save."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as fp:
            torch.save(value, fp)
        os.replace(tmp, path)
        return path.stat().st_size
    finally:
        tmp.unlink(missing_ok=True)


def stage(rank: int, epoch: int, label: str, *, detail: str = ""):
    print(f"[CHECKPOINT] rank={rank} epoch={epoch} stage={label} "
          f"unix={time.time():.3f} {detail}", flush=True)


def should_save_last(epoch_one_based: int, every: int, *, is_final=False):
    if every < 1:
        raise ValueError("save-last-every must be >= 1")
    return bool(is_final or epoch_one_based % every == 0)
