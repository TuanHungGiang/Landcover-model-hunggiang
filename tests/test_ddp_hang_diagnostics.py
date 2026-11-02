"""Structural regression tests for v3 DDP hang diagnostics without running CUDA."""
import ast
from pathlib import Path
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "train_loveda_ddp_v3.py").read_text()


def _func(name):
    tree = ast.parse(SOURCE)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == name)
    ns = {"DataLoader": torch.utils.data.DataLoader, "time": __import__("time")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "<extracted>", "exec"), ns)
    return ns[name]


class TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index


def test_loader_timeout_enabled_only_with_workers():
    make_loader = _func("make_loader")
    no_workers = make_loader(TinyDataset(), 2, 0, timeout_seconds=12)
    workers = make_loader(TinyDataset(), 2, 1, timeout_seconds=12)
    try:
        assert no_workers.timeout == 0
        assert workers.timeout == 12
    finally:
        del workers


def test_rank_heartbeats_no_collectives(capsys):
    beat = _func("heartbeat")
    beat(1, 8, 126, 158, "fetch_start")
    assert "[HEARTBEAT] rank=1 epoch=8 step=126/158 phase=fetch_start" in capsys.readouterr().out


def test_collective_logging_and_cleanup():
    assert SOURCE.index('"epoch_loss_allreduce_enter"') < SOURCE.index(
        'dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)') < SOURCE.index(
        '"epoch_loss_allreduce_exit"')
    assert "faulthandler.dump_traceback_later" in SOURCE
    assert "faulthandler.cancel_dump_traceback_later()" in SOURCE
    assert 'train_iterator = iter(train_dl)' in SOURCE
    assert 'timeout=timedelta(seconds=a.ddp_timeout_seconds)' in SOURCE
