"""Regression tests for DDP v3 initialization (CPU-only; no NCCL needed)."""
import ast
from pathlib import Path
import pytest
import torch


def _loader_from_source():
    filename = Path(__file__).resolve().parents[1] / "scripts" / "train_loveda_ddp_v3.py"
    tree = ast.parse(filename.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "load_init_weights")
    module = ast.Module(body=[func], type_ignores=[])
    scope = {"Path": Path, "torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), scope)
    return scope["load_init_weights"]


def test_v2_checkpoint_preflight(tmp_path):
    fn = _loader_from_source()
    checkpoint = tmp_path / "best.pt"
    state = {"weight": torch.ones(2, 3)}
    torch.save({"model": state, "epoch": 8, "model_config": {
        "architecture_version": "r34_layer4_bi_v2", "context_type": "conv"}}, checkpoint)
    loaded = fn(str(checkpoint), "conv", 1)
    assert torch.equal(loaded["weight"], state["weight"])


def test_missing_checkpoint_fails_before_nccl(tmp_path):
    fn = _loader_from_source()
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        fn(str(tmp_path / "missing.pt"), "conv", 0)


def test_mismatched_context_fails_before_nccl(tmp_path):
    fn = _loader_from_source()
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": {"x": torch.ones(1)}, "model_config": {
        "architecture_version": "r34_layer4_bi_v2", "context_type": "mamba"}}, checkpoint)
    with pytest.raises(ValueError, match="context_type"):
        fn(str(checkpoint), "conv", 0)


def test_startup_and_lr_log_order():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_loveda_ddp_v3.py").read_text()
    assert src.index('initial_weights = (load_init_weights(') < src.index('dist.init_process_group(')
    assert src.count("raw = SDRMambaNet(cfg)") == 1
    assert src.index('stats["lr_pretrained"] = epoch_lrs[0]') < src.index('print(f"Epoch {epoch+1}/{a.epochs}:')
