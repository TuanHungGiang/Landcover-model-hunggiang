"""CPU tests for atomic checkpoints and no-NCCL checkpoint barrier design."""
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn
from scripts.checkpoint_io import (
    snapshot_cpu, atomic_torch_save, should_save_last, stage,
)

SOURCE=(Path(__file__).resolve().parents[1] /
        "scripts" / "train_loveda_ddp_hcr.py").read_text()


def test_snapshot_cpu_is_independent_and_preserves_metadata():
    m=nn.Sequential(nn.Linear(2,3),nn.BatchNorm1d(3))
    weights=m.state_dict()
    frozen=snapshot_cpu(weights)
    assert isinstance(frozen,OrderedDict)
    assert hasattr(frozen,"_metadata")
    assert frozen._metadata==weights._metadata
    old=frozen["0.weight"].clone()
    with torch.no_grad():
        m[0].weight.add_(1.)
    torch.testing.assert_close(frozen["0.weight"],old)
    clone=nn.Sequential(nn.Linear(2,3),nn.BatchNorm1d(3))
    clone.load_state_dict(frozen,strict=True)
    torch.testing.assert_close(clone[0].weight,old)


def test_optimizer_resume_snapshot_roundtrip(tmp_path):
    m=nn.Linear(3,2)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    (m(torch.randn(4,3)).square().mean()).backward()
    opt.step()
    checkpoint={"model":snapshot_cpu(m.state_dict()),
                "optimizer":snapshot_cpu(opt.state_dict()),
                "epoch":3,
                "model_config":{"architecture_version":"example"}}
    path=tmp_path/"last.pt"
    count=atomic_torch_save(checkpoint,path)
    assert count>0 and path.is_file()
    stored=torch.load(path,map_location="cpu",weights_only=True)
    model=nn.Linear(3,2)
    model.load_state_dict(stored["model"],strict=True)
    optim=torch.optim.AdamW(model.parameters(),lr=1e-3)
    optim.load_state_dict(stored["optimizer"])
    assert len(optim.state_dict()["state"])==len(opt.state_dict()["state"])


def test_atomic_save_does_not_replace_a_good_checkpoint_on_failure(tmp_path,monkeypatch):
    target=tmp_path/"best.pt"
    atomic_torch_save({"good":torch.ones(2)},target)
    def broken(*args,**kwargs):
        raise RuntimeError("disk failure simulation")
    monkeypatch.setattr(torch,"save",broken)
    with pytest.raises(RuntimeError,match="disk failure"):
        atomic_torch_save({"bad":True},target)
    assert target.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert "good" in torch.load(target,map_location="cpu",weights_only=True)


def test_saving_cadence_and_cpu_group_control(capsys):
    assert not should_save_last(1,5)
    assert should_save_last(5,5)
    assert should_save_last(1,5,is_final=True)
    stage(0,1,"model_cpu_snapshot_begin")
    assert "model_cpu_snapshot_begin" in capsys.readouterr().out
    assert 'backend="gloo"' in SOURCE
    assert 'dist.broadcast(status, src=0, group=control_group)' in SOURCE
    assert '"control_gloo_enter"' in SOURCE
    assert '"last_pt_write_begin"' in SOURCE
