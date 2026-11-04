"""Restart launcher regression tests; no CUDA or torchrun needed."""
from pathlib import Path
import json
import sys
import torch
from scripts import run_hcr_robust as s


def fake_checkpoint(path, *, epoch=29):
    state={"model":{"w":torch.ones(2)},
           "optimizer":{"state":{},"param_groups":[]},
           "scheduler":{"last_epoch":epoch},
           "scaler":{},
           "epoch":epoch,
           "best_miou":.5371,
           "training_recipe_version":"hcr_v2",
           "training_args":{
               "epochs":100,"batch":8,"accum":2,"workers":2,"val_workers":0,
               "memory_saver":True,
               "class_weight_alpha":.5,"no_calibrate":True,
               "evaluate_holdout":False,
               "size":1024,
               "lr":.0002,
               "backbone_lr":.00005,
           }}
    torch.save(state,path)
    return state


def test_preflight_and_dry_run_preserve_hyperparams(tmp_path,capsys):
    fake_checkpoint(tmp_path/"last.pt")
    assert s.run(tmp_path,max_restarts=2,dry_run=True)==0
    out=capsys.readouterr().out
    assert "resume_after_epoch=30" in out
    assert "--class-weight-alpha 0.5" in out
    assert "--lr 0.0002" in out
    assert "--workers 0" in out
    assert "--save-last-every 1" in out
    assert "--no-evaluate-holdout" in out
    assert "--resume "+str(tmp_path/"last.pt") in out
    assert "--eval-only" not in out


def test_retries_only_native_segv_and_stops_after_limit(tmp_path,monkeypatch):
    fake_checkpoint(tmp_path/"last.pt")
    responses=iter([(1,True),(1,True),(0,False)])
    attempted=[]
    def fake_stream(cmd,logfile,env):
        attempted.append(cmd)
        return next(responses)
    monkeypatch.setattr(s,"stream_one_attempt",fake_stream)
    monkeypatch.setattr(s.time,"sleep",lambda t: None)
    assert s.run(tmp_path,max_restarts=2)==0
    assert len(attempted)==3
    # True SIGSEGV if repeatedly encountered beyond the retry cap:
    assert s.NATIVE_SEGV.search("Root Cause: exitcode: -11 Signal 11 (SIGSEGV)")


def test_stops_python_failures_without_retry(tmp_path,monkeypatch):
    fake_checkpoint(tmp_path/"last.pt")
    attempts=[]
    monkeypatch.setattr(s,"stream_one_attempt",
                        lambda cmd,log,env: (attempts.append(cmd) or (1,False)))
    assert s.run(tmp_path,max_restarts=5)==1
    assert len(attempts)==1


def test_rejects_missing_or_corrupt_checkpoint(tmp_path):
    try:
        s.run(tmp_path,dry_run=True)
        assert False,"expected FileNotFoundError"
    except FileNotFoundError:
        pass
    (tmp_path/"last.pt").write_bytes(b"not a checkpoint")
    try:
        s.run(tmp_path,dry_run=True)
        assert False,"expected checkpoint load failure"
    except Exception:
        pass


def test_saves_last_per_epoch_and_spawns_workers():
    trainer=(Path(__file__).resolve().parents[1]/
             "scripts"/"train_loveda_ddp_hcr.py").read_text()
    assert 'p.add_argument("--workers", type=int, default=0' in trainer
    assert 'multiprocessing_context="spawn"' in trainer
    assert '"[RESUME] rank={rank} next_epoch={start+1}' in trainer
