"""Regression tests for HCR validation handoff / no eval worker fork."""
import ast
from pathlib import Path

import torch
from scripts import train_loveda_ddp_hcr as hcr


def test_val_traces_mark_forward_and_finish(monkeypatch,capsys):
    monkeypatch.setattr(hcr.dist,"all_reduce",lambda *args,**kwargs: None)
    # When there is no initialized DDP, the helper skips BN broadcast.
    model=Tiny()
    image=torch.randn(1,3,16,16)
    target=torch.zeros(1,16,16,dtype=torch.long)
    stats,heads,graph=hcr.validate_hcr_heads(
        model,[(image,target)],torch.device("cpu"),rank=0,epoch=1,progress_every=1)
    text=capsys.readouterr().out
    for stage in ("gpu_synchronize_begin","gpu_synchronize_end",
                  "bn_buffer_sync_begin","bn_buffer_sync_end",
                  "loader_iter_begin","loader_iter_end",
                  "val_fetch_begin","val_fetch_end","val_forward_begin",
                  "val_forward_end","val_allreduce_begin","val_allreduce_end",
                  "val_complete"):
        assert "phase="+stage in text
    assert stats is not heads["final"]


class Tiny(torch.nn.Module):
    def forward(self,x):
        n,_,h,w=x.shape
        out=torch.zeros((n,7,h,w),dtype=x.dtype)
        out[:,0]=3.
        return {"flat_logits":out,"hier_probs":out.softmax(1),
                "mixed_logits":out,"logits":out,
                "risk_logits":torch.zeros(n,1,h//8,w//8)}


def test_validation_defaults_to_main_process_not_fork():
    code=(Path(__file__).resolve().parents[1]/"scripts"/"train_loveda_ddp_hcr.py").read_text()
    tree=ast.parse(code)
    assert "p.add_argument(\"--val-workers\", type=int, default=0" in code
    assert "val_dl = make_loader(val_ds, 1, a.val_workers, val_sampler" in code
    assert 'faulthandler.enable(file=fatal_log, all_threads=True)' in code
