from pathlib import Path
import json
from torch import nn
import torch
from scripts.training_diagnostics import class_health, gate_scales, DiagnosticsWriter, selected_grad_norms, StepCudaTimer


def test_class_health_collapse():
    cm = torch.zeros(7, 7, dtype=torch.int64)
    cm[0, 0] = 950
    cm[1, 0] = 50
    rows, alerts = class_health(cm, epoch=6)
    assert len(rows) == 7
    assert any(s.startswith("class_collapse:building") for s in alerts)
    assert rows[1]["gt_fraction"] == 0.05
    assert rows[1]["pred_fraction"] == 0.0
    assert not any(s.startswith("class_collapse:road") for s in alerts)


def test_gate_scales_and_selected_gradient_norms():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.bi8 = nn.Module()
            self.backbone.bi8.s2d_scale = nn.Parameter(torch.tensor(0.1))
            self.backbone.bi8.d2s_scale = nn.Parameter(torch.tensor(0.2))
            self.backbone.cross16 = nn.Module()
            self.backbone.cross16.deep_scale = nn.Parameter(torch.tensor(0.3))
            self.backbone.cross16.detail_scale = nn.Parameter(torch.tensor(0.4))
            self.backbone.detail = nn.Linear(2, 2)
            self.backbone.semantic = nn.Module()
            self.backbone.semantic.layer4 = nn.Linear(2, 2)
            self.fusion = nn.Module()
            self.fusion.semantic_scale = nn.Parameter(torch.tensor(0.5))
            self.decoder = nn.Linear(2, 2)
    m=Dummy()
    gates=gate_scales(m)
    assert gates["S_to_D_1_8"] == 0.1
    assert gates["D8_to_S16"] == 0.4
    ((m.decoder(torch.randn(2,2)).sum()) + m.backbone.detail(torch.randn(2,2)).sum()).backward()
    grads=selected_grad_norms(m)
    assert "decoder" in grads and "detail" in grads
    assert grads["decoder"].item() > 0


def test_log_file(tmp_path: Path):
    writer=DiagnosticsWriter(tmp_path,rank=1)
    writer.write("train_step",epoch=2,loss=.52)
    row=json.loads((tmp_path / "diagnostics_rank1.jsonl").read_text().strip())
    assert row["rank"] == 1 and row["epoch"] == 2
    assert row["loss"] == .52
    assert StepCudaTimer is not None