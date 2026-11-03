"""Regression for HCR Val JSON circular-reference crash after epoch 1."""
import json

import torch
from scripts import train_loveda_ddp_hcr as trainer


class DummyModel(torch.nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        flat = torch.zeros((b, 7, h, w), device=x.device)
        flat[:, 0, :, :] = 3.0
        return {
            "flat_logits": flat,
            "mixed_logits": flat,
            "logits": flat,
            "hier_probs": flat.softmax(dim=1),
            "risk_logits": torch.zeros((b, 1, h // 8, w // 8), device=x.device),
        }


def test_epoch1_hcr_validation_serializes_without_cycle(monkeypatch):
    # DDP all_reduce is identity for a single mocked rank.
    monkeypatch.setattr(trainer.dist, "all_reduce", lambda *args, **kwargs: None)
    loader = [(torch.zeros(1, 3, 16, 16),
               torch.zeros(1, 16, 16, dtype=torch.long))]
    stats, heads, graph = trainer.validate_hcr_heads(
        DummyModel(), loader, torch.device("cpu"))
    # This EXACT assignment was producing circular reference in the trainer.
    stats["head_metrics"] = heads
    stats["graph_metrics"] = graph
    assert stats is not heads["final"]
    assert json.loads(json.dumps(stats))["head_metrics"]["final"]["miou"] == 1.0


def test_json_history_with_serialization_and_checkpoint_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer.dist, "all_reduce", lambda *args, **kwargs: None)
    loader = [(torch.zeros(1, 3, 16, 16),
               torch.zeros(1, 16, 16, dtype=torch.long))]
    stats, heads, graph = trainer.validate_hcr_heads(
        DummyModel(), loader, torch.device("cpu"))
    stats.update(head_metrics=heads, graph_metrics=graph,
                 hierarchy_strength=0.0, effective_hierarchy_mix=0.0,
                 epoch=1, train_loss=3.4259)
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps(stats, allow_nan=False) + "\n")
    result = json.loads(history.read_text())
    assert result["epoch"] == 1
    assert result["head_metrics"]["final"]["miou"] == 1.0
