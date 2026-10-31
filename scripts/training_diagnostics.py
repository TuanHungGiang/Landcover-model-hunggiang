"""Low-overhead JSONL diagnostics for LoveDA DDP training.

GPU timing is CUDA-event based and sampled; event synchronization only happens
on profiled optimizer steps. Timings are diagnostics, not a CUDA-profiler trace.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import torch

CLASS_NAMES = ("background", "building", "road", "water", "barren", "forest", "agriculture")
STAGES = (
    ("detail", "backbone.detail"),
    ("resnet_stem", "backbone.semantic.stem"),
    ("resnet_layer1", "backbone.semantic.layer1"),
    ("resnet_layer2", "backbone.semantic.layer2"),
    ("resnet_layer3", "backbone.semantic.layer3"),
    ("resnet_layer4", "backbone.semantic.layer4"),
    ("context_1_32", "backbone.semantic.context"),
    ("fusion_bi8", "backbone.bi8"),
    ("fusion_cross16", "backbone.cross16"),
    ("fusion_final", "fusion"),
    ("decoder", "decoder"),
)


def _resolve(model, dotted: str):
    for key in dotted.split("."):
        model = getattr(model, key)
    return model


def class_health(confusion, *, epoch: int = 0, low_iou: float = .10):
    """All metrics from the global confusion; no additional GPU forward pass."""
    cm = torch.as_tensor(confusion, dtype=torch.float64)
    total = max(float(cm.sum()), 1.)
    truth = cm.sum(1) / total
    pred = cm.sum(0) / total
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), 0.)
    rows = []
    alerts = []
    for i, name in enumerate(CLASS_NAMES):
        d = {"class": name, "iou": round(float(iou[i]), 5),
             "gt_fraction": round(float(truth[i]), 5),
             "pred_fraction": round(float(pred[i]), 5),
             "present": bool(union[i] > 0)}
        rows.append(d)
        if epoch >= 3 and d["gt_fraction"] >= .01 and d["pred_fraction"] < .001:
            alerts.append(f"class_collapse:{name} gt={d['gt_fraction']:.3f}, pred={d['pred_fraction']:.4f}")
        elif epoch >= 5 and d["gt_fraction"] >= .01 and d["iou"] < low_iou:
            alerts.append(f"low_IoU:{name}={d['iou']:.3f}")
    return rows, alerts


def gate_scales(model):
    """Learned residual scales; near zero means the corresponding path is weak."""
    paths = {
        "S_to_D_1_8": "backbone.bi8.s2d_scale",
        "D_to_S_1_8": "backbone.bi8.d2s_scale",
        "S32_to_S16": "backbone.cross16.deep_scale",
        "D8_to_S16": "backbone.cross16.detail_scale",
        "S16_to_D8": "fusion.semantic_scale",
    }
    result = {}
    with torch.no_grad():
        for key, path in paths.items():
            result[key] = round(float(_resolve(model, path).detach().float().cpu()), 6)
    return result


def selected_grad_norms(model):
    """Call only at profiled synchronization steps, before optimizer.zero_grad()."""
    paths = (
        ("detail", "backbone.detail"),
        ("resnet_layer4", "backbone.semantic.layer4"),
        ("fusion_bi8", "backbone.bi8"),
        ("fusion_cross16", "backbone.cross16"),
        ("fusion_final", "fusion"),
        ("decoder", "decoder"),
    )
    result = {}
    with torch.no_grad():
        for name, path in paths:
            mod = _resolve(model, path)
            grads = [p.grad for p in mod.parameters() if p.grad is not None]
            if grads:
                squared = torch.stack([g.detach().float().square().sum() for g in grads]).sum()
                result[name] = squared.sqrt()
    return result


class StepCudaTimer:
    """Optional sampled GPU timing by events, no synchronize on ordinary steps."""
    def __init__(self, model):
        self.active = False
        self.handles = []
        self._pairs = {}
        self._events = {}
        for name, path in STAGES:
            mod = _resolve(model, path)
            self.handles.append(mod.register_forward_pre_hook(self._start_hook(name)))
            self.handles.append(mod.register_forward_hook(self._end_hook(name)))

    def _start_hook(self, name):
        def hook(module, args):
            if self.active:
                self._pairs[name] = [torch.cuda.Event(enable_timing=True), None]
                self._pairs[name][0].record()
        return hook

    def _end_hook(self, name):
        def hook(module, args, output):
            if self.active and name in self._pairs:
                self._pairs[name][1] = torch.cuda.Event(enable_timing=True)
                self._pairs[name][1].record()
        return hook

    def begin(self):
        self._pairs = {}
        self._events = {}
        self.active = True

    def mark(self, key):
        if self.active:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._events[key] = e

    def finish(self, device):
        self.active = False
        torch.cuda.synchronize(device)
        result = {}
        for name, (start, end) in self._pairs.items():
            if end is not None:
                result[name] = round(start.elapsed_time(end), 3)
        keys = ("copy_begin", "copy_end", "forward_begin", "forward_end",
                "backward_begin", "backward_end", "optimizer_begin", "optimizer_end")
        for label, a, b in (("h2d_ms", "copy_begin", "copy_end"),
                            ("forward_ms", "forward_begin", "forward_end"),
                            ("backward_ms", "backward_begin", "backward_end"),
                            ("optimizer_ms", "optimizer_begin", "optimizer_end")):
            if a in self._events and b in self._events:
                result[label] = round(self._events[a].elapsed_time(self._events[b]), 3)
        return result

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class DiagnosticsWriter:
    def __init__(self, folder: Path, rank: int, *, enabled: bool = True):
        self.rank = rank
        self.enabled = enabled
        self.path = Path(folder) / f"diagnostics_rank{rank}.jsonl"

    def write(self, kind: str, **fields):
        if not self.enabled:
            return
        record = {"time_unix": round(time.time(), 3), "rank": self.rank,
                  "type": kind, **fields}
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")