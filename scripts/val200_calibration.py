"""LoveDA validation subset selection and multiclass threshold tuning.

Selection: seeded 100 Rural + 100 Urban from official Val, remaining images excluded.
Calibration: argmax(softmax(logits) / class_threshold), class-0 threshold anchored to 1.
This changes the multiclass *decision rule*, not the probability calibration.
"""
import json
import random
from pathlib import Path
import numpy as np
import torch

NAMES = ("background", "building", "road", "water", "barren", "forest", "agriculture")

def select_validation_200(pairs, root, output, seed=42, count=200, resume=False):
    if count < 2 or count % 2:
        raise ValueError("Use even --val-count, e.g. 200")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / ("val_split_200.json" if count == 200 else f"val_split_{count}.json")
    by_name = {str(p[0].relative_to(root)): p for p in pairs}
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data["seed"] != seed or data["count"] != count:
            raise ValueError("Existing manifest seed/count mismatch; use another --output")
        names = data["selected_images"]
    else:
        if resume:
            raise FileNotFoundError("Cannot resume without the original split manifest")
        rng = random.Random(seed)
        chosen = []
        for scene in ("Rural", "Urban"):
            group = sorted((p for p in pairs if p[2] == scene), key=lambda p: str(p[0]))
            if len(group) < count // 2:
                raise ValueError(f"Not enough {scene} Val images for balanced selection")
            chosen.extend(rng.sample(group, count // 2))
        names = sorted(str(p[0].relative_to(root)) for p in chosen)
        manifest.write_text(json.dumps({
            "seed": seed, "count": count, "rural": count//2, "urban": count//2,
            "selected_images": names
        }, indent=2))
    if len(names) != count or len(set(names)) != count or any(n not in by_name for n in names):
        raise ValueError("Invalid or stale validation split manifest")
    selected = {n for n in names}
    val = [p for p in pairs if str(p[0].relative_to(root)) in selected]
    holdout = [p for p in pairs if str(p[0].relative_to(root)) not in selected]
    if len(val) != count:
        raise ValueError("Validation size mismatch")
    print(f"Selected Val={len(val)} (Rural={sum(p[2]=='Rural' for p in val)}, "
          f"Urban={sum(p[2]=='Urban' for p in val)}); untouched holdout={len(holdout)}",
          flush=True)
    return val, holdout

def confusion_metrics(cm):
    cm = np.asarray(cm, np.int64)
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union != 0)
    return {
        "miou": float(iou.mean()),
        "oa": float(tp.sum()/max(int(cm.sum()),1)),
        "iou_per_class": {k: float(v) for k, v in zip(NAMES, iou)},
        "confusion": cm.tolist(),
    }

def predict_threshold(probs, thresholds):
    return (np.asarray(probs, np.float32)/np.asarray(thresholds, np.float32)).argmax(axis=-1)

def tune_thresholds(probs, gt, classes=7, passes=2):
    probs = np.asarray(probs, np.float32)
    gt = np.asarray(gt, np.int64)
    if probs.ndim != 2 or probs.shape != (gt.size, classes) or gt.size == 0:
        raise ValueError("Invalid calibration input")
    grid = np.array([.55,.65,.75,.85,.95,1.,1.05,1.15,1.30,1.50],np.float32)
    threshold = np.ones(classes,np.float32)
    def score(t):
        pred = predict_threshold(probs, t)
        cm = np.bincount(gt*classes+pred, minlength=classes*classes).reshape(classes,classes)
        return confusion_metrics(cm)["miou"]
    baseline = current = score(threshold)
    for _ in range(passes):
        changed = False
        for c in range(1,classes):  # anchor class 0 to avoid scale ambiguity
            winning = float(threshold[c])
            for candidate in grid:
                candidate_ts = threshold.copy()
                candidate_ts[c] = candidate
                s = score(candidate_ts)
                if s > current + 1e-10:
                    current,winning,changed = s,float(candidate),True
            threshold[c] = winning
        if not changed:
            break
    return threshold, {"sampled_pixels":int(gt.size), "raw_sample_miou":baseline,
                       "tuned_sample_miou":current, "grid":grid.tolist(),
                       "rule":"argmax(softmax(logits)/class_thresholds)"}

@torch.no_grad()
def sample_pixels(model, loader, device, per_tile=2048, seed=42):
    model.eval()
    rng=np.random.default_rng(seed)
    probs_list, labels_list=[],[]
    for x, y in loader:
        with torch.autocast("cuda", enabled=device.type=="cuda"):
            logits=model(x.to(device,non_blocking=True))["logits"]
        probs=logits.float().softmax(1)[0].permute(1,2,0).cpu().numpy().reshape(-1,7)
        labels=y[0].numpy().reshape(-1)
        valid=np.flatnonzero(labels!=255)
        if not len(valid): continue
        take=rng.choice(valid,size=min(per_tile,len(valid)),replace=False)
        probs_list.append(probs[take].astype(np.float16))
        labels_list.append(labels[take].astype(np.uint8))
    if not probs_list: raise RuntimeError("No valid calibration labels")
    return np.concatenate(probs_list),np.concatenate(labels_list)

@torch.no_grad()
def evaluate_thresholds(model, loader, device, thresholds):
    model.eval()
    raw=np.zeros((7,7),np.int64)
    tuned=np.zeros((7,7),np.int64)
    t=torch.tensor(thresholds,device=device,dtype=torch.float32).view(1,7,1,1)
    for x,y in loader:
        with torch.autocast("cuda",enabled=device.type=="cuda"):
            logits=model(x.to(device,non_blocking=True))["logits"]
        prob=logits.float().softmax(1)
        pred0=prob.argmax(1).cpu().numpy().reshape(-1)
        pred1=(prob/t).argmax(1).cpu().numpy().reshape(-1)
        gt=y.numpy().reshape(-1)
        ok=gt!=255
        raw+=np.bincount(gt[ok]*7+pred0[ok],minlength=49).reshape(7,7)
        tuned+=np.bincount(gt[ok]*7+pred1[ok],minlength=49).reshape(7,7)
    return {"argmax":confusion_metrics(raw),"calibrated":confusion_metrics(tuned)}
