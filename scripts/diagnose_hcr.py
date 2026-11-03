#!/usr/bin/env python3
"""Diagnose hierarchy, graph, and risk at ONE HCR checkpoint; selected Val only."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.loveda_fast import ROOT, NAMES, FastLoveDA, find_pairs, confusion_metrics
from scripts.val200_calibration import select_validation_200
from sdr_mambanet.hcr_model import HCRConfig, HCRSDRNet

STAGES = ("flat", "hierarchy", "mixed", "final")
TRANSITIONS = (("flat", "mixed"), ("mixed", "final"), ("flat", "final"))
BINS = 20
IGNORE = 255


def parse_args():
    p = argparse.ArgumentParser(description="HCR post-hoc diagnosis on original Val 200")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-tiles", type=int, default=0, help="Debug only: 0=all")
    p.add_argument("--progress-every", type=int, default=20)
    return p.parse_args()


def predictions_from_output(out, hierarchy_mix):
    """Reconstruct stage probabilities exactly as HCRSDRNet.forward does."""
    flat_prob = out["flat_logits"].float().softmax(1)
    hier_prob = F.interpolate(out["hier_probs"].float(), flat_prob.shape[-2:],
                              mode="bilinear", align_corners=False)
    # v2 persists the actual pre-graph mixed distribution; v1 reconstructs it.
    mixed_prob = (out["mixed_logits"].float().exp() if "mixed_logits" in out else
                  (1-hierarchy_mix)*flat_prob + hierarchy_mix*hier_prob)
    return {
        "flat": flat_prob.argmax(1),
        "hierarchy": hier_prob.argmax(1),
        "mixed": mixed_prob.argmax(1),
        "final": out["logits"].argmax(1),
    }


def add_cm(cm, pred, gt, classes=7):
    valid = gt != IGNORE
    code = torch.where(valid, gt.long()*classes + pred.long(), classes*classes)
    cm.add_(torch.bincount(code.flatten(), minlength=classes*classes+1)[:classes*classes]
            .reshape(classes, classes))


def compare_pixel_labels(gt, before, after, classes=7):
    valid = gt != IGNORE
    a = before == gt
    b = after == gt
    corrected = valid & ~a & b
    damaged = valid & a & ~b
    changed = valid & (before != after)
    return {
        "valid_pixels": int(valid.sum()),
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "damaged": int(damaged.sum()),
        "wrong_to_different_wrong": int((changed & ~a & ~b).sum()),
        "corrected_per_class": torch.bincount(gt[corrected], minlength=classes).cpu().tolist(),
        "damaged_per_class": torch.bincount(gt[damaged], minlength=classes).cpu().tolist(),
    }


def add_transitions(acc, item):
    for k in ("valid_pixels", "changed", "corrected", "damaged",
              "wrong_to_different_wrong"):
        acc[k] += item[k]
    for k in ("corrected_per_class", "damaged_per_class"):
        acc[k] = [x+y for x,y in zip(acc[k], item[k])]


def empty_transition(classes=7):
    return dict(valid_pixels=0, changed=0, corrected=0, damaged=0,
                wrong_to_different_wrong=0,
                corrected_per_class=[0]*classes, damaged_per_class=[0]*classes)


def complete_transition(d):
    n = max(d["valid_pixels"], 1)
    return {**d, "changed_rate": d["changed"]/n,
            "corrected_rate": d["corrected"]/n,
            "damaged_rate": d["damaged"]/n,
            "net_correct": d["corrected"] - d["damaged"],
            "corrected_over_damaged": (
                d["corrected"]/d["damaged"] if d["damaged"] else None)}


class RiskStats:
    """Native 1/8 error-fraction labels match the original supervised risk loss."""

    def __init__(self, bins=BINS):
        self.bins = bins
        self.count = np.zeros(bins, np.int64)
        self.p_sum = np.zeros(bins, np.float64)
        self.y_sum = np.zeros(bins, np.float64)
        self.pos = np.zeros(bins, np.int64)
        self.neg = np.zeros(bins, np.int64)
        self.n = 0
        self.sums = np.zeros(6, np.float64)  # p,y,p²,y²,p*y,(p-y)²
        self.corrected_n = self.damaged_n = 0
        self.corrected_r = self.damaged_r = 0.

    @torch.no_grad()
    def add(self, risk_logits, flat, gt, mixed, final):
        valid = (gt!=IGNORE)
        sz = risk_logits.shape[-2:]
        coverage = F.adaptive_avg_pool2d(valid[:,None].float(), sz)
        mistakes = F.adaptive_avg_pool2d(((flat!=gt)&valid)[:,None].float(), sz)
        wrong_fraction = mistakes / coverage.clamp_min(1e-6)
        keep = coverage>=0.5
        r = risk_logits.float().sigmoid()[keep]
        y = wrong_fraction[keep]
        if r.numel():
            self.n += r.numel()
            vals = (r, y, r*r, y*y, r*y, (r-y).square())
            self.sums += np.asarray([v.sum().item() for v in vals])
            idx = (r.clamp(0, 1-1e-7)*self.bins).long()
            self.count += torch.bincount(idx, minlength=self.bins).cpu().numpy()
            self.p_sum += torch.bincount(idx, weights=r.double(), minlength=self.bins).cpu().numpy()
            self.y_sum += torch.bincount(idx, weights=y.double(), minlength=self.bins).cpu().numpy()
            positive = y>=0.5
            self.pos += torch.bincount(idx[positive], minlength=self.bins).cpu().numpy()
            self.neg += torch.bincount(idx[~positive], minlength=self.bins).cpu().numpy()
        up = F.interpolate(risk_logits.float().sigmoid(), gt.shape[-2:],
                           mode="bilinear", align_corners=False)[:,0]
        fix = valid & (mixed!=gt) & (final==gt)
        harm = valid & (mixed==gt) & (final!=gt)
        self.corrected_n += int(fix.sum())
        self.damaged_n += int(harm.sum())
        self.corrected_r += float(up[fix].sum())
        self.damaged_r += float(up[harm].sum())

    def report(self):
        n = max(self.n, 1)
        mp,my,mpp,myy,mpy,mse = self.sums/n
        vr,vw = max(mpp-mp*mp,0.),max(myy-my*my,0.)
        correlation = ((mpy-mp*my)/np.sqrt(vr*vw) if vr*vw>0 else None)
        tp=self.pos[::-1].cumsum()
        fp=self.neg[::-1].cumsum()
        auroc=auprc=None
        if tp.size and tp[-1]>0 and fp[-1]>0:
            tpr=np.r_[0,tp/tp[-1]]
            fpr=np.r_[0,fp/fp[-1]]
            auroc=float(np.trapezoid(tpr,fpr))
            precision=tp/(tp+fp).clip(min=1)
            recall=tp/tp[-1]
            auprc=float(np.sum(precision*np.diff(np.r_[0,recall])))
        buckets=[
            {"risk_range": [i/self.bins,(i+1)/self.bins],
             "nodes":int(self.count[i]),
             "predicted_risk":self.p_sum[i]/max(int(self.count[i]),1),
             "observed_wrong_fraction":self.y_sum[i]/max(int(self.count[i]),1)}
            for i in range(self.bins)]
        return {
            "native_resolution_nodes":self.n,
            "mean_predicted_risk":float(mp), "mean_observed_wrong_fraction":float(my),
            "brier_error_fraction":float(mse), "risk_error_correlation":(
                float(correlation) if correlation is not None else None),
            "auroc_approx":auroc,"auprc_approx":auprc,
            "positive_node_definition":"At least 50% of labeled 1/8 pixels wrong",
            "mean_risk_on_graph_corrections":self.corrected_r/max(self.corrected_n,1),
            "mean_risk_on_graph_damage":self.damaged_r/max(self.damaged_n,1),
            "buckets":buckets,
        }


def top_confusions(cm, limit=15):
    cm=np.asarray(cm, np.int64)
    denom=cm.sum(1).clip(min=1)
    pairs=[
        {"gt":NAMES[a], "pred":NAMES[b],
         "count":int(cm[a,b]),"fraction_of_gt":float(cm[a,b]/denom[a])}
        for a in range(7) for b in range(7) if a!=b
    ]
    return sorted(pairs,key=lambda z:z["fraction_of_gt"],reverse=True)[:limit]


def make_markdown(report):
    stages=report["stages"]
    tr=report["transitions"]
    risk=report["risk"]
    s=["# HCR diagnosis — same checkpoint, selected Val only","",
       "This decomposes inference, NOT a causal retraining ablation.",
       "","## Four outputs","",
       "| Flat | Hierarchy | Mixed before graph | Final after graph |",
       "|---:|---:|---:|---:|",
       "|"+"|".join(f"{stages[k]['miou']:.4f}" for k in STAGES)+"|",
       "","## Pixel-level stage transitions","",
       "| Transition | Wrong→Correct | Correct→Wrong | Wrong→Other Wrong | Net | Changed (%) |",
       "|---|---:|---:|---:|---:|---:|"]
    for a,b in TRANSITIONS:
        v=tr[f"{a}_to_{b}"]
        s.append(f"| {a}→{b} | {v['corrected']:,} | {v['damaged']:,} | "
                 f"{v['wrong_to_different_wrong']:,} | {v['net_correct']:+,} | "
                 f"{100*v['changed_rate']:.3f} |")
    s+=["","## Per-class IoU","",
        "| Class | Flat | Hierarchy | Mixed | Final | Graph Delta |",
        "|---|---:|---:|---:|---:|---:|"]
    for cls in NAMES:
        x=[stages[k]["iou_per_class"][cls] for k in STAGES]
        s.append(f"| {cls} | "+" | ".join(f"{v:.4f}" for v in x)+
                 f" | {x[3]-x[2]:+.4f} |")
    s+=["","## Risk head","",
        f"- Mean predicted risk: {risk['mean_predicted_risk']:.4f}",
        f"- Actual flat error fraction: {risk['mean_observed_wrong_fraction']:.4f}",
        f"- Risk/error correlation: {risk['risk_error_correlation']}",
        f"- Approx. AUROC: {risk['auroc_approx']}; AUPRC: {risk['auprc_approx']}",
        f"- Mean risk where graph corrected labels: {risk['mean_risk_on_graph_corrections']:.4f}",
        f"- Mean risk where graph damaged labels: {risk['mean_risk_on_graph_damage']:.4f}",
        "","## Possible bottlenecks",""]
    hd=stages["mixed"]["miou"]-stages["flat"]["miou"]
    gd=stages["final"]["miou"]-stages["mixed"]["miou"]
    s.append(f"- Hierarchy mix: {hd:+.4f} mIoU vs flat. "
             "If negative, inspect confusion grouping and probability mixing.")
    s.append(f"- Graph: {gd:+.4f} mIoU vs mixed. "
             "Check corrected/damaged by class and top damaging images.")
    if tr["mixed_to_final"]["changed"]==0:
        s.append("- Graph changed ZERO hard labels. Inspect risk, correction magnitude and eta.")
    if risk["risk_error_correlation"] is not None and risk["risk_error_correlation"]<.2:
        s.append("- Risk scores correlate weakly with flat prediction errors.")
    s+=["","## Dataset/evaluation safeguards",
        "- The original selected validation manifest is reused.",
        "- No holdout data are opened.",
        "- Raw comparisons use the same uncalibrated decision rule.",
        "- Flat vs final from jointly trained weights does not isolate training-time causality."]
    return "\n".join(s)+"\n"


@torch.inference_mode()
def main():
    a=parse_args()
    if a.workers<0 or a.limit_tiles<0 or a.progress_every<1:
        raise ValueError("Invalid diagnostic flags")
    run=Path(a.run_dir)
    train_args=json.loads((run/"args.json").read_text())
    count=int(train_args["val_count"])
    seed=int(train_args["seed"])
    size=int(train_args["size"])
    manifest=run/("val_split_200.json" if count==200 else f"val_split_{count}.json")
    if not manifest.is_file():
        raise FileNotFoundError("Original validation split manifest required: "+str(manifest))
    ckpt_path=Path(a.checkpoint) if a.checkpoint else run/"best.pt"
    ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=True)
    conf=dict(ckpt["model_config"])
    if conf.get("architecture_version") not in ("r34_layer4_bi_hcr_v1", "r34_layer4_bi_hcr_v2"):
        raise ValueError("Expected HCR v1 or v2 checkpoint")
    conf.update(pretrained_semantic=False,semantic_weights_path=None,memory_saver=False)
    cfg=HCRConfig(**conf)
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; specify --device cpu")
    model=HCRSDRNet(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"],strict=True)
    root=Path(a.root)
    selected,_=select_validation_200(find_pairs(root,"val"),root,run,
                                      seed=seed,count=count,resume=True)
    ds=FastLoveDA(selected,size,train=False,cache_size=2)
    dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=a.workers,
                  pin_memory=device.type=="cuda")
    limit=min(a.limit_tiles,len(dl)) if a.limit_tiles else len(dl)
    cm={s:torch.zeros(7,7,dtype=torch.int64,device=device) for s in STAGES}
    cm_domain={d:{s:torch.zeros_like(cm[s]) for s in STAGES}
               for d in ("Rural","Urban")}
    transitions={f"{x}_to_{y}":empty_transition() for x,y in TRANSITIONS}
    risk=RiskStats()
    rows=[]
    for index,(image,gt) in enumerate(dl):
        if index>=limit:
            break
        pair_id,x,y=ds.tiles[index]
        path,_,domain=ds.pairs[pair_id]
        image=image.to(device,non_blocking=True)
        gt=gt.to(device,non_blocking=True)
        with torch.autocast("cuda",enabled=device.type=="cuda"):
            out=model(image)
        preds=predictions_from_output(out,cfg.hierarchy_mix)
        for stage in STAGES:
            add_cm(cm[stage],preds[stage],gt)
            add_cm(cm_domain[domain][stage],preds[stage],gt)
        row={"image":str(path.relative_to(root)),
             "domain":domain,"tile_x":x,"tile_y":y}
        for old,new in TRANSITIONS:
            tag=f"{old}_to_{new}"
            update=compare_pixel_labels(gt,preds[old],preds[new])
            add_transitions(transitions[tag],update)
            row[tag+"_fixed"]=update["corrected"]
            row[tag+"_broken"]=update["damaged"]
            row[tag+"_net"]=update["corrected"]-update["damaged"]
        risk_source = preds["mixed"] if cfg.architecture_version.endswith("v2") else preds["flat"]
        risk.add(out["risk_logits"],risk_source,gt,preds["mixed"],preds["final"])
        rows.append(row)
        if (index+1)%a.progress_every==0 or index+1==limit:
            print(f"[HCR-DIAG] tiles {index+1}/{limit}",flush=True)
    metrics={s:confusion_metrics(cm[s]) for s in STAGES}
    groups=defaultdict(list)
    group_by_class=model.class_groups.cpu().tolist()
    for cls,gid in enumerate(group_by_class):
        groups[str(gid)].append(NAMES[cls])
    summary={
        "checkpoint":str(ckpt_path),"checkpoint_epoch":int(ckpt["epoch"])+1,
        "tile_size":size,"selected_images":len(selected),"evaluated_tiles":limit,
        "complete_selected_validation":limit==len(ds),"manifest":str(manifest),
        "split":"selected Val ONLY, no holdout",
        "class_groups":dict(groups),"class_to_group":group_by_class,
        "training_confusion":model.class_train_confusion.cpu().tolist(),
        "stages":metrics,"domains":{
            d:{s:confusion_metrics(cm_domain[d][s]) for s in STAGES}
            for d in cm_domain},
        "transitions":{key:complete_transition(t) for key,t in transitions.items()},
        "risk":risk.report(),
        "top_confusions":{s:top_confusions(cm[s].cpu()) for s in STAGES},
        "worst_graph_tiles":sorted(rows,key=lambda r:r["mixed_to_final_net"])[:15],
        "best_graph_tiles":sorted(rows,key=lambda r:r["mixed_to_final_net"],reverse=True)[:15],
    }
    outdir=Path(a.out_dir) if a.out_dir else run/"diagnostics_selected200"
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"hcr_diagnostic.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (outdir/"hcr_diagnostic.md").write_text(make_markdown(summary))
    if rows:
        with (outdir/"hcr_tile_metrics.csv").open("w",newline="",encoding="utf-8") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("[mIoU] "+" ".join(f"{s}={metrics[s]['miou']:.4f}" for s in STAGES))
    v=summary["transitions"]["mixed_to_final"]
    print(f"[GRAPH] corrected={v['corrected']} damaged={v['damaged']} net={v['net_correct']:+d}")
    print("[RISK] "+str({k:summary["risk"][k] for k in
                         ("auroc_approx","auprc_approx","risk_error_correlation")}))
    print("[SAVED] "+str(outdir))


if __name__=="__main__":
    main()
