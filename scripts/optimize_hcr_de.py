#!/usr/bin/env python3
"""Budgeted DE hyperparameter exploration with fresh HCR runs on 2 GPUs.

Each trial trains independently, selects only on the saved 200-image Val mIoU,
never evaluates the remaining holdout. No training is launched by importing this.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

BOUNDS=np.array([[0.0,0.75],[0.05,0.40],[0.05,0.40],[0.02,0.25]],dtype=float)
KEYS=("class_weight_alpha","hierarchy_weight","flat_weight","risk_weight")


def propose(population, scores, index, rng):
    """DE/rand/1/bin mutation with bounds, at least one gene crossing."""
    n, dim=population.shape
    if n<4:
        raise ValueError("DE requires at least four candidates")
    others=[k for k in range(n) if k!=index]
    a,b,c=rng.choice(others,3,replace=False)
    mutant=np.clip(population[a]+.7*(population[b]-population[c]),
                   BOUNDS[:,0],BOUNDS[:,1])
    cross=rng.random(dim)<.8
    cross[int(rng.integers(dim))]=True
    return np.where(cross,mutant,population[index])


def command(x, args, run_dir):
    theta=dict(zip(KEYS,map(float,x)))
    return [sys.executable,"-m","torch.distributed.run",
            "--standalone","--nproc_per_node=2",
            "scripts/train_loveda_ddp_hcr.py",
            "--context","conv","--size",str(args.size),
            "--batch",str(args.batch),"--accum",str(args.accum),
            "--workers",str(args.workers),"--val-workers","0",
            "--memory-saver","--epochs",str(args.epochs),
            "--warmup-epochs",str(args.warmup_epochs),
            "--group-warmup-epochs",str(args.group_warmup_epochs),
            "--hierarchy-ramp-epochs",str(args.hierarchy_ramp_epochs),
            "--confusion-min-score","0.02",
            "--confusion-min-reciprocal","0.015",
            "--hierarchy-mix","0.25","--graph-steps","3",
            "--graph-lambda","0.5","--graph-eta","0.5",
            "--lr","0.0002","--backbone-lr","0.00005",
            "--min-lr-ratio","0.1","--augment","strong",
            "--focus-prob","0.5","--focus-classes","4,6",
            "--zoom-min","1.0","--zoom-max","1.35",
            "--early-stop-patience","0","--val-count","200",
            "--seed",str(args.seed),"--log-every",str(args.log_every),
            "--heartbeat-every","0","--profile-every","0",
            "--save-last-every",str(args.epochs),
            "--class-counts-json",str(Path(args.counts).resolve()),
            "--no-evaluate-holdout","--no-calibrate",
            "--output",str(run_dir.resolve()),
            *sum(([f"--{k.replace('_','-')}",f"{v:.6f}"] for k,v in theta.items()),[])]


def run_trial(x,args,trial_id):
    root=Path(args.output).resolve()
    path=root/f"trial_{trial_id:03d}"
    path.mkdir(parents=True,exist_ok=True)
    parameter_path=path/"candidate.json"
    result_path=path/"trial_result.json"
    theta=dict(zip(KEYS,map(float,x)))
    if result_path.is_file():
        obj=json.loads(result_path.read_text())
        if obj["parameters"]!=theta:
            raise RuntimeError("Trial ID reused with different parameters")
        print(f"[REUSE] {path} mIoU={obj['miou']:.4f}",flush=True)
        return float(obj["miou"])
    if parameter_path.is_file():
        raise RuntimeError(f"Interrupted trial {path}; inspect logs and restart in a NEW output directory")
    parameter_path.write_text(json.dumps(theta,indent=2))
    cmd=command(x,args,path)
    (path/"command.txt").write_text(" ".join(cmd)+"\n")
    print(f"[DE-TRIAL] {trial_id:03d} {theta}",flush=True)
    env=dict(os.environ, PYTHONUNBUFFERED="1",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (path/"train.log").open("w",buffering=1) as log:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,bufsize=1,env=env)
        for line in proc.stdout:
            log.write(line)
            if ("Epoch " in line or "[VAL-HCR]" in line or
                "[CHECKPOINT-ERROR]" in line or "Traceback" in line):
                print(f"[TRIAL {trial_id}] {line.rstrip()}",flush=True)
        rc=proc.wait()
    if rc:
        raise RuntimeError(f"Trial {trial_id} failed (exit {rc}); see {path/'train.log'}")
    info=json.loads((path/"best_metrics.json").read_text())
    score=float(info["miou"])
    result_path.write_text(json.dumps({"parameters":theta,"miou":score,
                                       "best_epoch":info.get("epoch")},indent=2))
    print(f"[DE-SCORE] trial={trial_id} mIoU={score:.5f}",flush=True)
    return score


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--counts",required=True,help="Precomputed training counts JSON")
    p.add_argument("--output",default="/kaggle/working/hcr_de_search")
    p.add_argument("--size",type=int,default=1024,choices=[512,1024])
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--accum",type=int,default=2)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--epochs",type=int,default=20,help="Fixed per-trial budget")
    p.add_argument("--warmup-epochs",type=int,default=3)
    p.add_argument("--group-warmup-epochs",type=int,default=5)
    p.add_argument("--hierarchy-ramp-epochs",type=int,default=4)
    p.add_argument("--population",type=int,default=4)
    p.add_argument("--generations",type=int,default=1)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--log-every",type=int,default=100)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.epochs<3 or a.group_warmup_epochs+a.hierarchy_ramp_epochs>a.epochs:
        p.error("Allow enough epochs for hierarchy ramp to finish")
    if a.population<4 or a.generations<0:
        p.error("Population>=4 and generations>=0 required")
    if not Path(a.counts).is_file():
        p.error("Precompute train counts with scripts/count_loveda_train.py")
    rng=np.random.default_rng(a.seed)
    pop=rng.uniform(BOUNDS[:,0],BOUNDS[:,1],size=(a.population,4))
    # Include original loss coefficient baseline (alpha=0) for reference.
    pop[0]=[0.0,0.2,0.2,0.15]
    total=a.population*(a.generations+1)
    print(f"[DE] planned_trials={total} per_trial_epochs={a.epochs} "
          f"size={a.size} batch/GPU={a.batch} holdout=OFF",flush=True)
    if a.dry_run:
        for i,x in enumerate(pop):
            print("[DRY-RUN]",i,dict(zip(KEYS,map(float,x))))
        return
    Path(a.output).mkdir(parents=True,exist_ok=True)
    scores=np.array([run_trial(x,a,i) for i,x in enumerate(pop)])
    n=a.population
    for generation in range(a.generations):
        for i in range(a.population):
            candidate=propose(pop,scores,i,rng)
            s=run_trial(candidate,a,n)
            n+=1
            if s>scores[i]:
                pop[i],scores[i]=candidate,s
        print(f"[DE-GENERATION] {generation+1} best={scores.max():.5f}",flush=True)
    top=int(scores.argmax())
    result={"best_val_miou":float(scores[top]),
            "parameters":dict(zip(KEYS,map(float,pop[top]))),
            "selection":"selected Val 200 only; no holdout",
            "warning":"Short-trial ranking may not transfer to full 100-epoch schedule"}
    out=Path(a.output)/"de_best.json"
    out.write_text(json.dumps(result,indent=2))
    print("[DE-BEST] "+json.dumps(result),flush=True)
    print("[SAVED] "+str(out),flush=True)


if __name__=="__main__":
    main()
