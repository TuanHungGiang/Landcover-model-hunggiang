#!/usr/bin/env python3
"""Restart RSF-HCRNet after recoverable native SIGSEGV without losing saved epochs.

For an EXISTING run, load the training recipe from its last.pt, launch
torchrun as a child process, and retry only on an observed native SIGSEGV.
This cannot prevent CUDA/driver crashes or guarantee training completion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


NATIVE_SEGV = re.compile(
    r"(?:SIGSEGV|Segmentation fault|exitcode:\s*-11|exitcode\s*=\s*-11)",
    flags=re.IGNORECASE,
)
OMIT_KEYS = frozenset({"resume", "eval_only", "init_from", "output"})
SAFE_OVERRIDES = {
    "workers": 0,
    "val_workers": 0,
    "save_last_every": 1,
    "heartbeat_every": 5,
    "log_every": 25,
    "no_calibrate": True,
    "evaluate_holdout": False,
}


def read_complete_checkpoint(path: Path) -> dict:
    """Reject partial/incompatible snapshots before resuming anything."""
    if torch is None:
        raise RuntimeError("PyTorch is required to verify checkpoint")
    if not path.is_file():
        raise FileNotFoundError(f"Missing full optimizer checkpoint: {path}")
    # CHECKPOINTS MUST BE TRUSTED. weights_only protects against most arbitrary
    # pickle objects, but this is not a general untrusted-file validator.
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = ("model", "optimizer", "scheduler", "scaler",
                "epoch", "training_args", "training_recipe_version")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Checkpoint missing {missing}: {path}")
    if state["training_recipe_version"] != "hcr_v2":
        raise ValueError("Expected HCR v2 full resume checkpoint")
    if not isinstance(state["epoch"], int) or state["epoch"] < 0:
        raise ValueError("Invalid checkpoint epoch")
    if not isinstance(state["training_args"], dict):
        raise ValueError("Missing checkpoint training recipe")
    return state


def args_to_cli(saved: dict, run_dir: Path) -> list[str]:
    """Serialize trainer arguments losslessly, with only I/O/safety overrides."""
    config = dict(saved)
    config.update(SAFE_OVERRIDES)
    config["output"] = str(run_dir.resolve())
    cli = []
    for key, value in config.items():
        if key in OMIT_KEYS or value is None:
            continue
        if key == "evaluate_holdout":
            cli.append("--evaluate-holdout" if value else "--no-evaluate-holdout")
        elif isinstance(value, bool):
            if value:
                cli.append("--" + key.replace("_", "-"))
        else:
            cli.extend(["--" + key.replace("_", "-"), str(value)])
    cli.extend(["--output", str(run_dir.resolve()),
                "--resume", str((run_dir/"last.pt").resolve())])
    return cli


def command_for_checkpoint(state: dict, run_dir: Path) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=2",
            "scripts/train_loveda_ddp_hcr.py",
            *args_to_cli(state["training_args"],run_dir)]


def stream_one_attempt(cmd, logfile: Path, env: dict) -> tuple[int, bool]:
    crashed_native = False
    with logfile.open("w", encoding="utf-8", buffering=1) as fp:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        try:
            for line in proc.stdout:
                fp.write(line)
                print(line, end="", flush=True)
                crashed_native |= bool(NATIVE_SEGV.search(line))
            status = proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
    return status, crashed_native


def run(run_dir: Path, *, max_restarts=2, log_dir: Path | None = None,
        retry_pause=3.0, dry_run=False) -> int:
    if max_restarts < 0:
        raise ValueError("max-restarts must be >= 0")
    run_dir = Path(run_dir).resolve()
    log_dir = (Path(log_dir).resolve() if log_dir is not None else
               run_dir/"robust_launcher_logs")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Before touching CUDA, validate the most recent complete optimizer state.
    previous = read_complete_checkpoint(run_dir/"last.pt")
    print(f"[ROBUST] recovered_checkpoint_epoch={previous['epoch']+1} "
          f"best_mIoU={previous.get('best_miou','?')}"
          f" restart_budget={max_restarts}",flush=True)
    del previous
    # The saved recipe is authoritative; do not infer hyperparameters from
    # an args.json overwritten by later --resume attempts.
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1",PYTHONFAULTHANDLER="1",
               OMP_NUM_THREADS="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               TORCH_NCCL_ASYNC_ERROR_HANDLING="1")

    for attempt in range(max_restarts+1):
        state = read_complete_checkpoint(run_dir/"last.pt")
        saved_epoch = state["epoch"]+1
        cmd = command_for_checkpoint(state,run_dir)
        del state
        print(f"[ROBUST] attempt={attempt+1}/{max_restarts+1} "
              f"resume_after_epoch={saved_epoch} workers=0 "
              f"save_last_every=1",flush=True)
        if dry_run:
            print("[ROBUST] dry-run command: "+" ".join(cmd),flush=True)
            return 0
        logfile = log_dir/f"attempt_{attempt:02d}.log"
        print(f"[ROBUST] log={logfile}",flush=True)
        status, native = stream_one_attempt(cmd,logfile,env)
        if status == 0:
            print(f"[ROBUST] completed successfully; log={logfile}",flush=True)
            return 0

        # Never restart after Python validation, configuration, checkpoint,
        # OOM, or other errors: they will not be resolved by rerunning.
        if not native:
            print(f"[ROBUST-STOP] non-SIGSEGV failure exit={status}; "
                  f"inspect {logfile} and fatal_stacks_rank*.log",flush=True)
            return status if status>0 else 1
        if attempt>=max_restarts:
            print(f"[ROBUST-STOP] SIGSEGV repeated {attempt+1} times; "
                  "checkpoint preserved; inspect native stack trace",flush=True)
            return status if status>0 else 1
        # Native failure must not destroy the latest atomic last.pt. Only
        # proceed if it passes complete checkpoint validation.
        restored = read_complete_checkpoint(run_dir/"last.pt")
        new_epoch = restored["epoch"]+1
        del restored
        print(f"[ROBUST-RESTART] native SIGSEGV exit={status} "
              f"checkpoint_epoch={new_epoch} "
              f"lost_at_most_one_epoch_since_last_save=True",flush=True)
        time.sleep(retry_pause)
    return 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--from-run",required=True,
                   help="EXISTING run folder containing full last.pt")
    p.add_argument("--max-restarts",type=int,default=2)
    p.add_argument("--log-dir",default=None)
    p.add_argument("--retry-pause",type=float,default=3.0)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    raise SystemExit(run(Path(a.from_run),max_restarts=a.max_restarts,
                         log_dir=Path(a.log_dir) if a.log_dir else None,
                         retry_pause=a.retry_pause,dry_run=a.dry_run))


if __name__=="__main__":
    main()
