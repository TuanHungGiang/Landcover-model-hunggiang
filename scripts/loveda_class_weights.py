"""Count original LoveDA *Train* masks and create stable CE weights.

Counts use full ORIGINAL mask pixel frequencies (ignore 0/255), not augmented
crops; this is a deliberately reproducible frequency prior, not the stochastic
augmentation distribution. No Validation/Holdout file may be passed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.loveda_fast import NAMES


def train_signature(pairs, root):
    root=Path(root).resolve()
    sha=hashlib.sha256()
    for _, mask, domain in pairs:
        mask=Path(mask).resolve()
        if "train" not in {p.lower() for p in mask.relative_to(root).parts}:
            raise ValueError(f"Not a Train mask: {mask}")
        st=mask.stat()
        sha.update(f"{mask.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}|{domain}\n".encode())
    return sha.hexdigest()


def count_train_masks(pairs,root):
    if not pairs:
        raise ValueError("Train image list is empty")
    signature=train_signature(pairs,root)
    counts=np.zeros(7,dtype=np.int64)
    by_domain={}
    for _,mask_path,domain in pairs:
        with Image.open(mask_path) as img:
            mask=np.asarray(img)
        if mask.ndim!=2 or not np.isin(mask,(0,1,2,3,4,5,6,7,255)).all():
            raise ValueError(f"Invalid LoveDA label mask {mask_path}")
        c=np.bincount((mask[(mask>=1)&(mask<=7)]-1).astype(np.int64),minlength=7)
        counts+=c
        by_domain.setdefault(domain,np.zeros(7,dtype=np.int64))
        by_domain[domain]+=c
    if np.any(counts==0):
        raise ValueError(f"Missing class in Train labels: {np.flatnonzero(counts==0).tolist()}")
    return {
        "source":"Train masks ONLY; raw ids 1..7; ignore 0 and 255",
        "train_images":len(pairs),"signature":signature,
        "counts":counts.tolist(),
        "freq":(counts/counts.sum()).tolist(),
        "by_domain":{d:v.tolist() for d,v in by_domain.items()},
    }


def load_or_count_train(pairs,root,path,allow_compute=True):
    path=Path(path)
    expected=train_signature(pairs,root)
    if path.exists():
        obj=json.loads(path.read_text())
        if obj.get("signature")!=expected or obj.get("train_images")!=len(pairs):
            raise ValueError(f"Cached class counts do not match CURRENT Train: {path}")
        if len(obj.get("counts",[]))!=len(NAMES) or any(v<=0 for v in obj["counts"]):
            raise ValueError("Class counts missing or invalid")
        return obj
    if not allow_compute:
        raise FileNotFoundError(f"Rank0 did not create Train counts: {path}")
    result=count_train_masks(pairs,root)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(result,indent=2))
    temp.replace(path)
    return result


def make_class_weights(counts,alpha: float,low=.5,high=2.5):
    if not 0<=alpha<=1:
        raise ValueError("alpha must be in [0,1]")
    freq=np.asarray(counts,dtype=np.float64)
    if freq.shape!=(7,) or np.any(freq<=0):
        raise ValueError("Expected positive pixel counts for seven classes")
    freq=freq/freq.sum()
    w=np.clip((np.median(freq)/freq)**alpha,low,high)
    # Scales expected CE contribution by unity while retaining relative ratios.
    w=w/np.sum(freq*w)
    return torch.tensor(w,dtype=torch.float32)
