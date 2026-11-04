#!/usr/bin/env python3
"""Count class pixels only from ORIGINAL LoveDA Train masks (0/255 ignored)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.loveda_fast import ROOT, NAMES, find_pairs
from scripts.loveda_class_weights import load_or_count_train, make_class_weights

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=ROOT)
    p.add_argument("--output",default="/kaggle/working/loveda_train_class_counts.json")
    p.add_argument("--alpha",type=float,default=0.5)
    a=p.parse_args()
    record=load_or_count_train(find_pairs(Path(a.root),"train"),a.root,a.output)
    weights=make_class_weights(record["counts"],a.alpha)
    for name,count,freq,weight in zip(NAMES,record["counts"],record["freq"],weights):
        print(f"{name:12s} pixels={count:12,d} freq={freq:.5f} weight={weight:.5f}")
    print(f"[SAVED] {a.output}")

if __name__=="__main__":
    main()
