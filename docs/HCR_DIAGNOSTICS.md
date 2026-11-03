# Diagnose Hierarchical Head and Graph Refinement

Run on the ORIGINAL selected 200 LoveDA validation images, using the checkpoint saved by the completed HCR run. No training and no holdout access.

Example Kaggle cell:

    %cd /kaggle/working/Landcover-model-hunggiang
    !git pull origin main
    !python scripts/diagnose_hcr.py \
        --run-dir /kaggle/working/rsf_hcrnet_batch8_checkpointing \
        --device cuda --workers 2 --progress-every 20

Output under run-dir/diagnostics_selected200:
- hcr_diagnostic.md: brief interpretation and per-class mIoU
- hcr_diagnostic.json: all four confusion matrices, paired transitions, risk statistics, Rural and Urban
- hcr_tile_metrics.csv: tiles ranked by graph damage or correction

Four stages are from the SAME checkpoint: flat, hierarchy alone, mixed before graph, final.
Paired pixels reveal how many errors were fixed versus created by each stage.
Risk is evaluated at native 1/8 grid against the flat-head wrong-pixel fraction.
Approx. node AUROC/AUPRC uses >=50% wrong labeled pixels per node as positive.
No causal attribution to training-time modules is possible without retraining.
Optional --limit-tiles 3 is for smoke testing only, not valid comparison.
Never repeatedly tune model against untouched holdout.
