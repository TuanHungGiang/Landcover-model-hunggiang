# RSF-HCRNet v2: train from scratch + verbose diagnostics

## Fixes (research hypotheses, not guaranteed accuracy improvements)

- **Reciprocal class confusion**: no longer join two classes merely because one is frequently mistaken for the other. New `--confusion-min-reciprocal 0.015` enforces both directions.
- **Scheduled hierarchy**: train with the existing flat prediction initially, compute groups from train-only confusion after epoch 10, then ramp the probability mix over `--hierarchy-ramp-epochs 6` epochs. This avoids an abrupt prediction switch.
- **Risk supervision**: train error risk on **pre-graph mixed logits**, the distribution the graph will actually correct. v1 trained on Flat Head error, sometimes inconsistent with the mixed predictor.
- **Class-consistent graph**: remove v1's risk-dependent relaxation that could smooth across different class boundaries; use confidence-anchored regularization and class-group agreement.
- **VRAM**: retain v2 activation checkpointing and BatchNorm training statistics unchanged.

Changes are intentionally versioned as `r34_layer4_bi_hcr_v2`. Old v1 checkpoints are not resumable as v2. Start fresh with a **new output directory**.

## Kaggle 2×T4, 1024, batch 8 and ALL logs

```python
%cd /kaggle/working/Landcover-model-hunggiang
!git pull origin main
!git rev-parse --short HEAD

!mkdir -p /kaggle/working/hcr_v2_logs
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=2 scripts/train_loveda_ddp_hcr.py \
    --context conv --size 1024 --batch 8 --accum 2 --workers 2 \
    --memory-saver --epochs 100 --warmup-epochs 5 \
    --group-warmup-epochs 10 --hierarchy-ramp-epochs 6 \
    --confusion-min-score 0.02 --confusion-min-reciprocal 0.015 \
    --hierarchy-mix 0.25 --graph-steps 3 \
    --graph-lambda 0.5 --graph-eta 0.5 \
    --hierarchy-weight 0.2 --flat-weight 0.2 --risk-weight 0.15 \
    --lr 2e-4 --backbone-lr 5e-5 --min-lr-ratio 0.1 \
    --augment strong --focus-prob 0.5 --focus-classes 4,6 \
    --zoom-min 1.0 --zoom-max 1.35 --early-stop-patience 20 \
    --early-stop-min-delta 0.001 --val-count 200 --seed 42 \
    --log-every 1 --heartbeat-every 0 --profile-every 0 \
    --no-evaluate-holdout \
    --output /kaggle/working/rsf_hcrnet_v2_1024_batch8 \
    > /kaggle/working/hcr_v2_logs/console_full.log 2>&1

!tail -n 100 /kaggle/working/hcr_v2_logs/console_full.log
```

For LIVE streaming while saving, replace the redirect line with `2>&1 | tee /kaggle/working/hcr_v2_logs/console_full.log` in a notebook shell with `bash -o pipefail -c` if you need shell pipeline exit-code semantics. `--log-every 1` prints train metrics and HCR diagnostics for each local step on rank0. Rank-specific heartbeat, OOM, early-stop, and end-of-epoch confusion metrics are in the console log. Structured JSONL diagnostics are written under run-dir by DiagnosticsWriter.

The `[HCR-STEP]` line prints each step's hierarchy CE, Flat CE, risk loss, mean risk, mean graph logit correction, percentage of hard labels changed by graph, and hierarchy confidence.

### Post-training diagnosis on selected Val 200

```python
!python scripts/diagnose_hcr.py \
  --run-dir /kaggle/working/rsf_hcrnet_v2_1024_batch8 \
  --device cuda --workers 2 --progress-every 20
```

Review `diagnostics_selected200/hcr_diagnostic.md` and JSON after training to judge whether the new hierarchy and graph improve predictions.

**Important:** In this training run use `--no-evaluate-holdout` to preserve the untouched holdout. An earlier experimental run used it once, but repeated tuning on holdout would bias the final assessment.

**Limitations:** CPU smoke/gradient tests do not prove Kaggle VRAM or mIoU gains. No automatic guarantee of improved performance.
