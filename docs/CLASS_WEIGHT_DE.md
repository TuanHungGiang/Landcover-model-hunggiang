# RSF-HCRNet: Train-derived weights and budgeted DE search

This is an experimental optimizer, not a guaranteed improvement. Original LoveDA Train masks have labels 1..7 (0/255 ignored); internal ids are 0..6.

## Run on Kaggle (2xT4)

1. Update and run smoke tests:

```python
%cd /kaggle/working/Landcover-model-hunggiang
!git pull origin main
!python -m pytest -q tests/test_loveda_weight_search.py tests/test_hcr_model.py
```

2. Scan ALL original Train labels once (does not touch Validation/Holdout):

```python
!python scripts/count_loveda_train.py \
  --output /kaggle/working/loveda_train_class_counts.json --alpha 0.5
```

3. Check the DE plan WITHOUT launching GPU training:

```python
!python scripts/optimize_hcr_de.py \
  --counts /kaggle/working/loveda_train_class_counts.json \
  --epochs 20 --population 4 --generations 1 --dry-run
```

4. Run (4 initial + 4 trial candidates = 8 fresh 20-epoch runs; expensive):

```python
!PYTHONUNBUFFERED=1 python scripts/optimize_hcr_de.py \
  --counts /kaggle/working/loveda_train_class_counts.json \
  --size 1024 --batch 8 --accum 2 --workers 2 \
  --epochs 20 --population 4 --generations 1 \
  --output /kaggle/working/hcr_de_20epoch \
  2>&1 | tee /kaggle/working/hcr_de_search.log
```

Each trial stores its own train.log and best.pt; the search writes de_best.json.
Only selected Val 200 controls candidate selection; holdout is disabled.

For a **single fast experiment** before DE, train with `--class-weight-alpha 0.5` and
`--class-counts-json /kaggle/working/loveda_train_class_counts.json` along
with the existing training arguments (use a new --output directory).

### Important research considerations

- Weighted CE is applied consistently to final, flat, hierarchical heads;
  Dice loss is unchanged. Risk remains an error-probability supervision task.
- Train mask statistics count complete original images; stochastic focus crops
  change effective mini-batch frequencies. An alpha is therefore only a prior.
- Every candidate is trained from scratch; no untrained class-weights at inference.
- DE tunes (alpha, hierarchy_weight, flat_weight, risk_weight), not graph values.
- 20-epoch candidate rankings might not match 100-epoch performance and the
  group-warmup/ramp schedule differs from the 100-epoch baseline. This is a
  screening experiment, **not** a matched-length model comparison.
- On Kaggle sessions, 8x20 epochs may exceed the GPU quota/session duration.
  Run `--generations 0` first (4 trials) or train a single weighted model.
- The search is not robust to interrupted partial trials; choose a NEW output
  directory on restart rather than silently reusing a partial run.
- Avoid searching on holdout; it has already been inspected in prior work.
