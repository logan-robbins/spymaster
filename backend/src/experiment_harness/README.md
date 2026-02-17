# Experiment Harness

Config-driven offline experiment system for VP research.

This harness runs immutable/cached dataset experiments without changing the live runtime path. It supports:
- Statistical and ML signal runs
- Parameter sweeps and ablations
- Grid-variant regeneration
- MLflow tracking (canonical) with optional W&B mirroring

## Canonical entrypoint

Run from `backend/`:

```bash
uv run python -m src.experiment_harness.cli --help
```

## Launch experiments

### 1) List what is available

```bash
uv run python -m src.experiment_harness.cli list-signals
uv run python -m src.experiment_harness.cli list-datasets
```

### 2) Run one config

```bash
uv run python -m src.experiment_harness.cli run lake/research/vp_harness/configs/legacy/legacy_ads.yaml
```

### 3) Run full legacy suite (all historical agents)

```bash
uv run python -m src.experiment_harness.cli run lake/research/vp_harness/configs/legacy/legacy_full_suite.yaml
```

### 4) Compare best runs

```bash
uv run python -m src.experiment_harness.cli compare --dataset-id mnqh6_20260206_0925_1025 --min-signals 5
```

### 5) Optional online simulation

```bash
uv run python -m src.experiment_harness.cli online-sim --signal ads --dataset-id mnqh6_20260206_0925_1025 --bin-budget-ms 100
```

## Config locations

- General templates: `backend/lake/research/vp_harness/configs/`
- Legacy-mapped runs: `backend/lake/research/vp_harness/configs/legacy/`

Legacy-mapped configs:
- `legacy_ads.yaml`
- `legacy_spg.yaml`
- `legacy_erd.yaml`
- `legacy_pfp.yaml`
- `legacy_jad.yaml`
- `legacy_iirc.yaml`
- `legacy_svm_sp.yaml`
- `legacy_gbm_mf.yaml`
- `legacy_knn_cl.yaml`
- `legacy_lsvm_der.yaml`
- `legacy_xgb_snap.yaml`
- `legacy_pca_ad.yaml`
- `legacy_msd.yaml`
- `legacy_full_suite.yaml`

## Update workflow (for LLMs)

1. Duplicate the closest YAML config in `lake/research/vp_harness/configs/`.
2. Change only config fields; do not hardcode experiment params in Python.
3. Keep `datasets` pointed at immutable/gold dataset IDs unless explicitly running grid variants.
4. Add sweep axes under `sweep.per_signal.<signal_name>`.
5. Set `tracking.experiment_name` and tags for clear grouping.
6. Run config via CLI and validate with `compare`.

Fail-fast behavior:
- Unknown signal sweep parameters raise an error.
- Missing dataset/config files raise an error.

## Reporting workflow (for LLMs)

After a run, report:
1. Config path and git commit SHA.
2. Dataset IDs and signal list.
3. Sweep axes changed.
4. Top runs by TP rate and mean PnL (`compare` output).
5. Run IDs for reproducibility.
6. MLflow experiment name and tracking URI.

## Tracking

MLflow is the canonical store.

Config block:

```yaml
tracking:
  backend: mlflow
  experiment_name: vp/legacy/ads
  run_name_prefix: legacy_ads
  tags:
    stage: legacy
```

Optional W&B mirror:

```yaml
tracking:
  backend: mlflow
  wandb_mirror: true
  wandb_project: your-project
  wandb_entity: your-entity
```

## Notes on legacy parity

- Statistical signals with distribution-driven thresholds (`spg`, `pfp`, `iirc`, `msd`) publish adaptive thresholds in metadata and the runner uses them automatically.
- `erd` supports `variant: a|b` for direct ablation.
- `msd` is implemented as the legacy results-compatible spatial-vacuum signal (`variant=weighted` matches legacy `results.json` behavior).
