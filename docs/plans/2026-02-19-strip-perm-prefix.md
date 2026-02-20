# Strip `perm_` Prefix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the meaningless `perm_` prefix from all runtime model parameters, signal names, class names, CLI flags, URL query params, and YAML configs across the entire codebase.

**Architecture:** Pure mechanical rename. `perm_X` becomes `X` everywhere. `PermDerivative*` classes become `Derivative*`. The signal registry name `perm_derivative` becomes `derivative`. No logic changes.

**Tech Stack:** Python (backend), TypeScript (frontend), YAML (configs)

---

## Rename Map

| Old | New |
|-----|-----|
| `perm_runtime_enabled` | `runtime_enabled` |
| `perm_center_exclusion_radius` | `center_exclusion_radius` |
| `perm_spatial_decay_power` | `spatial_decay_power` |
| `perm_zscore_window_bins` | `zscore_window_bins` |
| `perm_zscore_min_periods` | `zscore_min_periods` |
| `perm_tanh_scale` | `tanh_scale` |
| `perm_d1_weight` | `d1_weight` |
| `perm_d2_weight` | `d2_weight` |
| `perm_d3_weight` | `d3_weight` |
| `perm_bull_pressure_weight` | `bull_pressure_weight` |
| `perm_bull_vacuum_weight` | `bull_vacuum_weight` |
| `perm_bear_pressure_weight` | `bear_pressure_weight` |
| `perm_bear_vacuum_weight` | `bear_vacuum_weight` |
| `perm_mixed_weight` | `mixed_weight` |
| `perm_enable_weighted_blend` | `enable_weighted_blend` |
| `perm_microstate_id` | `microstate_id` |
| `perm_state5_code` | `state5_code` |
| `perm_state5_distribution` | `state5_distribution` |
| `perm_micro9_distribution` | `micro9_distribution` |
| `perm_state5_transition_matrix` | `state5_transition_matrix` |
| `perm_state5_labels` | `state5_labels` |
| `perm_taxonomy_version` | `taxonomy_version` |
| `DEFAULT_PERM_*` | `DEFAULT_*` |
| `PermDerivativeRuntimeParams` | `DerivativeRuntimeParams` |
| `PermDerivativeRuntimeOutput` | `DerivativeRuntimeOutput` |
| `PermDerivativeRuntime` | `DerivativeRuntime` |
| `PermDerivativeSignal` | `DerivativeSignal` |
| `perm_derivative` (signal name) | `derivative` |
| `--perm-*` (CLI flags) | `--*` (e.g. `--d1-weight`) |
| `_perm_microstate_id()` | `_microstate_id()` |
| `_perm_state5_code()` | `_state5_code()` |
| `_perm_runtime_params_from_config()` | `_runtime_params_from_config()` |
| `perm_overrides` (local var) | `overrides` |

## File Rename

| Old | New |
|-----|-----|
| `signals/statistical/perm_derivative.py` | `signals/statistical/derivative.py` |
| `tests/test_runtime_perm_model.py` | `tests/test_runtime_model.py` |
| `tests/test_experiment_harness/test_perm_derivative_signal.py` | `tests/test_experiment_harness/test_derivative_signal.py` |

## YAML Config File Renames

These are config files in `lake/research/vp_harness/configs/` that have `perm_derivative` in their name. They reference the signal by name and use `perm_` param names internally.

| Old | New |
|-----|-----|
| `configs/smoke_perm_derivative.yaml` | `configs/smoke_derivative.yaml` |
| `configs/tune_perm_derivative.yaml` | `configs/tune_derivative.yaml` |
| `configs/experiments/sweep_perm_derivative_rr20.yaml` | `configs/experiments/sweep_derivative_rr20.yaml` |
| `configs/serving/perm_derivative_baseline.yaml` | `configs/serving/derivative_baseline.yaml` |

Inside each YAML: rename signal name references from `perm_derivative` to `derivative`, rename any `perm_` param keys, rename `name:` fields.

## Existing Results Parquet

`lake/research/vp_harness/results/runs_meta.parquet` contains columns:
- `perm_state5_distribution_json` → `state5_distribution_json`
- `perm_micro9_distribution_json` → `micro9_distribution_json`
- `perm_state5_transition_matrix_json` → `state5_transition_matrix_json`
- `perm_state5_labels_json` → `state5_labels_json`
- `perm_taxonomy_version` → `taxonomy_version`

This is regenerable data. Delete `runs_meta.parquet` and `runs.parquet` and regenerate from experiments.

---

## Tasks

### Task 1: instrument.yaml

Strip `perm_` prefix from all keys.

**Files:**
- Modify: `backend/src/vacuum_pressure/instrument.yaml`

**Step 1:** Rename all `perm_*` keys to remove prefix. `perm_runtime_enabled` → `runtime_enabled`, etc.

**Step 2:** Commit.

---

### Task 2: config.py (VPRuntimeConfig + defaults + parser + validation)

**Files:**
- Modify: `backend/src/vacuum_pressure/config.py`

**Step 1:** Rename all `DEFAULT_PERM_*` constants to `DEFAULT_*`.

**Step 2:** Rename all `perm_*` fields on VPRuntimeConfig dataclass to drop prefix.

**Step 3:** Rename all `perm_*` keys in the `_build_fields()` parser function.

**Step 4:** Rename all `perm_*` references in validation function.

**Step 5:** Rename all `perm_*` keys in `to_dict()` method.

**Step 6:** Run: `cd backend && uv run python -c "from src.vacuum_pressure.config import VPRuntimeConfig; print('OK')"`

**Step 7:** Commit.

---

### Task 3: runtime_model.py (class renames + internal references)

**Files:**
- Modify: `backend/src/vacuum_pressure/runtime_model.py`

**Step 1:** Rename classes: `PermDerivativeRuntimeParams` → `DerivativeRuntimeParams`, `PermDerivativeRuntimeOutput` → `DerivativeRuntimeOutput`, `PermDerivativeRuntime` → `DerivativeRuntime`.

**Step 2:** Rename all error message strings that reference `perm_*` param names.

**Step 3:** Rename the `("perm_d1_weight", self.d1_weight)` validation tuples — strip `perm_` prefix from the string keys.

**Step 4:** Rename `perm_state5_code` parameter to `state5_code`.

**Step 5:** Change output name from `"perm_derivative"` to `"derivative"`.

**Step 6:** Update module docstring.

**Step 7:** Run: `cd backend && uv run python -c "from src.vacuum_pressure.runtime_model import DerivativeRuntime; print('OK')"`

**Step 8:** Commit.

---

### Task 4: stream_pipeline.py

**Files:**
- Modify: `backend/src/vacuum_pressure/stream_pipeline.py`

**Step 1:** Update imports: `PermDerivativeRuntime` → `DerivativeRuntime`, etc.

**Step 2:** Rename functions: `_perm_microstate_id` → `_microstate_id`, `_perm_state5_code` → `_state5_code`, `_perm_runtime_params_from_config` → `_runtime_params_from_config`.

**Step 3:** Rename all `config.perm_*` field accesses to `config.*` (matches Task 2 renames).

**Step 4:** Rename grid column keys: `perm_microstate_id` → `microstate_id`, `perm_state5_code` → `state5_code`.

**Step 5:** Rename local variables: `perm_runtime` → `runtime`, `perm_state5` → `state5`.

**Step 6:** Run: `cd backend && uv run python -c "from src.vacuum_pressure.stream_pipeline import stream_events; print('OK')"`

**Step 7:** Commit.

---

### Task 5: server.py

**Files:**
- Modify: `backend/src/vacuum_pressure/server.py`

**Step 1:** Rename Arrow schema fields: `perm_microstate_id` → `microstate_id`, `perm_state5_code` → `state5_code`.

**Step 2:** Rename the serving-to-config param mapping dict values (strip `perm_` prefix from the config key side).

**Step 3:** Rename WebSocket handler params: all `perm_*` function parameters drop prefix.

**Step 4:** Rename the `perm_overrides` local variable to `overrides`.

**Step 5:** Rename all `config.perm_*` field accesses.

**Step 6:** Rename the `runtime_config` JSON payload: drop `perm_` from all keys, change `"name": "perm_derivative"` → `"name": "derivative"`.

**Step 7:** Remove the `perm_derivative` signal name check in `_build_streaming_url()` or update to `derivative`.

**Step 8:** Run: `cd backend && uv run python -c "from src.vacuum_pressure.server import app; print('OK')"`

**Step 9:** Commit.

---

### Task 6: run_vacuum_pressure.py (CLI flags)

**Files:**
- Modify: `backend/scripts/run_vacuum_pressure.py`

**Step 1:** Rename all `--perm-*` argparse flags to strip prefix: `--perm-d1-weight` → `--d1-weight`, etc.

**Step 2:** Rename all `args.perm_*` references to `args.*`.

**Step 3:** Rename all `perm_*` keys in the overrides dict.

**Step 4:** Rename all `perm_*` references in the query string builder.

**Step 5:** Run: `cd backend && uv run scripts/run_vacuum_pressure.py --help` and verify no `perm` in output.

**Step 6:** Commit.

---

### Task 7: experiment_config.py

**Files:**
- Modify: `backend/src/vacuum_pressure/experiment_config.py`

**Step 1:** Change the fallback signal name from `"perm_derivative"` to `"derivative"`.

**Step 2:** Run: `cd backend && uv run python -c "from src.vacuum_pressure.experiment_config import ExperimentSpec; print('OK')"`

**Step 3:** Commit.

---

### Task 8: Harness signal file rename + internals

**Files:**
- Rename: `backend/src/experiment_harness/signals/statistical/perm_derivative.py` → `derivative.py`
- Modify: the new `derivative.py`
- Modify: `backend/src/experiment_harness/signals/statistical/__init__.py`

**Step 1:** `git mv signals/statistical/perm_derivative.py signals/statistical/derivative.py`

**Step 2:** In `derivative.py`: rename `PermDerivativeSignal` → `DerivativeSignal`, change `return "perm_derivative"` → `return "derivative"`, rename all `perm_state5_code` → `state5_code`, `perm_microstate_id` → `microstate_id`, rename metadata keys (`perm_state5_distribution` → `state5_distribution`, etc.), update `register_signal("perm_derivative", ...)` → `register_signal("derivative", ...)`.

**Step 3:** In `__init__.py`: update import from `perm_derivative` → `derivative`.

**Step 4:** Run: `cd backend && uv run python -c "from src.experiment_harness.signals.statistical.derivative import DerivativeSignal; print('OK')"`

**Step 5:** Commit.

---

### Task 9: runner.py, tracking.py, grid_generator.py

**Files:**
- Modify: `backend/src/experiment_harness/runner.py`
- Modify: `backend/src/experiment_harness/tracking.py`
- Modify: `backend/src/experiment_harness/grid_generator.py`

**Step 1:** In `runner.py`: rename `perm_state5_distribution` → `state5_distribution`, `perm_micro9_distribution` → `micro9_distribution`, `perm_state5_transition_matrix` → `state5_transition_matrix`, `perm_state5_labels` → `state5_labels`, `perm_taxonomy_version` → `taxonomy_version`, `perm_state5_distribution_json` → `state5_distribution_json`, `perm_micro9_distribution_json` → `micro9_distribution_json`, `perm_state5_transition_matrix_json` → `state5_transition_matrix_json`, `perm_state5_labels_json` → `state5_labels_json`.

**Step 2:** In `tracking.py`: same metadata key renames as Step 1.

**Step 3:** In `grid_generator.py`: rename `perm_microstate_id` → `microstate_id`, `perm_state5_code` → `state5_code`.

**Step 4:** Run: `cd backend && uv run python -c "from src.experiment_harness.runner import ExperimentRunner; print('OK')"`

**Step 5:** Commit.

---

### Task 10: Frontend — vacuum-pressure.ts

**Files:**
- Modify: `frontend/src/vacuum-pressure.ts`

**Step 1:** In `StreamParams` interface: rename all `perm_*` fields to drop prefix.

**Step 2:** In the `PERM_OVERRIDE_KEYS` array (or equivalent): rename all `'perm_*'` strings to drop prefix.

**Step 3:** In the WebSocket URL builder: rename all `perm_*` query param names.

**Step 4:** Run: `cd frontend && npx tsc --noEmit`

**Step 5:** Commit.

---

### Task 11: Frontend — experiments.ts

**Files:**
- Modify: `frontend/src/experiments.ts`

**Step 1:** Remove comment mentioning `perm_*`.

**Step 2:** Change `perm_derivative` signal name check to `derivative`.

**Step 3:** Run: `cd frontend && npx tsc --noEmit`

**Step 4:** Commit.

---

### Task 12: Test file renames + content updates

**Files:**
- Rename: `backend/tests/test_runtime_perm_model.py` → `test_runtime_model.py`
- Rename: `backend/tests/test_experiment_harness/test_perm_derivative_signal.py` → `test_derivative_signal.py`
- Modify: the renamed files + `test_runtime_config_overrides.py`, `test_stream_pipeline_perf.py`, `test_server_arrow_serialization.py`, `test_experiment_harness/test_runner_core.py`

**Step 1:** `git mv` both test files.

**Step 2:** In `test_runtime_model.py` (renamed): update imports (`PermDerivativeRuntime` → `DerivativeRuntime`, etc.), rename test functions, rename all `perm_*` string literals.

**Step 3:** In `test_derivative_signal.py` (renamed): update import path (from `perm_derivative` → `derivative`), rename `PermDerivativeSignal` → `DerivativeSignal`, rename all `perm_*` column names, rename test functions.

**Step 4:** In `test_runtime_config_overrides.py`: rename all `perm_*` config keys and field accesses.

**Step 5:** In `test_stream_pipeline_perf.py`: rename all `perm_*` column/field references, rename `perm_runtime_enabled` override.

**Step 6:** In `test_server_arrow_serialization.py`: rename `perm_microstate_id` → `microstate_id`, `perm_state5_code` → `state5_code`.

**Step 7:** In `test_runner_core.py`: rename `perm_derivative` signal reference → `derivative`.

**Step 8:** Run: `cd backend && uv run pytest tests/ -x -q`

**Step 9:** Commit.

---

### Task 13: YAML config renames + content updates

**Files:**
- Rename + modify: all 4 YAML config files listed in the file rename table above
- Modify: content of all old-format configs that reference `perm_derivative`

**Step 1:** `git mv` each config file.

**Step 2:** In each renamed file: update `name:` field, `signal:` references from `perm_derivative` → `derivative`, any `perm_*` param key names, description text.

**Step 3:** In old-format configs (`smoke_perm_derivative.yaml`, `tune_perm_derivative.yaml`): rename signal references and sweep keys. Also rename the files themselves.

**Step 4:** Commit.

---

### Task 14: Delete stale results + update READMEs

**Files:**
- Delete: `backend/lake/research/vp_harness/results/runs_meta.parquet`
- Delete: `backend/lake/research/vp_harness/results/runs.parquet`
- Modify: `README.md`
- Modify: `backend/src/experiment_harness/README.md`

**Step 1:** Delete stale parquet files (they contain `perm_*` column names and will be regenerated on next experiment run).

**Step 2:** In root `README.md`: strip all `perm_*` and `--perm-*` references.

**Step 3:** In harness `README.md`: no `perm_` references exist (already clean).

**Step 4:** Commit.

---

### Task 15: Full verification

**Step 1:** `cd backend && uv run pytest tests/ -x -q`

**Step 2:** `cd frontend && npx tsc --noEmit`

**Step 3:** `grep -r 'perm_\|perm-\|PermDerivative\|perm_derivative' backend/src/ backend/tests/ backend/scripts/ frontend/src/ --include='*.py' --include='*.ts' --include='*.yaml'` — expect zero matches.

**Step 4:** `grep -r 'perm_\|perm-' backend/src/vacuum_pressure/instrument.yaml` — expect zero matches.

**Step 5:** `grep -r 'perm_\|perm-' README.md backend/src/experiment_harness/README.md` — expect zero matches.
