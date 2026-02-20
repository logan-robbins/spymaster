# Remove `perm_` Prefix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strip the vestigial `perm_` / `perm-` / `Perm` prefix from every identifier, field, flag, filename, and YAML key in the codebase. There is only one runtime model — the prefix is meaningless.

**Architecture:** Two rename domains: (1) `perm_` config prefix on ~15 runtime params (instrument.yaml fields, VPRuntimeConfig fields, CLI flags, TS interface, WebSocket query params, grid bucket fields), and (2) `perm_derivative` signal name + class names. Both get renamed everywhere simultaneously.

**Tech Stack:** Python (dataclass fields, Click CLI, signal registry), TypeScript (interface fields, URL params), YAML configs.

**Rename mapping:**

Config param prefix `perm_` → drop prefix entirely:
- `perm_runtime_enabled` → `runtime_enabled`
- `perm_center_exclusion_radius` → `center_exclusion_radius`
- `perm_spatial_decay_power` → `spatial_decay_power`
- `perm_zscore_window_bins` → `zscore_window_bins`
- `perm_zscore_min_periods` → `zscore_min_periods`
- `perm_tanh_scale` → `tanh_scale`
- `perm_d1_weight` → `d1_weight`
- `perm_d2_weight` → `d2_weight`
- `perm_d3_weight` → `d3_weight`
- `perm_bull_pressure_weight` → `bull_pressure_weight`
- `perm_bull_vacuum_weight` → `bull_vacuum_weight`
- `perm_bear_pressure_weight` → `bear_pressure_weight`
- `perm_bear_vacuum_weight` → `bear_vacuum_weight`
- `perm_mixed_weight` → `mixed_weight`
- `perm_enable_weighted_blend` → `enable_weighted_blend`

CLI flags `--perm-*` → drop prefix:
- `--perm-runtime-enabled` → `--runtime-enabled`
- `--perm-zscore-window-bins` → `--zscore-window-bins`
- etc. (same 15 params)

Grid data fields:
- `perm_state5_code` → `state5_code`
- `perm_microstate_id` → `microstate_id`

Signal/class renames:
- `perm_derivative` (signal name) → `derivative`
- `PermDerivativeSignal` → `DerivativeSignal`
- `PermDerivativeRuntime` → `DerivativeRuntime`
- `PermDerivativeRuntimeParams` → `DerivativeRuntimeParams`
- `PermDerivativeRuntimeOutput` → `DerivativeRuntimeOutput`

File renames:
- `backend/src/experiment_harness/signals/statistical/perm_derivative.py` → `derivative.py`
- `backend/tests/test_experiment_harness/test_perm_derivative_signal.py` → `test_derivative_signal.py`
- `backend/tests/test_runtime_perm_model.py` → `test_runtime_model.py`

YAML config renames (filenames + internal refs):
- `configs/serving/perm_derivative_baseline.yaml` → `derivative_baseline.yaml`
- `configs/experiments/sweep_perm_derivative_rr20.yaml` → `sweep_derivative_rr20.yaml`
- `configs/smoke_perm_derivative.yaml` → `smoke_derivative.yaml`
- `configs/tune_perm_derivative.yaml` → `tune_derivative.yaml`

Tracking fields:
- `perm_state5_distribution_json` → `state5_distribution_json`
- `perm_micro9_distribution_json` → `micro9_distribution_json`
- `perm_state5_transition_matrix_json` → `state5_transition_matrix_json`
- `perm_state5_labels_json` → `state5_labels_json`
- `perm_state5_count_*` → `state5_count_*`
- `perm_micro9_count_*` → `micro9_count_*`
- `perm_taxonomy_version` → `taxonomy_version`

Helper functions:
- `_perm_microstate_id()` → `_microstate_id()`
- `_perm_state5_code()` → `_state5_code()`
- `_perm_runtime_params_from_config()` → `_runtime_params_from_config()`

---

### Task 1: instrument.yaml — strip perm_ prefix from all keys

**Files:**
- Modify: `backend/src/vacuum_pressure/instrument.yaml`

**Step 1: Rename all perm_ keys**

Remove the `perm_` prefix from every key in instrument.yaml. `perm_runtime_enabled` → `runtime_enabled`, `perm_d1_weight` → `d1_weight`, etc.

**Step 2: Commit**

```bash
git add backend/src/vacuum_pressure/instrument.yaml
git commit -m "refactor: strip perm_ prefix from instrument.yaml keys"
```

---

### Task 2: config.py — rename DEFAULT constants and VPRuntimeConfig fields

**Files:**
- Modify: `backend/src/vacuum_pressure/config.py`

**Step 1: Rename all DEFAULT_PERM_* constants to DEFAULT_***

Replace every `DEFAULT_PERM_` prefix with `DEFAULT_`. Replace every `perm_` field in the VPRuntimeConfig dataclass.

**Step 2: Commit**

```bash
git add backend/src/vacuum_pressure/config.py
git commit -m "refactor: strip perm_ prefix from VPRuntimeConfig fields and defaults"
```

---

### Task 3: runtime_model.py — rename classes, params, helpers

**Files:**
- Modify: `backend/src/vacuum_pressure/runtime_model.py`

**Step 1: Rename classes and all perm_ references**

- `PermDerivativeRuntimeParams` → `DerivativeRuntimeParams`
- `PermDerivativeRuntimeOutput` → `DerivativeRuntimeOutput`
- `PermDerivativeRuntime` → `DerivativeRuntime`
- All `perm_` field references in validation messages → strip prefix
- `name="perm_derivative"` → `name="derivative"`
- `perm_state5_code` param name → `state5_code`

**Step 2: Commit**

```bash
git add backend/src/vacuum_pressure/runtime_model.py
git commit -m "refactor: strip Perm/perm_ prefix from runtime model classes"
```

---

### Task 4: stream_pipeline.py — rename helper functions and grid fields

**Files:**
- Modify: `backend/src/vacuum_pressure/stream_pipeline.py`

**Step 1: Rename helpers and all perm_ field references**

- `_perm_microstate_id()` → `_microstate_id()`
- `_perm_state5_code()` → `_state5_code()`
- `_perm_runtime_params_from_config()` → `_runtime_params_from_config()`
- All `perm_state5_code` bucket keys → `state5_code`
- All `perm_microstate_id` bucket keys → `microstate_id`
- All `config.perm_*` references → `config.*` (matching new config field names)

**Step 2: Commit**

```bash
git add backend/src/vacuum_pressure/stream_pipeline.py
git commit -m "refactor: strip perm_ prefix from stream pipeline helpers and grid fields"
```

---

### Task 5: server.py — rename Arrow schema fields and param mapping

**Files:**
- Modify: `backend/src/vacuum_pressure/server.py`

**Step 1: Rename all perm_ references**

- Arrow schema field names: `perm_state5_code` → `state5_code`, `perm_microstate_id` → `microstate_id`
- `_SIGNAL_PARAM_TO_WS` mapping: all `perm_*` values → strip prefix
- Any `perm_*` URL param parsing → strip prefix
- `perm_derivative` string → `derivative`

**Step 2: Commit**

```bash
git add backend/src/vacuum_pressure/server.py
git commit -m "refactor: strip perm_ prefix from server Arrow schema and param mapping"
```

---

### Task 6: run_vacuum_pressure.py — rename CLI flags

**Files:**
- Modify: `backend/scripts/run_vacuum_pressure.py`

**Step 1: Rename all --perm-* CLI flags**

Every `--perm-*` flag → drop the `perm-` prefix. Also update the variable names from `perm_*` to match.

**Step 2: Commit**

```bash
git add backend/scripts/run_vacuum_pressure.py
git commit -m "refactor: strip --perm- prefix from CLI flags"
```

---

### Task 7: cache_vp_output.py — rename Arrow schema fields

**Files:**
- Modify: `backend/scripts/cache_vp_output.py`

**Step 1: Rename perm_ field names in Arrow schema**

`perm_state5_code` → `state5_code`, `perm_microstate_id` → `microstate_id`

**Step 2: Commit**

```bash
git add backend/scripts/cache_vp_output.py
git commit -m "refactor: strip perm_ prefix from cache script Arrow schema"
```

---

### Task 8: Signal file rename and class rename

**Files:**
- Rename: `backend/src/experiment_harness/signals/statistical/perm_derivative.py` → `derivative.py`
- Modify: `backend/src/experiment_harness/signals/statistical/__init__.py`
- Modify: `backend/src/experiment_harness/signals/statistical/derivative.py` (the renamed file)

**Step 1: Rename the file**

```bash
cd backend
git mv src/experiment_harness/signals/statistical/perm_derivative.py src/experiment_harness/signals/statistical/derivative.py
```

**Step 2: Update __init__.py import**

Change `from . import perm_derivative` → `from . import derivative`

**Step 3: Update derivative.py internals**

- `PermDerivativeSignal` → `DerivativeSignal`
- `name` property: return `"derivative"` instead of `"perm_derivative"`
- `required_grid_columns`: `"perm_state5_code"` → `"state5_code"`, `"perm_microstate_id"` → `"microstate_id"`
- `register_signal("perm_derivative", PermDerivativeSignal)` → `register_signal("derivative", DerivativeSignal)`
- All `perm_state5_*` and `perm_micro9_*` metadata keys → strip prefix
- Class docstring: remove "permutation-" prefix

**Step 4: Commit**

```bash
git add -A src/experiment_harness/signals/statistical/
git commit -m "refactor: rename perm_derivative signal to derivative"
```

---

### Task 9: experiment_config.py — update default signal name

**Files:**
- Modify: `backend/src/vacuum_pressure/experiment_config.py`

**Step 1: Update perm_derivative fallback**

Line 242: `["perm_derivative"]` → `["derivative"]`

**Step 2: Commit**

```bash
git add backend/src/vacuum_pressure/experiment_config.py
git commit -m "refactor: update default signal name from perm_derivative to derivative"
```

---

### Task 10: tracking.py — rename metric keys

**Files:**
- Modify: `backend/src/experiment_harness/tracking.py`

**Step 1: Strip perm_ prefix from all tracking metric keys**

All `perm_state5_*` → `state5_*`, `perm_micro9_*` → `micro9_*`, `perm_taxonomy_version` → `taxonomy_version`.

**Step 2: Commit**

```bash
git add backend/src/experiment_harness/tracking.py
git commit -m "refactor: strip perm_ prefix from tracking metric keys"
```

---

### Task 11: grid_generator.py — rename grid columns

**Files:**
- Modify: `backend/src/experiment_harness/grid_generator.py`

**Step 1: Rename perm_ column names**

`perm_microstate_id` → `microstate_id`, `perm_state5_code` → `state5_code`

**Step 2: Commit**

```bash
git add backend/src/experiment_harness/grid_generator.py
git commit -m "refactor: strip perm_ prefix from grid generator columns"
```

---

### Task 12: Frontend vacuum-pressure.ts — rename interface fields and URL param handling

**Files:**
- Modify: `frontend/src/vacuum-pressure.ts`

**Step 1: Rename all perm_ references**

- `StreamParams` interface: all `perm_*` properties → strip prefix
- `parseStreamParams()`: all `perm_*` param parsing → strip prefix
- `permKeys` array → `runtimeKeys`, contents strip `perm_` prefix
- URL param forwarding: all `perm_*` → strip prefix
- Comment references to `perm_*` → strip prefix

**Step 2: Commit**

```bash
git add frontend/src/vacuum-pressure.ts
git commit -m "refactor: strip perm_ prefix from frontend interface and URL params"
```

---

### Task 13: Frontend experiments.ts — update signal name reference

**Files:**
- Modify: `frontend/src/experiments.ts`

**Step 1: Update perm_derivative references**

- `perm_derivative` signal name → `derivative`
- Comment about `perm_*` params → strip prefix

**Step 2: Commit**

```bash
git add frontend/src/experiments.ts
git commit -m "refactor: strip perm_ prefix from experiments.ts"
```

---

### Task 14: Test file renames and content updates

**Files:**
- Rename: `backend/tests/test_runtime_perm_model.py` → `test_runtime_model.py`
- Rename: `backend/tests/test_experiment_harness/test_perm_derivative_signal.py` → `test_derivative_signal.py`
- Modify: both renamed files (update class/function/field references)
- Modify: `backend/tests/test_runtime_config_overrides.py` (update perm_ field references)
- Modify: `backend/tests/test_stream_pipeline_perf.py` (update perm_ field references)
- Modify: `backend/tests/test_server_arrow_serialization.py` (update perm_ field references)
- Modify: `backend/tests/test_scoring_equivalence.py` (update any perm_ references)

**Step 1: Rename test files**

```bash
cd backend
git mv tests/test_runtime_perm_model.py tests/test_runtime_model.py
git mv tests/test_experiment_harness/test_perm_derivative_signal.py tests/test_experiment_harness/test_derivative_signal.py
```

**Step 2: Update all perm_ references in all test files**

Strip `perm_` prefix from all field names, class names, config keys. Update `PermDerivative*` → `Derivative*`. Update `"perm_derivative"` → `"derivative"`.

**Step 3: Commit**

```bash
git add tests/
git commit -m "refactor: strip perm_ prefix from all test files"
```

---

### Task 15: YAML config file renames and content updates

**Files:**
- Rename: `backend/lake/research/vp_harness/configs/smoke_perm_derivative.yaml` → `smoke_derivative.yaml`
- Rename: `backend/lake/research/vp_harness/configs/tune_perm_derivative.yaml` → `tune_derivative.yaml`
- Rename: `backend/lake/research/vp_harness/configs/serving/perm_derivative_baseline.yaml` → `derivative_baseline.yaml`
- Rename: `backend/lake/research/vp_harness/configs/experiments/sweep_perm_derivative_rr20.yaml` → `sweep_derivative_rr20.yaml`
- Modify: all 4 files (update internal name/signal/serving references)

**Step 1: Rename all 4 config files**

```bash
cd backend
git mv lake/research/vp_harness/configs/smoke_perm_derivative.yaml lake/research/vp_harness/configs/smoke_derivative.yaml
git mv lake/research/vp_harness/configs/tune_perm_derivative.yaml lake/research/vp_harness/configs/tune_derivative.yaml
git mv lake/research/vp_harness/configs/serving/perm_derivative_baseline.yaml lake/research/vp_harness/configs/serving/derivative_baseline.yaml
git mv lake/research/vp_harness/configs/experiments/sweep_perm_derivative_rr20.yaml lake/research/vp_harness/configs/experiments/sweep_derivative_rr20.yaml
```

**Step 2: Update internal references**

- All `name:` values: strip `perm_` prefix
- All `serving:` references: `perm_derivative_baseline` → `derivative_baseline`
- All signal references: `perm_derivative` → `derivative`
- All `experiment_name:` values: strip `perm_`
- All `perm_*` signal param keys → strip prefix

**Step 3: Commit**

```bash
git add lake/research/vp_harness/configs/
git commit -m "refactor: strip perm_ prefix from config YAML filenames and contents"
```

---

### Task 16: README updates

**Files:**
- Modify: `README.md`
- Modify: `backend/src/experiment_harness/README.md`

**Step 1: Strip all remaining perm_ / --perm- references from both READMEs**

Replace `--perm-zscore-window-bins` → `--zscore-window-bins`, `perm_*` → runtime model params, etc.

**Step 2: Commit**

```bash
git add README.md backend/src/experiment_harness/README.md
git commit -m "docs: strip perm_ prefix from READMEs"
```

---

### Task 17: Run all tests and typecheck

**Step 1: Run backend tests**

```bash
cd backend
uv run pytest tests/ -v
```

Expected: all pass

**Step 2: Run frontend typecheck**

```bash
cd frontend
npx tsc --noEmit
```

Expected: clean

**Step 3: Verify CLI loads**

```bash
cd backend
uv run python -m src.experiment_harness.cli --help
```

Expected: all commands listed

**Step 4: Final grep — confirm zero perm_ references remain**

```bash
cd /Users/logan.robbins/research/spymaster
grep -r "perm_\|perm-\|Perm" --include="*.py" --include="*.ts" --include="*.yaml" --include="*.md" --exclude-dir=.venv --exclude-dir=node_modules --exclude-dir=__pycache__ --exclude-dir=.git --exclude-dir=docs/plans .
```

Expected: zero matches
