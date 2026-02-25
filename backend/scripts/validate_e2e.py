"""
End-to-end pipeline validation. Runs entirely offline against the MNQ 2026-02-06
.dbn file. No WebSocket, no browser, no mocks.

Usage (from backend/):
    uv run scripts/validate_e2e.py [--lake-root PATH] [--skip-generate] [--skip-experiment]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Resolve backend root so relative imports work when invoked from backend/
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_ROOT))

PIPELINE_CONFIG_NAME = "mnq_60m_baseline"
SERVING_CONFIG_NAME = "derivative_baseline"
EXPERIMENT_CONFIG_NAME = "sweep_derivative_rr20"


def _resolve_dataset_id(pipeline_config_path: Path) -> str:
    """Derive DATASET_ID from the pipeline spec (deterministic, no hardcoding)."""
    from src.qmachina.pipeline_config import PipelineSpec

    return PipelineSpec.from_yaml(pipeline_config_path).dataset_id()

# Gold columns required in gold_grid.parquet
REQUIRED_GOLD_COLS = {
    "bin_seq",
    "k",
    "pressure_variant",
    "vacuum_variant",
    "composite",
    "composite_d1",
    "composite_d2",
    "composite_d3",
    "state5_code",
    "flow_score",
    "flow_state_code",
}

# runtime_snapshot keys required in a published serving spec.
# gold params are inlined (c1_v_add, etc.), not nested under gold_config.
# stream_schema is used (not grid_schema_fields). config_hash is in source.
REQUIRED_SNAPSHOT_KEYS = {
    "visualization",
    "model_id",
    "stream_schema",
    "cell_width_ms",
    "n_absolute_ticks",
    "config_version",
    "flow_windows",
    "flow_rollup_weights",
}


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    status: str = "SKIP"  # "PASS" | "FAIL" | "SKIP"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.status = "FAIL"
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def passed(self) -> bool:
        return self.status == "PASS"


def _pass(result: StageResult) -> StageResult:
    if result.status != "FAIL":
        result.status = "PASS"
    return result


# ---------------------------------------------------------------------------
# Stage 0: Config validation
# ---------------------------------------------------------------------------


def stage0_config_validation(lake_root: Path) -> StageResult:
    r = StageResult(name="Stage 0 — Config validation")
    try:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.qmachina.validate_configs",
                "--lake-root",
                str(lake_root),
                "--log-level",
                "WARNING",
            ],
            capture_output=True,
            text=True,
            cwd=str(_BACKEND_ROOT),
        )
        if proc.returncode != 0:
            r.fail(f"validate_configs exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
        else:
            r.status = "PASS"
    except Exception as exc:
        r.fail(f"Failed to run validate_configs: {exc}")
    return r


# ---------------------------------------------------------------------------
# Stage 1: Dataset existence / generation + assertions
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_bins(r: StageResult, bins_path: Path) -> None:
    required_cols = {
        "ts_ns",
        "bin_seq",
        "bin_start_ns",
        "bin_end_ns",
        "bin_event_count",
        "event_id",
        "mid_price",
        "spot_ref_price_int",
        "best_bid_price_int",
        "best_ask_price_int",
        "book_valid",
    }
    df = pd.read_parquet(bins_path)
    missing = required_cols - set(df.columns)
    if missing:
        r.fail(f"bins.parquet missing columns: {sorted(missing)}")
    if len(df) == 0:
        r.fail("bins.parquet has 0 rows")
        return

    # bin_seq monotonically increasing from 0
    if df["bin_seq"].iloc[0] != 0:
        r.fail(f"bin_seq does not start at 0, starts at {df['bin_seq'].iloc[0]}")
    if not (df["bin_seq"].diff().iloc[1:] > 0).all():
        r.fail("bin_seq is not strictly monotonically increasing")

    # mid_price > 0 for book_valid rows
    valid = df[df["book_valid"] == True]
    if len(valid) > 0 and (valid["mid_price"] <= 0).any():
        r.fail("mid_price <= 0 for some book_valid rows")

    # No NaN in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            r.fail(f"bins.parquet has NaN in column '{col}'")

    # spread > 0 for book_valid rows
    if len(valid) > 0:
        spread = valid["best_ask_price_int"] - valid["best_bid_price_int"]
        if (spread <= 0).any():
            r.fail("best_ask_price_int <= best_bid_price_int for some book_valid rows")


def _assert_grid_clean(r: StageResult, grid_path: Path, grid_radius_ticks: int) -> None:
    from src.qmachina.stage_schema import SILVER_FLOAT_COLS, SILVER_INT_COL_DTYPES

    required_silver = set(SILVER_FLOAT_COLS) | set(SILVER_INT_COL_DTYPES.keys())
    df = pd.read_parquet(grid_path)
    actual_cols = set(df.columns)

    missing_silver = required_silver - actual_cols
    if missing_silver:
        r.warn(
            f"grid_clean.parquet missing {len(missing_silver)} silver cols "
            f"(dataset may pre-date current schema): {sorted(missing_silver)}"
        )

    if "k" not in actual_cols:
        r.fail("grid_clean.parquet missing 'k' column")
        return

    # k range
    k_min, k_max = df["k"].min(), df["k"].max()
    if k_min < -grid_radius_ticks or k_max > grid_radius_ticks:
        r.fail(
            f"k values out of range [-{grid_radius_ticks}, {grid_radius_ticks}]: "
            f"got [{k_min}, {k_max}]"
        )

    # rest_depth >= 0
    if "rest_depth" in actual_cols and (df["rest_depth"] < 0).any():
        r.fail("rest_depth < 0 in grid_clean.parquet")

    # reprice signs in {-1, 0, 1}
    for sign_col in ("ask_reprice_sign", "bid_reprice_sign"):
        if sign_col in actual_cols:
            valid_vals = {-1, 0, 1}
            actual_vals = set(df[sign_col].unique())
            bad = actual_vals - valid_vals
            if bad:
                r.fail(f"{sign_col} has values outside {{-1, 0, 1}}: {bad}")

    # No NaN in float cols for book_valid rows
    if "book_valid" in actual_cols:
        valid_mask = df["book_valid"] == True
        float_cols = [c for c in SILVER_FLOAT_COLS if c in actual_cols]
        for col in float_cols:
            if df.loc[valid_mask, col].isna().any():
                r.fail(f"NaN in '{col}' for book_valid rows")
                break  # one message is enough

    # Row count check: n_bins * (2 * grid_radius + 1) within ±5%
    n_grid_k = 2 * grid_radius_ticks + 1
    n_bins_approx = len(df) / n_grid_k
    if "bin_seq" in actual_cols:
        n_bins = df["bin_seq"].nunique()
        expected = n_bins * n_grid_k
        if abs(len(df) - expected) / expected > 0.05:
            r.warn(
                f"grid_clean row count {len(df)} differs >5% from "
                f"n_bins*grid_width = {n_bins}*{n_grid_k} = {expected}"
            )


def _assert_manifest(r: StageResult, manifest_path: Path) -> int:
    """Assert manifest structure; return grid_radius_ticks or 50 as fallback.

    Supports both old nested manifest format (source_manifest.part_manifests)
    and new flat format (n_bins, grid_radius_ticks at top or via pipeline config).
    """
    try:
        data = json.loads(manifest_path.read_text())
    except Exception as exc:
        r.fail(f"manifest.json not parseable: {exc}")
        return 50

    # dataset_id is required in both formats
    if "dataset_id" not in data:
        r.fail("manifest.json missing key 'dataset_id'")

    # Try to find grid_radius_ticks in order of preference:
    # 1. Top-level (new format)
    # 2. source_manifest.part_manifests[0] (old multi-part format)
    grid_radius = 50  # fallback
    if "grid_radius_ticks" in data:
        grid_radius = int(data["grid_radius_ticks"])
    elif "source_manifest" in data:
        try:
            parts = data["source_manifest"].get("part_manifests", [])
            if parts:
                grid_radius = int(parts[0].get("grid_radius_ticks", 50))
            else:
                r.warn("Could not extract grid_radius_ticks from manifest; defaulting to 50")
        except (KeyError, IndexError, TypeError):
            r.warn("Could not extract grid_radius_ticks from manifest; defaulting to 50")
    else:
        r.warn("grid_radius_ticks not found in manifest; defaulting to 50")

    return grid_radius


def _assert_checksums(r: StageResult, checksums_path: Path, dataset_dir: Path) -> None:
    try:
        data = json.loads(checksums_path.read_text())
    except Exception as exc:
        r.fail(f"checksums.json not parseable: {exc}")
        return

    # Support both key formats:
    #   old: {"bins_parquet_sha256": "...", "grid_clean_parquet_sha256": "..."}
    #   new: {"bins.parquet": "...", "grid_clean.parquet": "..."}
    key_map_old = {"bins_parquet_sha256": "bins.parquet", "grid_clean_parquet_sha256": "grid_clean.parquet"}
    key_map_new = {"bins.parquet": "bins.parquet", "grid_clean.parquet": "grid_clean.parquet"}

    # Determine which format is in use
    if any(k in data for k in key_map_old):
        key_map = key_map_old
    else:
        key_map = key_map_new

    for key, filename in key_map.items():
        if key not in data:
            r.warn(f"checksums.json missing key '{key}'")
            continue
        file_path = dataset_dir / filename
        if not file_path.exists():
            r.fail(f"checksums.json references {filename} but file not found")
            continue
        actual_hash = _sha256_file(file_path)
        stored_hash = data[key]
        if actual_hash != stored_hash:
            r.fail(
                f"SHA256 mismatch for {filename}: "
                f"stored={stored_hash[:12]}... actual={actual_hash[:12]}..."
            )


def stage1_dataset(
    lake_root: Path,
    pipeline_config_path: Path,
    dataset_id: str,
    skip_generate: bool,
) -> StageResult:
    r = StageResult(name="Stage 1 — Dataset")

    from src.experiment_harness.dataset_registry import DatasetRegistry

    registry = DatasetRegistry(lake_root)
    try:
        paths = registry.resolve(dataset_id)
    except FileNotFoundError:
        paths = None

    if paths is None or not paths.bins_parquet.exists():
        if skip_generate:
            r.fail(f"Dataset '{dataset_id}' not found and --skip-generate is set")
            return r
        # Run generation via subprocess
        r.warn(f"Dataset '{dataset_id}' not found; generating (this may take a while)...")
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.experiment_harness.cli",
                "generate",
                str(pipeline_config_path),
                "--lake-root",
                str(lake_root),
            ],
            capture_output=True,
            text=True,
            cwd=str(_BACKEND_ROOT),
        )
        if proc.returncode != 0:
            r.fail(f"Dataset generation failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
            return r
        try:
            paths = registry.resolve(dataset_id)
        except FileNotFoundError:
            r.fail(f"Dataset '{dataset_id}' still not found after generation")
            return r

    dataset_dir = paths.bins_parquet.parent

    # Assert manifest.json
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        r.fail("manifest.json not found in dataset directory")
        grid_radius_ticks = 50
    else:
        grid_radius_ticks = _assert_manifest(r, manifest_path)
        # Fall back to pipeline config if manifest doesn't carry grid_radius_ticks
        if grid_radius_ticks == 50:
            try:
                from src.qmachina.pipeline_config import PipelineSpec
                spec = PipelineSpec.from_yaml(pipeline_config_path)
                cfg = spec.resolve_runtime_config()
                grid_radius_ticks = cfg.grid_radius_ticks
            except Exception:
                pass

    # Assert checksums.json
    checksums_path = dataset_dir / "checksums.json"
    if not checksums_path.exists():
        r.warn("checksums.json not found; skipping checksum verification")
    else:
        _assert_checksums(r, checksums_path, dataset_dir)

    # Assert bins.parquet
    if not paths.bins_parquet.exists():
        r.fail("bins.parquet does not exist")
    else:
        _assert_bins(r, paths.bins_parquet)

    # Assert grid_clean.parquet
    if not paths.grid_clean_parquet.exists():
        r.fail("grid_clean.parquet does not exist")
    else:
        _assert_grid_clean(r, paths.grid_clean_parquet, grid_radius_ticks)

    return _pass(r)


# ---------------------------------------------------------------------------
# Stage 2: Gold generation + assertions
# ---------------------------------------------------------------------------


def _assert_gold_grid(r: StageResult, gold_path: Path, grid_clean_path: Path) -> None:
    df = pd.read_parquet(gold_path)
    actual_cols = set(df.columns)

    missing = REQUIRED_GOLD_COLS - actual_cols
    if missing:
        r.fail(f"gold_grid.parquet missing columns: {sorted(missing)}")

    if len(df) == 0:
        r.fail("gold_grid.parquet has 0 rows")
        return

    # composite is an unbounded pressure/vacuum ratio — just check finite (no NaN/inf)
    if "composite" in actual_cols:
        if not np.isfinite(df["composite"]).all():
            r.fail("composite contains NaN or inf")

    # flow_score in [-1, 1] (tanh output)
    if "flow_score" in actual_cols:
        out_of_range = df[(df["flow_score"] < -1.0) | (df["flow_score"] > 1.0)]
        if len(out_of_range) > 0:
            r.fail(f"flow_score out of [-1, 1] for {len(out_of_range)} rows")

    # state5_code in {-2, -1, 0, 1, 2}
    if "state5_code" in actual_cols:
        valid_codes = {-2, -1, 0, 1, 2}
        actual_codes = set(df["state5_code"].unique())
        bad = actual_codes - valid_codes
        if bad:
            r.fail(f"state5_code has values outside {{-2,-1,0,1,2}}: {bad}")

    # No NaN in any column
    for col in REQUIRED_GOLD_COLS & actual_cols:
        if df[col].isna().any():
            r.fail(f"NaN in gold column '{col}'")

    # Row count matches grid_clean
    if grid_clean_path.exists():
        grid_df = pd.read_parquet(grid_clean_path, columns=["bin_seq"])
        if len(df) != len(grid_df):
            r.fail(
                f"gold_grid row count {len(df)} != grid_clean row count {len(grid_df)}"
            )

    # bin_seq x k pairs unique
    if "bin_seq" in actual_cols and "k" in actual_cols:
        n_unique = df.groupby(["bin_seq", "k"]).ngroups
        if n_unique != len(df):
            r.fail(
                f"Duplicate (bin_seq, k) pairs in gold_grid: "
                f"{len(df) - n_unique} duplicates"
            )


def stage2_gold(
    lake_root: Path,
    pipeline_config_path: Path,
    dataset_id: str,
) -> StageResult:
    r = StageResult(name="Stage 2 — Gold generation")

    from src.experiment_harness.dataset_registry import DatasetRegistry
    from src.experiment_harness.gold_builder import generate_gold_dataset
    from src.qmachina.gold_config import GoldFeatureConfig
    from src.qmachina.pipeline_config import PipelineSpec

    registry = DatasetRegistry(lake_root)
    try:
        paths = registry.resolve(dataset_id)
    except FileNotFoundError:
        r.fail(f"Dataset '{dataset_id}' not found; run Stage 1 first")
        return r

    gold_path = paths.gold_grid_parquet
    if not gold_path.exists():
        r.warn("gold_grid.parquet absent; generating via gold_builder (Python API)...")
        try:
            spec = PipelineSpec.from_yaml(pipeline_config_path)
            config = spec.resolve_runtime_config()
            gold_cfg = GoldFeatureConfig.from_runtime_config(config)
            generate_gold_dataset(paths, gold_cfg)
        except Exception as exc:
            r.fail(f"gold generation failed: {exc}")
            return r
        if not gold_path.exists():
            r.fail("gold_grid.parquet still absent after generation")
            return r

    _assert_gold_grid(r, gold_path, paths.grid_clean_parquet)
    return _pass(r)


# ---------------------------------------------------------------------------
# Stage 3: Experiment run + assertions
# ---------------------------------------------------------------------------


def stage3_experiment(
    lake_root: Path,
    experiment_config_path: Path,
    skip_experiment: bool,
) -> StageResult:
    r = StageResult(name="Stage 3 — Experiment run")

    results_root = lake_root / "research" / "harness" / "results"
    meta_path = results_root / "runs_meta.parquet"
    runs_path = results_root / "runs.parquet"

    if skip_experiment:
        r.warn("--skip-experiment set; skipping experiment execution")
        # Assert existing results only if present; otherwise treat as SKIP
        if not meta_path.exists():
            r.status = "SKIP"
            return r
    else:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.experiment_harness.cli",
                "run",
                str(experiment_config_path),
                "--lake-root",
                str(lake_root),
            ],
            capture_output=True,
            text=True,
            cwd=str(_BACKEND_ROOT),
            timeout=7200,
        )
        if proc.returncode != 0:
            r.fail(
                f"Experiment run failed (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
            return r

    # Assert runs_meta.parquet
    if not meta_path.exists():
        r.fail("runs_meta.parquet not found after experiment run")
        return r

    meta = pd.read_parquet(meta_path)
    exp_name = EXPERIMENT_CONFIG_NAME
    exp_rows = meta[meta["experiment_name"] == exp_name] if "experiment_name" in meta.columns else pd.DataFrame()
    if "experiment_name" not in meta.columns or len(exp_rows) == 0:
        if skip_experiment:
            # Results from this experiment simply haven't been written yet — acceptable.
            r.status = "SKIP"
            return r
        r.fail(f"No rows with experiment_name == '{exp_name}' in runs_meta.parquet")
    else:
        # run_id, config_hash, signal_name non-null
        for col in ("run_id", "config_hash", "signal_name"):
            if col in exp_rows.columns and exp_rows[col].isna().any():
                r.fail(f"runs_meta has null '{col}' for experiment rows")

        # eval_tp_ticks=8, eval_sl_ticks=4
        if "eval_tp_ticks" in exp_rows.columns:
            bad_tp = exp_rows[exp_rows["eval_tp_ticks"] != 8]
            if len(bad_tp) > 0:
                r.fail(f"eval_tp_ticks != 8 in {len(bad_tp)} rows (expected 8)")
        if "eval_sl_ticks" in exp_rows.columns:
            bad_sl = exp_rows[exp_rows["eval_sl_ticks"] != 4]
            if len(bad_sl) > 0:
                r.fail(f"eval_sl_ticks != 4 in {len(bad_sl)} rows (expected 4)")

    # Assert runs.parquet
    if not runs_path.exists():
        r.fail("runs.parquet not found")
        return r

    runs = pd.read_parquet(runs_path)
    if "run_id" in meta.columns and "run_id" in runs.columns:
        valid_run_ids = set(exp_rows["run_id"]) if "experiment_name" in meta.columns else set()
        if valid_run_ids:
            exp_runs = runs[runs["run_id"].isin(valid_run_ids)]
            if len(exp_runs) == 0:
                r.fail("No rows in runs.parquet joined to experiment run_ids")
            else:
                # tp_rate in [0, 1]
                if "tp_rate" in exp_runs.columns:
                    bad = exp_runs[(exp_runs["tp_rate"] < 0) | (exp_runs["tp_rate"] > 1)]
                    if len(bad) > 0:
                        r.fail(f"tp_rate out of [0, 1] in {len(bad)} rows")
                # sl_rate in [0, 1]
                if "sl_rate" in exp_runs.columns:
                    bad = exp_runs[(exp_runs["sl_rate"] < 0) | (exp_runs["sl_rate"] > 1)]
                    if len(bad) > 0:
                        r.fail(f"sl_rate out of [0, 1] in {len(bad)} rows")
                # tp + sl + timeout ~ 1.0 for rows with n_signals > 0
                rate_cols = {"tp_rate", "sl_rate", "timeout_rate"}
                if rate_cols <= set(exp_runs.columns) and "n_signals" in exp_runs.columns:
                    active = exp_runs[exp_runs["n_signals"] > 0]
                    if len(active) > 0:
                        total = active["tp_rate"] + active["sl_rate"] + active["timeout_rate"]
                        off = (total - 1.0).abs()
                        if (off > 1e-6).any():
                            worst = off.max()
                            r.fail(
                                f"tp_rate + sl_rate + timeout_rate != 1.0 for "
                                f"{(off > 1e-6).sum()} rows; max deviation={worst:.2e}"
                            )
                # mean_pnl_ticks finite
                if "mean_pnl_ticks" in exp_runs.columns:
                    if not np.isfinite(exp_runs["mean_pnl_ticks"]).all():
                        r.fail("mean_pnl_ticks is NaN/inf in some rows")

    return _pass(r)


# ---------------------------------------------------------------------------
# Stage 4: Serving registration + assertions
# ---------------------------------------------------------------------------


def stage4_serving(lake_root: Path) -> StageResult:
    r = StageResult(name="Stage 4 — Serving registration")

    proc = subprocess.run(
        [
            "uv",
            "run",
            "scripts/register_serving.py",
            SERVING_CONFIG_NAME,
            "--lake-root",
            str(lake_root),
        ],
        capture_output=True,
        text=True,
        cwd=str(_BACKEND_ROOT),
    )
    if proc.returncode != 0:
        r.fail(
            f"register_serving.py failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
        return r

    # Assert SQLite registry
    registry_path = lake_root / "research" / "harness" / "serving_registry.sqlite"
    if not registry_path.exists():
        r.fail(f"serving_registry.sqlite not found at {registry_path}")
        return r

    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT serving_id FROM serving_aliases WHERE alias = ?",
            (SERVING_CONFIG_NAME,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        r.fail(f"No alias '{SERVING_CONFIG_NAME}' in serving_aliases table")
        return r

    serving_id = str(row["serving_id"])

    # Assert PublishedServingSpec YAML
    versions_dir = lake_root / "research" / "harness" / "configs" / "serving_versions"
    spec_path = versions_dir / f"{serving_id}.yaml"
    if not spec_path.exists():
        r.fail(f"PublishedServingSpec YAML not found at {spec_path}")
        return r

    try:
        spec_data = yaml.safe_load(spec_path.read_text())
    except Exception as exc:
        r.fail(f"Failed to parse {spec_path}: {exc}")
        return r

    # Required top-level keys (PublishedServingSpec has no `name` field; uses serving_id)
    for key in ("serving_id", "description", "runtime_snapshot", "source"):
        if key not in spec_data:
            r.fail(f"PublishedServingSpec missing key '{key}'")

    snapshot = spec_data.get("runtime_snapshot", {})

    # Required snapshot keys
    missing_snapshot = REQUIRED_SNAPSHOT_KEYS - set(snapshot.keys())
    if missing_snapshot:
        r.fail(f"runtime_snapshot missing keys: {sorted(missing_snapshot)}")

    # model_id == "vacuum_pressure"
    if snapshot.get("model_id") != "vacuum_pressure":
        r.fail(
            f"runtime_snapshot.model_id expected 'vacuum_pressure', "
            f"got '{snapshot.get('model_id')}'"
        )

    # Gold params are inlined in runtime_snapshot (not nested under gold_config).
    # Check that the key canonical gold fields are present directly.
    expected_gold_inlined = {"c1_v_add", "c7_a_pull", "flow_windows", "flow_rollup_weights"}
    missing_gold_inlined = expected_gold_inlined - set(snapshot.keys())
    if missing_gold_inlined:
        r.warn(f"runtime_snapshot missing expected gold param keys: {sorted(missing_gold_inlined)}")

    # source must carry config_hash
    source = spec_data.get("source", {})
    if isinstance(source, dict) and not source.get("config_hash"):
        r.warn("PublishedServingSpec source.config_hash is missing or empty")

    return _pass(r)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_table(results: list[StageResult]) -> None:
    print()
    print("=" * 70)
    print("  E2E Validation Results")
    print("=" * 70)

    for res in results:
        marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(res.status, "[????]")
        print(f"  {marker}  {res.name}")
        for warn in res.warnings:
            print(f"           WARN: {warn}")
        for err in res.errors:
            # Truncate very long error messages
            preview = err[:300] + ("..." if len(err) > 300 else "")
            print(f"           ERROR: {preview}")

    print("-" * 70)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    print(f"  Total: {len(results)}  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end offline pipeline validation for qMachina.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lake-root",
        default="",
        help="Path to lake root (default: backend/lake/)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip dataset generation if dataset is missing (fail instead)",
    )
    parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Skip experiment run; only validate pre-existing results",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    lake_root = Path(args.lake_root) if args.lake_root else _BACKEND_ROOT / "lake"
    configs_dir = lake_root / "research" / "harness" / "configs"
    pipeline_config_path = configs_dir / "pipelines" / f"{PIPELINE_CONFIG_NAME}.yaml"
    experiment_config_path = configs_dir / "experiments" / f"{EXPERIMENT_CONFIG_NAME}.yaml"

    if not pipeline_config_path.exists():
        print(f"ERROR: Pipeline config not found at {pipeline_config_path}", file=sys.stderr)
        sys.exit(1)
    if not experiment_config_path.exists():
        print(f"ERROR: Experiment config not found at {experiment_config_path}", file=sys.stderr)
        sys.exit(1)

    dataset_id = _resolve_dataset_id(pipeline_config_path)
    results: list[StageResult] = []

    print(f"\nRunning E2E validation (lake_root={lake_root})")
    print(f"  dataset_id:   {dataset_id}")
    print(f"  pipeline:     {PIPELINE_CONFIG_NAME}")
    print(f"  experiment:   {EXPERIMENT_CONFIG_NAME}")
    print(f"  serving:      {SERVING_CONFIG_NAME}")

    # Stage 0
    print("\n[Stage 0] Config validation...", flush=True)
    r0 = stage0_config_validation(lake_root)
    results.append(r0)

    # Stage 1
    print("[Stage 1] Dataset existence & assertions...", flush=True)
    r1 = stage1_dataset(lake_root, pipeline_config_path, dataset_id, args.skip_generate)
    results.append(r1)

    # Stage 2
    print("[Stage 2] Gold generation & assertions...", flush=True)
    r2 = stage2_gold(lake_root, pipeline_config_path, dataset_id)
    results.append(r2)

    # Stage 3
    print("[Stage 3] Experiment run & assertions...", flush=True)
    r3 = stage3_experiment(lake_root, experiment_config_path, args.skip_experiment)
    results.append(r3)

    # Stage 4
    print("[Stage 4] Serving registration & assertions...", flush=True)
    r4 = stage4_serving(lake_root)
    results.append(r4)

    _print_table(results)

    any_fail = any(r.status == "FAIL" for r in results)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
