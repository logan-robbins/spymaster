"""Offline gold feature builder: silver parquet -> gold_grid.parquet.

Computes all gold features from a silver dataset (grid_clean.parquet):
  1. Force block: pressure_variant, vacuum_variant, composite, composite_d1/d2/d3
  2. State code: state5_code (from BBO signs + k position)
  3. Scoring block: flow_score, flow_state_code

Each relative-k cell is treated as an independent time series across bins.
This is the same approximation used by the live stream (IndependentCellSpectrum
processes absolute ticks; relative-k cells shift with spot but are approximated
as stable for offline scoring purposes).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .dataset_registry import DatasetPaths

from ..qmachina.gold_config import GoldFeatureConfig
from ..qmachina.stage_schema import (
    GOLD_SCORING_FLOAT_COLS,
    GOLD_SCORING_INT_COL_DTYPES,
    GOLD_FORCE_FLOAT_COLS,
    GOLD_FORCE_INT_COL_DTYPES,
)
from ..models.vacuum_pressure.scoring import SpectrumScorer
from ..qmachina.serving_config import ScoringConfig

logger = logging.getLogger(__name__)

_EPS = 1e-12

# 5-state directional code lookup tables
_STATE5_BULL_VACUUM = 2
_STATE5_BULL_PRESSURE = 1
_STATE5_MIXED = 0
_STATE5_BEAR_PRESSURE = -1
_STATE5_BEAR_VACUUM = -2

_ABOVE_STATE5_BY_SIGNS: dict[tuple[int, int], int] = {
    (1, 1): _STATE5_BULL_VACUUM,
    (1, 0): _STATE5_BEAR_PRESSURE,
    (1, -1): _STATE5_BEAR_PRESSURE,
    (0, 1): _STATE5_BULL_VACUUM,
    (0, 0): _STATE5_MIXED,
    (0, -1): _STATE5_BEAR_PRESSURE,
    (-1, 1): _STATE5_MIXED,
    (-1, 0): _STATE5_BEAR_PRESSURE,
    (-1, -1): _STATE5_BEAR_PRESSURE,
}

_BELOW_STATE5_BY_SIGNS: dict[tuple[int, int], int] = {
    (1, 1): _STATE5_BULL_PRESSURE,
    (1, 0): _STATE5_BEAR_VACUUM,
    (1, -1): _STATE5_BEAR_VACUUM,
    (0, 1): _STATE5_BULL_PRESSURE,
    (0, 0): _STATE5_MIXED,
    (0, -1): _STATE5_BEAR_VACUUM,
    (-1, 1): _STATE5_MIXED,
    (-1, 0): _STATE5_BEAR_VACUUM,
    (-1, -1): _STATE5_BEAR_VACUUM,
}


def _compute_state5_series(
    k_arr: np.ndarray,
    ask_sign_arr: np.ndarray,
    bid_sign_arr: np.ndarray,
) -> np.ndarray:
    """Vectorize state5_code from k + BBO movement signs.

    Args:
        k_arr: Relative tick positions (int32).
        ask_sign_arr: Ask repricing signs in {-1, 0, 1} (int8).
        bid_sign_arr: Bid repricing signs in {-1, 0, 1} (int8).

    Returns:
        state5_code array (int8).
    """
    n = len(k_arr)
    result = np.full(n, _STATE5_MIXED, dtype=np.int8)
    for i in range(n):
        k = int(k_arr[i])
        ask = int(ask_sign_arr[i])
        bid = int(bid_sign_arr[i])
        key = (ask, bid)
        if k > 0:
            result[i] = np.int8(_ABOVE_STATE5_BY_SIGNS[key])
        elif k < 0:
            result[i] = np.int8(_BELOW_STATE5_BY_SIGNS[key])
        # k == 0: MIXED (already set)
    return result


def _compute_force_block_for_k_series(
    silver_k: pd.DataFrame,
    cfg: GoldFeatureConfig,
    cell_width_s: float,
) -> pd.DataFrame:
    """Compute force block for one k-level time series.

    Args:
        silver_k: Silver rows for one k value, sorted by bin_seq.
        cfg: GoldFeatureConfig with force and spectrum params.
        cell_width_s: Bin width in seconds (dt for derivative computation).

    Returns:
        DataFrame with columns: bin_seq, k, pressure_variant, vacuum_variant,
        composite, composite_d1, composite_d2, composite_d3.
    """
    n = len(silver_k)

    v_add = silver_k["v_add"].to_numpy(dtype=np.float64)
    v_pull = silver_k["v_pull"].to_numpy(dtype=np.float64)
    v_fill = silver_k["v_fill"].to_numpy(dtype=np.float64)
    v_rest = silver_k["v_rest_depth"].to_numpy(dtype=np.float64)
    a_add = silver_k["a_add"].to_numpy(dtype=np.float64)
    a_pull = silver_k["a_pull"].to_numpy(dtype=np.float64)

    pressure = (
        cfg.c1_v_add * v_add
        + cfg.c2_v_rest_pos * np.maximum(v_rest, 0.0)
        + cfg.c3_a_add * np.maximum(a_add, 0.0)
    )
    vacuum = (
        cfg.c4_v_pull * v_pull
        + cfg.c5_v_fill * v_fill
        + cfg.c6_v_rest_neg * np.maximum(-v_rest, 0.0)
        + cfg.c7_a_pull * np.maximum(a_pull, 0.0)
    )

    composite_raw = (pressure - vacuum) / (np.abs(pressure) + np.abs(vacuum) + _EPS)

    # Rolling composite: weighted average of multiple windows
    windows = cfg.flow_windows
    weights = np.asarray(cfg.flow_rollup_weights, dtype=np.float64)
    max_hist = max(windows)

    rolled = np.zeros(n, dtype=np.float64)
    for idx, (window, w) in enumerate(zip(windows, weights)):
        win_sum = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        cum = 0.0
        for i in range(n):
            cum += composite_raw[i]
            if i >= window:
                cum -= composite_raw[i - window]
            cnt = min(i + 1, window)
            win_sum[i] = cum
            counts[i] = cnt
        rolled += w * (win_sum / np.maximum(counts, 1.0))

    # Temporal derivatives (finite differences, scale by dt)
    dt = cell_width_s
    d1 = np.zeros(n, dtype=np.float64)
    d2 = np.zeros(n, dtype=np.float64)
    d3 = np.zeros(n, dtype=np.float64)

    if n > 1:
        d1[1:] = (rolled[1:] - rolled[:-1]) / dt
    if n > 2:
        d2[2:] = (d1[2:] - d1[1:-1]) / dt
    if n > 3:
        d3[3:] = (d2[3:] - d2[2:-1]) / dt

    out = pd.DataFrame({
        "bin_seq": silver_k["bin_seq"].to_numpy(),
        "k": silver_k["k"].to_numpy(),
        "pressure_variant": pressure,
        "vacuum_variant": vacuum,
        "composite": rolled,
        "composite_d1": d1,
        "composite_d2": d2,
        "composite_d3": d3,
    })
    return out


def build_gold_grid(
    silver_path: Path,
    bins_path: Path,
    cfg: GoldFeatureConfig,
) -> pd.DataFrame:
    """Compute all gold features from a silver grid_clean.parquet.

    Args:
        silver_path: Path to grid_clean.parquet (silver stage output).
        bins_path: Path to bins.parquet (for cell_width_ms).
        cfg: GoldFeatureConfig specifying all gold computation parameters.

    Returns:
        DataFrame with columns: bin_seq, k, + all gold columns.
        Sorted by (bin_seq, k).

    Raises:
        KeyError: If required silver columns are missing.
    """
    import pyarrow.parquet as pq

    required_cols = [
        "bin_seq", "k",
        "v_add", "v_pull", "v_fill", "v_rest_depth", "a_add", "a_pull",
        "ask_reprice_sign", "bid_reprice_sign",
    ]
    schema_names = set(pq.read_schema(silver_path).names)
    missing = [c for c in required_cols if c not in schema_names]
    if missing:
        raise KeyError(f"silver grid_clean.parquet missing required columns: {missing}")

    bins_df = pd.read_parquet(bins_path, columns=["bin_seq", "bin_start_ns", "bin_end_ns"])
    if len(bins_df) < 2:
        cell_width_s = 0.1
    else:
        cell_ns = int(bins_df["bin_end_ns"].iloc[0]) - int(bins_df["bin_start_ns"].iloc[0])
        cell_width_s = max(float(cell_ns) / 1e9, 1e-6)

    grid_df = pd.read_parquet(silver_path, columns=required_cols)
    grid_df = grid_df.sort_values(["k", "bin_seq"]).reset_index(drop=True)

    n_total = len(grid_df)
    k_vals = grid_df["k"].unique()
    n_k = len(k_vals)
    logger.info(
        "Building gold grid: %d rows, %d k-levels, cell_width_s=%.4f",
        n_total, n_k, cell_width_s,
    )

    # -----------------------------------------------------------------------
    # 1. Force block + composite derivatives (per k-level)
    # -----------------------------------------------------------------------
    force_block_parts: list[pd.DataFrame] = []
    for k in sorted(k_vals):
        k_mask = grid_df["k"] == k
        silver_k = grid_df[k_mask].sort_values("bin_seq")
        force_block_k = _compute_force_block_for_k_series(silver_k, cfg, cell_width_s)
        force_block_parts.append(force_block_k)

    force_block_df = pd.concat(force_block_parts, ignore_index=True)

    # -----------------------------------------------------------------------
    # 2. State5 code (from silver BBO signs + k)
    # -----------------------------------------------------------------------
    state5_df = grid_df[["bin_seq", "k", "ask_reprice_sign", "bid_reprice_sign"]].copy()
    state5_df["state5_code"] = _compute_state5_series(
        grid_df["k"].to_numpy(dtype=np.int32),
        grid_df["ask_reprice_sign"].to_numpy(dtype=np.int8),
        grid_df["bid_reprice_sign"].to_numpy(dtype=np.int8),
    )

    # -----------------------------------------------------------------------
    # 3. Scoring block: flow_score / flow_state_code (per k-level)
    # -----------------------------------------------------------------------
    scoring_cfg = ScoringConfig(
        zscore_window_bins=cfg.flow_zscore_window_bins,
        zscore_min_periods=cfg.flow_zscore_min_periods,
        derivative_weights=cfg.flow_derivative_weights,
        tanh_scale=cfg.flow_tanh_scale,
        neutral_threshold=cfg.flow_neutral_threshold,
    )

    scored_parts: list[pd.DataFrame] = []
    for k in sorted(k_vals):
        k_mask = force_block_df["k"] == k
        force_block_k = force_block_df[k_mask].sort_values("bin_seq").reset_index(drop=True)

        scorer = SpectrumScorer(scoring_cfg, n_cells=1)
        flow_scores = np.zeros(len(force_block_k), dtype=np.float64)
        flow_states = np.zeros(len(force_block_k), dtype=np.int8)

        for i in range(len(force_block_k)):
            d1 = np.array([force_block_k["composite_d1"].iloc[i]], dtype=np.float64)
            d2 = np.array([force_block_k["composite_d2"].iloc[i]], dtype=np.float64)
            d3 = np.array([force_block_k["composite_d3"].iloc[i]], dtype=np.float64)
            score, state = scorer.update(d1, d2, d3)
            flow_scores[i] = float(score[0])
            flow_states[i] = int(state[0])

        scored_parts.append(pd.DataFrame({
            "bin_seq": force_block_k["bin_seq"].to_numpy(),
            "k": force_block_k["k"].to_numpy(),
            "flow_score": flow_scores,
            "flow_state_code": flow_states,
        }))

    scored_df = pd.concat(scored_parts, ignore_index=True)

    # -----------------------------------------------------------------------
    # 4. Merge all gold blocks on (bin_seq, k)
    # -----------------------------------------------------------------------
    gold = (
        force_block_df
        .merge(
            state5_df[["bin_seq", "k", "state5_code"]],
            on=["bin_seq", "k"],
            how="left",
        )
        .merge(
            scored_df[["bin_seq", "k", "flow_score", "flow_state_code"]],
            on=["bin_seq", "k"],
            how="left",
        )
    )

    # Cast dtypes to match stage_schema contracts
    gold["state5_code"] = gold["state5_code"].fillna(0).astype(np.int8)
    gold["flow_state_code"] = gold["flow_state_code"].fillna(0).astype(np.int8)

    gold = gold.sort_values(["bin_seq", "k"]).reset_index(drop=True)

    logger.info("Gold grid built: %d rows", len(gold))
    return gold


def generate_gold_dataset(
    paths: "DatasetPaths",
    cfg: GoldFeatureConfig,
    *,
    force: bool = False,
) -> Path:
    """Generate gold_grid.parquet from existing silver dataset files.

    Idempotent: if gold_grid.parquet already exists with matching config hash
    recorded in manifest.json, returns immediately (unless force=True).

    Args:
        paths: DatasetPaths with resolved silver and bins parquet paths.
        cfg: GoldFeatureConfig specifying gold computation parameters.
        force: If True, recompute even if gold_grid.parquet exists.

    Returns:
        Path to the written gold_grid.parquet.

    Raises:
        FileNotFoundError: If bins.parquet or grid_clean.parquet are missing.
        KeyError: If required silver columns are missing.
    """
    bins_path = paths.bins_parquet
    silver_path = paths.grid_clean_parquet
    gold_path = paths.gold_grid_parquet
    dataset_dir = bins_path.parent

    if not bins_path.exists():
        raise FileNotFoundError(f"bins.parquet not found in {dataset_dir}")
    if not silver_path.exists():
        raise FileNotFoundError(f"grid_clean.parquet not found in {dataset_dir}")

    # Check idempotency via manifest
    manifest_path = dataset_dir / "manifest.json"
    config_hash = cfg.config_hash()
    if not force and gold_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("gold_config_hash") == config_hash:
                logger.info(
                    "Gold grid already exists with matching config hash %s: %s",
                    config_hash,
                    gold_path,
                )
                return gold_path
        except Exception:
            pass  # Re-generate if manifest is unreadable

    logger.info("Generating gold grid: dataset_dir=%s config_hash=%s", dataset_dir, config_hash)
    gold_df = build_gold_grid(silver_path, bins_path, cfg)
    gold_df.to_parquet(gold_path, index=False)
    logger.info("  gold_grid.parquet: %d rows -> %s", len(gold_df), gold_path)

    # Update manifest with gold config hash
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest["gold_config_hash"] = config_hash
    manifest["gold_config"] = cfg.model_dump()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")

    return gold_path
