"""Derivative-only micro-regime analysis for vacuum-pressure dense grids.

This script evaluates *state transitions* rather than directional return
prediction. It uses deterministic rules over derivative-only descriptors
(velocity/acceleration/jerk fields) with no model training path.

Usage:
    cd backend
    uv run scripts/analyze_vp_signals.py
    uv run scripts/analyze_vp_signals.py --product-type future_mbo --symbol MNQH6

Methodology (regime mode only):
    1. Capture VP dense-grid snapshots from stream_events().
    2. Build derivative-only spatial descriptors:
       - energy_raw, coherence_raw, asymmetry_raw, shock_raw.
    3. Normalize online with rolling robust z-score (median/MAD).
    4. Run deterministic hysteresis state machine:
       stable_chop / directional_build / directional_drain / transition_shock.
    5. Evaluate transition-event quality against deterministic shock references
       and report state stability metrics.
    6. Optionally run a secondary non-overlapping directional sanity check.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup (same pattern as warm_cache.py / run_vacuum_pressure.py)
# ---------------------------------------------------------------------------
backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

logger = logging.getLogger("analyze_vp_signals")

from src.vacuum_pressure.event_engine import (
    C1_V_ADD,
    C2_V_REST_POS,
    C3_A_ADD,
    C4_V_PULL,
    C5_V_FILL,
    C6_V_REST_NEG,
    C7_A_PULL,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dense grid width is fixed by stream pipeline default K=50.
N_BUCKETS: int = 101
K_MAX: int = 50

# Bucket field names in grid dict (must match event_engine output keys)
PRESSURE_FIELD: str = "pressure_variant"
VACUUM_FIELD: str = "vacuum_variant"
REST_DEPTH_FIELD: str = "rest_depth"

VELOCITY_FIELDS: List[str] = ["v_add", "v_pull", "v_fill", "v_rest_depth"]
ACCEL_FIELDS: List[str] = ["a_add", "a_pull", "a_fill", "a_rest_depth"]
JERK_FIELDS: List[str] = ["j_add", "j_pull", "j_fill", "j_rest_depth"]

REGIME_STABLE_CHOP = "stable_chop"
REGIME_DIRECTIONAL_BUILD = "directional_build"
REGIME_DIRECTIONAL_DRAIN = "directional_drain"
REGIME_TRANSITION_SHOCK = "transition_shock"

VALID_REGIMES = (
    REGIME_STABLE_CHOP,
    REGIME_DIRECTIONAL_BUILD,
    REGIME_DIRECTIONAL_DRAIN,
    REGIME_TRANSITION_SHOCK,
)


@dataclass(frozen=True)
class RegimeThresholds:
    """Deterministic thresholds for online state transitions."""

    enter_shock: float = 2.5
    exit_shock: float = 1.5
    enter_directional: float = 1.4
    exit_directional: float = 0.7
    enter_energy: float = 0.4
    exit_energy: float = 0.1
    coherence_floor: float = 0.0
    min_dwell_snapshots: int = 8
    cooldown_snapshots: int = 4


@dataclass(frozen=True)
class TransitionReferenceConfig:
    """Reference-event extraction settings for transition metrics."""

    shock_threshold_z: float = 3.0
    min_run_snapshots: int = 3
    match_tolerance_snapshots: int = 12


# ---------------------------------------------------------------------------
# Data capture
# ---------------------------------------------------------------------------

def capture_grids(
    lake_root: Path,
    config: Any,
    dt: str,
    start_time: str,
    throttle_ms: float = 25.0,
    end_time_et: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capture VP grid snapshots into dense numpy arrays.

    Args:
        lake_root: Path to lake directory.
        config: VPRuntimeConfig.
        dt: Date string YYYY-MM-DD.
        start_time: Emit start HH:MM in ET.
        throttle_ms: Throttle for grid emission.
        end_time_et: Optional stop time HH:MM in ET. None = run to end of file.

    Returns:
        Tuple of:
            ts_ns_arr: (N,) int64 array of event timestamps in nanoseconds.
            mid_price_arr: (N,) float64 array of mid prices in dollars.
            bucket_data: (N, 101, 15) float64 array of bucket fields.
                Ordered fields:
                pressure_variant, vacuum_variant, rest_depth,
                v_add, v_pull, v_fill, v_rest_depth,
                a_add, a_pull, a_fill, a_rest_depth,
                j_add, j_pull, j_fill, j_rest_depth.
    """
    from src.vacuum_pressure.stream_pipeline import stream_events

    if throttle_ms <= 0.0:
        raise ValueError(f"--throttle-ms must be > 0, got {throttle_ms}")

    end_ns: int = 0
    if end_time_et:
        et_end = pd.Timestamp(f"{dt} {end_time_et}:00", tz="America/New_York").tz_convert("UTC")
        end_ns = int(et_end.value)

    bucket_fields = [
        PRESSURE_FIELD,
        VACUUM_FIELD,
        REST_DEPTH_FIELD,
        *VELOCITY_FIELDS,
        *ACCEL_FIELDS,
        *JERK_FIELDS,
    ]
    n_fields = len(bucket_fields)

    capacity = 80_000
    ts_ns_arr = np.empty(capacity, dtype=np.int64)
    mid_price_arr = np.empty(capacity, dtype=np.float64)
    bucket_arr = np.empty((capacity, N_BUCKETS, n_fields), dtype=np.float64)

    count = 0
    t_start = time.monotonic()

    for _event_id, grid in stream_events(
        lake_root=lake_root,
        config=config,
        dt=dt,
        start_time=start_time,
        throttle_ms=throttle_ms,
    ):
        ts_ns = int(grid["ts_ns"])
        if end_ns > 0 and ts_ns >= end_ns:
            break

        if not grid["book_valid"]:
            continue

        mid = float(grid["mid_price"])
        if mid <= 0.0:
            continue

        if count >= capacity:
            capacity = int(capacity * 1.5)
            ts_ns_arr = np.resize(ts_ns_arr, capacity)
            mid_price_arr = np.resize(mid_price_arr, capacity)
            new_bucket = np.empty((capacity, N_BUCKETS, n_fields), dtype=np.float64)
            new_bucket[:count] = bucket_arr[:count]
            bucket_arr = new_bucket

        ts_ns_arr[count] = ts_ns
        mid_price_arr[count] = mid

        buckets = grid["buckets"]
        if len(buckets) != N_BUCKETS:
            raise RuntimeError(
                f"Expected {N_BUCKETS} buckets from stream pipeline, got {len(buckets)}. "
                "This analysis script assumes K=50 fixed runtime grid."
            )

        for i, bucket in enumerate(buckets):
            for j, field_name in enumerate(bucket_fields):
                bucket_arr[count, i, j] = float(bucket[field_name])

        count += 1
        if count % 10_000 == 0:
            elapsed = time.monotonic() - t_start
            logger.info(
                "Captured %d snapshots (%.1fs, %.0f snap/s)",
                count,
                elapsed,
                count / elapsed if elapsed > 0 else 0.0,
            )

    elapsed = time.monotonic() - t_start
    logger.info(
        "Capture complete: %d snapshots in %.2fs (%.0f snap/s)",
        count,
        elapsed,
        count / elapsed if elapsed > 0 else 0.0,
    )

    return (
        ts_ns_arr[:count].copy(),
        mid_price_arr[:count].copy(),
        bucket_arr[:count].copy(),
    )


# ---------------------------------------------------------------------------
# Derivative-only feature engineering
# ---------------------------------------------------------------------------

# Field index mapping must match capture_grids() extraction order.
_FIELD_IDX: Dict[str, int] = {
    "pressure_variant": 0,
    "vacuum_variant": 1,
    "rest_depth": 2,
    "v_add": 3,
    "v_pull": 4,
    "v_fill": 5,
    "v_rest_depth": 6,
    "a_add": 7,
    "a_pull": 8,
    "a_fill": 9,
    "a_rest_depth": 10,
    "j_add": 11,
    "j_pull": 12,
    "j_fill": 13,
    "j_rest_depth": 14,
}


def _k_axis_for_grid(n_buckets: int) -> np.ndarray:
    """Return integer relative tick index axis for bucket dimension."""
    if n_buckets % 2 == 0:
        raise ValueError(f"Expected odd bucket count, got {n_buckets}")
    half = n_buckets // 2
    return np.arange(-half, half + 1)


def _family_tilt(
    add: np.ndarray,
    pull: np.ndarray,
    fill: np.ndarray,
    rest: np.ndarray,
    bid_mask: np.ndarray,
    ask_mask: np.ndarray,
) -> np.ndarray:
    """Directional tilt combining add/pull/fill/rest derivative asymmetry."""
    return (
        add[:, bid_mask].sum(axis=1) - add[:, ask_mask].sum(axis=1)
        + pull[:, ask_mask].sum(axis=1) - pull[:, bid_mask].sum(axis=1)
        + fill[:, ask_mask].sum(axis=1) - fill[:, bid_mask].sum(axis=1)
        + rest[:, bid_mask].sum(axis=1) - rest[:, ask_mask].sum(axis=1)
    )


def _family_energy(
    add: np.ndarray,
    pull: np.ndarray,
    fill: np.ndarray,
    rest: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Mean absolute activity magnitude for one derivative family."""
    total = (
        np.abs(add[:, active_mask]).sum(axis=1)
        + np.abs(pull[:, active_mask]).sum(axis=1)
        + np.abs(fill[:, active_mask]).sum(axis=1)
        + np.abs(rest[:, active_mask]).sum(axis=1)
    )
    denom = float(active_mask.sum() * 4)
    if denom <= 0.0:
        raise ValueError("Active mask is empty; cannot compute family energy.")
    return total / denom


def _rolling_mad(values: np.ndarray) -> float:
    """Median absolute deviation for rolling apply (raw=True)."""
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def online_robust_zscore(
    values: np.ndarray,
    window: int,
    min_periods: int,
) -> np.ndarray:
    """Online robust z-score using trailing rolling median/MAD.

    z_t = (x_t - median_t) / (1.4826 * MAD_t)

    This uses trailing windows only (no look-ahead).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")
    if min_periods > window:
        raise ValueError(
            f"min_periods={min_periods} cannot exceed window={window}"
        )

    series = pd.Series(values, dtype="float64")
    roll = series.rolling(window=window, min_periods=min_periods)
    med = roll.median()
    mad = roll.apply(_rolling_mad, raw=True)
    scale = 1.4826 * mad
    scale = scale.where(scale > 1e-9, np.nan)
    z = (series - med) / scale
    return z.values.astype(np.float64)


def compute_derivative_descriptors(
    ts_ns: np.ndarray,
    bucket_data: np.ndarray,
    normalization_window: int,
    normalization_min_periods: int,
) -> pd.DataFrame:
    """Compute derivative-only descriptors and online robust z-scores.

    Returns DataFrame with raw and z-scored descriptor columns:
        energy_raw, coherence_raw, asymmetry_raw, directional_raw, shock_raw
        energy_raw_z, coherence_raw_z, asymmetry_raw_z, directional_raw_z, shock_raw_z
    """
    if bucket_data.ndim != 3:
        raise ValueError(
            f"bucket_data must be 3D (N, buckets, fields), got shape={bucket_data.shape}"
        )
    if len(ts_ns) != bucket_data.shape[0]:
        raise ValueError(
            f"ts_ns length {len(ts_ns)} does not match bucket_data rows {bucket_data.shape[0]}"
        )

    n_samples, n_buckets, _n_fields = bucket_data.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 snapshots to compute descriptors.")

    k_axis = _k_axis_for_grid(n_buckets)
    bid_mask = k_axis < 0
    ask_mask = k_axis > 0
    active_mask = k_axis != 0

    v_add = bucket_data[:, :, _FIELD_IDX["v_add"]]
    v_pull = bucket_data[:, :, _FIELD_IDX["v_pull"]]
    v_fill = bucket_data[:, :, _FIELD_IDX["v_fill"]]
    v_rest = bucket_data[:, :, _FIELD_IDX["v_rest_depth"]]

    a_add = bucket_data[:, :, _FIELD_IDX["a_add"]]
    a_pull = bucket_data[:, :, _FIELD_IDX["a_pull"]]
    a_fill = bucket_data[:, :, _FIELD_IDX["a_fill"]]
    a_rest = bucket_data[:, :, _FIELD_IDX["a_rest_depth"]]

    j_add = bucket_data[:, :, _FIELD_IDX["j_add"]]
    j_pull = bucket_data[:, :, _FIELD_IDX["j_pull"]]
    j_fill = bucket_data[:, :, _FIELD_IDX["j_fill"]]
    j_rest = bucket_data[:, :, _FIELD_IDX["j_rest_depth"]]

    v_tilt = _family_tilt(v_add, v_pull, v_fill, v_rest, bid_mask, ask_mask)
    a_tilt = _family_tilt(a_add, a_pull, a_fill, a_rest, bid_mask, ask_mask)
    j_tilt = _family_tilt(j_add, j_pull, j_fill, j_rest, bid_mask, ask_mask)

    v_energy = _family_energy(v_add, v_pull, v_fill, v_rest, active_mask)
    a_energy = _family_energy(a_add, a_pull, a_fill, a_rest, active_mask)
    j_energy = _family_energy(j_add, j_pull, j_fill, j_rest, active_mask)

    # Scale-aware but instrument-portable descriptor components.
    energy_raw = v_energy + 0.8 * a_energy + 0.6 * j_energy
    tilt_combo = 0.5 * v_tilt + 0.3 * a_tilt + 0.2 * j_tilt

    # Coherence in [-1, +1]: positive when derivative families align by sign.
    coherence_num = v_tilt * a_tilt + a_tilt * j_tilt + v_tilt * j_tilt
    coherence_den = (
        np.abs(v_tilt * a_tilt)
        + np.abs(a_tilt * j_tilt)
        + np.abs(v_tilt * j_tilt)
        + 1e-12
    )
    coherence_raw = coherence_num / coherence_den

    # Directional asymmetry normalized by total derivative energy.
    asymmetry_raw = tilt_combo / (energy_raw + 1e-12)
    directional_raw = asymmetry_raw * np.clip(coherence_raw, 0.0, None)

    # Shock intensity from time-normalized jumps in energy + coherence.
    ts_f = ts_ns.astype(np.float64)
    dt_s = np.diff(ts_f, prepend=np.nan) / 1e9
    safe_dt = np.where(np.isfinite(dt_s) & (dt_s > 1e-6), dt_s, np.nan)

    energy_jump = np.abs(np.diff(energy_raw, prepend=energy_raw[0]))
    coherence_jump = np.abs(np.diff(coherence_raw, prepend=coherence_raw[0]))
    shock_raw = np.where(
        np.isfinite(safe_dt),
        0.7 * (energy_jump / safe_dt) + 0.3 * (coherence_jump / safe_dt),
        0.0,
    )
    shock_raw[0] = 0.0

    raw_df = pd.DataFrame(
        {
            "v_tilt": v_tilt,
            "a_tilt": a_tilt,
            "j_tilt": j_tilt,
            "v_energy": v_energy,
            "a_energy": a_energy,
            "j_energy": j_energy,
            "energy_raw": energy_raw,
            "coherence_raw": coherence_raw,
            "asymmetry_raw": asymmetry_raw,
            "directional_raw": directional_raw,
            "shock_raw": shock_raw,
        }
    )

    z_cols: Dict[str, np.ndarray] = {}
    for col in ["energy_raw", "coherence_raw", "asymmetry_raw", "directional_raw", "shock_raw"]:
        z_cols[f"{col}_z"] = online_robust_zscore(
            raw_df[col].values.astype(np.float64),
            window=normalization_window,
            min_periods=normalization_min_periods,
        )

    z_df = pd.DataFrame(z_cols, index=raw_df.index)
    return pd.concat([raw_df, z_df], axis=1)


# ---------------------------------------------------------------------------
# Forward projection (per bucket, per horizon)
# ---------------------------------------------------------------------------

def parse_int_list(raw: str, arg_name: str, min_value: int = 1) -> List[int]:
    """Parse comma-separated integer list with strict validation."""
    items: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            val = int(tok)
        except ValueError as exc:
            raise ValueError(f"{arg_name} contains non-integer token '{tok}'.") from exc
        if val < min_value:
            raise ValueError(
                f"{arg_name} values must be >= {min_value}, got {val}."
            )
        items.append(val)

    if not items:
        raise ValueError(f"{arg_name} must contain at least one integer value.")
    return sorted(set(items))


def _trailing_mean_2d(values: np.ndarray, window: int) -> np.ndarray:
    """Compute trailing mean over axis=0 for (N, K) array."""
    if values.ndim != 2:
        raise ValueError(f"_trailing_mean_2d expects 2D input, got {values.shape}")
    if window < 1:
        raise ValueError(f"window must be >=1, got {window}")

    n, _k = values.shape
    prefix = np.vstack([
        np.zeros((1, values.shape[1]), dtype=np.float64),
        np.cumsum(values, axis=0, dtype=np.float64),
    ])
    idx = np.arange(n)
    start = np.maximum(0, idx - window + 1)
    totals = prefix[idx + 1] - prefix[start]
    counts = (idx - start + 1).astype(np.float64)[:, None]
    return totals / counts


def _rolled_derivative_field(
    field_vals: np.ndarray,
    rollup_windows: List[int],
) -> np.ndarray:
    """Build one rolled derivative field from multi-window trailing means."""
    if field_vals.ndim != 2:
        raise ValueError(f"field_vals must be 2D (N,K), got {field_vals.shape}")

    weights = np.array([1.0 / np.sqrt(float(w)) for w in rollup_windows], dtype=np.float64)
    wsum = float(weights.sum())
    rolled = np.zeros_like(field_vals, dtype=np.float64)
    for w, weight in zip(rollup_windows, weights):
        rolled += weight * _trailing_mean_2d(field_vals, w)
    rolled /= wsum
    return rolled


def _composite_force(pressure: np.ndarray, vacuum: np.ndarray) -> np.ndarray:
    """Map pressure-vacuum pair to [-1, +1] composite color scalar."""
    denom = pressure + vacuum + 1e-12
    return (pressure - vacuum) / denom


def compute_bucket_forward_projection(
    ts_ns: np.ndarray,
    bucket_data: np.ndarray,
    eval_mask: np.ndarray,
    projection_horizons_ms: List[int],
    projection_rollup_windows: List[int],
) -> Dict[str, Any]:
    """Compute forward projection composites per bucket/horizon.

    Projection model:
        P_hat = clip(P0 + dP_dt*h + 0.5*d2P_dt2*h^2, 0, inf)
        V_hat = clip(V0 + dV_dt*h + 0.5*d2V_dt2*h^2, 0, inf)
        C_hat = (P_hat - V_hat) / (P_hat + V_hat + eps)

    Derivative rollups are computed per bucket from multi-window trailing means.
    """
    n, k_count, _ = bucket_data.shape
    if len(ts_ns) != n:
        raise ValueError("ts_ns length must match bucket_data rows.")
    if len(eval_mask) != n:
        raise ValueError("eval_mask length must match bucket_data rows.")
    if k_count != N_BUCKETS:
        raise ValueError(f"Expected {N_BUCKETS} buckets, got {k_count}.")
    if not eval_mask.any():
        raise ValueError("eval_mask contains no True values.")

    pressure_0 = bucket_data[:, :, _FIELD_IDX["pressure_variant"]].astype(np.float64)
    vacuum_0 = bucket_data[:, :, _FIELD_IDX["vacuum_variant"]].astype(np.float64)
    composite_0 = _composite_force(pressure_0, vacuum_0)

    # Multi-window rolled derivative signals (per bucket).
    v_rest = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["v_rest_depth"]].astype(np.float64),
        projection_rollup_windows,
    )
    a_add = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["a_add"]].astype(np.float64),
        projection_rollup_windows,
    )
    a_pull = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["a_pull"]].astype(np.float64),
        projection_rollup_windows,
    )
    a_fill = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["a_fill"]].astype(np.float64),
        projection_rollup_windows,
    )
    a_rest = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["a_rest_depth"]].astype(np.float64),
        projection_rollup_windows,
    )
    j_add = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["j_add"]].astype(np.float64),
        projection_rollup_windows,
    )
    j_pull = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["j_pull"]].astype(np.float64),
        projection_rollup_windows,
    )
    j_fill = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["j_fill"]].astype(np.float64),
        projection_rollup_windows,
    )
    j_rest = _rolled_derivative_field(
        bucket_data[:, :, _FIELD_IDX["j_rest_depth"]].astype(np.float64),
        projection_rollup_windows,
    )

    # Time derivatives of pressure/vacuum components based on VP formulas.
    dP_dt = (
        C1_V_ADD * a_add
        + C2_V_REST_POS * np.where(v_rest > 0.0, a_rest, 0.0)
        + C3_A_ADD * np.where(a_add > 0.0, j_add, 0.0)
    )
    dV_dt = (
        C4_V_PULL * a_pull
        + C5_V_FILL * a_fill
        + C6_V_REST_NEG * np.where(v_rest < 0.0, -a_rest, 0.0)
        + C7_A_PULL * np.where(a_pull > 0.0, j_pull, 0.0)
    )

    d2P_dt2 = (
        C1_V_ADD * j_add
        + C2_V_REST_POS * np.where(v_rest > 0.0, j_rest, 0.0)
    )
    d2V_dt2 = (
        C4_V_PULL * j_pull
        + C5_V_FILL * j_fill
        + C6_V_REST_NEG * np.where(v_rest < 0.0, -j_rest, 0.0)
    )

    eval_idx = np.where(eval_mask)[0]
    last_eval_idx = int(eval_idx[-1])
    k_axis = _k_axis_for_grid(k_count).tolist()

    per_horizon_last: Dict[int, np.ndarray] = {}
    per_horizon_metrics: Dict[int, Dict[str, float]] = {}

    for horizon_ms in projection_horizons_ms:
        h_s = float(horizon_ms) / 1000.0

        p_hat = np.clip(
            pressure_0 + dP_dt * h_s + 0.5 * d2P_dt2 * h_s * h_s,
            0.0,
            np.inf,
        )
        v_hat = np.clip(
            vacuum_0 + dV_dt * h_s + 0.5 * d2V_dt2 * h_s * h_s,
            0.0,
            np.inf,
        )
        c_hat = _composite_force(p_hat, v_hat)

        per_horizon_last[horizon_ms] = c_hat[last_eval_idx].astype(np.float32)

        # Directional consistency of projected composite change (eval window).
        target_ts = ts_ns + int(horizon_ms * 1_000_000)
        j_idx = np.searchsorted(ts_ns, target_ts, side="left")
        valid_rows = eval_mask & (j_idx < n)
        if valid_rows.any():
            pred_delta = c_hat[valid_rows] - composite_0[valid_rows]
            real_delta = composite_0[j_idx[valid_rows]] - composite_0[valid_rows]
            active = (np.abs(pred_delta) > 1e-6) & (np.abs(real_delta) > 1e-6)
            if active.any():
                hit_rate = float(
                    np.mean(np.sign(pred_delta[active]) == np.sign(real_delta[active]))
                )
                n_pairs = int(active.sum())
            else:
                hit_rate = np.nan
                n_pairs = 0
        else:
            hit_rate = np.nan
            n_pairs = 0

        per_horizon_metrics[horizon_ms] = {
            "delta_sign_hit_rate": hit_rate,
            "n_pairs": float(n_pairs),
        }

    return {
        "k_axis": k_axis,
        "current_last": composite_0[last_eval_idx].astype(np.float32),
        "projected_last_by_horizon": per_horizon_last,
        "horizon_metrics": per_horizon_metrics,
        "last_eval_idx": float(last_eval_idx),
    }


# ---------------------------------------------------------------------------
# Deterministic state machine
# ---------------------------------------------------------------------------

def _candidate_state(
    energy_z: float,
    directional_z: float,
    coherence_raw: float,
    shock_z: float,
    thresholds: RegimeThresholds,
) -> str:
    """Compute candidate regime state from current descriptor values."""
    if np.isfinite(shock_z) and shock_z >= thresholds.enter_shock:
        return REGIME_TRANSITION_SHOCK

    directional_ok = (
        np.isfinite(directional_z)
        and np.isfinite(energy_z)
        and np.isfinite(coherence_raw)
        and energy_z >= thresholds.enter_energy
        and coherence_raw >= thresholds.coherence_floor
    )

    if directional_ok and directional_z >= thresholds.enter_directional:
        return REGIME_DIRECTIONAL_BUILD
    if directional_ok and directional_z <= -thresholds.enter_directional:
        return REGIME_DIRECTIONAL_DRAIN

    if (
        np.isfinite(energy_z)
        and np.isfinite(directional_z)
        and np.isfinite(shock_z)
        and energy_z <= thresholds.exit_energy
        and abs(directional_z) <= thresholds.exit_directional
        and shock_z <= thresholds.exit_shock
    ):
        return REGIME_STABLE_CHOP

    # Ambiguous region: hold prior state via caller logic.
    return ""


def infer_regimes(
    ts_ns: np.ndarray,
    descriptors: pd.DataFrame,
    thresholds: RegimeThresholds,
) -> pd.DataFrame:
    """Run deterministic regime state machine with hysteresis/cooldown.

    Required descriptor columns:
        energy_raw_z, directional_raw_z, coherence_raw, shock_raw_z.
    """
    needed = ["energy_raw_z", "directional_raw_z", "coherence_raw", "shock_raw_z"]
    missing = [c for c in needed if c not in descriptors.columns]
    if missing:
        raise ValueError(f"Missing descriptor columns: {missing}")

    n = len(descriptors)
    if len(ts_ns) != n:
        raise ValueError(f"ts_ns length {len(ts_ns)} != descriptor rows {n}")
    if n == 0:
        raise ValueError("No descriptor rows provided.")

    energy_z = descriptors["energy_raw_z"].values
    directional_z = descriptors["directional_raw_z"].values
    coherence_raw = descriptors["coherence_raw"].values
    shock_z = descriptors["shock_raw_z"].values

    states = np.empty(n, dtype=object)
    transition_event = np.zeros(n, dtype=bool)
    transition_type = np.empty(n, dtype=object)
    dwell_ms = np.zeros(n, dtype=np.float64)

    state = REGIME_STABLE_CHOP
    state_start_idx = 0
    last_transition_idx = -10_000_000

    for i in range(n):
        cand = _candidate_state(
            energy_z=float(energy_z[i]),
            directional_z=float(directional_z[i]),
            coherence_raw=float(coherence_raw[i]),
            shock_z=float(shock_z[i]),
            thresholds=thresholds,
        )
        if not cand:
            cand = state

        # Exit hysteresis: stay in current state while still above exit threshold.
        # Shock has strict priority and must not be overridden by other hysteresis.
        if cand != REGIME_TRANSITION_SHOCK:
            if state == REGIME_TRANSITION_SHOCK and np.isfinite(shock_z[i]) and shock_z[i] >= thresholds.exit_shock:
                cand = REGIME_TRANSITION_SHOCK
            if state == REGIME_DIRECTIONAL_BUILD and np.isfinite(directional_z[i]) and directional_z[i] >= thresholds.exit_directional:
                cand = REGIME_DIRECTIONAL_BUILD
            if state == REGIME_DIRECTIONAL_DRAIN and np.isfinite(directional_z[i]) and directional_z[i] <= -thresholds.exit_directional:
                cand = REGIME_DIRECTIONAL_DRAIN
            if (
                state == REGIME_STABLE_CHOP
                and np.isfinite(energy_z[i])
                and np.isfinite(shock_z[i])
                and energy_z[i] <= thresholds.enter_energy
                and shock_z[i] <= thresholds.enter_shock
            ):
                cand = REGIME_STABLE_CHOP

        if cand != state:
            dwell_snapshots = i - state_start_idx
            cooldown_elapsed = i - last_transition_idx
            can_switch = (
                dwell_snapshots >= thresholds.min_dwell_snapshots
                and cooldown_elapsed >= thresholds.cooldown_snapshots
            )
            force_switch = cand == REGIME_TRANSITION_SHOCK

            if can_switch or force_switch:
                prev = state
                state = cand
                state_start_idx = i
                last_transition_idx = i
                transition_event[i] = True
                transition_type[i] = f"{prev}->{state}"
            else:
                transition_type[i] = ""
        else:
            transition_type[i] = ""

        states[i] = state
        dwell_ms[i] = max(0.0, (ts_ns[i] - ts_ns[state_start_idx]) / 1e6)

    result = descriptors.copy()
    result["state"] = states
    result["transition_event"] = transition_event
    result["transition_type"] = transition_type
    result["dwell_ms"] = dwell_ms
    return result


# ---------------------------------------------------------------------------
# Transition evaluation
# ---------------------------------------------------------------------------

def _extract_reference_events(
    shock_z: np.ndarray,
    mask: np.ndarray,
    cfg: TransitionReferenceConfig,
) -> List[int]:
    """Reference transitions from sustained shock exceedances."""
    if len(shock_z) != len(mask):
        raise ValueError("shock_z and mask lengths must match.")

    cond = mask & np.isfinite(shock_z) & (shock_z >= cfg.shock_threshold_z)
    idx = np.where(cond)[0]
    if len(idx) == 0:
        return []

    events: List[int] = []
    run_start = idx[0]
    prev = idx[0]
    run_len = 1
    for i in idx[1:]:
        if i == prev + 1:
            run_len += 1
        else:
            if run_len >= cfg.min_run_snapshots:
                events.append(run_start)
            run_start = i
            run_len = 1
        prev = i
    if run_len >= cfg.min_run_snapshots:
        events.append(run_start)
    return events


def _match_events(
    predicted: Sequence[int],
    reference: Sequence[int],
    tolerance: int,
) -> List[Tuple[int, int]]:
    """Greedy one-to-one matching under absolute index tolerance."""
    matches: List[Tuple[int, int]] = []
    used_pred: set[int] = set()

    for ref_idx, ref in enumerate(reference):
        best_pred_idx = -1
        best_abs_diff = tolerance + 1
        for j, pred in enumerate(predicted):
            if j in used_pred:
                continue
            diff = abs(pred - ref)
            if diff <= tolerance and diff < best_abs_diff:
                best_abs_diff = diff
                best_pred_idx = j
        if best_pred_idx >= 0:
            used_pred.add(best_pred_idx)
            matches.append((ref_idx, best_pred_idx))
    return matches


def _state_run_durations_ms(
    ts_ns: np.ndarray,
    states: Sequence[str],
    mask: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute per-state run durations within a masked interval."""
    idx = np.where(mask)[0]
    durations: Dict[str, List[float]] = {s: [] for s in VALID_REGIMES}
    if len(idx) == 0:
        return durations

    run_state = states[idx[0]]
    run_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if states[i] != run_state:
            dur_ms = max(0.0, (ts_ns[prev] - ts_ns[run_start]) / 1e6)
            durations[run_state].append(dur_ms)
            run_state = states[i]
            run_start = i
        prev = i

    dur_ms = max(0.0, (ts_ns[prev] - ts_ns[run_start]) / 1e6)
    durations[run_state].append(dur_ms)
    return durations


def evaluate_transition_quality(
    ts_ns: np.ndarray,
    regime_df: pd.DataFrame,
    eval_mask: np.ndarray,
    ref_cfg: TransitionReferenceConfig,
) -> Dict[str, float]:
    """Evaluate transition event quality and state stability."""
    needed = ["shock_raw_z", "state", "transition_event", "dwell_ms"]
    missing = [c for c in needed if c not in regime_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in regime_df: {missing}")

    shock_z = regime_df["shock_raw_z"].values
    states = regime_df["state"].values
    transitions = regime_df["transition_event"].values

    predicted_events = np.where(
        eval_mask & transitions & (states == REGIME_TRANSITION_SHOCK)
    )[0].tolist()
    reference_events = _extract_reference_events(shock_z, eval_mask, ref_cfg)

    matched_pairs = _match_events(
        predicted=predicted_events,
        reference=reference_events,
        tolerance=ref_cfg.match_tolerance_snapshots,
    )

    matched_pred = {predicted_events[pair[1]] for pair in matched_pairs}
    matched_ref = {reference_events[pair[0]] for pair in matched_pairs}

    n_pred = len(predicted_events)
    n_ref = len(reference_events)
    n_match = len(matched_pairs)

    precision = n_match / n_pred if n_pred > 0 else 0.0
    recall = n_match / n_ref if n_ref > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) >= 2:
        eval_minutes = max(1e-9, (ts_ns[eval_idx[-1]] - ts_ns[eval_idx[0]]) / 60e9)
    else:
        eval_minutes = 1e-9
    false_transition_rate = (n_pred - n_match) / eval_minutes

    latency_ms: List[float] = []
    for ref_idx, pred_idx in matched_pairs:
        ref_i = reference_events[ref_idx]
        pred_i = predicted_events[pred_idx]
        latency_ms.append((ts_ns[pred_i] - ts_ns[ref_i]) / 1e6)
    mean_latency_ms = float(np.mean(latency_ms)) if latency_ms else np.nan
    median_latency_ms = float(np.median(latency_ms)) if latency_ms else np.nan

    total_transitions_eval = int((eval_mask & transitions).sum())
    transition_rate_per_min = total_transitions_eval / eval_minutes
    median_dwell_ms = float(np.nanmedian(regime_df.loc[eval_mask, "dwell_ms"].values)) if eval_mask.any() else np.nan

    state_counts = regime_df.loc[eval_mask, "state"].value_counts().to_dict()
    run_durations = _state_run_durations_ms(ts_ns, states, eval_mask)
    state_median_dwell = {
        state: (float(np.median(vals)) if vals else np.nan)
        for state, vals in run_durations.items()
    }

    return {
        "n_pred_events": float(n_pred),
        "n_ref_events": float(n_ref),
        "n_matched_events": float(n_match),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_latency_ms": mean_latency_ms,
        "median_latency_ms": median_latency_ms,
        "false_transition_rate_per_min": false_transition_rate,
        "transition_rate_per_min": transition_rate_per_min,
        "median_dwell_ms": median_dwell_ms,
        "state_share_stable_chop": float(state_counts.get(REGIME_STABLE_CHOP, 0) / max(eval_mask.sum(), 1)),
        "state_share_directional_build": float(state_counts.get(REGIME_DIRECTIONAL_BUILD, 0) / max(eval_mask.sum(), 1)),
        "state_share_directional_drain": float(state_counts.get(REGIME_DIRECTIONAL_DRAIN, 0) / max(eval_mask.sum(), 1)),
        "state_share_transition_shock": float(state_counts.get(REGIME_TRANSITION_SHOCK, 0) / max(eval_mask.sum(), 1)),
        "state_median_dwell_stable_ms": state_median_dwell[REGIME_STABLE_CHOP],
        "state_median_dwell_build_ms": state_median_dwell[REGIME_DIRECTIONAL_BUILD],
        "state_median_dwell_drain_ms": state_median_dwell[REGIME_DIRECTIONAL_DRAIN],
        "state_median_dwell_shock_ms": state_median_dwell[REGIME_TRANSITION_SHOCK],
        "n_matched_pred_events": float(len(matched_pred)),
        "n_matched_ref_events": float(len(matched_ref)),
    }


# ---------------------------------------------------------------------------
# Secondary directional sanity (non-overlapping)
# ---------------------------------------------------------------------------

def non_overlapping_directional_sanity(
    mid_price: np.ndarray,
    directional_signal: np.ndarray,
    eval_mask: np.ndarray,
    horizon_snapshots: int,
    tick_size: float,
) -> Dict[str, float]:
    """Compute secondary directional checks on non-overlapping horizons."""
    if horizon_snapshots < 1:
        raise ValueError(
            f"horizon_snapshots must be >= 1, got {horizon_snapshots}"
        )
    if tick_size <= 0.0:
        raise ValueError(f"tick_size must be > 0, got {tick_size}")

    n = len(mid_price)
    if len(directional_signal) != n or len(eval_mask) != n:
        raise ValueError("mid_price, directional_signal, and eval_mask lengths must match.")

    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) < horizon_snapshots + 1:
        return {"n_samples": 0.0, "hit_rate": np.nan, "spearman_ic": np.nan}

    start = int(eval_idx[0])
    end = int(eval_idx[-1] - horizon_snapshots)
    if end < start:
        return {"n_samples": 0.0, "hit_rate": np.nan, "spearman_ic": np.nan}

    sample_indices = np.arange(start, end + 1, horizon_snapshots, dtype=int)
    valid = eval_mask[sample_indices] & eval_mask[sample_indices + horizon_snapshots]
    sample_indices = sample_indices[valid]
    if len(sample_indices) == 0:
        return {"n_samples": 0.0, "hit_rate": np.nan, "spearman_ic": np.nan}

    returns_ticks = (
        mid_price[sample_indices + horizon_snapshots] - mid_price[sample_indices]
    ) / tick_size
    signal = directional_signal[sample_indices]

    finite = np.isfinite(signal) & np.isfinite(returns_ticks)
    signal = signal[finite]
    returns_ticks = returns_ticks[finite]

    if len(signal) == 0:
        return {"n_samples": 0.0, "hit_rate": np.nan, "spearman_ic": np.nan}

    nonzero = (signal != 0.0) & (returns_ticks != 0.0)
    hit_rate = (
        float(np.mean(np.sign(signal[nonzero]) == np.sign(returns_ticks[nonzero])))
        if nonzero.any() else np.nan
    )
    spearman_ic = (
        float(stats.spearmanr(signal, returns_ticks)[0])
        if len(signal) >= 10 else np.nan
    )

    return {
        "n_samples": float(len(signal)),
        "hit_rate": hit_rate,
        "spearman_ic": spearman_ic,
    }


# ---------------------------------------------------------------------------
# Instrument parsing / orchestration
# ---------------------------------------------------------------------------

SPECTRUM_PRESSURE = "pressure"
SPECTRUM_TRANSITION = "transition"
SPECTRUM_VACUUM = "vacuum"

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_FLAT = "flat"


def _parse_et_timestamp_ns(dt: str, hhmm: str) -> int:
    """Parse YYYY-MM-DD + HH:MM in ET and return UTC nanoseconds."""
    return int(
        pd.Timestamp(f"{dt} {hhmm}:00", tz="America/New_York")
        .tz_convert("UTC")
        .value
    )


def _resolve_eval_end_time(eval_start: str, eval_end: str | None, eval_minutes: int | None) -> str:
    """Resolve eval end time from explicit end or fallback minutes."""
    if eval_end:
        return eval_end
    if eval_minutes is None:
        raise ValueError("Provide either --eval-end or --eval-minutes.")
    if eval_minutes < 1:
        raise ValueError(f"--eval-minutes must be >= 1, got {eval_minutes}")
    hh, mm = map(int, eval_start.split(":"))
    total = hh * 60 + mm + eval_minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def _trailing_mean_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Compute trailing mean for one series."""
    if values.ndim != 1:
        raise ValueError(f"_trailing_mean_1d expects 1D input, got shape {values.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n = len(values)
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    idx = np.arange(n)
    start = np.maximum(0, idx - window + 1)
    totals = prefix[idx + 1] - prefix[start]
    counts = (idx - start + 1).astype(np.float64)
    return totals / counts


def _rolled_signal_1d(values: np.ndarray, windows: List[int]) -> np.ndarray:
    """Combine multi-window trailing means with deterministic weights."""
    weights = np.array([1.0 / np.sqrt(float(w)) for w in windows], dtype=np.float64)
    weights /= float(weights.sum())
    out = np.zeros_like(values, dtype=np.float64)
    for w, weight in zip(windows, weights):
        out += weight * _trailing_mean_1d(values, w)
    return out


def _time_derivative(values: np.ndarray, ts_ns: np.ndarray) -> np.ndarray:
    """Time derivative with event-time deltas; non-finite deltas map to 0."""
    if values.ndim != 1 or ts_ns.ndim != 1:
        raise ValueError("values and ts_ns must be 1D arrays.")
    if len(values) != len(ts_ns):
        raise ValueError("values and ts_ns lengths must match.")

    out = np.zeros(len(values), dtype=np.float64)
    dv = np.diff(values, prepend=values[0])
    dt_s = np.diff(ts_ns.astype(np.float64), prepend=np.nan) / 1e9
    valid = np.isfinite(dt_s) & (dt_s > 1e-6)
    out[valid] = dv[valid] / dt_s[valid]
    out[~valid] = 0.0
    out[0] = 0.0
    return out


def _state_from_score(score: np.ndarray, threshold: float) -> np.ndarray:
    """Map signed score to pressure/transition/vacuum states."""
    out = np.full(len(score), SPECTRUM_TRANSITION, dtype=object)
    out[score >= threshold] = SPECTRUM_PRESSURE
    out[score <= -threshold] = SPECTRUM_VACUUM
    return out


def compute_directional_spectrum(
    ts_ns: np.ndarray,
    bucket_data: np.ndarray,
    directional_bands: List[int],
    micro_windows: List[int],
    normalization_window: int,
    normalization_min_periods: int,
    spectrum_threshold: float,
    directional_edge_threshold: float,
) -> pd.DataFrame:
    """Compute derivative-slope directional spectrum across above/below bands."""
    n, k_count, _ = bucket_data.shape
    if len(ts_ns) != n:
        raise ValueError("ts_ns length must match bucket_data rows.")
    if k_count != N_BUCKETS:
        raise ValueError(f"Expected {N_BUCKETS} buckets, got {k_count}.")

    k_axis = _k_axis_for_grid(k_count)
    pressure = bucket_data[:, :, _FIELD_IDX["pressure_variant"]].astype(np.float64)
    vacuum = bucket_data[:, :, _FIELD_IDX["vacuum_variant"]].astype(np.float64)

    band_weights = np.array([1.0 / np.sqrt(float(b)) for b in directional_bands], dtype=np.float64)
    band_weights /= float(band_weights.sum())

    out = pd.DataFrame(index=np.arange(n))
    above_score_cols: List[str] = []
    below_score_cols: List[str] = []

    for band in directional_bands:
        above_mask = (k_axis > 0) & (k_axis <= band)
        below_mask = (k_axis < 0) & (k_axis >= -band)
        if not above_mask.any() or not below_mask.any():
            raise ValueError(f"Band +/-{band} has empty mask for K={K_MAX}.")

        for side_name, mask in (("above", above_mask), ("below", below_mask)):
            p_layer = pressure[:, mask].mean(axis=1)
            v_layer = vacuum[:, mask].mean(axis=1)
            c_layer = _composite_force(p_layer, v_layer)
            c_roll = _rolled_signal_1d(c_layer, micro_windows)

            d1 = _time_derivative(c_roll, ts_ns)
            d2 = _time_derivative(d1, ts_ns)
            d3 = _time_derivative(d2, ts_ns)

            d1_z = online_robust_zscore(d1, normalization_window, normalization_min_periods)
            d2_z = online_robust_zscore(d2, normalization_window, normalization_min_periods)
            d3_z = online_robust_zscore(d3, normalization_window, normalization_min_periods)

            score = (
                0.55 * np.tanh(np.nan_to_num(d1_z, nan=0.0) / 3.0)
                + 0.30 * np.tanh(np.nan_to_num(d2_z, nan=0.0) / 3.0)
                + 0.15 * np.tanh(np.nan_to_num(d3_z, nan=0.0) / 3.0)
            )
            state = _state_from_score(score, spectrum_threshold)

            prefix = f"{side_name}_{band}"
            out[f"{prefix}_pressure"] = p_layer
            out[f"{prefix}_vacuum"] = v_layer
            out[f"{prefix}_composite"] = c_layer
            out[f"{prefix}_rolled"] = c_roll
            out[f"{prefix}_d1"] = d1
            out[f"{prefix}_d2"] = d2
            out[f"{prefix}_d3"] = d3
            out[f"{prefix}_score"] = score
            out[f"{prefix}_state"] = state

            if side_name == "above":
                above_score_cols.append(f"{prefix}_score")
            else:
                below_score_cols.append(f"{prefix}_score")

    above_scores = out[above_score_cols].values.astype(np.float64)
    below_scores = out[below_score_cols].values.astype(np.float64)
    out["above_side_score"] = above_scores @ band_weights
    out["below_side_score"] = below_scores @ band_weights
    out["above_side_state"] = _state_from_score(out["above_side_score"].values, spectrum_threshold)
    out["below_side_state"] = _state_from_score(out["below_side_score"].values, spectrum_threshold)

    direction_edge = out["below_side_score"].values - out["above_side_score"].values
    out["direction_edge"] = direction_edge

    direction_state = np.full(n, DIRECTION_FLAT, dtype=object)
    direction_state[direction_edge >= directional_edge_threshold] = DIRECTION_UP
    direction_state[direction_edge <= -directional_edge_threshold] = DIRECTION_DOWN
    out["direction_state"] = direction_state

    posture = np.full(n, "transition", dtype=object)
    above_state = out["above_side_state"].values.astype(str)
    below_state = out["below_side_state"].values.astype(str)
    posture[(above_state == SPECTRUM_VACUUM) & (below_state == SPECTRUM_PRESSURE)] = "bullish_release"
    posture[(above_state == SPECTRUM_PRESSURE) & (below_state == SPECTRUM_VACUUM)] = "bearish_release"
    posture[(above_state == SPECTRUM_VACUUM) & (below_state == SPECTRUM_VACUUM)] = "two_sided_vacuum"
    posture[(above_state == SPECTRUM_PRESSURE) & (below_state == SPECTRUM_PRESSURE)] = "two_sided_pressure"
    out["posture_state"] = posture

    return out


def detect_direction_switch_events(
    eval_mask: np.ndarray,
    direction_state: np.ndarray,
    cooldown_snapshots: int,
) -> List[Dict[str, Any]]:
    """Detect directional switch events from up/down/flat regime changes."""
    if len(eval_mask) != len(direction_state):
        raise ValueError("eval_mask and direction_state lengths must match.")
    if cooldown_snapshots < 0:
        raise ValueError("cooldown_snapshots must be >= 0.")

    events: List[Dict[str, Any]] = []
    last_event_idx = -10_000_000
    prev_state = DIRECTION_FLAT

    for i in np.where(eval_mask)[0]:
        cur = str(direction_state[i])
        if (
            cur in (DIRECTION_UP, DIRECTION_DOWN)
            and cur != prev_state
            and (i - last_event_idx) >= cooldown_snapshots
        ):
            events.append({
                "event_idx": int(i),
                "direction": cur,
            })
            last_event_idx = i
        prev_state = cur
    return events


def evaluate_trade_targets(
    ts_ns: np.ndarray,
    mid_price: np.ndarray,
    events: List[Dict[str, Any]],
    tick_size: float,
    tp_ticks: int,
    sl_ticks: int,
    max_hold_snapshots: int,
) -> pd.DataFrame:
    """Evaluate TP/SL/timeout outcomes for directional switch events."""
    if tick_size <= 0.0:
        raise ValueError(f"tick_size must be > 0, got {tick_size}")
    if tp_ticks < 1 or sl_ticks < 1:
        raise ValueError("tp_ticks and sl_ticks must be >= 1.")
    if max_hold_snapshots < 1:
        raise ValueError("max_hold_snapshots must be >= 1.")

    n = len(mid_price)
    rows: List[Dict[str, Any]] = []

    for ev in events:
        i = int(ev["event_idx"])
        direction = str(ev["direction"])
        if i < 0 or i >= n:
            continue

        entry_price = float(mid_price[i])
        tp_delta = float(tp_ticks) * tick_size
        sl_delta = float(sl_ticks) * tick_size

        if direction == DIRECTION_UP:
            tp_price = entry_price + tp_delta
            sl_price = entry_price - sl_delta
        elif direction == DIRECTION_DOWN:
            tp_price = entry_price - tp_delta
            sl_price = entry_price + sl_delta
        else:
            continue

        exit_idx = min(n - 1, i + max_hold_snapshots)
        outcome = "timeout"
        resolved_idx = exit_idx
        exit_price = float(mid_price[exit_idx])

        for j in range(i + 1, exit_idx + 1):
            px = float(mid_price[j])
            if direction == DIRECTION_UP:
                if px >= tp_price:
                    outcome = "tp_before_sl"
                    resolved_idx = j
                    exit_price = px
                    break
                if px <= sl_price:
                    outcome = "sl_before_tp"
                    resolved_idx = j
                    exit_price = px
                    break
            else:
                if px <= tp_price:
                    outcome = "tp_before_sl"
                    resolved_idx = j
                    exit_price = px
                    break
                if px >= sl_price:
                    outcome = "sl_before_tp"
                    resolved_idx = j
                    exit_price = px
                    break

        rows.append({
            "event_idx": int(i),
            "direction": direction,
            "outcome": outcome,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "exit_idx": int(resolved_idx),
            "hold_snapshots": int(resolved_idx - i),
            "time_to_outcome_ms": max(0.0, (ts_ns[resolved_idx] - ts_ns[i]) / 1e6),
        })

    return pd.DataFrame(rows)


def summarize_trade_metrics(
    outcomes: pd.DataFrame,
    duration_hours: float,
) -> Dict[str, float]:
    """Summarize trader-style TP/SL outcome metrics."""
    n = int(len(outcomes))
    if n == 0:
        return {
            "n_events": 0.0,
            "events_per_hour": 0.0,
            "tp_before_sl_rate": np.nan,
            "sl_before_tp_rate": np.nan,
            "timeout_rate": np.nan,
            "median_time_to_outcome_ms": np.nan,
        }

    tp = int((outcomes["outcome"] == "tp_before_sl").sum())
    sl = int((outcomes["outcome"] == "sl_before_tp").sum())
    timeout = int((outcomes["outcome"] == "timeout").sum())

    return {
        "n_events": float(n),
        "events_per_hour": float(n / max(duration_hours, 1e-9)),
        "tp_before_sl_rate": float(tp / n),
        "sl_before_tp_rate": float(sl / n),
        "timeout_rate": float(timeout / n),
        "median_time_to_outcome_ms": float(np.median(outcomes["time_to_outcome_ms"].values)),
    }


def _hour_windows_ns(
    dt: str,
    eval_start: str,
    eval_end: str,
) -> List[Tuple[str, int, int]]:
    """Build contiguous hourly ET windows clipped to eval boundaries."""
    start = pd.Timestamp(f"{dt} {eval_start}:00", tz="America/New_York")
    end = pd.Timestamp(f"{dt} {eval_end}:00", tz="America/New_York")
    if end <= start:
        raise ValueError(f"Evaluation end must be after start: {eval_start} -> {eval_end}")

    windows: List[Tuple[str, int, int]] = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(hours=1), end)
        label = f"{cur.strftime('%H:%M')}-{nxt.strftime('%H:%M')}"
        windows.append((label, int(cur.tz_convert("UTC").value), int(nxt.tz_convert("UTC").value)))
        cur = nxt
    return windows


def summarize_hourly_trade_metrics(
    ts_ns: np.ndarray,
    outcomes: pd.DataFrame,
    dt: str,
    eval_start: str,
    eval_end: str,
) -> List[Dict[str, float]]:
    """Summarize TP/SL outcomes across hourly subwindows."""
    windows = _hour_windows_ns(dt, eval_start, eval_end)
    rows: List[Dict[str, float]] = []

    if len(outcomes) == 0:
        for label, start_ns, end_ns in windows:
            duration_h = max(1e-9, (end_ns - start_ns) / 3.6e12)
            m = summarize_trade_metrics(pd.DataFrame(), duration_h)
            m["window"] = label
            rows.append(m)
        return rows

    event_idx = outcomes["event_idx"].values.astype(int)
    event_ts_ns = ts_ns[event_idx]
    for label, start_ns, end_ns in windows:
        msk = (event_ts_ns >= start_ns) & (event_ts_ns < end_ns)
        sub = outcomes.loc[msk].copy()
        duration_h = max(1e-9, (end_ns - start_ns) / 3.6e12)
        m = summarize_trade_metrics(sub, duration_h)
        m["window"] = label
        rows.append(m)
    return rows


def evaluate_hourly_stability(
    hourly_metrics: List[Dict[str, float]],
    max_drift: float,
    min_signals_per_hour: int,
) -> Dict[str, Any]:
    """Compute hourly consistency gate over event rate and TP hit rate."""
    if max_drift < 0.0:
        raise ValueError("max_drift must be >= 0.")
    if min_signals_per_hour < 0:
        raise ValueError("min_signals_per_hour must be >= 0.")

    reasons: List[str] = []
    passed = True
    for row in hourly_metrics:
        if int(row["n_events"]) < min_signals_per_hour:
            passed = False
            reasons.append(
                f"{row['window']}: n_events={int(row['n_events'])} < min_signals_per_hour={min_signals_per_hour}"
            )

    def _rel_drift(values: List[float]) -> float:
        arr = np.array(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) <= 1:
            return np.nan
        baseline = max(abs(float(arr.mean())), 1e-9)
        return float((arr.max() - arr.min()) / baseline)

    tp_drift = _rel_drift([float(r["tp_before_sl_rate"]) for r in hourly_metrics])
    event_rate_drift = _rel_drift([float(r["events_per_hour"]) for r in hourly_metrics])

    if np.isfinite(tp_drift) and tp_drift > max_drift:
        passed = False
        reasons.append(f"tp_before_sl_rate drift {tp_drift:.3f} exceeds max {max_drift:.3f}")
    if np.isfinite(event_rate_drift) and event_rate_drift > max_drift:
        passed = False
        reasons.append(f"events_per_hour drift {event_rate_drift:.3f} exceeds max {max_drift:.3f}")

    return {
        "passed": bool(passed),
        "tp_before_sl_rate_drift": float(tp_drift) if np.isfinite(tp_drift) else np.nan,
        "events_per_hour_drift": float(event_rate_drift) if np.isfinite(event_rate_drift) else np.nan,
        "reasons": reasons,
    }


def _print_instrument_header(
    product_type: str,
    symbol: str,
    dt: str,
    start_time: str,
    eval_start: str,
    end_time_et: str,
    throttle_ms: float,
    tick_size: float,
    normalization_window: int,
) -> None:
    print()
    print("=" * 120)
    print("  VP DERIVATIVE MICRO-REGIME ANALYSIS")
    print("=" * 120)
    print(f"  Instrument:         {product_type}:{symbol}")
    print(f"  Date:               {dt}")
    print(f"  Capture:            {start_time} - {end_time_et} ET")
    print(f"  Evaluation:         {eval_start} - {end_time_et} ET")
    print(f"  Throttle:           {throttle_ms}ms")
    print(f"  Tick size:          {tick_size}")
    print(f"  Normalization win:  {normalization_window} snapshots")
    print("=" * 120)


def _print_directional_metrics(
    directional_df: pd.DataFrame,
    eval_mask: np.ndarray,
    directional_bands: List[int],
    trade_metrics: Dict[str, float],
    hourly_metrics: List[Dict[str, float]],
    stability: Dict[str, Any],
    tp_ticks: int,
    sl_ticks: int,
    max_hold_snapshots: int,
) -> None:
    last_eval_idx = int(np.where(eval_mask)[0][-1])
    print("\n  Directional Spectrum (last eval snapshot)")
    print("  " + "-" * 88)
    for band in directional_bands:
        a_state = directional_df.loc[last_eval_idx, f"above_{band}_state"]
        b_state = directional_df.loc[last_eval_idx, f"below_{band}_state"]
        a_score = float(directional_df.loc[last_eval_idx, f"above_{band}_score"])
        b_score = float(directional_df.loc[last_eval_idx, f"below_{band}_score"])
        print(
            f"  +/-{band:>2d}  above={a_state:>10s} ({a_score:+0.3f})"
            f"   below={b_state:>10s} ({b_score:+0.3f})"
        )
    print(f"  side above: {directional_df.loc[last_eval_idx, 'above_side_state']:>10s} ({float(directional_df.loc[last_eval_idx, 'above_side_score']):+0.3f})")
    print(f"  side below: {directional_df.loc[last_eval_idx, 'below_side_state']:>10s} ({float(directional_df.loc[last_eval_idx, 'below_side_score']):+0.3f})")
    print(f"  posture:    {directional_df.loc[last_eval_idx, 'posture_state']}")
    print(f"  direction:  {directional_df.loc[last_eval_idx, 'direction_state']} (edge={float(directional_df.loc[last_eval_idx, 'direction_edge']):+0.3f})")

    print("\n  Trade-Style Evaluation (eval window)")
    print("  " + "-" * 88)
    print(f"  Target / Stop (ticks):      +{tp_ticks} / -{sl_ticks}")
    print(f"  Max hold (snapshots):       {max_hold_snapshots}")
    print(f"  Direction switch events:    {int(trade_metrics['n_events'])}")
    print(f"  Events / hour:              {trade_metrics['events_per_hour']:.3f}")
    print(f"  TP before SL rate:          {trade_metrics['tp_before_sl_rate']:.3f}")
    print(f"  SL before TP rate:          {trade_metrics['sl_before_tp_rate']:.3f}")
    print(f"  Timeout rate:               {trade_metrics['timeout_rate']:.3f}")
    print(f"  Median time to outcome ms:  {trade_metrics['median_time_to_outcome_ms']:.2f}")

    print("\n  Hourly Stability")
    print("  " + "-" * 88)
    print(f"  Gate pass:                  {stability['passed']}")
    print(f"  TP-rate drift:              {stability['tp_before_sl_rate_drift']:.3f}")
    print(f"  Event-rate drift:           {stability['events_per_hour_drift']:.3f}")
    if stability["reasons"]:
        for reason in stability["reasons"]:
            print(f"  Gate reason:                {reason}")

    print(f"  {'Window':>12s} {'Events':>10s} {'TP<SL':>10s} {'SL<TP':>10s} {'Timeout':>10s}")
    for row in hourly_metrics:
        tp = row["tp_before_sl_rate"]
        sl = row["sl_before_tp_rate"]
        tout = row["timeout_rate"]
        tp_s = f"{tp:.3f}" if np.isfinite(tp) else "nan"
        sl_s = f"{sl:.3f}" if np.isfinite(sl) else "nan"
        to_s = f"{tout:.3f}" if np.isfinite(tout) else "nan"
        print(
            f"  {row['window']:>12s} {int(row['n_events']):>10d} {tp_s:>10s} {sl_s:>10s} {to_s:>10s}"
        )
    print()


def _print_projection_summary(
    projection: Dict[str, Any],
    selected_buckets: List[int],
) -> None:
    """Print per-bucket forward projection composites for selected buckets."""
    current_last = projection["current_last"]
    proj_last = projection["projected_last_by_horizon"]
    horizon_metrics = projection["horizon_metrics"]

    print("  Forward Projection (last eval snapshot)")
    print("  " + "-" * 88)
    print("  Composite color scale: +1 pressure / 0 chop / -1 vacuum")

    for k in selected_buckets:
        idx = k + K_MAX
        if idx < 0 or idx >= len(current_last):
            print(f"  k={k:+d}: out of range for K={K_MAX}, skipped.")
            continue
        side = "ABOVE" if k > 0 else "BELOW" if k < 0 else "SPOT"
        print(f"  k={k:+d} ({side})")
        print(f"    current: {float(current_last[idx]):+0.4f}")
        for h_ms in sorted(proj_last.keys()):
            print(f"    {h_ms:>5d}ms: {float(proj_last[h_ms][idx]):+0.4f}")

    print("\n  Projection Directional Consistency (eval window)")
    print("  " + "-" * 88)
    print(f"  {'Horizon':>10s} {'Delta-Sign Hit':>16s} {'Pairs':>10s}")
    for h_ms in sorted(horizon_metrics.keys()):
        m = horizon_metrics[h_ms]
        hit = m["delta_sign_hit_rate"]
        hit_str = f"{hit:0.3f}" if np.isfinite(hit) else "nan"
        print(f"  {h_ms:>7d}ms {hit_str:>16s} {int(m['n_pairs']):>10d}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derivative-only micro-regime analysis for VP dense grids.",
    )
    parser.add_argument("--mode", default="regime", choices=["regime"], help="Analysis mode.")
    parser.add_argument("--product-type", default="future_mbo", help="Product type for single-instrument mode.")
    parser.add_argument("--symbol", default="MNQH6", help="Symbol for single-instrument mode.")
    parser.add_argument("--dt", default="2026-02-06", help="Date YYYY-MM-DD.")
    parser.add_argument("--start-time", default="09:00", help="Capture start HH:MM ET.")
    parser.add_argument("--eval-start", default="09:00", help="Evaluation start HH:MM ET.")
    parser.add_argument("--eval-end", default="12:00", help="Evaluation end HH:MM ET.")
    parser.add_argument("--eval-minutes", type=int, default=None, help="Fallback eval duration if --eval-end is omitted.")
    parser.add_argument("--throttle-ms", type=float, default=25.0, help="Grid throttle in milliseconds.")

    parser.add_argument("--normalization-window", type=int, default=300, help="Trailing window for robust z-score.")
    parser.add_argument(
        "--normalization-min-periods",
        type=int,
        default=75,
        help="Minimum trailing samples for robust z-score.",
    )
    parser.add_argument(
        "--directional-bands",
        default="4,8,16",
        help="Comma-separated symmetric tick bands for above/below side rollups.",
    )
    parser.add_argument(
        "--micro-windows",
        default="25,50,100,200",
        help="Comma-separated trailing windows for derivative rollups.",
    )
    parser.add_argument("--spectrum-threshold", type=float, default=0.15, help="Threshold for pressure/transition/vacuum spectrum states.")
    parser.add_argument("--directional-edge-threshold", type=float, default=0.20, help="Threshold for up/down directional edge state.")
    parser.add_argument("--signal-cooldown", type=int, default=8, help="Minimum snapshots between directional switch events.")
    parser.add_argument("--tp-ticks", type=int, default=8, help="Take-profit ticks from signal event.")
    parser.add_argument("--sl-ticks", type=int, default=4, help="Stop-loss ticks from signal event.")
    parser.add_argument("--max-hold-snapshots", type=int, default=1200, help="Maximum snapshots to hold event before timeout.")
    parser.add_argument("--stability-max-drift", type=float, default=0.35, help="Max allowed relative hourly drift for key metrics.")
    parser.add_argument("--stability-min-signals-per-hour", type=int, default=5, help="Minimum event count required in each hourly bucket.")

    parser.add_argument(
        "--projection-horizons-ms",
        default="250,500,1000,2500",
        help="Comma-separated projection horizons in milliseconds.",
    )
    parser.add_argument(
        "--projection-rollup-windows",
        default="8,32,96",
        help="Comma-separated trailing rollup windows in snapshots for derivative rollups.",
    )
    parser.add_argument(
        "--projection-buckets",
        default="-8,8",
        help="Comma-separated relative bucket indices (k) to print in projection summary.",
    )
    parser.add_argument("--json-output", default=None, help="Optional path to write machine-readable metrics JSON.")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.normalization_window < 2:
        raise ValueError("--normalization-window must be >= 2.")
    if args.normalization_min_periods < 2:
        raise ValueError("--normalization-min-periods must be >= 2.")
    if args.normalization_min_periods > args.normalization_window:
        raise ValueError("--normalization-min-periods cannot exceed --normalization-window.")
    if args.spectrum_threshold <= 0.0:
        raise ValueError("--spectrum-threshold must be > 0.")
    if args.directional_edge_threshold <= 0.0:
        raise ValueError("--directional-edge-threshold must be > 0.")
    if args.signal_cooldown < 0:
        raise ValueError("--signal-cooldown must be >= 0.")
    if args.tp_ticks < 1:
        raise ValueError("--tp-ticks must be >= 1.")
    if args.sl_ticks < 1:
        raise ValueError("--sl-ticks must be >= 1.")
    if args.max_hold_snapshots < 1:
        raise ValueError("--max-hold-snapshots must be >= 1.")
    if args.stability_max_drift < 0.0:
        raise ValueError("--stability-max-drift must be >= 0.")
    if args.stability_min_signals_per_hour < 0:
        raise ValueError("--stability-min-signals-per-hour must be >= 0.")

    projection_horizons_ms = parse_int_list(
        args.projection_horizons_ms, "--projection-horizons-ms", min_value=1
    )
    projection_rollup_windows = parse_int_list(
        args.projection_rollup_windows, "--projection-rollup-windows", min_value=1
    )
    projection_buckets = parse_int_list(
        args.projection_buckets, "--projection-buckets", min_value=-K_MAX
    )
    projection_buckets = [k for k in projection_buckets if -K_MAX <= k <= K_MAX]
    if not projection_buckets:
        raise ValueError(
            f"--projection-buckets must include at least one bucket in range [-{K_MAX}, {K_MAX}]."
        )
    directional_bands = parse_int_list(
        args.directional_bands, "--directional-bands", min_value=1
    )
    directional_bands = [b for b in directional_bands if b <= K_MAX]
    if not directional_bands:
        raise ValueError(f"--directional-bands must contain values <= {K_MAX}.")
    micro_windows = parse_int_list(
        args.micro_windows, "--micro-windows", min_value=1
    )

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"
    lake_root = backend_root / "lake"

    from src.vacuum_pressure.config import resolve_config

    end_time_et = _resolve_eval_end_time(args.eval_start, args.eval_end, args.eval_minutes)
    eval_start_ns = _parse_et_timestamp_ns(args.dt, args.eval_start)
    eval_end_ns = _parse_et_timestamp_ns(args.dt, end_time_et)
    if eval_end_ns <= eval_start_ns:
        raise ValueError("--eval-end must be later than --eval-start.")

    t0 = time.monotonic()
    config = resolve_config(args.product_type, args.symbol, products_yaml_path)
    if config.tick_size <= 0.0:
        raise ValueError(
            f"Resolved tick_size must be > 0 for {args.product_type}:{args.symbol}, got {config.tick_size}"
        )

    _print_instrument_header(
        product_type=args.product_type,
        symbol=args.symbol,
        dt=args.dt,
        start_time=args.start_time,
        eval_start=args.eval_start,
        end_time_et=end_time_et,
        throttle_ms=args.throttle_ms,
        tick_size=float(config.tick_size),
        normalization_window=args.normalization_window,
    )

    t_capture = time.monotonic()
    ts_ns, mid_price, bucket_data = capture_grids(
        lake_root=lake_root,
        config=config,
        dt=args.dt,
        start_time=args.start_time,
        throttle_ms=args.throttle_ms,
        end_time_et=end_time_et,
    )
    capture_elapsed = time.monotonic() - t_capture

    n = len(ts_ns)
    if n < 200:
        raise RuntimeError(
            f"Insufficient snapshots for {args.product_type}:{args.symbol} on {args.dt}. "
            f"Need >=200, got {n}. Adjust --start-time/--eval-start/--eval-end."
        )

    eval_mask = (ts_ns >= eval_start_ns) & (ts_ns < eval_end_ns)
    n_eval = int(eval_mask.sum())
    if n_eval < 50:
        raise RuntimeError(
            f"Insufficient evaluation snapshots for {args.product_type}:{args.symbol}. "
            f"Need >=50, got {n_eval}. Adjust --eval-start/--eval-end."
        )

    print(f"\n  Capture complete: {n:,} snapshots in {capture_elapsed:.2f}s")
    print(f"  Evaluation snapshots: {n_eval:,}")
    print(
        f"  Mid range: ${mid_price.min():.2f} - ${mid_price.max():.2f} "
        f"({(mid_price.max() - mid_price.min()) / config.tick_size:.1f} ticks)"
    )

    t_dir = time.monotonic()
    directional_df = compute_directional_spectrum(
        ts_ns=ts_ns,
        bucket_data=bucket_data,
        directional_bands=directional_bands,
        micro_windows=micro_windows,
        normalization_window=args.normalization_window,
        normalization_min_periods=args.normalization_min_periods,
        spectrum_threshold=args.spectrum_threshold,
        directional_edge_threshold=args.directional_edge_threshold,
    )
    dir_elapsed = time.monotonic() - t_dir
    print(f"  Directional spectrum compute: {dir_elapsed:.2f}s ({len(directional_df.columns)} columns)")

    events = detect_direction_switch_events(
        eval_mask=eval_mask,
        direction_state=directional_df["direction_state"].values,
        cooldown_snapshots=args.signal_cooldown,
    )
    outcomes = evaluate_trade_targets(
        ts_ns=ts_ns,
        mid_price=mid_price,
        events=events,
        tick_size=float(config.tick_size),
        tp_ticks=args.tp_ticks,
        sl_ticks=args.sl_ticks,
        max_hold_snapshots=args.max_hold_snapshots,
    )

    eval_duration_hours = max(1e-9, (eval_end_ns - eval_start_ns) / 3.6e12)
    trade_metrics = summarize_trade_metrics(outcomes, eval_duration_hours)
    hourly_metrics = summarize_hourly_trade_metrics(
        ts_ns=ts_ns,
        outcomes=outcomes,
        dt=args.dt,
        eval_start=args.eval_start,
        eval_end=end_time_et,
    )
    stability = evaluate_hourly_stability(
        hourly_metrics=hourly_metrics,
        max_drift=args.stability_max_drift,
        min_signals_per_hour=args.stability_min_signals_per_hour,
    )
    _print_directional_metrics(
        directional_df=directional_df,
        eval_mask=eval_mask,
        directional_bands=directional_bands,
        trade_metrics=trade_metrics,
        hourly_metrics=hourly_metrics,
        stability=stability,
        tp_ticks=args.tp_ticks,
        sl_ticks=args.sl_ticks,
        max_hold_snapshots=args.max_hold_snapshots,
    )

    projection = compute_bucket_forward_projection(
        ts_ns=ts_ns,
        bucket_data=bucket_data,
        eval_mask=eval_mask,
        projection_horizons_ms=projection_horizons_ms,
        projection_rollup_windows=projection_rollup_windows,
    )
    _print_projection_summary(
        projection=projection,
        selected_buckets=projection_buckets,
    )

    if args.json_output:
        import json

        payload = {
            "instrument": {
                "product_type": args.product_type,
                "symbol": args.symbol,
                "dt": args.dt,
            },
            "window": {
                "start_time": args.start_time,
                "eval_start": args.eval_start,
                "eval_end": end_time_et,
            },
            "directional_params": {
                "directional_bands": directional_bands,
                "micro_windows": micro_windows,
                "spectrum_threshold": args.spectrum_threshold,
                "directional_edge_threshold": args.directional_edge_threshold,
                "signal_cooldown": args.signal_cooldown,
            },
            "trade_target_params": {
                "tp_ticks": args.tp_ticks,
                "sl_ticks": args.sl_ticks,
                "max_hold_snapshots": args.max_hold_snapshots,
            },
            "trade_metrics": trade_metrics,
            "hourly_metrics": hourly_metrics,
            "stability": stability,
            "projection_horizon_metrics": projection["horizon_metrics"],
            "event_count": int(len(events)),
        }
        out_path = Path(args.json_output).expanduser().resolve()
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Metrics JSON: {out_path}")

    total_elapsed = time.monotonic() - t0
    print(f"Total analysis time: {total_elapsed:.2f}s\n")


if __name__ == "__main__":
    main()
