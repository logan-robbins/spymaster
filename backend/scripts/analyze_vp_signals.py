"""Predictive power analysis of vacuum-pressure derivative signals.

Evaluates spatial imbalance, depth asymmetry, velocity/acceleration tilts,
and jerk features from the VP dense grid for directional mid-price prediction.

Usage:
    cd backend
    uv run scripts/analyze_vp_signals.py
    uv run scripts/analyze_vp_signals.py --start-time 09:30 --eval-start 09:50 --eval-minutes 5

Methodology:
    1. Capture all VP grid snapshots over a training + evaluation window.
    2. Engineer spatial aggregation features per snapshot (tilt/imbalance).
    3. Apply temporal smoothing via EWM at multiple lookback horizons.
    4. Compute z-scored signals and rank-IC against forward returns.
    5. Evaluate composite signal on held-out evaluation window.

All computations are vectorized (numpy/pandas). No Python loops over snapshots.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup (same pattern as warm_cache.py / run_vacuum_pressure.py)
# ---------------------------------------------------------------------------
backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

logger = logging.getLogger("analyze_vp_signals")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_SIZE_DOLLARS: float = 0.25
"""MNQ tick size in dollars. Forward returns are reported in ticks."""

N_BUCKETS: int = 101
"""Dense grid width: 2*50 + 1 = 101 buckets (k = -50 .. +50)."""

K_MAX: int = 50
"""Grid half-width."""

# Spatial zone boundaries (inclusive)
NEAR_MAX: int = 5
MID_MAX: int = 20
FAR_MAX: int = 50

# EWM lookback windows (in snapshots)
LOOKBACK_WINDOWS: List[int] = [5, 15, 50, 150]

# Forward return horizons (in snapshots)
FORWARD_HORIZONS: List[int] = [25, 100, 500]

# Bucket field names in the grid dict
PRESSURE_FIELD: str = "pressure_variant"
VACUUM_FIELD: str = "vacuum_variant"
REST_DEPTH_FIELD: str = "rest_depth"

VELOCITY_FIELDS: List[str] = ["v_add", "v_pull", "v_fill", "v_rest_depth"]
ACCEL_FIELDS: List[str] = ["a_add", "a_pull", "a_fill", "a_rest_depth"]
JERK_FIELDS: List[str] = ["j_add", "j_pull", "j_fill", "j_rest_depth"]


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
            bucket_data: (N, 101, n_fields) float64 array of bucket fields.
                Fields ordered as: pressure_variant, vacuum_variant, rest_depth,
                v_add, v_pull, v_fill, v_rest_depth,
                a_add, a_pull, a_fill, a_rest_depth,
                j_add, j_pull, j_fill, j_rest_depth
    """
    from src.vacuum_pressure.stream_pipeline import stream_events

    # Compute end boundary in nanoseconds
    end_ns: int = 0
    if end_time_et:
        et_end = pd.Timestamp(f"{dt} {end_time_et}:00", tz="America/New_York").tz_convert("UTC")
        end_ns = int(et_end.value)

    # Field extraction order for bucket_data
    bucket_fields = [
        PRESSURE_FIELD, VACUUM_FIELD, REST_DEPTH_FIELD,
        *VELOCITY_FIELDS, *ACCEL_FIELDS, *JERK_FIELDS,
    ]
    n_fields = len(bucket_fields)

    # Pre-allocate with generous capacity, then trim
    capacity = 80_000
    ts_ns_list = np.empty(capacity, dtype=np.int64)
    mid_price_list = np.empty(capacity, dtype=np.float64)
    bucket_list = np.empty((capacity, N_BUCKETS, n_fields), dtype=np.float64)

    count = 0
    t_start = time.monotonic()

    for _event_id, grid in stream_events(
        lake_root=lake_root,
        config=config,
        dt=dt,
        start_time=start_time,
        throttle_ms=throttle_ms,
    ):
        ts_ns = grid["ts_ns"]

        # Stop if past end boundary
        if end_ns > 0 and ts_ns >= end_ns:
            break

        # Skip invalid book states
        if not grid["book_valid"]:
            continue

        mid = grid["mid_price"]
        if mid <= 0.0:
            continue

        # Grow arrays if needed
        if count >= capacity:
            capacity = int(capacity * 1.5)
            ts_ns_list = np.resize(ts_ns_list, capacity)
            mid_price_list = np.resize(mid_price_list, capacity)
            new_bucket = np.empty((capacity, N_BUCKETS, n_fields), dtype=np.float64)
            new_bucket[:count] = bucket_list[:count]
            bucket_list = new_bucket

        ts_ns_list[count] = ts_ns
        mid_price_list[count] = mid

        # Extract bucket fields — buckets are pre-sorted by k (-50..+50)
        buckets = grid["buckets"]
        for i, b in enumerate(buckets):
            for j, field_name in enumerate(bucket_fields):
                bucket_list[count, i, j] = b[field_name]

        count += 1
        if count % 10000 == 0:
            elapsed = time.monotonic() - t_start
            logger.info(
                "Captured %d snapshots (%.1fs, %.0f snap/s)",
                count, elapsed, count / elapsed,
            )

    elapsed = time.monotonic() - t_start
    logger.info(
        "Capture complete: %d snapshots in %.2fs (%.0f snap/s)",
        count, elapsed, count / elapsed if elapsed > 0 else 0,
    )

    return (
        ts_ns_list[:count].copy(),
        mid_price_list[:count].copy(),
        bucket_list[:count].copy(),
    )


# ---------------------------------------------------------------------------
# Feature engineering — spatial aggregation
# ---------------------------------------------------------------------------

# Field index mapping (must match bucket_fields order in capture_grids)
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


def _build_zone_masks() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build boolean masks for bid/ask sides across spatial zones.

    Returns:
        Dict mapping zone name -> (bid_mask, ask_mask) where each is
        a (101,) boolean array indexing into the bucket dimension.
        k < 0 = bid side (below spot), k > 0 = ask side (above spot).
    """
    k_arr = np.arange(-K_MAX, K_MAX + 1)  # shape (101,)

    zones: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Full grid (excluding k=0)
    zones["full"] = (k_arr < 0, k_arr > 0)

    # Near field: |k| <= 5
    zones["near"] = (
        (k_arr < 0) & (np.abs(k_arr) <= NEAR_MAX),
        (k_arr > 0) & (np.abs(k_arr) <= NEAR_MAX),
    )

    # Mid field: 5 < |k| <= 20
    zones["mid"] = (
        (k_arr < 0) & (np.abs(k_arr) > NEAR_MAX) & (np.abs(k_arr) <= MID_MAX),
        (k_arr > 0) & (np.abs(k_arr) > NEAR_MAX) & (np.abs(k_arr) <= MID_MAX),
    )

    # Far field: 20 < |k| <= 50
    zones["far"] = (
        (k_arr < 0) & (np.abs(k_arr) > MID_MAX) & (np.abs(k_arr) <= FAR_MAX),
        (k_arr > 0) & (np.abs(k_arr) > MID_MAX) & (np.abs(k_arr) <= FAR_MAX),
    )

    return zones


def _build_k_weights() -> np.ndarray:
    """Build 1/max(1,|k|) weight array for k-weighted features.

    Returns:
        (101,) float64 array.
    """
    k_arr = np.arange(-K_MAX, K_MAX + 1, dtype=np.float64)
    return 1.0 / np.maximum(1.0, np.abs(k_arr))


def compute_spatial_features(
    bucket_data: np.ndarray,
) -> pd.DataFrame:
    """Compute all spatial aggregation features across snapshots.

    All operations are fully vectorized over the snapshot dimension.

    Args:
        bucket_data: (N, 101, n_fields) array from capture_grids.

    Returns:
        DataFrame with N rows and one column per spatial feature.
        Feature semantics:
            Positive = bullish signal (upward price pressure).
            Negative = bearish signal (downward price pressure).
    """
    N = bucket_data.shape[0]
    zones = _build_zone_masks()
    k_weights = _build_k_weights()  # (101,)

    features: Dict[str, np.ndarray] = {}

    # --- Pressure / Vacuum imbalance ---
    # pv_net_edge = pressure_below - pressure_above + vacuum_above - vacuum_below
    # Interpretation: net bullish = support building + resistance draining
    P = bucket_data[:, :, _FIELD_IDX["pressure_variant"]]  # (N, 101)
    V = bucket_data[:, :, _FIELD_IDX["vacuum_variant"]]    # (N, 101)

    for zone_name, (bid_mask, ask_mask) in zones.items():
        suffix = "" if zone_name == "full" else f"_{zone_name}"

        # PV net edge: pressure below + vacuum above - pressure above - vacuum below
        pv_net = (
            P[:, bid_mask].sum(axis=1)
            - P[:, ask_mask].sum(axis=1)
            + V[:, ask_mask].sum(axis=1)
            - V[:, bid_mask].sum(axis=1)
        )
        features[f"pv_net_edge{suffix}"] = pv_net

    # K-weighted PV net edge (full grid only)
    bid_full, ask_full = zones["full"]
    pv_net_kw = (
        (P[:, bid_full] * k_weights[bid_full]).sum(axis=1)
        - (P[:, ask_full] * k_weights[ask_full]).sum(axis=1)
        + (V[:, ask_full] * k_weights[ask_full]).sum(axis=1)
        - (V[:, bid_full] * k_weights[bid_full]).sum(axis=1)
    )
    features["pv_net_edge_kw"] = pv_net_kw

    # --- Depth asymmetry ---
    D = bucket_data[:, :, _FIELD_IDX["rest_depth"]]  # (N, 101)

    for zone_name, (bid_mask, ask_mask) in zones.items():
        suffix = "" if zone_name == "full" else f"_{zone_name}"

        # depth_tilt: bid depth - ask depth (more bid depth = bullish)
        depth_tilt = D[:, bid_mask].sum(axis=1) - D[:, ask_mask].sum(axis=1)
        features[f"depth_tilt{suffix}"] = depth_tilt

    # --- Velocity tilts ---
    # v_add_tilt: bid add velocity excess (more adding below = bullish)
    v_add = bucket_data[:, :, _FIELD_IDX["v_add"]]
    features["v_add_tilt"] = (
        v_add[:, bid_full].sum(axis=1) - v_add[:, ask_full].sum(axis=1)
    )

    # v_pull_tilt: ask pull excess (more pulling above = bullish — resistance leaving)
    v_pull = bucket_data[:, :, _FIELD_IDX["v_pull"]]
    features["v_pull_tilt"] = (
        v_pull[:, ask_full].sum(axis=1) - v_pull[:, bid_full].sum(axis=1)
    )

    # v_fill_tilt: ask fill excess (fills eating asks = bullish)
    v_fill = bucket_data[:, :, _FIELD_IDX["v_fill"]]
    features["v_fill_tilt"] = (
        v_fill[:, ask_full].sum(axis=1) - v_fill[:, bid_full].sum(axis=1)
    )

    # v_depth_tilt: bid rest_depth velocity excess (depth growing below = bullish)
    v_rest = bucket_data[:, :, _FIELD_IDX["v_rest_depth"]]
    features["v_depth_tilt"] = (
        v_rest[:, bid_full].sum(axis=1) - v_rest[:, ask_full].sum(axis=1)
    )

    # --- Acceleration tilts (same structure as velocity) ---
    a_add = bucket_data[:, :, _FIELD_IDX["a_add"]]
    features["a_add_tilt"] = (
        a_add[:, bid_full].sum(axis=1) - a_add[:, ask_full].sum(axis=1)
    )

    a_pull = bucket_data[:, :, _FIELD_IDX["a_pull"]]
    features["a_pull_tilt"] = (
        a_pull[:, ask_full].sum(axis=1) - a_pull[:, bid_full].sum(axis=1)
    )

    a_fill = bucket_data[:, :, _FIELD_IDX["a_fill"]]
    features["a_fill_tilt"] = (
        a_fill[:, ask_full].sum(axis=1) - a_fill[:, bid_full].sum(axis=1)
    )

    a_rest = bucket_data[:, :, _FIELD_IDX["a_rest_depth"]]
    features["a_depth_tilt"] = (
        a_rest[:, bid_full].sum(axis=1) - a_rest[:, ask_full].sum(axis=1)
    )

    # --- Jerk magnitude (regime change detector) ---
    # max |j_add| + max |j_pull| across all k
    j_add = bucket_data[:, :, _FIELD_IDX["j_add"]]
    j_pull = bucket_data[:, :, _FIELD_IDX["j_pull"]]
    features["jerk_magnitude"] = (
        np.abs(j_add).max(axis=1) + np.abs(j_pull).max(axis=1)
    )

    # Directional jerk: j_add tilt + j_pull tilt (signed)
    features["j_add_tilt"] = (
        j_add[:, bid_full].sum(axis=1) - j_add[:, ask_full].sum(axis=1)
    )
    features["j_pull_tilt"] = (
        j_pull[:, ask_full].sum(axis=1) - j_pull[:, bid_full].sum(axis=1)
    )

    return pd.DataFrame(features)


# ---------------------------------------------------------------------------
# Temporal smoothing — EWM z-scores
# ---------------------------------------------------------------------------

def compute_ewm_zscores(
    raw_features: pd.DataFrame,
    lookback_windows: List[int],
) -> pd.DataFrame:
    """Apply EWM smoothing and z-score normalization at multiple lookbacks.

    For each raw feature F and lookback L:
        alpha = 1 - exp(-ln(2) / L)
        ewm_L = EWM(F, halflife=L)
        mu_L  = rolling_mean(F, window=L)
        sigma_L = rolling_std(F, window=L, min_periods=max(L//2, 2))
        z_L   = (ewm_L - mu_L) / sigma_L

    This captures the deviation of the smoothed signal from its recent
    average, normalized by recent volatility. Positive z = signal is
    elevated relative to recent history.

    Args:
        raw_features: DataFrame with N rows, one column per spatial feature.
        lookback_windows: List of lookback window sizes (in snapshots).

    Returns:
        DataFrame with N rows and columns named "{feature}_L{lookback}".
        NaN rows at the start correspond to insufficient warmup.
    """
    result_cols: Dict[str, np.ndarray] = {}

    for feat_name in raw_features.columns:
        series = raw_features[feat_name]

        for L in lookback_windows:
            # EWM with half-life parameterization
            # alpha = 1 - exp(-ln(2) / L), matching pandas ewm(halflife=L)
            ewm_val = series.ewm(halflife=L, adjust=True).mean()

            # Rolling statistics for z-score denominator
            min_periods = max(L // 2, 2)
            roll_mean = series.rolling(window=L, min_periods=min_periods).mean()
            roll_std = series.rolling(window=L, min_periods=min_periods).std()

            # Z-score: (ewm - rolling_mean) / rolling_std
            # Clamp std to avoid division by zero
            safe_std = roll_std.where(roll_std > 1e-12, np.nan)
            z = (ewm_val - roll_mean) / safe_std

            col_name = f"{feat_name}_L{L}"
            result_cols[col_name] = z.values

    return pd.DataFrame(result_cols, index=raw_features.index)


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def compute_forward_returns(
    mid_price: np.ndarray,
    horizons: List[int],
) -> Dict[int, np.ndarray]:
    """Compute forward mid-price returns in ticks.

    r(t, H) = (mid_price[t+H] - mid_price[t]) / TICK_SIZE_DOLLARS

    Args:
        mid_price: (N,) float64 array of mid prices in dollars.
        horizons: List of forward horizons in snapshots.

    Returns:
        Dict mapping horizon -> (N,) float64 array of forward returns in ticks.
        Entries beyond N-H are NaN.
    """
    N = len(mid_price)
    fwd_returns: Dict[int, np.ndarray] = {}

    for H in horizons:
        ret = np.full(N, np.nan, dtype=np.float64)
        if H < N:
            ret[:N - H] = (mid_price[H:] - mid_price[:N - H]) / TICK_SIZE_DOLLARS
        fwd_returns[H] = ret

    return fwd_returns


# ---------------------------------------------------------------------------
# Prediction evaluation
# ---------------------------------------------------------------------------

def evaluate_signals(
    signals: pd.DataFrame,
    fwd_returns: Dict[int, np.ndarray],
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    horizons: List[int],
) -> pd.DataFrame:
    """Compute rank IC, hit rate, and t-stat for all signal x horizon pairs.

    Rank IC = Spearman correlation between signal rank and forward return rank.
    Hit rate = fraction of times sign(signal) == sign(forward_return).
    t-stat = IC / sqrt((1 - IC^2) / (n - 2)), where n is the number of valid pairs.

    Uses training window for IC computation and reports both train and eval metrics.

    Args:
        signals: DataFrame with N rows, one column per signal.
        fwd_returns: Dict mapping horizon -> (N,) forward return array.
        train_mask: (N,) boolean mask for training window.
        eval_mask: (N,) boolean mask for evaluation window.
        horizons: List of forward horizons.

    Returns:
        DataFrame with columns: signal, horizon, train_ic, train_hit, train_tstat,
            eval_ic, eval_hit, eval_tstat, n_train, n_eval.
    """
    rows: List[Dict[str, Any]] = []

    for sig_name in signals.columns:
        sig_vals = signals[sig_name].values

        for H in horizons:
            fwd = fwd_returns[H]

            for phase, mask in [("train", train_mask), ("eval", eval_mask)]:
                # Valid pairs: both signal and forward return are non-NaN, finite
                valid = mask & np.isfinite(sig_vals) & np.isfinite(fwd)
                n_valid = valid.sum()

                if n_valid < 10:
                    ic, hit_rate, t_stat = np.nan, np.nan, np.nan
                else:
                    s = sig_vals[valid]
                    r = fwd[valid]

                    # Spearman rank IC
                    ic, _pval = stats.spearmanr(s, r)

                    # Hit rate: directional accuracy (exclude zero signal/return)
                    nonzero = (s != 0) & (r != 0)
                    if nonzero.sum() > 0:
                        hit_rate = float(np.mean(np.sign(s[nonzero]) == np.sign(r[nonzero])))
                    else:
                        hit_rate = 0.5

                    # t-stat of IC: t = IC * sqrt(n-2) / sqrt(1 - IC^2)
                    denom = 1.0 - ic * ic
                    if denom > 1e-12 and n_valid > 2:
                        t_stat = ic * math.sqrt(n_valid - 2) / math.sqrt(denom)
                    else:
                        t_stat = np.nan

                # Find or create the row dict for this signal/horizon
                existing = None
                for row in rows:
                    if row["signal"] == sig_name and row["horizon"] == H:
                        existing = row
                        break

                if existing is None:
                    existing = {"signal": sig_name, "horizon": H}
                    rows.append(existing)

                existing[f"{phase}_ic"] = ic
                existing[f"{phase}_hit"] = hit_rate
                existing[f"{phase}_tstat"] = t_stat
                existing[f"n_{phase}"] = int(n_valid)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def build_composite_signal(
    signals: pd.DataFrame,
    fwd_returns: Dict[int, np.ndarray],
    train_mask: np.ndarray,
    horizon: int,
    top_k: int = 5,
) -> Tuple[np.ndarray, List[str]]:
    """Build equal-weight composite from top-K features ranked by train IC.

    Selection:
        1. Compute absolute Spearman IC for each signal on training data.
        2. Select top_k signals by |IC|.
        3. Sign-flip signals with negative IC (so all contribute positively).
        4. Equal-weight average of selected (sign-corrected) z-scores.

    Args:
        signals: DataFrame of z-scored signals.
        fwd_returns: Forward return dict.
        train_mask: Boolean mask for training window.
        horizon: Forward horizon to optimize for.
        top_k: Number of signals to include.

    Returns:
        Tuple of:
            composite: (N,) array of composite signal values.
            selected_names: List of selected signal names (for reporting).
    """
    fwd = fwd_returns[horizon]
    ic_scores: Dict[str, float] = {}

    for sig_name in signals.columns:
        s = signals[sig_name].values
        valid = train_mask & np.isfinite(s) & np.isfinite(fwd)
        if valid.sum() < 20:
            continue
        ic, _ = stats.spearmanr(s[valid], fwd[valid])
        if np.isfinite(ic):
            ic_scores[sig_name] = ic

    # Sort by absolute IC, take top-K
    sorted_sigs = sorted(ic_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    selected = sorted_sigs[:top_k]

    if not selected:
        logger.warning("No valid signals found for composite construction.")
        return np.zeros(len(signals), dtype=np.float64), []

    # Build composite: sign-corrected equal-weight average
    composite = np.zeros(len(signals), dtype=np.float64)
    count_valid = np.zeros(len(signals), dtype=np.float64)

    selected_names = []
    for sig_name, ic in selected:
        vals = signals[sig_name].values.copy()
        # Sign-flip if IC is negative (make signal consistently bullish-positive)
        if ic < 0:
            vals = -vals
        finite_mask = np.isfinite(vals)
        composite[finite_mask] += vals[finite_mask]
        count_valid[finite_mask] += 1.0
        selected_names.append(f"{sig_name} (IC={ic:+.4f})")

    # Average where we have valid data
    nonzero_count = count_valid > 0
    composite[nonzero_count] /= count_valid[nonzero_count]
    composite[~nonzero_count] = np.nan

    return composite, selected_names


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def regime_analysis(
    composite: np.ndarray,
    fwd_returns: Dict[int, np.ndarray],
    eval_mask: np.ndarray,
    jerk_magnitude: np.ndarray,
    horizon: int,
) -> Dict[str, Dict[str, float]]:
    """Analyze prediction quality conditional on VP regime.

    Regimes:
        DIRECTIONAL: jerk_magnitude > median (high-activity, transitional)
        CHOP: jerk_magnitude <= median (low-activity, range-bound)

    Args:
        composite: (N,) composite signal array.
        fwd_returns: Forward return dict.
        eval_mask: Boolean mask for evaluation window.
        jerk_magnitude: (N,) jerk magnitude array (unsigned regime indicator).
        horizon: Forward horizon.

    Returns:
        Dict mapping regime name -> dict of {ic, hit_rate, n, pnl_ticks}.
    """
    fwd = fwd_returns[horizon]
    valid = eval_mask & np.isfinite(composite) & np.isfinite(fwd) & np.isfinite(jerk_magnitude)

    if valid.sum() < 20:
        return {"insufficient_data": {"n": int(valid.sum())}}

    jerk_valid = jerk_magnitude[valid]
    median_jerk = np.median(jerk_valid)

    results: Dict[str, Dict[str, float]] = {}

    for regime_name, regime_cond in [
        ("DIRECTIONAL", jerk_valid > median_jerk),
        ("CHOP", jerk_valid <= median_jerk),
    ]:
        # Map regime condition back to valid indices
        regime_idx = np.where(valid)[0][regime_cond]
        n_regime = len(regime_idx)

        if n_regime < 10:
            results[regime_name] = {"n": n_regime, "ic": np.nan, "hit_rate": np.nan, "pnl_ticks": np.nan}
            continue

        s = composite[regime_idx]
        r = fwd[regime_idx]

        ic, _ = stats.spearmanr(s, r)

        nonzero = (s != 0) & (r != 0)
        if nonzero.sum() > 0:
            hit_rate = float(np.mean(np.sign(s[nonzero]) == np.sign(r[nonzero])))
        else:
            hit_rate = 0.5

        # PnL: sign(composite) * forward_return (cumulative ticks)
        pnl = float(np.sum(np.sign(s) * r))

        results[regime_name] = {
            "n": n_regime,
            "ic": ic,
            "hit_rate": hit_rate,
            "pnl_ticks": pnl,
        }

    return results


# ---------------------------------------------------------------------------
# PnL computation
# ---------------------------------------------------------------------------

def compute_cumulative_pnl(
    composite: np.ndarray,
    fwd_returns: Dict[int, np.ndarray],
    mask: np.ndarray,
    horizon: int,
) -> Tuple[float, np.ndarray]:
    """Compute cumulative PnL of composite signal over masked window.

    PnL per snapshot = sign(composite[t]) * forward_return[t]
    Cumulative PnL = sum of per-snapshot PnL (in ticks).

    NOTE: This is a simplified PnL that assumes:
    - Position taken at each snapshot, closed H snapshots later.
    - No transaction costs, no slippage.
    - Overlapping positions (multiple positions open simultaneously).
    This is NOT a realistic backtest — it measures signal quality only.

    Args:
        composite: (N,) composite signal array.
        fwd_returns: Forward return dict.
        mask: Boolean mask for evaluation window.
        horizon: Forward horizon.

    Returns:
        Tuple of (total_pnl_ticks, per_snapshot_pnl_array).
    """
    fwd = fwd_returns[horizon]
    valid = mask & np.isfinite(composite) & np.isfinite(fwd)

    per_snap_pnl = np.zeros_like(composite)
    per_snap_pnl[valid] = np.sign(composite[valid]) * fwd[valid]

    total = float(per_snap_pnl[valid].sum())
    return total, per_snap_pnl


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results_table(results_df: pd.DataFrame) -> None:
    """Print formatted results table sorted by absolute train IC."""
    # Sort by absolute train IC descending
    results_df = results_df.copy()
    results_df["abs_train_ic"] = results_df["train_ic"].abs()
    results_df = results_df.sort_values(
        ["horizon", "abs_train_ic"], ascending=[True, False]
    )

    print()
    print("=" * 120)
    print("  SIGNAL PREDICTIVE POWER ANALYSIS")
    print("=" * 120)

    for H in sorted(results_df["horizon"].unique()):
        subset = results_df[results_df["horizon"] == H]
        print(f"\n  Forward Horizon: {H} snapshots (~{H * 25 / 1000:.1f}s)")
        print(f"  {'Signal':<40s} {'Train IC':>10s} {'Train Hit':>10s} {'Train t':>10s} "
              f"{'Eval IC':>10s} {'Eval Hit':>10s} {'Eval t':>10s} {'n_train':>8s} {'n_eval':>8s}")
        print("  " + "-" * 116)

        for _, row in subset.iterrows():
            sig = row["signal"]
            # Truncate long signal names
            if len(sig) > 38:
                sig = sig[:35] + "..."
            print(
                f"  {sig:<40s} "
                f"{row['train_ic']:>+10.4f} "
                f"{row['train_hit']:>10.1%} "
                f"{row['train_tstat']:>+10.2f} "
                f"{row['eval_ic']:>+10.4f} "
                f"{row['eval_hit']:>10.1%} "
                f"{row['eval_tstat']:>+10.2f} "
                f"{int(row['n_train']):>8d} "
                f"{int(row['n_eval']):>8d}"
            )

    print()


def print_composite_results(
    selected_names: List[str],
    eval_ic: float,
    eval_hit: float,
    total_pnl: float,
    n_eval: int,
    horizon: int,
    regime_results: Dict[str, Dict[str, float]],
) -> None:
    """Print composite signal evaluation results."""
    print("=" * 120)
    print("  COMPOSITE SIGNAL PERFORMANCE")
    print("=" * 120)
    print(f"\n  Horizon: {horizon} snapshots (~{horizon * 25 / 1000:.1f}s)")
    print(f"  Components (top-5 by |IC| on training set):")
    for name in selected_names:
        print(f"    - {name}")

    print(f"\n  Evaluation Window:")
    print(f"    Rank IC:         {eval_ic:+.4f}")
    print(f"    Hit Rate:        {eval_hit:.1%}")
    print(f"    Cumulative PnL:  {total_pnl:+.1f} ticks")
    print(f"    PnL ($):         ${total_pnl * TICK_SIZE_DOLLARS * 2:+.2f} (MNQ $2/tick)")
    print(f"    n_eval:          {n_eval}")

    print(f"\n  Regime Analysis (eval window, median jerk split):")
    print(f"  {'Regime':<16s} {'IC':>10s} {'Hit Rate':>10s} {'PnL (ticks)':>12s} {'n':>8s}")
    print("  " + "-" * 58)
    for regime_name, metrics in regime_results.items():
        if "ic" not in metrics:
            print(f"  {regime_name:<16s} insufficient data (n={metrics.get('n', 0)})")
            continue
        print(
            f"  {regime_name:<16s} "
            f"{metrics['ic']:>+10.4f} "
            f"{metrics['hit_rate']:>10.1%} "
            f"{metrics['pnl_ticks']:>+12.1f} "
            f"{int(metrics['n']):>8d}"
        )

    print()
    print("=" * 120)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze VP signal predictive power for directional price prediction.",
    )
    parser.add_argument("--product-type", default="future_mbo", help="Product type")
    parser.add_argument("--symbol", default="MNQH6", help="Instrument symbol")
    parser.add_argument("--dt", default="2026-02-06", help="Date YYYY-MM-DD")
    parser.add_argument("--start-time", default="09:25", help="Capture start HH:MM ET")
    parser.add_argument("--eval-start", default="09:45", help="Eval window start HH:MM ET")
    parser.add_argument("--eval-minutes", type=int, default=3, help="Eval window duration (minutes)")
    parser.add_argument("--throttle-ms", type=float, default=25.0, help="Grid throttle ms")
    parser.add_argument("--composite-horizon", type=int, default=100,
                        help="Forward horizon for composite signal (snapshots)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K signals for composite")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"
    lake_root = backend_root / "lake"

    from src.vacuum_pressure.config import resolve_config

    config = resolve_config(args.product_type, args.symbol, products_yaml_path)

    # Compute end time for capture (eval_start + eval_minutes)
    eval_start_h, eval_start_m = map(int, args.eval_start.split(":"))
    end_total_m = eval_start_h * 60 + eval_start_m + args.eval_minutes
    end_time_et = f"{end_total_m // 60:02d}:{end_total_m % 60:02d}"

    print()
    print("=" * 120)
    print("  VP SIGNAL PREDICTIVE POWER ANALYSIS")
    print("=" * 120)
    print(f"  Symbol:       {args.symbol}")
    print(f"  Date:         {args.dt}")
    print(f"  Train:        {args.start_time} - {args.eval_start} ET")
    print(f"  Eval:         {args.eval_start} - {end_time_et} ET")
    print(f"  Throttle:     {args.throttle_ms}ms")
    print(f"  Horizons:     {FORWARD_HORIZONS} snapshots")
    print(f"  Lookbacks:    {LOOKBACK_WINDOWS} snapshots")
    print("=" * 120)
    print()

    # -----------------------------------------------------------------------
    # Step 1: Data capture
    # -----------------------------------------------------------------------
    logger.info("Phase 1: Capturing VP grid snapshots...")
    t0 = time.monotonic()

    ts_ns, mid_price, bucket_data = capture_grids(
        lake_root=lake_root,
        config=config,
        dt=args.dt,
        start_time=args.start_time,
        throttle_ms=args.throttle_ms,
        end_time_et=end_time_et,
    )

    N = len(ts_ns)
    capture_time = time.monotonic() - t0
    print(f"  Captured {N:,} snapshots in {capture_time:.1f}s")
    print(f"  Mid price range: ${mid_price.min():.2f} - ${mid_price.max():.2f}")
    print(f"  Total price movement: {(mid_price.max() - mid_price.min()) / TICK_SIZE_DOLLARS:.1f} ticks")
    print()

    if N < 200:
        print("ERROR: Insufficient snapshots for analysis. Need at least 200.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Build train/eval masks
    # -----------------------------------------------------------------------
    eval_start_ns = int(
        pd.Timestamp(f"{args.dt} {args.eval_start}:00", tz="America/New_York")
        .tz_convert("UTC").value
    )

    train_mask = ts_ns < eval_start_ns
    eval_mask = ts_ns >= eval_start_ns

    n_train = train_mask.sum()
    n_eval = eval_mask.sum()
    print(f"  Training snapshots:   {n_train:,}")
    print(f"  Evaluation snapshots: {n_eval:,}")
    print()

    # -----------------------------------------------------------------------
    # Step 3: Feature engineering
    # -----------------------------------------------------------------------
    logger.info("Phase 2: Computing spatial features...")
    t1 = time.monotonic()
    raw_features = compute_spatial_features(bucket_data)
    feat_time = time.monotonic() - t1
    print(f"  Computed {len(raw_features.columns)} spatial features in {feat_time:.2f}s")

    logger.info("Phase 3: Computing EWM z-scores...")
    t2 = time.monotonic()
    zscore_signals = compute_ewm_zscores(raw_features, LOOKBACK_WINDOWS)
    zscore_time = time.monotonic() - t2
    print(f"  Computed {len(zscore_signals.columns)} z-scored signals in {zscore_time:.2f}s")
    print()

    # -----------------------------------------------------------------------
    # Step 4: Forward returns
    # -----------------------------------------------------------------------
    logger.info("Phase 4: Computing forward returns...")
    fwd_returns = compute_forward_returns(mid_price, FORWARD_HORIZONS)

    for H in FORWARD_HORIZONS:
        valid_count = np.isfinite(fwd_returns[H]).sum()
        if valid_count > 0:
            fwd_std = np.nanstd(fwd_returns[H])
            print(f"  H={H:>4d}: {valid_count:,} valid returns, std={fwd_std:.2f} ticks")
    print()

    # -----------------------------------------------------------------------
    # Step 5: Evaluate all signals
    # -----------------------------------------------------------------------
    logger.info("Phase 5: Evaluating signal predictive power...")
    t3 = time.monotonic()
    results_df = evaluate_signals(
        zscore_signals, fwd_returns, train_mask, eval_mask, FORWARD_HORIZONS
    )
    eval_time = time.monotonic() - t3
    print(f"  Evaluated {len(results_df)} signal x horizon pairs in {eval_time:.2f}s")

    print_results_table(results_df)

    # -----------------------------------------------------------------------
    # Step 6: Composite signal
    # -----------------------------------------------------------------------
    logger.info("Phase 6: Building composite signal...")

    # Use the specified horizon for composite
    comp_horizon = args.composite_horizon
    if comp_horizon not in FORWARD_HORIZONS:
        logger.warning(
            "Composite horizon %d not in FORWARD_HORIZONS %s, using closest.",
            comp_horizon, FORWARD_HORIZONS,
        )
        comp_horizon = min(FORWARD_HORIZONS, key=lambda h: abs(h - args.composite_horizon))

    composite, selected_names = build_composite_signal(
        zscore_signals, fwd_returns, train_mask, comp_horizon, top_k=args.top_k
    )

    # Evaluate composite on eval window
    fwd = fwd_returns[comp_horizon]
    valid_eval = eval_mask & np.isfinite(composite) & np.isfinite(fwd)
    n_valid_eval = valid_eval.sum()

    if n_valid_eval >= 10:
        comp_ic, _ = stats.spearmanr(composite[valid_eval], fwd[valid_eval])
        nonzero = (composite[valid_eval] != 0) & (fwd[valid_eval] != 0)
        if nonzero.sum() > 0:
            comp_hit = float(np.mean(
                np.sign(composite[valid_eval][nonzero]) == np.sign(fwd[valid_eval][nonzero])
            ))
        else:
            comp_hit = 0.5
    else:
        comp_ic = np.nan
        comp_hit = np.nan

    total_pnl, _ = compute_cumulative_pnl(
        composite, fwd_returns, eval_mask, comp_horizon
    )

    # Regime analysis
    jerk_mag = raw_features["jerk_magnitude"].values
    regime_results = regime_analysis(
        composite, fwd_returns, eval_mask, jerk_mag, comp_horizon
    )

    print_composite_results(
        selected_names, comp_ic, comp_hit, total_pnl,
        int(n_valid_eval), comp_horizon, regime_results,
    )

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    total_time = time.monotonic() - t0
    print(f"  Total analysis time: {total_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
