"""Shared evaluation harness for VP regime-detection experiments.

All 6 experiment scripts import this module for:
- Dataset loading (bins + grid_clean pivoted to (n_bins, 101) arrays)
- Signal detection (threshold crossing with cooldown)
- TP/SL evaluation (identical to analyze_vp_signals.py)
- Threshold sweep + JSON output
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

DATASET_ID = "mnqh6_20260206_0925_1025"
BASE_DIR = Path(__file__).resolve().parent
IMMUTABLE_DIR = BASE_DIR.parent.parent / "vp_immutable" / DATASET_ID

TICK_SIZE = 0.25
TP_TICKS = 8
SL_TICKS = 4
MAX_HOLD_BINS = 1200
WARMUP_BINS = 300  # skip first 300 bins (30s) for rolling windows to fill

# k column values: -50 to +50 inclusive = 101 ticks
N_TICKS = 101
K_MIN = -50
K_MAX = 50


def load_dataset(
    columns: List[str] | None = None,
) -> Dict[str, Any]:
    """Load bins + grid_clean and pivot grid columns to (n_bins, 101) arrays.

    Args:
        columns: grid_clean columns to pivot (besides k and bin_seq).
                 If None, loads all numeric columns.

    Returns:
        dict with:
          - 'bins': DataFrame (n_bins rows)
          - 'mid_price': ndarray (n_bins,)
          - 'ts_ns': ndarray (n_bins,)
          - 'n_bins': int
          - '<col>': ndarray (n_bins, 101) for each requested column
          - 'k_values': ndarray (101,) the k axis
    """
    bins_path = IMMUTABLE_DIR / "bins.parquet"
    grid_path = IMMUTABLE_DIR / "grid_clean.parquet"

    bins_df = pd.read_parquet(bins_path)
    bins_df = bins_df.sort_values("bin_seq").reset_index(drop=True)
    n_bins = len(bins_df)

    mid_price = bins_df["mid_price"].values.astype(np.float64)
    ts_ns = bins_df["ts_ns"].values.astype(np.int64)

    # Determine columns to load from grid
    read_cols = ["bin_seq", "k"]
    if columns is None:
        grid_sample = pd.read_parquet(grid_path, columns=["bin_seq", "k"])
        raise ValueError("Must specify columns to load")
    read_cols.extend(columns)

    grid_df = pd.read_parquet(grid_path, columns=read_cols)

    # Build bin_seq -> row index mapping
    bin_seq_to_idx = {seq: i for i, seq in enumerate(bins_df["bin_seq"].values)}

    # k -> column index mapping: k=-50 maps to col 0, k=+50 maps to col 100
    k_values = np.arange(K_MIN, K_MAX + 1, dtype=np.int32)

    result: Dict[str, Any] = {
        "bins": bins_df,
        "mid_price": mid_price,
        "ts_ns": ts_ns,
        "n_bins": n_bins,
        "k_values": k_values,
    }

    # Pivot each column to (n_bins, 101) array
    for col in columns:
        arr = np.zeros((n_bins, N_TICKS), dtype=np.float64)
        col_vals = grid_df[col].values.astype(np.float64)
        bin_seqs = grid_df["bin_seq"].values
        k_vals = grid_df["k"].values

        for row_i in range(len(grid_df)):
            b_idx = bin_seq_to_idx.get(bin_seqs[row_i])
            if b_idx is None:
                continue
            k_idx = k_vals[row_i] - K_MIN
            if 0 <= k_idx < N_TICKS:
                arr[b_idx, k_idx] = col_vals[row_i]

        result[col] = arr

    return result


def detect_signals(
    signal: np.ndarray,
    threshold: float,
    cooldown_bins: int,
) -> List[Dict[str, Any]]:
    """Detect directional signals from a continuous signal array.

    Identical logic to _detect_directional_signals in analyze_vp_signals.py.

    Signal fires when value crosses +/-threshold from neutral/opposite state.
    """
    signals: List[Dict[str, Any]] = []
    prev_state = "flat"
    last_signal_bin = -cooldown_bins

    for i in range(len(signal)):
        if signal[i] >= threshold:
            cur_state = "up"
        elif signal[i] <= -threshold:
            cur_state = "down"
        else:
            cur_state = "flat"

        if cur_state != "flat" and cur_state != prev_state:
            if i - last_signal_bin >= cooldown_bins:
                signals.append({"bin_idx": i, "direction": cur_state})
                last_signal_bin = i

        prev_state = cur_state

    return signals


def evaluate_tp_sl(
    *,
    signals: List[Dict[str, Any]],
    mid_price: np.ndarray,
    ts_ns: np.ndarray,
    tp_ticks: int = TP_TICKS,
    sl_ticks: int = SL_TICKS,
    tick_size: float = TICK_SIZE,
    max_hold_bins: int = MAX_HOLD_BINS,
) -> Dict[str, Any]:
    """Evaluate TP/SL outcomes. Identical to _evaluate_tp_sl in analyze_vp_signals.py."""
    tp_dollars = tp_ticks * tick_size
    sl_dollars = sl_ticks * tick_size

    outcomes: List[Dict[str, Any]] = []

    for sig in signals:
        idx = sig["bin_idx"]
        direction = sig["direction"]
        entry_price = mid_price[idx]
        entry_ts = ts_ns[idx]

        if entry_price <= 0:
            continue

        end_idx = min(idx + max_hold_bins, len(mid_price))
        outcome = "timeout"
        outcome_idx = end_idx - 1
        outcome_price = mid_price[outcome_idx] if outcome_idx < len(mid_price) else entry_price

        for j in range(idx + 1, end_idx):
            price = mid_price[j]
            if price <= 0:
                continue
            if direction == "up":
                if price >= entry_price + tp_dollars:
                    outcome = "tp"
                    outcome_idx = j
                    outcome_price = price
                    break
                if price <= entry_price - sl_dollars:
                    outcome = "sl"
                    outcome_idx = j
                    outcome_price = price
                    break
            else:
                if price <= entry_price - tp_dollars:
                    outcome = "tp"
                    outcome_idx = j
                    outcome_price = price
                    break
                if price >= entry_price + sl_dollars:
                    outcome = "sl"
                    outcome_idx = j
                    outcome_price = price
                    break

        time_to_outcome_ms = float(ts_ns[outcome_idx] - entry_ts) / 1e6

        outcomes.append({
            "bin_idx": idx,
            "direction": direction,
            "entry_price": float(entry_price),
            "outcome": outcome,
            "outcome_price": float(outcome_price),
            "time_to_outcome_ms": time_to_outcome_ms,
            "pnl_ticks": (outcome_price - entry_price) / tick_size
                         * (1.0 if direction == "up" else -1.0),
        })

    n_total = len(outcomes)
    if n_total == 0:
        return {
            "n_signals": 0,
            "tp_rate": float("nan"),
            "sl_rate": float("nan"),
            "timeout_rate": float("nan"),
            "events_per_hour": 0.0,
            "mean_pnl_ticks": float("nan"),
            "median_time_to_outcome_ms": float("nan"),
            "outcomes": [],
        }

    n_tp = sum(1 for o in outcomes if o["outcome"] == "tp")
    n_sl = sum(1 for o in outcomes if o["outcome"] == "sl")
    n_timeout = sum(1 for o in outcomes if o["outcome"] == "timeout")

    eval_duration_hours = float(ts_ns[-1] - ts_ns[0]) / 3.6e12
    events_per_hour = n_total / eval_duration_hours if eval_duration_hours > 0 else 0.0

    times = [o["time_to_outcome_ms"] for o in outcomes if o["outcome"] != "timeout"]
    pnls = [o["pnl_ticks"] for o in outcomes]

    return {
        "n_signals": n_total,
        "n_up": sum(1 for o in outcomes if o["direction"] == "up"),
        "n_down": sum(1 for o in outcomes if o["direction"] == "down"),
        "tp_rate": n_tp / n_total,
        "sl_rate": n_sl / n_total,
        "timeout_rate": n_timeout / n_total,
        "events_per_hour": events_per_hour,
        "median_time_to_outcome_ms": float(np.median(times)) if times else float("nan"),
        "mean_pnl_ticks": float(np.mean(pnls)),
        "outcomes": outcomes,
    }


def sweep_thresholds(
    *,
    signal: np.ndarray,
    thresholds: List[float],
    cooldown_bins: int,
    mid_price: np.ndarray,
    ts_ns: np.ndarray,
    warmup_bins: int = WARMUP_BINS,
) -> List[Dict[str, Any]]:
    """Run detect+evaluate across a grid of thresholds.

    Returns list of result dicts (one per threshold), each augmented
    with 'threshold' and 'cooldown_bins' fields.
    """
    results = []
    for thr in thresholds:
        sigs = detect_signals(signal[warmup_bins:], thr, cooldown_bins)
        # Adjust bin_idx back to global index
        for s in sigs:
            s["bin_idx"] += warmup_bins

        eval_result = evaluate_tp_sl(
            signals=sigs,
            mid_price=mid_price,
            ts_ns=ts_ns,
        )
        # Remove full trade log from sweep (too large); keep summary
        outcomes = eval_result.pop("outcomes", [])
        eval_result["threshold"] = thr
        eval_result["cooldown_bins"] = cooldown_bins
        results.append(eval_result)

    return results


def write_results(
    agent_name: str,
    experiment_name: str,
    params: Dict[str, Any],
    results_by_threshold: List[Dict[str, Any]],
    best_trades: List[Dict[str, Any]] | None = None,
) -> Path:
    """Write experiment results JSON to agent outputs directory."""
    agent_dir = BASE_DIR / "agents" / agent_name / "outputs"
    agent_dir.mkdir(parents=True, exist_ok=True)
    out_path = agent_dir / "results.json"

    # Find best threshold by TP rate (with min 5 signals)
    valid = [r for r in results_by_threshold if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
    else:
        best = results_by_threshold[0] if results_by_threshold else {}

    payload = {
        "experiment_name": experiment_name,
        "agent_name": agent_name,
        "dataset_id": DATASET_ID,
        "params": params,
        "results_by_threshold": results_by_threshold,
        "best_threshold": best.get("threshold"),
        "best_tp_rate": best.get("tp_rate"),
        "best_n_signals": best.get("n_signals"),
        "best_mean_pnl_ticks": best.get("mean_pnl_ticks"),
        "best_events_per_hour": best.get("events_per_hour"),
    }
    if best_trades is not None:
        payload["best_trades"] = best_trades

    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def rolling_ols_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling OLS slope of a 1D array.

    Uses the formula: slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    where x = [0, 1, ..., w-1] and y = arr values in the window.

    Returns array of same length with NaN for the first window-1 elements.
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < 2:
        return result

    # Pre-compute x constants
    x = np.arange(window, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = window * sum_x2 - sum_x * sum_x

    if abs(denom) < 1e-30:
        return result

    # Rolling computation
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        sum_y = y.sum()
        sum_xy = np.dot(x, y)
        result[i] = (window * sum_xy - sum_x * sum_y) / denom

    return result


def robust_zscore(arr: np.ndarray, window: int, min_periods: int = 30) -> np.ndarray:
    """Robust z-score using rolling median and MAD."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        seg = arr[start: i + 1]
        if len(seg) < min_periods:
            result[i] = 0.0
            continue
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        scale = 1.4826 * mad
        if scale < 1e-12:
            result[i] = 0.0
        else:
            result[i] = (arr[i] - med) / scale

    return result
