"""Jerk-Acceleration Divergence (JAD) experiment.

Tests the core thesis: when jerk (d3) of add/pull diverges between bid
and ask sides, it signals regime inflection before acceleration changes sign.

Signal construction:
    1. Distance-weighted spatial aggregation:
       w(k) = 1/|k| for bid cols 26..49 (k=-24..-1) and ask cols 51..74 (k=+1..+24).
       Weighted mean of j_add, j_pull, a_add, a_pull per side.
    2. Divergence signals (bullish-positive orientation):
       jerk_add_div  = J_add_bid  - J_add_ask   (more add jerk on bid = bullish)
       jerk_pull_div = J_pull_ask - J_pull_bid   (more pull jerk on ask = bullish)
       accel_add_div = A_add_bid  - A_add_ask
       accel_pull_div = A_pull_ask - A_pull_bid
    3. Combined:
       jerk_signal  = 0.5 * jerk_add_div + 0.5 * jerk_pull_div
       accel_signal = 0.5 * accel_add_div + 0.5 * accel_pull_div
    4. Agreement/disagreement weighting:
       If sign(jerk) == sign(accel): raw = 0.4*jerk + 0.6*accel (confirmed)
       Else (jerk leading): raw = 0.8*jerk + 0.2*accel (early reversal)
    5. Robust z-score (300-bin window), then signal = tanh(z / 3.0).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_harness import (
    WARMUP_BINS,
    load_dataset,
    robust_zscore,
    sweep_thresholds,
    write_results,
)

# ---------------------------------------------------------------------------
# Spatial aggregation config
# ---------------------------------------------------------------------------
# Bid cols 26..49 correspond to k=-24..-1, ask cols 51..74 correspond to k=+1..+24.
BID_COLS = np.arange(26, 50)   # 24 columns
ASK_COLS = np.arange(51, 75)   # 24 columns

# Pre-compute inverse-distance weights: w(k) = 1/|k|
# Bid: col c -> k = c - 50, so |k| = 50 - c for c in 26..49 -> |k| = 24..1
BID_K_ABS = 50 - BID_COLS      # [24, 23, ..., 1]
BID_WEIGHTS = 1.0 / BID_K_ABS.astype(np.float64)
BID_WEIGHTS_NORM = BID_WEIGHTS / BID_WEIGHTS.sum()

# Ask: col c -> k = c - 50, so |k| = c - 50 for c in 51..74 -> |k| = 1..24
ASK_K_ABS = ASK_COLS - 50      # [1, 2, ..., 24]
ASK_WEIGHTS = 1.0 / ASK_K_ABS.astype(np.float64)
ASK_WEIGHTS_NORM = ASK_WEIGHTS / ASK_WEIGHTS.sum()

# Signal construction params
ZSCORE_WINDOW: int = 300
TANH_SCALE: float = 3.0

# Sweep params
THRESHOLDS: list[float] = [0.05, 0.10, 0.15, 0.20, 0.30]
COOLDOWN_BINS: int = 25


def weighted_mean_cols(
    grid: np.ndarray,
    cols: np.ndarray,
    weights_norm: np.ndarray,
) -> np.ndarray:
    """Compute distance-weighted mean across selected columns for all bins.

    Args:
        grid: (n_bins, 101) array.
        cols: column indices to aggregate.
        weights_norm: normalized weights, same length as cols.

    Returns:
        (n_bins,) weighted mean per time bin.
    """
    # grid[:, cols] -> (n_bins, len(cols)), matmul with weights -> (n_bins,)
    return grid[:, cols] @ weights_norm


def compute_signal(
    j_add: np.ndarray,
    j_pull: np.ndarray,
    a_add: np.ndarray,
    a_pull: np.ndarray,
) -> np.ndarray:
    """Compute JAD signal from jerk and acceleration grids.

    Args:
        j_add: (n_bins, 101) jerk of add velocity.
        j_pull: (n_bins, 101) jerk of pull velocity.
        a_add: (n_bins, 101) acceleration of add velocity.
        a_pull: (n_bins, 101) acceleration of pull velocity.

    Returns:
        (n_bins,) final signal in roughly [-1, +1].
    """
    # Step 2: Distance-weighted spatial aggregation
    print("  aggregating jerk ...", flush=True)
    j_add_bid = weighted_mean_cols(j_add, BID_COLS, BID_WEIGHTS_NORM)
    j_add_ask = weighted_mean_cols(j_add, ASK_COLS, ASK_WEIGHTS_NORM)
    j_pull_bid = weighted_mean_cols(j_pull, BID_COLS, BID_WEIGHTS_NORM)
    j_pull_ask = weighted_mean_cols(j_pull, ASK_COLS, ASK_WEIGHTS_NORM)

    print("  aggregating acceleration ...", flush=True)
    a_add_bid = weighted_mean_cols(a_add, BID_COLS, BID_WEIGHTS_NORM)
    a_add_ask = weighted_mean_cols(a_add, ASK_COLS, ASK_WEIGHTS_NORM)
    a_pull_bid = weighted_mean_cols(a_pull, BID_COLS, BID_WEIGHTS_NORM)
    a_pull_ask = weighted_mean_cols(a_pull, ASK_COLS, ASK_WEIGHTS_NORM)

    # Step 3: Divergence signals (bullish-positive)
    print("  computing divergences ...", flush=True)
    jerk_add_div = j_add_bid - j_add_ask
    jerk_pull_div = j_pull_ask - j_pull_bid
    accel_add_div = a_add_bid - a_add_ask
    accel_pull_div = a_pull_ask - a_pull_bid

    # Step 4: Combined signals
    jerk_signal = 0.5 * jerk_add_div + 0.5 * jerk_pull_div
    accel_signal = 0.5 * accel_add_div + 0.5 * accel_pull_div

    # Step 5: Agreement/disagreement weighting (vectorized)
    print("  agreement/disagreement weighting ...", flush=True)
    same_sign = np.sign(jerk_signal) == np.sign(accel_signal)
    raw_signal = np.where(
        same_sign,
        0.4 * jerk_signal + 0.6 * accel_signal,   # confirmed direction
        0.8 * jerk_signal + 0.2 * accel_signal,    # jerk leading
    )

    # Step 6: Robust z-score + tanh compression
    print(f"  robust z-score (window={ZSCORE_WINDOW}) ...", flush=True)
    z = robust_zscore(raw_signal, ZSCORE_WINDOW)
    signal = np.tanh(z / TANH_SCALE)

    return signal


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset (j_add, j_pull, a_add, a_pull) ...", flush=True)
    ds = load_dataset(columns=["j_add", "j_pull", "a_add", "a_pull"])
    j_add: np.ndarray = ds["j_add"]
    j_pull: np.ndarray = ds["j_pull"]
    a_add: np.ndarray = ds["a_add"]
    a_pull: np.ndarray = ds["a_pull"]
    mid_price: np.ndarray = ds["mid_price"]
    ts_ns: np.ndarray = ds["ts_ns"]
    n_bins: int = ds["n_bins"]
    print(f"  n_bins={n_bins}, grid shape=({n_bins}, 101)", flush=True)
    print(f"  load time: {time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 2. Compute JAD signal
    # ------------------------------------------------------------------
    print("Computing JAD signal ...", flush=True)
    t1 = time.perf_counter()
    signal = compute_signal(j_add, j_pull, a_add, a_pull)
    print(f"  signal done in {time.perf_counter() - t1:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Signal distribution
    # ------------------------------------------------------------------
    active = signal[WARMUP_BINS:]
    pcts = np.nanpercentile(active, [5, 25, 50, 75, 95])
    print(
        f"Signal distribution (post-warmup): "
        f"p5={pcts[0]:.4f}  p25={pcts[1]:.4f}  p50={pcts[2]:.4f}  "
        f"p75={pcts[3]:.4f}  p95={pcts[4]:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4. Threshold sweep
    # ------------------------------------------------------------------
    print(
        f"Sweeping thresholds {THRESHOLDS} (cooldown={COOLDOWN_BINS}, "
        f"warmup={WARMUP_BINS}) ...",
        flush=True,
    )
    results = sweep_thresholds(
        signal=signal,
        thresholds=THRESHOLDS,
        cooldown_bins=COOLDOWN_BINS,
        mid_price=mid_price,
        ts_ns=ts_ns,
    )

    # ------------------------------------------------------------------
    # 5. Report + write
    # ------------------------------------------------------------------
    for r in results:
        tp = r["tp_rate"]
        tp_str = f"{tp:.1%}" if not np.isnan(tp) else "N/A"
        print(
            f"  thr={r['threshold']:.2f}  n={r['n_signals']:>4d}  "
            f"TP={tp_str}  mean_pnl={r['mean_pnl_ticks']:+.2f}t  "
            f"evts/hr={r['events_per_hour']:.1f}",
            flush=True,
        )

    params = {
        "columns": ["j_add", "j_pull", "a_add", "a_pull"],
        "bid_cols": BID_COLS.tolist(),
        "ask_cols": ASK_COLS.tolist(),
        "bid_k_range": "k=-24..-1",
        "ask_k_range": "k=+1..+24",
        "weighting": "inverse_distance: w(k) = 1/|k|",
        "jerk_accel_agreement_weights": {"confirmed": [0.4, 0.6], "disagreement": [0.8, 0.2]},
        "zscore_window": ZSCORE_WINDOW,
        "tanh_scale": TANH_SCALE,
        "cooldown_bins": COOLDOWN_BINS,
        "warmup_bins": WARMUP_BINS,
    }

    out_path = write_results(
        agent_name="jad",
        experiment_name="jerk_acceleration_divergence",
        params=params,
        results_by_threshold=results,
    )

    # Best threshold (min 5 signals)
    valid = [r for r in results if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
        print(
            f"\nBest: thr={best['threshold']:.2f}  "
            f"TP={best['tp_rate']:.1%}  "
            f"n={best['n_signals']}  "
            f"mean_pnl={best['mean_pnl_ticks']:+.2f}t",
        )
    else:
        print("\nNo threshold produced >= 5 signals.")

    print(f"Results written to {out_path}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
