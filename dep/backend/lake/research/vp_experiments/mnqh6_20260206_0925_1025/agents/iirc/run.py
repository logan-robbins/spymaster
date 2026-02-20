"""Intensity Imbalance Rate-of-Change (IIRC) experiment.

Captures order-flow toxicity momentum: the rate of change of the add-to-pull
intensity ratio on each side of the book.

Signal construction:
    1. Sum v_add, v_pull, v_fill by side for band [1..16] ticks from spot.
    2. Intensity ratio per side with Laplace smoothing (eps=1.0):
       ratio_bid = add_rate_bid / (pull_rate_bid + fill_rate_bid + 1.0)
       ratio_ask = add_rate_ask / (pull_rate_ask + fill_rate_ask + 1.0)
    3. Log imbalance:
       imbalance = log(ratio_bid + 1.0) - log(ratio_ask + 1.0)
       Positive = bid adding more relative to pulling (bullish).
    4. Rate of change via rolling OLS slope (windows 10, 30).
    5. Combined signal: 0.6 * d_fast + 0.4 * d_slow.
    6. Noise floor filter: zero out where |imbalance| < 0.1.
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
    rolling_ols_slope,
    sweep_thresholds,
    write_results,
)

# ---------------------------------------------------------------------------
# Column slices: (n_bins, 101) grid. col 50 = k=0 (spot).
# Bid band k=-16..-1 -> cols 34..49 (slice 34:50)
# Ask band k=+1..+16 -> cols 51..66 (slice 51:67)
# ---------------------------------------------------------------------------
BID_SLICE = slice(34, 50)  # 16 ticks below spot
ASK_SLICE = slice(51, 67)  # 16 ticks above spot

EPS = 1.0  # Laplace smoothing in intensity ratio denominator

FAST_WINDOW = 10
SLOW_WINDOW = 30
FAST_WEIGHT = 0.6
SLOW_WEIGHT = 0.4

NOISE_FLOOR = 0.1  # zero out signal where |imbalance| < this

COOLDOWN_BINS = 20


def compute_intensity_ratios(
    v_add: np.ndarray,
    v_pull: np.ndarray,
    v_fill: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute add-to-(pull+fill) intensity ratios per side.

    Args:
        v_add: (n_bins, 101) add velocity grid.
        v_pull: (n_bins, 101) pull velocity grid.
        v_fill: (n_bins, 101) fill velocity grid.

    Returns:
        (ratio_bid, ratio_ask) each (n_bins,).
        ratio = add_rate / (pull_rate + fill_rate + eps)
    """
    # Sum velocities across the 16-tick band on each side
    add_rate_bid = v_add[:, BID_SLICE].sum(axis=1)
    pull_rate_bid = v_pull[:, BID_SLICE].sum(axis=1)
    fill_rate_bid = v_fill[:, BID_SLICE].sum(axis=1)

    add_rate_ask = v_add[:, ASK_SLICE].sum(axis=1)
    pull_rate_ask = v_pull[:, ASK_SLICE].sum(axis=1)
    fill_rate_ask = v_fill[:, ASK_SLICE].sum(axis=1)

    ratio_bid = add_rate_bid / (pull_rate_bid + fill_rate_bid + EPS)
    ratio_ask = add_rate_ask / (pull_rate_ask + fill_rate_ask + EPS)

    return ratio_bid, ratio_ask


def compute_log_imbalance(
    ratio_bid: np.ndarray,
    ratio_ask: np.ndarray,
) -> np.ndarray:
    """Log imbalance between bid and ask intensity ratios.

    Formula:
        imbalance = log(ratio_bid + 1.0) - log(ratio_ask + 1.0)

    Positive = bid side net adding more relative to pulling (bullish).

    Args:
        ratio_bid: (n_bins,) bid-side intensity ratio.
        ratio_ask: (n_bins,) ask-side intensity ratio.

    Returns:
        (n_bins,) log imbalance.
    """
    return np.log(ratio_bid + 1.0) - np.log(ratio_ask + 1.0)


def compute_signal(imbalance: np.ndarray) -> np.ndarray:
    """Compute rate-of-change signal from log imbalance.

    Steps:
        1. Fast and slow rolling OLS slopes of imbalance.
        2. Weighted combination: 0.6 * fast + 0.4 * slow.
        3. NaN replaced with 0.
        4. Noise floor filter: zero where |imbalance| < 0.1.

    Args:
        imbalance: (n_bins,) log imbalance time series.

    Returns:
        (n_bins,) final signal.
    """
    print(f"  rolling OLS slope (fast window={FAST_WINDOW}) ...", flush=True)
    d_fast = rolling_ols_slope(imbalance, FAST_WINDOW)

    print(f"  rolling OLS slope (slow window={SLOW_WINDOW}) ...", flush=True)
    d_slow = rolling_ols_slope(imbalance, SLOW_WINDOW)

    # Weighted combination, NaN -> 0
    raw_signal = FAST_WEIGHT * d_fast + SLOW_WEIGHT * d_slow
    raw_signal = np.nan_to_num(raw_signal, nan=0.0)

    # Noise floor: zero out where imbalance is balanced (no directional info)
    noise_mask = np.abs(imbalance) >= NOISE_FLOOR
    signal = raw_signal * noise_mask.astype(np.float64)

    return signal


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset (v_add, v_pull, v_fill) ...", flush=True)
    ds = load_dataset(columns=["v_add", "v_pull", "v_fill"])
    v_add: np.ndarray = ds["v_add"]
    v_pull: np.ndarray = ds["v_pull"]
    v_fill: np.ndarray = ds["v_fill"]
    mid_price: np.ndarray = ds["mid_price"]
    ts_ns: np.ndarray = ds["ts_ns"]
    n_bins: int = ds["n_bins"]
    print(f"  n_bins={n_bins}, grid shape=({n_bins}, 101)", flush=True)
    print(f"  load time: {time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 2. Intensity ratios
    # ------------------------------------------------------------------
    print("Computing intensity ratios ...", flush=True)
    t1 = time.perf_counter()
    ratio_bid, ratio_ask = compute_intensity_ratios(v_add, v_pull, v_fill)
    print(f"  done in {time.perf_counter() - t1:.3f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Log imbalance
    # ------------------------------------------------------------------
    print("Computing log imbalance ...", flush=True)
    t2 = time.perf_counter()
    imbalance = compute_log_imbalance(ratio_bid, ratio_ask)
    print(f"  done in {time.perf_counter() - t2:.3f}s", flush=True)

    # ------------------------------------------------------------------
    # 4. Rate-of-change signal
    # ------------------------------------------------------------------
    print("Computing rate-of-change signal ...", flush=True)
    t3 = time.perf_counter()
    signal = compute_signal(imbalance)
    print(f"  done in {time.perf_counter() - t3:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 5. Signal distribution (non-zero values)
    # ------------------------------------------------------------------
    nonzero = signal[signal != 0.0]
    if len(nonzero) > 0:
        pcts = np.percentile(nonzero, [5, 25, 50, 75, 95])
        print(
            f"\nSignal distribution (n_nonzero={len(nonzero)} / {len(signal)}):",
            flush=True,
        )
        print(
            f"  p5={pcts[0]:.6f}  p25={pcts[1]:.6f}  p50={pcts[2]:.6f}  "
            f"p75={pcts[3]:.6f}  p95={pcts[4]:.6f}",
            flush=True,
        )
    else:
        print("\nWARNING: All signal values are zero.", flush=True)

    # ------------------------------------------------------------------
    # 6. Calibrate thresholds from distribution
    # ------------------------------------------------------------------
    # Start with specified thresholds; adjust if signal range is different
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]

    # If p95 of abs(nonzero) is much larger/smaller, add appropriate levels
    if len(nonzero) > 0:
        abs_p95 = np.percentile(np.abs(nonzero), 95)
        if abs_p95 > 0.1:
            thresholds.extend([0.08, 0.10])
        if abs_p95 > 0.5:
            thresholds.extend([0.20, 0.50])
        thresholds = sorted(set(thresholds))

    print(
        f"\nSweeping thresholds {thresholds} "
        f"(cooldown={COOLDOWN_BINS}, warmup={WARMUP_BINS}) ...",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 7. Threshold sweep
    # ------------------------------------------------------------------
    results = sweep_thresholds(
        signal=signal,
        thresholds=thresholds,
        cooldown_bins=COOLDOWN_BINS,
        mid_price=mid_price,
        ts_ns=ts_ns,
    )

    # ------------------------------------------------------------------
    # 8. Report + write
    # ------------------------------------------------------------------
    print("\nResults:", flush=True)
    for r in results:
        tp = r["tp_rate"]
        tp_str = f"{tp:.1%}" if not np.isnan(tp) else "N/A"
        print(
            f"  thr={r['threshold']:.4f}  n={r['n_signals']:>4d}  "
            f"TP={tp_str}  mean_pnl={r['mean_pnl_ticks']:+.2f}t  "
            f"evts/hr={r['events_per_hour']:.1f}",
            flush=True,
        )

    params = {
        "columns": ["v_add", "v_pull", "v_fill"],
        "bid_slice": [34, 50],
        "ask_slice": [51, 67],
        "band_width_ticks": 16,
        "eps": EPS,
        "fast_window": FAST_WINDOW,
        "slow_window": SLOW_WINDOW,
        "fast_weight": FAST_WEIGHT,
        "slow_weight": SLOW_WEIGHT,
        "noise_floor": NOISE_FLOOR,
        "cooldown_bins": COOLDOWN_BINS,
        "warmup_bins": WARMUP_BINS,
    }

    out_path = write_results(
        agent_name="iirc",
        experiment_name="intensity_imbalance_rate_of_change",
        params=params,
        results_by_threshold=results,
    )

    # Best threshold (min 5 signals)
    valid = [r for r in results if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
        print(
            f"\nBest: thr={best['threshold']:.4f}  "
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
