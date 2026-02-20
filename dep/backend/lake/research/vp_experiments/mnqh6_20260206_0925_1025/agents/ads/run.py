"""Asymmetric Derivative Slope (ADS) experiment.

Computes directional asymmetry across three spatial bands (inner/mid/outer)
using v_add and v_pull grid columns, then takes rolling OLS slopes of the
combined asymmetry, robust z-scores them, and blends into a single signal.

Signal construction:
    1. Per-band asymmetry:
       add_asym(band) = mean(v_add[bid_cols]) - mean(v_add[ask_cols])
       pull_asym(band) = mean(v_pull[ask_cols]) - mean(v_pull[bid_cols])
    2. Combined: weighted sum of (add_asym + pull_asym) per band,
       weights = 1/sqrt(band_width), normalized.
    3. Rolling OLS slope over windows [10, 25, 50].
    4. Robust z-score each slope (200-bin window).
    5. Blend: 0.4*tanh(z10/3) + 0.35*tanh(z25/3) + 0.25*tanh(z50/3).
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
    robust_zscore,
    sweep_thresholds,
    write_results,
)

# ---------------------------------------------------------------------------
# Band definitions: column indices into the (n_bins, 101) grid.
# col 50 = k=0 (spot). Bid side < 50, ask side > 50.
# ---------------------------------------------------------------------------
BANDS: list[dict[str, object]] = [
    {
        "name": "inner",
        "bid_cols": list(range(47, 50)),     # k=-3..-1
        "ask_cols": list(range(51, 54)),      # k=+1..+3
        "width": 3,
    },
    {
        "name": "mid",
        "bid_cols": list(range(39, 47)),      # k=-11..-4
        "ask_cols": list(range(54, 62)),       # k=+4..+11
        "width": 8,
    },
    {
        "name": "outer",
        "bid_cols": list(range(27, 39)),      # k=-23..-12
        "ask_cols": list(range(62, 74)),       # k=+12..+23
        "width": 12,
    },
]

SLOPE_WINDOWS: list[int] = [10, 25, 50]
ZSCORE_WINDOW: int = 200
BLEND_WEIGHTS: list[float] = [0.40, 0.35, 0.25]
BLEND_SCALE: float = 3.0

THRESHOLDS: list[float] = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
COOLDOWN_BINS: int = 30


def compute_combined_asymmetry(
    v_add: np.ndarray,
    v_pull: np.ndarray,
) -> np.ndarray:
    """Compute bandwidth-weighted combined asymmetry across all bands.

    Args:
        v_add: (n_bins, 101) add velocity grid.
        v_pull: (n_bins, 101) pull velocity grid.

    Returns:
        (n_bins,) combined asymmetry signal.
    """
    n_bins = v_add.shape[0]
    combined = np.zeros(n_bins, dtype=np.float64)
    total_weight = 0.0

    for band in BANDS:
        bid_cols = band["bid_cols"]
        ask_cols = band["ask_cols"]
        bw = band["width"]
        w = 1.0 / np.sqrt(bw)
        total_weight += w

        # add_asym: positive = more adding on bid side = bullish
        add_asym = v_add[:, bid_cols].mean(axis=1) - v_add[:, ask_cols].mean(axis=1)

        # pull_asym: positive = more pulling on ask side = bullish
        pull_asym = v_pull[:, ask_cols].mean(axis=1) - v_pull[:, bid_cols].mean(axis=1)

        combined += w * (add_asym + pull_asym)

    combined /= total_weight
    return combined


def compute_signal(combined_asym: np.ndarray) -> np.ndarray:
    """From combined asymmetry, compute blended slope z-score signal.

    Steps:
        1. Rolling OLS slope at each window in SLOPE_WINDOWS.
        2. Robust z-score each slope series (ZSCORE_WINDOW).
        3. Blend via tanh compression and BLEND_WEIGHTS.

    Args:
        combined_asym: (n_bins,) combined asymmetry.

    Returns:
        (n_bins,) final signal in roughly [-1, +1].
    """
    n = len(combined_asym)
    signal = np.zeros(n, dtype=np.float64)

    for i, win in enumerate(SLOPE_WINDOWS):
        print(f"  slope window={win} ...", flush=True)
        slope = rolling_ols_slope(combined_asym, win)
        z = robust_zscore(slope, ZSCORE_WINDOW)
        signal += BLEND_WEIGHTS[i] * np.tanh(z / BLEND_SCALE)

    return signal


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset (v_add, v_pull) ...", flush=True)
    ds = load_dataset(columns=["v_add", "v_pull"])
    v_add: np.ndarray = ds["v_add"]
    v_pull: np.ndarray = ds["v_pull"]
    mid_price: np.ndarray = ds["mid_price"]
    ts_ns: np.ndarray = ds["ts_ns"]
    n_bins: int = ds["n_bins"]
    print(f"  n_bins={n_bins}, grid shape=({n_bins}, 101)", flush=True)
    print(f"  load time: {time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 2. Combined asymmetry
    # ------------------------------------------------------------------
    print("Computing combined asymmetry ...", flush=True)
    t1 = time.perf_counter()
    combined_asym = compute_combined_asymmetry(v_add, v_pull)
    print(f"  done in {time.perf_counter() - t1:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Slope + z-score + blend
    # ------------------------------------------------------------------
    print("Computing slopes, z-scores, and blended signal ...", flush=True)
    t2 = time.perf_counter()
    signal = compute_signal(combined_asym)
    print(f"  done in {time.perf_counter() - t2:.1f}s", flush=True)

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
        "columns": ["v_add", "v_pull"],
        "bands": [
            {"name": b["name"], "width": b["width"],
             "bid_cols": b["bid_cols"], "ask_cols": b["ask_cols"]}
            for b in BANDS
        ],
        "slope_windows": SLOPE_WINDOWS,
        "zscore_window": ZSCORE_WINDOW,
        "blend_weights": BLEND_WEIGHTS,
        "blend_scale": BLEND_SCALE,
        "cooldown_bins": COOLDOWN_BINS,
        "warmup_bins": WARMUP_BINS,
    }

    out_path = write_results(
        agent_name="ads",
        experiment_name="asymmetric_derivative_slope",
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
