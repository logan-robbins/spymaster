"""Pressure Front Propagation (PFP) experiment.

Detects when inner-tick order activity leads outer-tick activity, indicating
aggressive directional intent propagating outward from the BBO.

Signal construction:
    1. Zones:
       - Inner bid: k=-3..-1 (cols 47-49), Inner ask: k=+1..+3 (cols 51-53)
       - Outer bid: k=-12..-5 (cols 38-45), Outer ask: k=+5..+12 (cols 55-62)
    2. Activity intensity per zone:
       - I_zone[t] = mean(v_add[t, zone_cols] + v_fill[t, zone_cols])
    3. Lead-lag via EMA of lagged vs unlagged cross-products:
       - lead_metric_bid[t] = ema(I_inner_bid[t] * I_outer_bid[t-L]) /
                              (ema(I_inner_bid[t] * I_outer_bid[t]) + eps)
       - Ratio > 1 => inner activity L bins ago predicts outer activity now.
    4. Directional signal from add/fill (building) channel:
       - signal = lead_metric_bid - lead_metric_ask
       - Positive => bid-side inner leads outer more => bullish
    5. Pull channel (cancellation propagation):
       - Same zones/lead-lag on v_pull.
       - pull_signal = pull_lead_ask - pull_lead_bid
       - Pull on ask side leading => bullish
    6. Final = 0.6 * signal + 0.4 * pull_signal
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
    sweep_thresholds,
    write_results,
)

# ---------------------------------------------------------------------------
# Zone definitions: column indices into (n_bins, 101) grid.
# col 50 = k=0 (spot). Bid side < 50, ask side > 50.
# ---------------------------------------------------------------------------
INNER_BID_COLS: list[int] = [47, 48, 49]       # k = -3, -2, -1
INNER_ASK_COLS: list[int] = [51, 52, 53]       # k = +1, +2, +3
OUTER_BID_COLS: list[int] = list(range(38, 46)) # k = -12 .. -5
OUTER_ASK_COLS: list[int] = list(range(55, 63)) # k = +5 .. +12

# EMA / lead-lag parameters
LAG_BINS: int = 5
EMA_ALPHA: float = 0.1
EPS: float = 1e-12

# Blending weights: add/fill channel vs pull channel
ADD_WEIGHT: float = 0.6
PULL_WEIGHT: float = 0.4

# Sweep parameters
COOLDOWN_BINS: int = 30


def ema_1d(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average over a 1D array.

    EMA[0] = arr[0]
    EMA[t] = alpha * arr[t] + (1 - alpha) * EMA[t-1]

    Args:
        arr: Input array of shape (n,).
        alpha: Smoothing factor in (0, 1].

    Returns:
        EMA array of same shape.
    """
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    out[0] = arr[0]
    one_minus_alpha = 1.0 - alpha
    for i in range(1, n):
        out[i] = alpha * arr[i] + one_minus_alpha * out[i - 1]
    return out


def compute_zone_intensity(
    v_add: np.ndarray,
    v_fill: np.ndarray,
    zone_cols: list[int],
) -> np.ndarray:
    """Compute building-activity intensity for a zone.

    I_zone[t] = mean(v_add[t, zone_cols] + v_fill[t, zone_cols])

    Args:
        v_add: (n_bins, 101) add velocity grid.
        v_fill: (n_bins, 101) fill velocity grid.
        zone_cols: Column indices defining the zone.

    Returns:
        (n_bins,) intensity array.
    """
    return (v_add[:, zone_cols] + v_fill[:, zone_cols]).mean(axis=1)


def compute_lead_metric(
    inner: np.ndarray,
    outer: np.ndarray,
    lag: int,
    alpha: float,
) -> np.ndarray:
    """Compute EMA-based lead-lag metric between inner and outer zones.

    lead_metric[t] = ema(inner[t] * outer[t - lag]) /
                     (ema(inner[t] * outer[t]) + eps)

    When the ratio > 1, inner activity `lag` bins ago correlates more strongly
    with current outer activity than contemporaneous inner does, indicating
    that inner leads outer.

    The lagged product uses outer[t - lag] paired with inner[t]. For bins
    where t < lag, the lagged product is set to 0 (cold start).

    Args:
        inner: (n_bins,) inner zone intensity.
        outer: (n_bins,) outer zone intensity.
        lag: Number of bins to lag the outer series.
        alpha: EMA smoothing factor.

    Returns:
        (n_bins,) lead-lag ratio.
    """
    n = len(inner)

    # Unlagged product: inner[t] * outer[t]
    prod_unlagged = inner * outer

    # Lagged product: inner[t] * outer[t - lag]
    # For t < lag, outer[t-lag] doesn't exist -> use 0
    outer_lagged = np.zeros(n, dtype=np.float64)
    outer_lagged[lag:] = outer[:n - lag]
    prod_lagged = inner * outer_lagged

    ema_lagged = ema_1d(prod_lagged, alpha)
    ema_unlagged = ema_1d(prod_unlagged, alpha)

    return ema_lagged / (ema_unlagged + EPS)


def compute_pfp_signal(
    v_add: np.ndarray,
    v_fill: np.ndarray,
    v_pull: np.ndarray,
) -> np.ndarray:
    """Compute the full PFP signal.

    Steps:
        1. Zone intensities for add/fill and pull channels.
        2. Lead-lag metrics for bid and ask sides per channel.
        3. Directional signals per channel.
        4. Blend channels.

    Args:
        v_add: (n_bins, 101) add velocity grid.
        v_fill: (n_bins, 101) fill velocity grid.
        v_pull: (n_bins, 101) pull velocity grid.

    Returns:
        (n_bins,) final PFP signal. Positive = bullish.
    """
    # --- Add/fill (building) channel ---
    print("  Computing add/fill zone intensities ...", flush=True)
    i_inner_bid = compute_zone_intensity(v_add, v_fill, INNER_BID_COLS)
    i_inner_ask = compute_zone_intensity(v_add, v_fill, INNER_ASK_COLS)
    i_outer_bid = compute_zone_intensity(v_add, v_fill, OUTER_BID_COLS)
    i_outer_ask = compute_zone_intensity(v_add, v_fill, OUTER_ASK_COLS)

    print("  Computing add/fill lead-lag metrics ...", flush=True)
    lead_bid = compute_lead_metric(i_inner_bid, i_outer_bid, LAG_BINS, EMA_ALPHA)
    lead_ask = compute_lead_metric(i_inner_ask, i_outer_ask, LAG_BINS, EMA_ALPHA)

    # Positive = bid-side inner leads outer more => bullish
    add_signal = lead_bid - lead_ask

    # --- Pull (cancellation) channel ---
    print("  Computing pull zone intensities ...", flush=True)
    p_inner_bid = v_pull[:, INNER_BID_COLS].mean(axis=1)
    p_inner_ask = v_pull[:, INNER_ASK_COLS].mean(axis=1)
    p_outer_bid = v_pull[:, OUTER_BID_COLS].mean(axis=1)
    p_outer_ask = v_pull[:, OUTER_ASK_COLS].mean(axis=1)

    print("  Computing pull lead-lag metrics ...", flush=True)
    pull_lead_bid = compute_lead_metric(p_inner_bid, p_outer_bid, LAG_BINS, EMA_ALPHA)
    pull_lead_ask = compute_lead_metric(p_inner_ask, p_outer_ask, LAG_BINS, EMA_ALPHA)

    # Pull on ask side leading = bullish (aggressive ask-side cancellation)
    pull_signal = pull_lead_ask - pull_lead_bid

    # --- Blend ---
    final = ADD_WEIGHT * add_signal + PULL_WEIGHT * pull_signal
    return final


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
    # 2. Compute PFP signal
    # ------------------------------------------------------------------
    print("Computing PFP signal ...", flush=True)
    t1 = time.perf_counter()
    signal = compute_pfp_signal(v_add, v_fill, v_pull)
    print(f"  signal compute time: {time.perf_counter() - t1:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Percentile distribution for threshold calibration
    # ------------------------------------------------------------------
    active_signal = signal[WARMUP_BINS:]
    pcts = np.nanpercentile(active_signal, [5, 25, 50, 75, 95])
    print(
        f"Signal distribution (post-warmup): "
        f"p5={pcts[0]:.6f}  p25={pcts[1]:.6f}  p50={pcts[2]:.6f}  "
        f"p75={pcts[3]:.6f}  p95={pcts[4]:.6f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4. Calibrate thresholds from distribution
    # ------------------------------------------------------------------
    # Use percentile-based thresholds centered on observed spread
    abs_signal = np.abs(active_signal)
    abs_pcts = np.nanpercentile(abs_signal, [50, 70, 80, 90, 95, 99])
    thresholds = sorted(set(round(float(v), 6) for v in abs_pcts if v > 0))
    if not thresholds:
        thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    print(f"Calibrated thresholds: {thresholds}", flush=True)

    # ------------------------------------------------------------------
    # 5. Threshold sweep
    # ------------------------------------------------------------------
    print(
        f"Sweeping {len(thresholds)} thresholds "
        f"(cooldown={COOLDOWN_BINS}, warmup={WARMUP_BINS}) ...",
        flush=True,
    )
    results = sweep_thresholds(
        signal=signal,
        thresholds=thresholds,
        cooldown_bins=COOLDOWN_BINS,
        mid_price=mid_price,
        ts_ns=ts_ns,
    )

    # ------------------------------------------------------------------
    # 6. Report + write
    # ------------------------------------------------------------------
    for r in results:
        tp = r["tp_rate"]
        tp_str = f"{tp:.1%}" if not np.isnan(tp) else "N/A"
        print(
            f"  thr={r['threshold']:.6f}  n={r['n_signals']:>4d}  "
            f"TP={tp_str}  mean_pnl={r['mean_pnl_ticks']:+.2f}t  "
            f"evts/hr={r['events_per_hour']:.1f}",
            flush=True,
        )

    params = {
        "columns": ["v_add", "v_pull", "v_fill"],
        "inner_bid_cols": INNER_BID_COLS,
        "inner_ask_cols": INNER_ASK_COLS,
        "outer_bid_cols": OUTER_BID_COLS,
        "outer_ask_cols": OUTER_ASK_COLS,
        "lag_bins": LAG_BINS,
        "ema_alpha": EMA_ALPHA,
        "eps": EPS,
        "add_weight": ADD_WEIGHT,
        "pull_weight": PULL_WEIGHT,
        "cooldown_bins": COOLDOWN_BINS,
        "warmup_bins": WARMUP_BINS,
        "thresholds": thresholds,
    }

    out_path = write_results(
        agent_name="pfp",
        experiment_name="pressure_front_propagation",
        params=params,
        results_by_threshold=results,
    )

    # Best threshold (min 5 signals)
    valid = [r for r in results if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
        print(
            f"\nBest: thr={best['threshold']:.6f}  "
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
