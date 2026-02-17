"""Spatial Pressure Gradient (SPG) experiment.

Computes directional trading signals from the spatial first derivative
(central difference) of pressure and vacuum fields around the spot price.

Signal logic:
  - Pressure gradient above spot (ask side) indicates resistance walls.
  - Vacuum gradient above spot (ask side) indicates absorption pull.
  - Net signal blends wall and pull gradients with dual-EMA smoothing
    and a spatial curvature correction term.

Formula:
  wall_signal  = mean(dP/dk on bid side) - mean(dP/dk on ask side)
  pull_signal  = mean(dV/dk on ask side) - mean(dV/dk on bid side)
  net          = -wall_signal + pull_signal
  smoothed     = 0.6 * EMA_fast(net) + 0.4 * EMA_slow(net)
  curv_signal  = -mean(d2P/dk2 near spot), EMA-smoothed
  final        = 0.7 * smoothed + 0.3 * curv_signal
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# -- harness import ----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_harness import (
    load_dataset,
    sweep_thresholds,
    write_results,
    WARMUP_BINS,
    N_TICKS,
    K_MIN,
    K_MAX,
)


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

def ema_1d(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average over a 1-D array.

    Args:
        arr: Input signal, shape (n,).
        alpha: Smoothing factor in (0, 1]. Higher = more responsive.

    Returns:
        EMA-smoothed array of same shape, dtype float64.
    """
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- 1. Load dataset -----------------------------------------------------
    print("[SPG] Loading dataset (pressure_variant, vacuum_variant) ...")
    ds = load_dataset(columns=["pressure_variant", "vacuum_variant"])
    P = ds["pressure_variant"]   # (n_bins, 101)
    V = ds["vacuum_variant"]     # (n_bins, 101)
    mid_price = ds["mid_price"]  # (n_bins,)
    ts_ns = ds["ts_ns"]          # (n_bins,)
    n_bins = ds["n_bins"]
    print(f"[SPG] Loaded {n_bins} bins, grid shape {P.shape}")

    # -- 2. Spatial first derivative (central difference along k axis) -------
    #    np.gradient computes central differences for interior points and
    #    one-sided differences at boundaries. axis=1 = along k.
    print("[SPG] Computing spatial gradients dP/dk, dV/dk ...")
    dP_dk = np.gradient(P, axis=1)  # (n_bins, 101)
    dV_dk = np.gradient(V, axis=1)  # (n_bins, 101)

    # -- 3. Mean gradient in bands around spot (col 50 = k=0) ----------------
    #    Ask side (above spot): cols 51..66  => k = +1 .. +16
    #    Bid side (below spot): cols 34..49  => k = -16 .. -1
    ASK_LO, ASK_HI = 51, 67   # slice [51:67] = 16 columns
    BID_LO, BID_HI = 34, 50   # slice [34:50] = 16 columns

    grad_P_above = np.mean(dP_dk[:, ASK_LO:ASK_HI], axis=1)  # (n_bins,)
    grad_P_below = np.mean(dP_dk[:, BID_LO:BID_HI], axis=1)  # (n_bins,)
    grad_V_above = np.mean(dV_dk[:, ASK_LO:ASK_HI], axis=1)  # (n_bins,)
    grad_V_below = np.mean(dV_dk[:, BID_LO:BID_HI], axis=1)  # (n_bins,)

    # -- 4. Directional signals ----------------------------------------------
    #    wall_signal > 0 => stronger pressure wall above => bearish
    #    pull_signal > 0 => stronger vacuum pull above   => bullish
    wall_signal = grad_P_below - grad_P_above
    pull_signal = grad_V_above - grad_V_below
    net = -wall_signal + pull_signal
    print(f"[SPG] net signal: mean={np.mean(net):.6f}, std={np.std(net):.6f}")

    # -- 5. Dual EMA smoothing -----------------------------------------------
    FAST_SPAN = 5    # ~500ms at 100ms bins
    SLOW_SPAN = 20   # ~2s
    alpha_fast = 2.0 / (FAST_SPAN + 1)   # 0.333
    alpha_slow = 2.0 / (SLOW_SPAN + 1)   # 0.095

    ema_fast = ema_1d(net, alpha_fast)
    ema_slow = ema_1d(net, alpha_slow)
    smoothed = 0.6 * ema_fast + 0.4 * ema_slow

    # -- 6. Spatial curvature (second derivative near spot) ------------------
    #    d2P/dk2[i] = P[i+1] + P[i-1] - 2*P[i]
    #    Evaluate at k = -2, -1, 0, +1, +2 => cols 48, 49, 50, 51, 52
    CURV_COLS = [48, 49, 50, 51, 52]
    curv_raw = np.zeros(n_bins, dtype=np.float64)
    for c in CURV_COLS:
        curv_raw += P[:, c + 1] + P[:, c - 1] - 2.0 * P[:, c]
    curv_raw /= len(CURV_COLS)

    curv_signal_raw = -curv_raw  # negate: positive curvature = wall = bearish
    CURV_SPAN = 10
    alpha_curv = 2.0 / (CURV_SPAN + 1)  # 0.182
    curv_signal = ema_1d(curv_signal_raw, alpha_curv)

    # -- 7. Final composite signal -------------------------------------------
    final = 0.7 * smoothed + 0.3 * curv_signal

    # -- Diagnostics: signal percentile distribution -------------------------
    post_warmup = final[WARMUP_BINS:]
    pcts = np.percentile(post_warmup, [5, 25, 50, 75, 95])
    print(f"[SPG] final signal percentiles (post-warmup):")
    print(f"       p5={pcts[0]:.6f}  p25={pcts[1]:.6f}  p50={pcts[2]:.6f}"
          f"  p75={pcts[3]:.6f}  p95={pcts[4]:.6f}")

    # -- Adaptive thresholds based on signal scale ---------------------------
    abs_p75 = max(abs(pcts[1]), abs(pcts[3]))  # ~p75 of |signal|
    abs_p95 = max(abs(pcts[0]), abs(pcts[4]))  # ~p95 of |signal|
    iqr = pcts[3] - pcts[1]

    # Build thresholds spanning the signal's actual range
    thresholds = sorted(set([
        round(0.5 * iqr, 6),
        round(1.0 * iqr, 6),
        round(abs_p75, 6),
        round(0.5 * (abs_p75 + abs_p95), 6),
        round(abs_p95, 6),
    ]))
    # Remove near-zero thresholds
    thresholds = [t for t in thresholds if t > 1e-8]
    if not thresholds:
        print("[SPG] ERROR: signal is effectively zero, no valid thresholds")
        return
    print(f"[SPG] Adaptive thresholds: {thresholds}")

    # -- 8. Sweep thresholds -------------------------------------------------
    COOLDOWN_BINS = 20
    print(f"[SPG] Sweeping {len(thresholds)} thresholds, cooldown={COOLDOWN_BINS} bins ...")
    results = sweep_thresholds(
        signal=final,
        thresholds=thresholds,
        cooldown_bins=COOLDOWN_BINS,
        mid_price=mid_price,
        ts_ns=ts_ns,
    )

    # -- Report each threshold -----------------------------------------------
    for r in results:
        tp = r["tp_rate"]
        tp_str = f"{tp:.1%}" if not np.isnan(tp) else "n/a"
        print(f"  thr={r['threshold']:.6f}  signals={r['n_signals']:>4d}"
              f"  TP={tp_str}  SL={r.get('sl_rate', float('nan')):.1%}"
              f"  timeout={r.get('timeout_rate', float('nan')):.1%}"
              f"  ev/hr={r.get('events_per_hour', 0):.1f}")

    # -- Best result (min 5 signals) -----------------------------------------
    valid = [r for r in results if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
        print(f"\n[SPG] BEST: threshold={best['threshold']:.6f}"
              f"  TP={best['tp_rate']:.1%}  signals={best['n_signals']}"
              f"  mean_pnl={best.get('mean_pnl_ticks', float('nan')):.2f}t")
    else:
        best = results[0] if results else {}
        print("[SPG] No threshold produced >= 5 signals")

    # -- Write results -------------------------------------------------------
    params = {
        "ask_band": [ASK_LO, ASK_HI],
        "bid_band": [BID_LO, BID_HI],
        "fast_span": FAST_SPAN,
        "slow_span": SLOW_SPAN,
        "curv_cols": CURV_COLS,
        "curv_span": CURV_SPAN,
        "blend_smooth": 0.7,
        "blend_curv": 0.3,
        "ema_fast_weight": 0.6,
        "ema_slow_weight": 0.4,
        "cooldown_bins": COOLDOWN_BINS,
        "thresholds": thresholds,
    }
    out_path = write_results(
        agent_name="spg",
        experiment_name="spatial_pressure_gradient",
        params=params,
        results_by_threshold=results,
    )
    print(f"[SPG] Results written to {out_path}")


if __name__ == "__main__":
    main()
