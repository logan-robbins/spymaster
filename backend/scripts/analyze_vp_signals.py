"""Canonical fixed-bin spectrum analysis for vacuum-pressure.

Two modes:
  default  — per-cell spectrum health metrics (state shares, projections)
  regime   — directional micro-regime detection + TP/SL trade evaluation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

logger = logging.getLogger("analyze_vp_signals")

STATE_VACUUM = -1
STATE_NEUTRAL = 0
STATE_PRESSURE = 1


def _parse_et_timestamp_ns(dt: str, hhmm: str) -> int:
    return int(
        pd.Timestamp(f"{dt} {hhmm}:00", tz="America/New_York")
        .tz_convert("UTC")
        .value
    )


def _collect_bins(
    *,
    lake_root: Path,
    config: Any,
    dt: str,
    start_time: str | None,
    eval_end_ns: int,
    collect_pressure_vacuum: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray], np.ndarray,
           np.ndarray | None, np.ndarray | None]:
    from src.vacuum_pressure.stream_pipeline import stream_events

    ts_bins: List[int] = []
    mid_prices: List[float] = []
    scores: List[np.ndarray] = []
    states: List[np.ndarray] = []
    pressures: List[np.ndarray] = []
    vacuums: List[np.ndarray] = []
    projections: Dict[int, List[np.ndarray]] = {
        int(h): [] for h in config.projection_horizons_ms
    }

    for grid in stream_events(
        lake_root=lake_root,
        config=config,
        dt=dt,
        start_time=start_time,
    ):
        ts_ns = int(grid["ts_ns"])
        if ts_ns >= eval_end_ns:
            break

        buckets = grid["buckets"]
        if not buckets:
            continue

        ts_bins.append(ts_ns)
        mid_prices.append(float(grid["mid_price"]))

        score_row = np.array([float(b["spectrum_score"]) for b in buckets], dtype=np.float64)
        state_row = np.array([int(b["spectrum_state_code"]) for b in buckets], dtype=np.int8)
        scores.append(score_row)
        states.append(state_row)

        if collect_pressure_vacuum:
            pressures.append(np.array([float(b["pressure_variant"]) for b in buckets], dtype=np.float64))
            vacuums.append(np.array([float(b["vacuum_variant"]) for b in buckets], dtype=np.float64))

        for horizon in projections:
            key = f"proj_score_h{horizon}"
            projections[horizon].append(
                np.array([float(b[key]) for b in buckets], dtype=np.float64)
            )

    if not ts_bins:
        raise RuntimeError("No bins were captured for the requested window.")

    proj_out = {
        h: np.stack(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float64)
        for h, rows in projections.items()
    }

    pressure_out = np.stack(pressures, axis=0) if pressures else None
    vacuum_out = np.stack(vacuums, axis=0) if vacuums else None

    return (
        np.array(ts_bins, dtype=np.int64),
        np.array(mid_prices, dtype=np.float64),
        np.stack(scores, axis=0),
        proj_out,
        np.stack(states, axis=0),
        pressure_out,
        vacuum_out,
    )


def _projection_hit_rate(
    *,
    ts_ns: np.ndarray,
    score: np.ndarray,
    proj: np.ndarray,
    horizon_ms: int,
) -> Dict[str, float]:
    if len(ts_ns) == 0:
        return {"hit_rate": float("nan"), "n_pairs": 0.0}

    n = len(ts_ns)
    target_ts = ts_ns + int(horizon_ms * 1_000_000)
    j_idx = np.searchsorted(ts_ns, target_ts, side="left")
    valid = j_idx < n
    if not valid.any():
        return {"hit_rate": float("nan"), "n_pairs": 0.0}

    i_idx = np.where(valid)[0]
    pred_delta = proj[i_idx] - score[i_idx]
    real_delta = score[j_idx[valid]] - score[i_idx]

    pred_sign = np.sign(pred_delta)
    real_sign = np.sign(real_delta)

    active = (pred_sign != 0.0) & (real_sign != 0.0)
    if not active.any():
        return {"hit_rate": float("nan"), "n_pairs": 0.0}

    hit = np.mean(pred_sign[active] == real_sign[active])
    return {"hit_rate": float(hit), "n_pairs": float(active.sum())}


def _summarize(
    *,
    ts_ns: np.ndarray,
    mid_price: np.ndarray,
    score: np.ndarray,
    proj_by_h: Dict[int, np.ndarray],
    state_code: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, Any]:
    n_eval = int(eval_mask.sum())
    if n_eval < 2:
        raise RuntimeError("Insufficient evaluation bins (need at least 2).")

    score_eval = score[eval_mask]
    state_eval = state_code[eval_mask]
    mid_eval = mid_price[eval_mask]

    total_cells = float(score_eval.shape[0] * score_eval.shape[1])
    pressure_share = float((state_eval == STATE_PRESSURE).sum() / total_cells)
    neutral_share = float((state_eval == STATE_NEUTRAL).sum() / total_cells)
    vacuum_share = float((state_eval == STATE_VACUUM).sum() / total_cells)

    k_count = score_eval.shape[1]
    center = k_count // 2
    above = score_eval[:, center + 1 :]
    below = score_eval[:, :center]
    directional_edge = float(np.mean(below) - np.mean(above))

    mid_delta = np.diff(mid_eval)
    score_center = score_eval[:, center]
    score_delta = np.diff(score_center)
    nz = (score_delta != 0.0) & (mid_delta != 0.0)
    center_sign_hit_rate = float(np.mean(np.sign(score_delta[nz]) == np.sign(mid_delta[nz]))) if nz.any() else float("nan")

    projection_metrics: Dict[str, Dict[str, float]] = {}
    for horizon_ms, proj_all in proj_by_h.items():
        metrics = _projection_hit_rate(
            ts_ns=ts_ns[eval_mask],
            score=score_eval,
            proj=proj_all[eval_mask],
            horizon_ms=horizon_ms,
        )
        projection_metrics[str(horizon_ms)] = metrics

    return {
        "n_eval_bins": float(n_eval),
        "n_cells": float(k_count),
        "state_share_pressure": pressure_share,
        "state_share_neutral": neutral_share,
        "state_share_vacuum": vacuum_share,
        "directional_edge": directional_edge,
        "center_sign_hit_rate": center_sign_hit_rate,
        "projection": projection_metrics,
    }


# ---------------------------------------------------------------------------
# Regime mode: directional micro-regime detection + TP/SL trade evaluation
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _build_directional_edge(
    *,
    pressure: np.ndarray,
    vacuum: np.ndarray,
    bands: List[int],
    center: int,
    micro_windows: List[int],
) -> np.ndarray:
    """Build a smoothed directional edge signal from raw P/V per bin.

    For each band b in `bands`, computes composite C = (P-V)/(P+V+eps)
    averaged over the above-side [center+1, center+b] and below-side
    [center-b, center-1]. Band scores are aggregated with 1/sqrt(b) weights.

    direction_edge = below_composite - above_composite

    Then applies a multi-scale trailing mean rollup with 1/sqrt(w) weights.

    Args:
        pressure: (n_bins, n_cells) raw pressure_variant
        vacuum:   (n_bins, n_cells) raw vacuum_variant
        bands:    e.g. [4, 8, 16]
        center:   center cell index (k=0)
        micro_windows: e.g. [25, 50, 100, 200]

    Returns:
        (n_bins,) smoothed directional edge signal
    """
    n_bins = pressure.shape[0]
    n_cells = pressure.shape[1]

    # Per-band composites
    band_weights = np.array([1.0 / np.sqrt(b) for b in bands], dtype=np.float64)
    band_weights /= band_weights.sum()

    above_composite = np.zeros(n_bins, dtype=np.float64)
    below_composite = np.zeros(n_bins, dtype=np.float64)

    for i, b in enumerate(bands):
        # Above: cells center+1 to center+b (inclusive)
        a_start = min(center + 1, n_cells)
        a_end = min(center + b + 1, n_cells)
        if a_end > a_start:
            p_above = pressure[:, a_start:a_end]
            v_above = vacuum[:, a_start:a_end]
            c_above = (p_above - v_above) / (np.abs(p_above) + np.abs(v_above) + _EPS)
            above_composite += band_weights[i] * c_above.mean(axis=1)

        # Below: cells center-b to center-1 (inclusive)
        b_start = max(center - b, 0)
        b_end = center
        if b_end > b_start:
            p_below = pressure[:, b_start:b_end]
            v_below = vacuum[:, b_start:b_end]
            c_below = (p_below - v_below) / (np.abs(p_below) + np.abs(v_below) + _EPS)
            below_composite += band_weights[i] * c_below.mean(axis=1)

    raw_edge = below_composite - above_composite

    # Multi-scale trailing mean rollup
    window_weights = np.array([1.0 / np.sqrt(w) for w in micro_windows], dtype=np.float64)
    window_weights /= window_weights.sum()

    smoothed = np.zeros(n_bins, dtype=np.float64)
    for i, w in enumerate(micro_windows):
        kernel = np.ones(w) / w
        conv = np.convolve(raw_edge, kernel, mode="full")[:n_bins]
        # Align: for the first w-1 bins, use expanding mean
        for j in range(min(w - 1, n_bins)):
            conv[j] = raw_edge[:j + 1].mean()
        smoothed += window_weights[i] * conv

    return smoothed


def _detect_directional_signals(
    edge: np.ndarray,
    edge_threshold: float,
    cooldown_bins: int,
) -> List[Dict[str, Any]]:
    """Detect directional switch events from edge signal.

    A signal fires when edge crosses ±edge_threshold from a neutral/opposite
    state, subject to a cooldown.

    Returns list of dicts with: bin_idx, direction ("up" or "down").
    """
    signals: List[Dict[str, Any]] = []
    prev_state = "flat"
    last_signal_bin = -cooldown_bins

    for i in range(len(edge)):
        if edge[i] >= edge_threshold:
            cur_state = "up"
        elif edge[i] <= -edge_threshold:
            cur_state = "down"
        else:
            cur_state = "flat"

        if cur_state != "flat" and cur_state != prev_state:
            if i - last_signal_bin >= cooldown_bins:
                signals.append({"bin_idx": i, "direction": cur_state})
                last_signal_bin = i

        prev_state = cur_state

    return signals


def _evaluate_tp_sl(
    *,
    signals: List[Dict[str, Any]],
    mid_price: np.ndarray,
    ts_ns: np.ndarray,
    tp_ticks: int,
    sl_ticks: int,
    tick_size: float,
    max_hold_bins: int,
) -> Dict[str, Any]:
    """Evaluate TP/SL outcomes for directional signals.

    For each signal entry:
    - Enter at mid_price[bin_idx]
    - For "up": TP at entry + tp_ticks*tick, SL at entry - sl_ticks*tick
    - For "down": TP at entry - tp_ticks*tick, SL at entry + sl_ticks*tick
    - Timeout if neither hit within max_hold_bins

    Returns summary metrics.
    """
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
                continue  # skip bins with invalid BBO (snapshot/clear)
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
            else:  # down
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
            "tp_before_sl_rate": float("nan"),
            "sl_before_tp_rate": float("nan"),
            "timeout_rate": float("nan"),
            "events_per_hour": 0.0,
            "median_time_to_outcome_ms": float("nan"),
            "mean_pnl_ticks": float("nan"),
            "outcomes": [],
        }

    n_tp = sum(1 for o in outcomes if o["outcome"] == "tp")
    n_sl = sum(1 for o in outcomes if o["outcome"] == "sl")
    n_timeout = sum(1 for o in outcomes if o["outcome"] == "timeout")

    # Duration of evaluation window in hours
    eval_duration_hours = float(ts_ns[-1] - ts_ns[0]) / 3.6e12
    events_per_hour = n_total / eval_duration_hours if eval_duration_hours > 0 else 0.0

    times = [o["time_to_outcome_ms"] for o in outcomes if o["outcome"] != "timeout"]
    pnls = [o["pnl_ticks"] for o in outcomes]

    return {
        "n_signals": n_total,
        "n_up": sum(1 for o in outcomes if o["direction"] == "up"),
        "n_down": sum(1 for o in outcomes if o["direction"] == "down"),
        "tp_before_sl_rate": n_tp / n_total,
        "sl_before_tp_rate": n_sl / n_total,
        "timeout_rate": n_timeout / n_total,
        "events_per_hour": events_per_hour,
        "median_time_to_outcome_ms": float(np.median(times)) if times else float("nan"),
        "mean_pnl_ticks": float(np.mean(pnls)),
        "outcomes": outcomes,
    }


def _run_regime_mode(args: argparse.Namespace) -> None:
    """Directional micro-regime detection + TP/SL trade evaluation."""
    from src.vacuum_pressure.config import resolve_config

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"
    config = resolve_config(args.product_type, args.symbol, products_yaml_path)

    eval_start_ns = _parse_et_timestamp_ns(args.dt, args.eval_start)
    eval_end_ns = _parse_et_timestamp_ns(args.dt, args.eval_end)
    if eval_end_ns <= eval_start_ns:
        raise ValueError("eval_end must be strictly after eval_start")

    directional_bands = [int(x) for x in args.directional_bands.split(",")]
    micro_windows = [int(x) for x in args.micro_windows.split(",")]

    logger.info("Collecting bins for regime analysis...")
    ts_ns, mid_price, score, proj_by_h, state_code, pressure, vacuum = _collect_bins(
        lake_root=backend_root / "lake",
        config=config,
        dt=args.dt,
        start_time=args.start_time,
        eval_end_ns=eval_end_ns,
        collect_pressure_vacuum=True,
    )

    eval_mask = (ts_ns >= eval_start_ns) & (ts_ns < eval_end_ns)
    if not eval_mask.any():
        raise RuntimeError("No evaluation bins found in the requested interval.")

    # Extract eval window
    ts_eval = ts_ns[eval_mask]
    mid_eval = mid_price[eval_mask]
    score_eval = score[eval_mask]
    state_eval = state_code[eval_mask]
    pressure_eval = pressure[eval_mask]
    vacuum_eval = vacuum[eval_mask]

    n_cells = score_eval.shape[1]
    center = n_cells // 2

    # Spectrum health summary
    total_cells = float(score_eval.shape[0] * score_eval.shape[1])
    pressure_share = float((state_eval == STATE_PRESSURE).sum() / total_cells)
    neutral_share = float((state_eval == STATE_NEUTRAL).sum() / total_cells)
    vacuum_share = float((state_eval == STATE_VACUUM).sum() / total_cells)

    # Build directional edge signal from raw P/V
    edge = _build_directional_edge(
        pressure=pressure_eval,
        vacuum=vacuum_eval,
        bands=directional_bands,
        center=center,
        micro_windows=micro_windows,
    )

    # Detect directional signals
    signals = _detect_directional_signals(
        edge=edge,
        edge_threshold=args.edge_threshold,
        cooldown_bins=args.cooldown_bins,
    )
    logger.info("Detected %d directional signals", len(signals))

    # Evaluate TP/SL outcomes
    trade_results = _evaluate_tp_sl(
        signals=signals,
        mid_price=mid_eval,
        ts_ns=ts_eval,
        tp_ticks=args.tp_ticks,
        sl_ticks=args.sl_ticks,
        tick_size=config.tick_size,
        max_hold_bins=args.max_hold_snapshots,
    )

    # Print results
    print("\nDirectional Micro-Regime Analysis")
    print("=" * 72)
    print(f"Instrument:       {args.product_type}:{args.symbol}")
    print(f"Date:             {args.dt}")
    print(f"Window (ET):      {args.eval_start} - {args.eval_end}")
    print(f"Bands:            {directional_bands}")
    print(f"Micro windows:    {micro_windows}")
    print(f"Edge threshold:   {args.edge_threshold}")
    print(f"Cooldown bins:    {args.cooldown_bins}")
    print(f"TP ticks / SL ticks: {args.tp_ticks} / {args.sl_ticks}")
    print(f"Max hold bins:    {args.max_hold_snapshots}")

    print(f"\nSpectrum State Distribution")
    print("-" * 48)
    print(f"Eval bins:        {int(eval_mask.sum())}")
    print(f"Cells/bin:        {n_cells}")
    print(f"Pressure share:   {pressure_share:.4f}")
    print(f"Neutral share:    {neutral_share:.4f}")
    print(f"Vacuum share:     {vacuum_share:.4f}")

    print(f"\nDirectional Edge Signal")
    print("-" * 48)
    print(f"Mean edge:        {edge.mean():+.6f}")
    print(f"Std edge:         {edge.std():.6f}")
    print(f"Min/Max edge:     {edge.min():+.6f} / {edge.max():+.6f}")
    pct_up = (edge >= args.edge_threshold).sum() / len(edge)
    pct_down = (edge <= -args.edge_threshold).sum() / len(edge)
    pct_flat = 1.0 - pct_up - pct_down
    print(f"Time in up:       {pct_up:.4f}")
    print(f"Time in flat:     {pct_flat:.4f}")
    print(f"Time in down:     {pct_down:.4f}")

    print(f"\nTrade Evaluation (TP={args.tp_ticks} ticks, SL={args.sl_ticks} ticks)")
    print("-" * 48)
    tr = trade_results
    print(f"Total signals:    {tr['n_signals']}")
    if tr['n_signals'] > 0:
        print(f"  Up signals:     {tr['n_up']}")
        print(f"  Down signals:   {tr['n_down']}")
        print(f"TP before SL:     {tr['tp_before_sl_rate']:.4f}")
        print(f"SL before TP:     {tr['sl_before_tp_rate']:.4f}")
        print(f"Timeout:          {tr['timeout_rate']:.4f}")
        print(f"Events/hour:      {tr['events_per_hour']:.1f}")
        median_str = f"{tr['median_time_to_outcome_ms']:.0f}" if np.isfinite(tr['median_time_to_outcome_ms']) else "nan"
        print(f"Median time (ms): {median_str}")
        print(f"Mean PnL (ticks): {tr['mean_pnl_ticks']:+.2f}")

        # Per-direction breakdown
        for d in ["up", "down"]:
            d_outcomes = [o for o in tr["outcomes"] if o["direction"] == d]
            if d_outcomes:
                d_tp = sum(1 for o in d_outcomes if o["outcome"] == "tp")
                d_sl = sum(1 for o in d_outcomes if o["outcome"] == "sl")
                d_to = sum(1 for o in d_outcomes if o["outcome"] == "timeout")
                d_n = len(d_outcomes)
                d_pnl = np.mean([o["pnl_ticks"] for o in d_outcomes])
                print(f"\n  {d.upper()} signals ({d_n}):")
                print(f"    TP: {d_tp}/{d_n} ({d_tp/d_n:.4f})  SL: {d_sl}/{d_n} ({d_sl/d_n:.4f})  TO: {d_to}/{d_n} ({d_to/d_n:.4f})")
                print(f"    Mean PnL: {d_pnl:+.2f} ticks")

    # Individual trade log (first 20)
    if tr["outcomes"]:
        print(f"\nTrade Log (first 20 of {len(tr['outcomes'])})")
        print("-" * 72)
        for o in tr["outcomes"][:20]:
            ts_et = pd.Timestamp(ts_eval[o["bin_idx"]], unit="ns", tz="UTC").tz_convert("America/New_York")
            print(
                f"  {ts_et.strftime('%H:%M:%S.%f')[:-3]} {o['direction']:>4s} "
                f"entry=${o['entry_price']:.2f} -> {o['outcome']:>7s} "
                f"${o['outcome_price']:.2f} pnl={o['pnl_ticks']:+.1f}t "
                f"({o['time_to_outcome_ms']:.0f}ms)"
            )

    if args.json_output:
        out_payload = {
            "config": config.to_dict(),
            "mode": "regime",
            "dt": args.dt,
            "start_time": args.start_time,
            "eval_start": args.eval_start,
            "eval_end": args.eval_end,
            "params": {
                "directional_bands": directional_bands,
                "micro_windows": micro_windows,
                "edge_threshold": args.edge_threshold,
                "cooldown_bins": args.cooldown_bins,
                "tp_ticks": args.tp_ticks,
                "sl_ticks": args.sl_ticks,
                "max_hold_snapshots": args.max_hold_snapshots,
            },
            "spectrum": {
                "pressure_share": pressure_share,
                "neutral_share": neutral_share,
                "vacuum_share": vacuum_share,
            },
            "edge": {
                "mean": float(edge.mean()),
                "std": float(edge.std()),
                "pct_up": float(pct_up),
                "pct_flat": float(pct_flat),
                "pct_down": float(pct_down),
            },
            "trade_results": {
                k: v for k, v in trade_results.items() if k != "outcomes"
            },
            "trades": trade_results["outcomes"],
        }
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, indent=2, default=float))
        print(f"\nWrote JSON summary: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical fixed-bin per-cell spectrum analysis.",
    )
    parser.add_argument("--mode", default="default", choices=["default", "regime"])
    parser.add_argument("--product-type", default="future_mbo")
    parser.add_argument("--symbol", default="MNQH6")
    parser.add_argument("--dt", default="2026-02-06")
    parser.add_argument("--start-time", default="09:00")
    parser.add_argument("--eval-start", default="09:00")
    parser.add_argument("--eval-end", default="12:00")
    parser.add_argument("--json-output", default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # Regime mode parameters
    parser.add_argument("--directional-bands", default="4,8,16")
    parser.add_argument("--micro-windows", default="25,50,100,200")
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--cooldown-bins", type=int, default=50)
    parser.add_argument("--tp-ticks", type=int, default=8)
    parser.add_argument("--sl-ticks", type=int, default=4)
    parser.add_argument("--max-hold-snapshots", type=int, default=1200)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "regime":
        _run_regime_mode(args)
        return

    from src.vacuum_pressure.config import resolve_config

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"
    config = resolve_config(args.product_type, args.symbol, products_yaml_path)

    eval_start_ns = _parse_et_timestamp_ns(args.dt, args.eval_start)
    eval_end_ns = _parse_et_timestamp_ns(args.dt, args.eval_end)
    if eval_end_ns <= eval_start_ns:
        raise ValueError("eval_end must be strictly after eval_start")

    ts_ns, mid_price, score, proj_by_h, state_code, _, _ = _collect_bins(
        lake_root=backend_root / "lake",
        config=config,
        dt=args.dt,
        start_time=args.start_time,
        eval_end_ns=eval_end_ns,
    )

    eval_mask = (ts_ns >= eval_start_ns) & (ts_ns < eval_end_ns)
    if not eval_mask.any():
        raise RuntimeError("No evaluation bins found in the requested interval.")

    summary = _summarize(
        ts_ns=ts_ns,
        mid_price=mid_price,
        score=score,
        proj_by_h=proj_by_h,
        state_code=state_code,
        eval_mask=eval_mask,
    )

    print("\nCanonical Spectrum Analysis")
    print("-" * 64)
    print(f"Instrument:      {args.product_type}:{args.symbol}")
    print(f"Date:            {args.dt}")
    print(f"Window (ET):     {args.eval_start} - {args.eval_end}")
    print(f"Radius ticks:    {config.grid_radius_ticks}")
    print(f"Cell width (ms): {config.cell_width_ms}")
    print(f"Eval bins:       {int(summary['n_eval_bins'])}")
    print(f"Cells/bin:       {int(summary['n_cells'])}")
    print(f"Pressure share:  {summary['state_share_pressure']:.4f}")
    print(f"Neutral share:   {summary['state_share_neutral']:.4f}")
    print(f"Vacuum share:    {summary['state_share_vacuum']:.4f}")
    print(f"Directional edge:{summary['directional_edge']:+.6f}")
    print(f"Center hit-rate: {summary['center_sign_hit_rate']:.4f}")

    print("\nProjection sign hit-rates")
    for horizon_ms in sorted(config.projection_horizons_ms):
        m = summary["projection"][str(horizon_ms)]
        hit = m["hit_rate"]
        hit_str = f"{hit:.4f}" if np.isfinite(hit) else "nan"
        print(f"  {horizon_ms:>5d}ms -> hit_rate={hit_str} n_pairs={int(m['n_pairs'])}")

    if args.json_output:
        out_payload = {
            "config": config.to_dict(),
            "dt": args.dt,
            "start_time": args.start_time,
            "eval_start": args.eval_start,
            "eval_end": args.eval_end,
            "summary": summary,
        }
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, indent=2))
        print(f"\nWrote JSON summary: {out_path}")


if __name__ == "__main__":
    main()
