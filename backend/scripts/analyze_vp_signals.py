"""Canonical fixed-bin spectrum analysis for vacuum-pressure."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    from src.vacuum_pressure.stream_pipeline import stream_events

    ts_bins: List[int] = []
    mid_prices: List[float] = []
    scores: List[np.ndarray] = []
    states: List[np.ndarray] = []
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

    return (
        np.array(ts_bins, dtype=np.int64),
        np.array(mid_prices, dtype=np.float64),
        np.stack(scores, axis=0),
        proj_out,
        np.stack(states, axis=0),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical fixed-bin per-cell spectrum analysis.",
    )
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from src.vacuum_pressure.config import resolve_config

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"
    config = resolve_config(args.product_type, args.symbol, products_yaml_path)

    eval_start_ns = _parse_et_timestamp_ns(args.dt, args.eval_start)
    eval_end_ns = _parse_et_timestamp_ns(args.dt, args.eval_end)
    if eval_end_ns <= eval_start_ns:
        raise ValueError("eval_end must be strictly after eval_start")

    ts_ns, mid_price, score, proj_by_h, state_code = _collect_bins(
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
