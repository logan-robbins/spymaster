"""Deterministic replay evaluator for vacuum-pressure threshold tuning.

Runs a pure grid search over fixed gating thresholds and reports directional
quality for short horizons (2s/5s/10s) as machine-readable JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

from src.vacuum_pressure.config import resolve_config
from src.vacuum_pressure.engine import VacuumPressureEngine
from src.vacuum_pressure.evaluation import (
    DEFAULT_HORIZONS_SECONDS,
    parse_csv_bools,
    parse_csv_floats,
    parse_csv_ints,
    prepare_signal_frame,
    sweep_threshold_grid,
)
from src.vacuum_pressure.stream_pipeline import stream_windows


logger = logging.getLogger(__name__)


def _load_live_signals(
    lake_root: Path,
    products_yaml_path: Path,
    product_type: str,
    symbol: str,
    dt: str,
    start_time: str | None,
    max_windows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load deterministic live-path signal outputs via stream replay."""
    config = resolve_config(product_type, symbol, products_yaml_path)
    rows: list[dict[str, Any]] = []

    for wid, signals in stream_windows(
        lake_root=lake_root,
        config=config,
        dt=dt,
        start_time=start_time,
    ):
        row = dict(signals)
        row.setdefault("window_end_ts_ns", wid)
        rows.append(row)
        if max_windows is not None and len(rows) >= max_windows:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No signal rows produced from live replay stream.")
    return df, config.to_dict()


def _load_replay_signals(
    lake_root: Path,
    products_yaml_path: Path,
    product_type: str,
    symbol: str,
    dt: str,
    max_windows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load replay-path signal outputs from silver parquet compute path."""
    config = resolve_config(product_type, symbol, products_yaml_path)
    engine = VacuumPressureEngine(lake_root)
    _, _, df_signals = engine.compute_day(config, dt)
    if max_windows is not None:
        df_signals = df_signals.sort_values("window_end_ts_ns").head(max_windows)
    if df_signals.empty:
        raise ValueError("No signal rows produced from replay compute path.")
    return df_signals, config.to_dict()


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic vacuum-pressure directional quality and "
            "recommend fixed threshold gates."
        )
    )
    parser.add_argument(
        "--product-type",
        required=True,
        choices=["equity_mbo", "future_mbo"],
        help="Product type.",
    )
    parser.add_argument("--symbol", required=True, help="Instrument symbol.")
    parser.add_argument("--dt", required=True, help="Session date YYYY-MM-DD.")
    parser.add_argument(
        "--source",
        default="live",
        choices=["live", "replay"],
        help=(
            "Signal source path. live=replay raw DBN through incremental stream "
            "(default), replay=silver->run_full_pipeline path."
        ),
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional HH:MM ET emit start for live source warmup handling.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on windows for fast iteration.",
    )
    parser.add_argument(
        "--horizons",
        default="2,5,10",
        help="Comma-separated horizon seconds (default: 2,5,10).",
    )
    parser.add_argument(
        "--min-move-ticks",
        type=float,
        default=1.0,
        help="Directional success threshold in ticks (default: 1.0).",
    )
    parser.add_argument(
        "--net-lift-thresholds",
        default="0.25,0.5,0.75,1.0,1.5",
        help="Grid for |net_lift| threshold.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        default="0.0,0.2,0.4,0.6,0.8",
        help="Grid for cross_confidence threshold.",
    )
    parser.add_argument(
        "--d1-15s-thresholds",
        default="0.0,0.05,0.1,0.2",
        help="Grid for |d1_15s| threshold.",
    )
    parser.add_argument(
        "--require-regime-alignment",
        default="0,1",
        help="Grid for regime alignment gate values (0/1).",
    )
    parser.add_argument(
        "--primary-horizon",
        type=int,
        default=5,
        help="Primary horizon in seconds for recommendation ranking.",
    )
    parser.add_argument(
        "--min-alerts",
        type=int,
        default=100,
        help="Minimum primary-horizon alerts required to be rank-eligible.",
    )
    parser.add_argument(
        "--target-alert-rate",
        type=float,
        default=0.01,
        help="Target alert coverage (fraction of evaluable windows).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-N candidates to include in JSON output.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output file for the full JSON summary.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    parser = _arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    lake_root = backend_root / "lake"
    products_yaml_path = (
        backend_root / "src" / "data_eng" / "config" / "products.yaml"
    )

    horizons = parse_csv_ints(args.horizons)
    net_lift_thresholds = parse_csv_floats(args.net_lift_thresholds)
    confidence_thresholds = parse_csv_floats(args.confidence_thresholds)
    d1_15s_thresholds = parse_csv_floats(args.d1_15s_thresholds)
    require_regime_alignment_values = parse_csv_bools(args.require_regime_alignment)

    if args.source == "live":
        df_signals, config_dict = _load_live_signals(
            lake_root=lake_root,
            products_yaml_path=products_yaml_path,
            product_type=args.product_type,
            symbol=args.symbol,
            dt=args.dt,
            start_time=args.start_time,
            max_windows=args.max_windows,
        )
    else:
        df_signals, config_dict = _load_replay_signals(
            lake_root=lake_root,
            products_yaml_path=products_yaml_path,
            product_type=args.product_type,
            symbol=args.symbol,
            dt=args.dt,
            max_windows=args.max_windows,
        )

    frame = prepare_signal_frame(df_signals)
    tick_size = float(config_dict["tick_size"])

    grid_result = sweep_threshold_grid(
        frame=frame,
        horizons_s=horizons or list(DEFAULT_HORIZONS_SECONDS),
        tick_size=tick_size,
        min_move_ticks=float(args.min_move_ticks),
        net_lift_thresholds=net_lift_thresholds,
        confidence_thresholds=confidence_thresholds,
        d1_15s_thresholds=d1_15s_thresholds,
        require_regime_alignment_values=require_regime_alignment_values,
        primary_horizon_s=int(args.primary_horizon),
        min_alerts=int(args.min_alerts),
        target_alert_rate=float(args.target_alert_rate),
        top_k=int(args.top_k),
    )

    recommended = grid_result["recommended"]
    sample_start_ns = int(frame["window_end_ts_ns"].min())
    sample_end_ns = int(frame["window_end_ts_ns"].max())

    output = {
        "product_type": args.product_type,
        "symbol": args.symbol,
        "dt": args.dt,
        "source": args.source,
        "horizons_seconds": horizons,
        "primary_horizon_s": int(args.primary_horizon),
        "min_move_ticks": float(args.min_move_ticks),
        "sample": {
            "windows": int(len(frame)),
            "window_start_ts_ns": sample_start_ns,
            "window_end_ts_ns": sample_end_ns,
        },
        "config": config_dict,
        "search_grid": {
            "min_abs_net_lift": net_lift_thresholds,
            "min_cross_confidence": confidence_thresholds,
            "min_abs_d1_15s": d1_15s_thresholds,
            "require_regime_alignment": require_regime_alignment_values,
        },
        "search_space_size": int(grid_result["search_space_size"]),
        "recommendation": {
            "thresholds": recommended["thresholds"] if recommended else None,
            "objective_score": (
                float(recommended["objective_score"]) if recommended else None
            ),
            "metrics_by_horizon": recommended["horizons"] if recommended else None,
        },
        "top_candidates": [
            {
                "rank": i + 1,
                "thresholds": candidate["thresholds"],
                "objective_score": float(candidate["objective_score"]),
                "primary_metrics": candidate["horizons"][f"{args.primary_horizon}s"],
            }
            for i, candidate in enumerate(grid_result["top_candidates"])
        ],
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, indent=2 if args.pretty else None, sort_keys=True)
        )

    print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
