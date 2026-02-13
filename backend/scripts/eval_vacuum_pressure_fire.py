"""Deterministic FIRE-event experiment harness for vacuum-pressure streams.

Evaluates FIRE -> +/-N tick quality using first-touch barrier logic with
configurable horizon and target ticks on replay or live signal sources.
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
    DEFAULT_FIRE_HORIZONS_SECONDS,
    DEFAULT_FIRE_TARGET_TICKS,
    ensure_event_columns,
    parse_csv_floats,
    parse_csv_ints,
    prepare_signal_frame,
    sweep_fire_operating_grid,
)
from src.vacuum_pressure.formulas import GoldSignalConfig
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
    gold_signal_config: GoldSignalConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load live-path signal outputs via deterministic replay stream."""
    config = resolve_config(product_type, symbol, products_yaml_path)
    rows: list[dict[str, Any]] = []

    for wid, signals in stream_windows(
        lake_root=lake_root,
        config=config,
        dt=dt,
        gold_config=gold_signal_config,
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
    gold_signal_config: GoldSignalConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load replay-path signal outputs from silver parquet compute path."""
    config = resolve_config(product_type, symbol, products_yaml_path)
    engine = VacuumPressureEngine(lake_root)
    _, _, df_signals = engine.compute_day(
        config=config,
        dt=dt,
        gold_signal_config=gold_signal_config,
    )
    if max_windows is not None:
        df_signals = df_signals.sort_values("window_end_ts_ns").head(max_windows)
    if df_signals.empty:
        raise ValueError("No signal rows produced from replay compute path.")
    return df_signals, config.to_dict()


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic FIRE-event quality and recommend "
            "horizon/target operating thresholds."
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
        help="Signal source path (default: live).",
    )
    parser.add_argument(
        "--label",
        default="default",
        help="Optional experiment label for parameter-set comparisons.",
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
        default=",".join(str(v) for v in DEFAULT_FIRE_HORIZONS_SECONDS),
        help="Comma-separated horizon seconds grid.",
    )
    parser.add_argument(
        "--target-ticks",
        default=",".join(str(v) for v in DEFAULT_FIRE_TARGET_TICKS),
        help="Comma-separated target ticks grid.",
    )
    parser.add_argument(
        "--min-evaluable-fires",
        type=int,
        default=25,
        help="Minimum evaluable FIRE events to be rank-eligible.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-N candidates to include in JSON output.",
    )
    parser.add_argument(
        "--pre-smooth-span",
        type=int,
        default=None,
        help="Override GoldSignalConfig.pre_smooth_span.",
    )
    parser.add_argument(
        "--d1-span",
        type=int,
        default=None,
        help="Override GoldSignalConfig.d1_span.",
    )
    parser.add_argument(
        "--d2-span",
        type=int,
        default=None,
        help="Override GoldSignalConfig.d2_span.",
    )
    parser.add_argument(
        "--d3-span",
        type=int,
        default=None,
        help="Override GoldSignalConfig.d3_span.",
    )
    parser.add_argument(
        "--w-d1",
        type=float,
        default=None,
        help="Override GoldSignalConfig.w_d1.",
    )
    parser.add_argument(
        "--w-d2",
        type=float,
        default=None,
        help="Override GoldSignalConfig.w_d2.",
    )
    parser.add_argument(
        "--w-d3",
        type=float,
        default=None,
        help="Override GoldSignalConfig.w_d3.",
    )
    parser.add_argument(
        "--projection-horizon-s",
        type=float,
        default=None,
        help="Override GoldSignalConfig.projection_horizon_s.",
    )
    parser.add_argument(
        "--fast-projection-horizon-s",
        type=float,
        default=None,
        help="Override GoldSignalConfig.fast_projection_horizon_s.",
    )
    parser.add_argument(
        "--smooth-zscore-window",
        type=int,
        default=None,
        help="Override GoldSignalConfig.smooth_zscore_window.",
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


def _build_gold_signal_config(args: argparse.Namespace) -> GoldSignalConfig:
    kwargs: dict[str, float | int] = {}
    for key in (
        "pre_smooth_span",
        "d1_span",
        "d2_span",
        "d3_span",
        "w_d1",
        "w_d2",
        "w_d3",
        "projection_horizon_s",
        "fast_projection_horizon_s",
        "smooth_zscore_window",
    ):
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value
    cfg = GoldSignalConfig(**kwargs)
    cfg.validate()
    return cfg


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
    gold_signal_config = _build_gold_signal_config(args)

    horizons = parse_csv_ints(args.horizons)
    target_ticks_values = parse_csv_floats(args.target_ticks)

    if args.source == "live":
        df_signals, config_dict = _load_live_signals(
            lake_root=lake_root,
            products_yaml_path=products_yaml_path,
            product_type=args.product_type,
            symbol=args.symbol,
            dt=args.dt,
            start_time=args.start_time,
            max_windows=args.max_windows,
            gold_signal_config=gold_signal_config,
        )
    else:
        df_signals, config_dict = _load_replay_signals(
            lake_root=lake_root,
            products_yaml_path=products_yaml_path,
            product_type=args.product_type,
            symbol=args.symbol,
            dt=args.dt,
            max_windows=args.max_windows,
            gold_signal_config=gold_signal_config,
        )

    frame = prepare_signal_frame(
        df_signals=ensure_event_columns(df_signals),
        extra_columns=("event_state", "event_direction"),
    )
    tick_size = float(config_dict["tick_size"])

    fire_grid = sweep_fire_operating_grid(
        frame=frame,
        horizons_s=horizons,
        target_ticks_values=target_ticks_values,
        tick_size=tick_size,
        min_evaluable_fires=int(args.min_evaluable_fires),
        top_k=int(args.top_k),
    )

    recommended = fire_grid["recommended"]
    sample_start_ns = int(frame["window_end_ts_ns"].min())
    sample_end_ns = int(frame["window_end_ts_ns"].max())

    output = {
        "experiment": {
            "label": args.label,
            "product_type": args.product_type,
            "symbol": args.symbol,
            "dt": args.dt,
            "source": args.source,
        },
        "sample": {
            "windows": int(len(frame)),
            "window_start_ts_ns": sample_start_ns,
            "window_end_ts_ns": sample_end_ns,
        },
        "config": config_dict,
        "gold_signal_config": vars(gold_signal_config),
        "search_grid": {
            "horizons_s": horizons,
            "target_ticks": target_ticks_values,
            "min_evaluable_fires": int(args.min_evaluable_fires),
        },
        "search_space_size": int(fire_grid["search_space_size"]),
        "recommendation": {
            "thresholds": (
                {
                    "horizon_s": int(recommended["horizon_s"]),
                    "target_ticks": float(recommended["target_ticks"]),
                }
                if recommended
                else None
            ),
            "objective_score": (
                float(recommended["objective_score"]) if recommended else None
            ),
            "metrics": (recommended["overall"] if recommended else None),
            "by_event_direction": (
                recommended["by_event_direction"] if recommended else None
            ),
            "by_regime": (recommended["by_regime"] if recommended else None),
            "by_regime_and_direction": (
                recommended["by_regime_and_direction"] if recommended else None
            ),
        },
        "top_candidates": [
            {
                "rank": i + 1,
                "thresholds": {
                    "horizon_s": int(candidate["horizon_s"]),
                    "target_ticks": float(candidate["target_ticks"]),
                },
                "objective_score": float(candidate["objective_score"]),
                "metrics": candidate["overall"],
                "by_event_direction": candidate["by_event_direction"],
                "by_regime": candidate["by_regime"],
                "by_regime_and_direction": candidate["by_regime_and_direction"],
            }
            for i, candidate in enumerate(fire_grid["top_candidates"])
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
