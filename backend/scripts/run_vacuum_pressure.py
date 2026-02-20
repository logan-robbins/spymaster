"""Run the canonical vacuum-pressure fixed-bin websocket server.

Canonical runtime:
    PRE-PROD .dbn ingest adapter -> in-memory AbsoluteTickEngine -> fixed-bin dense grid
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))


def main() -> None:
    def parse_optional_bool(raw: str) -> bool:
        token = raw.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(
            f"Expected boolean true/false value, got: {raw!r}"
        )

    parser = argparse.ArgumentParser(
        description="Vacuum Pressure Canonical Fixed-Bin Stream (PRE-PROD DBN source)",
    )
    parser.add_argument(
        "--product-type",
        required=True,
        choices=["equity_mbo", "future_mbo"],
        help="Product type",
    )
    parser.add_argument("--symbol", default="MNQH6", help="Instrument symbol")
    parser.add_argument("--dt", default="2026-02-06", help="Date YYYY-MM-DD")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Emit start time HH:MM in ET. Warmup is processed in-memory before emit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--perf-latency-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL output path for producer-latency telemetry (disabled when omitted).",
    )
    parser.add_argument(
        "--perf-window-start-et",
        type=str,
        default=None,
        help="Optional ET window start HH:MM for telemetry records.",
    )
    parser.add_argument(
        "--perf-window-end-et",
        type=str,
        default=None,
        help="Optional ET window end HH:MM for telemetry records.",
    )
    parser.add_argument(
        "--perf-summary-every-bins",
        type=int,
        default=200,
        help="Emit producer latency percentile summary logs every N recorded bins.",
    )
    parser.add_argument(
        "--projection-use-cubic",
        action="store_true",
        help="Enable cubic score projection term (score_d3 * h^3).",
    )
    parser.add_argument(
        "--projection-horizons-bins",
        type=str,
        default=None,
        help=(
            "Optional runtime horizon override as comma-separated bin counts "
            "(for example: 1,2,3,4). Applied at stream time."
        ),
    )
    parser.add_argument(
        "--projection-cubic-scale",
        type=float,
        default=1.0 / 6.0,
        help="Scale applied to cubic projection term when --projection-use-cubic is enabled.",
    )
    parser.add_argument(
        "--projection-damping-lambda",
        type=float,
        default=0.0,
        help="Exponential damping coefficient lambda for projection: proj *= exp(-lambda*h).",
    )
    parser.add_argument(
        "--state-model-enabled",
        type=parse_optional_bool,
        default=None,
        help="Override runtime model enablement (true/false).",
    )
    parser.add_argument(
        "--state-model-center-exclusion-radius",
        type=int,
        default=None,
        help="Override center exclusion radius for runtime model spatial weights.",
    )
    parser.add_argument(
        "--state-model-spatial-decay-power",
        type=float,
        default=None,
        help="Override spatial decay power for runtime model spatial weights.",
    )
    parser.add_argument(
        "--state-model-zscore-window-bins",
        type=int,
        default=None,
        help="Override rolling robust-z window for runtime model.",
    )
    parser.add_argument(
        "--state-model-zscore-min-periods",
        type=int,
        default=None,
        help="Override rolling robust-z min periods for runtime model.",
    )
    parser.add_argument(
        "--state-model-tanh-scale",
        type=float,
        default=None,
        help="Override tanh compression scale for runtime model.",
    )
    parser.add_argument(
        "--state-model-d1-weight",
        type=float,
        default=None,
        help="Override d1 derivative blend weight for runtime model.",
    )
    parser.add_argument(
        "--state-model-d2-weight",
        type=float,
        default=None,
        help="Override d2 derivative blend weight for runtime model.",
    )
    parser.add_argument(
        "--state-model-d3-weight",
        type=float,
        default=None,
        help="Override d3 derivative blend weight for runtime model.",
    )
    parser.add_argument(
        "--state-model-bull-pressure-weight",
        type=float,
        default=None,
        help="Override bull-pressure intensity weight for runtime model.",
    )
    parser.add_argument(
        "--state-model-bull-vacuum-weight",
        type=float,
        default=None,
        help="Override bull-vacuum intensity weight for runtime model.",
    )
    parser.add_argument(
        "--state-model-bear-pressure-weight",
        type=float,
        default=None,
        help="Override bear-pressure intensity weight for runtime model.",
    )
    parser.add_argument(
        "--state-model-bear-vacuum-weight",
        type=float,
        default=None,
        help="Override bear-vacuum intensity weight for runtime model.",
    )
    parser.add_argument(
        "--state-model-mixed-weight",
        type=float,
        default=None,
        help="Override mixed-state damping weight for runtime model.",
    )
    parser.add_argument(
        "--state-model-enable-weighted-blend",
        type=parse_optional_bool,
        default=None,
        help="Override weighted blend toggle for runtime model (true/false).",
    )
    args = parser.parse_args()

    if args.perf_latency_jsonl is None and (
        args.perf_window_start_et is not None or args.perf_window_end_et is not None
    ):
        parser.error("--perf-window-start-et/--perf-window-end-et require --perf-latency-jsonl")
    if args.perf_summary_every_bins <= 0:
        parser.error("--perf-summary-every-bins must be > 0")
    if args.projection_cubic_scale < 0.0:
        parser.error("--projection-cubic-scale must be >= 0")
    if args.projection_damping_lambda < 0.0:
        parser.error("--projection-damping-lambda must be >= 0")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"

    from src.vacuum_pressure.config import (
        build_config_with_overrides,
        parse_projection_horizons_bins_override,
        resolve_config,
    )
    from src.vacuum_pressure.server import create_app

    config = resolve_config(args.product_type, args.symbol, products_yaml_path)
    try:
        projection_horizons_bins = parse_projection_horizons_bins_override(
            args.projection_horizons_bins
        )
    except ValueError as exc:
        parser.error(str(exc))
    if projection_horizons_bins is not None:
        config = build_config_with_overrides(
            config,
            {"projection_horizons_bins": list(projection_horizons_bins)},
        )

    print()
    print("=" * 64)
    print("  VACUUM PRESSURE CANONICAL FIXED-BIN STREAM")
    print("  PRE-PROD SOURCE ADAPTER: DATABENTO .DBN FILES")
    print("=" * 64)
    print(json.dumps(config.to_dict(), indent=2))
    print("-" * 64)
    print(
        json.dumps(
            {
                "dt": args.dt,
                "start_time": args.start_time,
                "grid_radius_ticks": config.grid_radius_ticks,
                "cell_width_ms": config.cell_width_ms,
                "perf_latency_jsonl": str(args.perf_latency_jsonl) if args.perf_latency_jsonl else None,
                "perf_window_start_et": args.perf_window_start_et,
                "perf_window_end_et": args.perf_window_end_et,
                "perf_summary_every_bins": args.perf_summary_every_bins,
                "projection_use_cubic": args.projection_use_cubic,
                "projection_horizons_bins": (
                    list(projection_horizons_bins)
                    if projection_horizons_bins is not None
                    else "default_from_config"
                ),
                "projection_cubic_scale": args.projection_cubic_scale,
                "projection_damping_lambda": args.projection_damping_lambda,
                "state_model_enabled": args.state_model_enabled,
                "state_model_center_exclusion_radius": args.state_model_center_exclusion_radius,
                "state_model_spatial_decay_power": args.state_model_spatial_decay_power,
                "state_model_zscore_window_bins": args.state_model_zscore_window_bins,
                "state_model_zscore_min_periods": args.state_model_zscore_min_periods,
                "state_model_tanh_scale": args.state_model_tanh_scale,
                "state_model_d1_weight": args.state_model_d1_weight,
                "state_model_d2_weight": args.state_model_d2_weight,
                "state_model_d3_weight": args.state_model_d3_weight,
                "state_model_bull_pressure_weight": args.state_model_bull_pressure_weight,
                "state_model_bull_vacuum_weight": args.state_model_bull_vacuum_weight,
                "state_model_bear_pressure_weight": args.state_model_bear_pressure_weight,
                "state_model_bear_vacuum_weight": args.state_model_bear_vacuum_weight,
                "state_model_mixed_weight": args.state_model_mixed_weight,
                "state_model_enable_weighted_blend": args.state_model_enable_weighted_blend,
            },
            indent=2,
        )
    )
    print("=" * 64)
    print()

    app = create_app(
        lake_root=backend_root / "lake",
        products_yaml_path=products_yaml_path,
        perf_latency_jsonl=args.perf_latency_jsonl,
        perf_window_start_et=args.perf_window_start_et,
        perf_window_end_et=args.perf_window_end_et,
        perf_summary_every_bins=args.perf_summary_every_bins,
        projection_horizons_bins_override=args.projection_horizons_bins,
        projection_use_cubic=args.projection_use_cubic,
        projection_cubic_scale=args.projection_cubic_scale,
        projection_damping_lambda=args.projection_damping_lambda,
    )

    qs_parts = [
        f"product_type={args.product_type}",
        f"symbol={args.symbol}",
        f"dt={args.dt}",
    ]
    if args.start_time:
        qs_parts.append(f"start_time={args.start_time}")
    if args.projection_horizons_bins:
        qs_parts.append(f"projection_horizons_bins={args.projection_horizons_bins}")
    if args.state_model_enabled is not None:
        qs_parts.append(f"state_model_enabled={str(args.state_model_enabled).lower()}")
    if args.state_model_center_exclusion_radius is not None:
        qs_parts.append(f"state_model_center_exclusion_radius={args.state_model_center_exclusion_radius}")
    if args.state_model_spatial_decay_power is not None:
        qs_parts.append(f"state_model_spatial_decay_power={args.state_model_spatial_decay_power}")
    if args.state_model_zscore_window_bins is not None:
        qs_parts.append(f"state_model_zscore_window_bins={args.state_model_zscore_window_bins}")
    if args.state_model_zscore_min_periods is not None:
        qs_parts.append(f"state_model_zscore_min_periods={args.state_model_zscore_min_periods}")
    if args.state_model_tanh_scale is not None:
        qs_parts.append(f"state_model_tanh_scale={args.state_model_tanh_scale}")
    if args.state_model_d1_weight is not None:
        qs_parts.append(f"state_model_d1_weight={args.state_model_d1_weight}")
    if args.state_model_d2_weight is not None:
        qs_parts.append(f"state_model_d2_weight={args.state_model_d2_weight}")
    if args.state_model_d3_weight is not None:
        qs_parts.append(f"state_model_d3_weight={args.state_model_d3_weight}")
    if args.state_model_bull_pressure_weight is not None:
        qs_parts.append(f"state_model_bull_pressure_weight={args.state_model_bull_pressure_weight}")
    if args.state_model_bull_vacuum_weight is not None:
        qs_parts.append(f"state_model_bull_vacuum_weight={args.state_model_bull_vacuum_weight}")
    if args.state_model_bear_pressure_weight is not None:
        qs_parts.append(f"state_model_bear_pressure_weight={args.state_model_bear_pressure_weight}")
    if args.state_model_bear_vacuum_weight is not None:
        qs_parts.append(f"state_model_bear_vacuum_weight={args.state_model_bear_vacuum_weight}")
    if args.state_model_mixed_weight is not None:
        qs_parts.append(f"state_model_mixed_weight={args.state_model_mixed_weight}")
    if args.state_model_enable_weighted_blend is not None:
        qs_parts.append(
            f"state_model_enable_weighted_blend={str(args.state_model_enable_weighted_blend).lower()}"
        )
    qs = "&".join(qs_parts)

    print(f"  WebSocket: ws://localhost:{args.port}/v1/vacuum-pressure/stream?{qs}")
    print(f"  Frontend:  http://localhost:5174/vacuum-pressure.html?{qs}")
    print("=" * 64)
    print()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
