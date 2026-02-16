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
    args = parser.parse_args()

    if args.perf_latency_jsonl is None and (
        args.perf_window_start_et is not None or args.perf_window_end_et is not None
    ):
        parser.error("--perf-window-start-et/--perf-window-end-et require --perf-latency-jsonl")
    if args.perf_summary_every_bins <= 0:
        parser.error("--perf-summary-every-bins must be > 0")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"

    from src.vacuum_pressure.config import resolve_config
    from src.vacuum_pressure.server import create_app

    config = resolve_config(args.product_type, args.symbol, products_yaml_path)

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
    )

    qs_parts = [
        f"product_type={args.product_type}",
        f"symbol={args.symbol}",
        f"dt={args.dt}",
    ]
    if args.start_time:
        qs_parts.append(f"start_time={args.start_time}")
    qs = "&".join(qs_parts)

    print(f"  WebSocket: ws://localhost:{args.port}/v1/vacuum-pressure/stream?{qs}")
    print(f"  Frontend:  http://localhost:5174/vacuum-pressure.html?{qs}")
    print("=" * 64)
    print()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
