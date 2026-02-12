"""CLI runner for the vacuum / pressure detection pipeline.

Usage:
    cd backend

    # Start WebSocket server for equity (default)
    uv run python scripts/run_vacuum_pressure.py \\
        --product-type equity_mbo --symbol QQQ --dt 2026-02-06 --port 8002

    # Start WebSocket server for futures
    uv run python scripts/run_vacuum_pressure.py \\
        --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --port 8002

    # Compute only -- save to parquet without serving
    uv run python scripts/run_vacuum_pressure.py \\
        --product-type equity_mbo --symbol QQQ --dt 2026-02-06 --compute-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure backend is on path
backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vacuum & Pressure Detection Pipeline",
    )
    parser.add_argument(
        "--product-type", required=True,
        choices=["equity_mbo", "future_mbo"],
        help="Product type: equity_mbo or future_mbo",
    )
    parser.add_argument(
        "--symbol", default="QQQ",
        help="Instrument symbol (default: QQQ)",
    )
    parser.add_argument(
        "--dt", default="2026-02-06",
        help="Date YYYY-MM-DD (default: 2026-02-06)",
    )
    parser.add_argument(
        "--port", type=int, default=8002,
        help="WebSocket server port (default: 8002)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Server bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--compute-only", action="store_true",
        help="Compute and save to parquet without starting server",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output dir for --compute-only (default: lake/gold/vacuum_pressure)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    lake_root = backend_root / "lake"
    products_yaml_path = (
        backend_root / "src" / "data_eng" / "config" / "products.yaml"
    )

    from src.vacuum_pressure.config import resolve_config

    # Resolve and print runtime config at startup (4.5)
    config = resolve_config(args.product_type, args.symbol, products_yaml_path)

    print()
    print("=" * 60)
    print("  VACUUM & PRESSURE DETECTOR -- Runtime Config")
    print("=" * 60)
    print(json.dumps(config.to_dict(), indent=2))
    print("=" * 60)
    print()

    from src.vacuum_pressure.engine import VacuumPressureEngine

    engine = VacuumPressureEngine(lake_root)

    if args.compute_only:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else lake_root / "gold" / "vacuum_pressure"
        )
        out_path = engine.save_signals(config, args.dt, output_dir)
        print(f"Saved vacuum/pressure signals to: {out_path}")
        return

    # Start WebSocket server
    import uvicorn

    from src.vacuum_pressure.server import create_app

    app = create_app(lake_root, products_yaml_path)

    print(f"  WebSocket: ws://localhost:{args.port}"
          f"/v1/vacuum-pressure/stream"
          f"?product_type={args.product_type}"
          f"&symbol={args.symbol}&dt={args.dt}")
    print(f"  Frontend:  http://localhost:5174/vacuum-pressure.html"
          f"?product_type={args.product_type}"
          f"&symbol={args.symbol}&dt={args.dt}")
    print("=" * 60)
    print()

    uvicorn.run(app, host=args.host, port=args.port,
                log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
