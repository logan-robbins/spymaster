"""CLI runner for the vacuum / pressure detection pipeline.

Usage:
    cd backend

    # Start WebSocket server (default)
    uv run python scripts/run_vacuum_pressure.py \\
        --symbol QQQ --dt 2026-02-06 --port 8002

    # Compute only — save to parquet without serving
    uv run python scripts/run_vacuum_pressure.py \\
        --symbol QQQ --dt 2026-02-06 --compute-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure backend is on path
backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vacuum & Pressure Detection Pipeline for Equity MBO",
    )
    parser.add_argument(
        "--symbol", default="QQQ",
        help="Ticker symbol (default: QQQ)",
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

    from src.vacuum_pressure.engine import VacuumPressureEngine

    engine = VacuumPressureEngine(lake_root)

    if args.compute_only:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else lake_root / "gold" / "vacuum_pressure"
        )
        out_path = engine.save_signals(args.symbol, args.dt, output_dir)
        print(f"Saved vacuum/pressure signals to: {out_path}")
        return

    # Start WebSocket server
    import uvicorn

    from src.vacuum_pressure.server import create_app

    app = create_app(lake_root)

    print()
    print("=" * 60)
    print("  VACUUM & PRESSURE DETECTOR")
    print("=" * 60)
    print(f"  Symbol:    {args.symbol}")
    print(f"  Date:      {args.dt}")
    print(f"  Port:      {args.port}")
    print(f"  WebSocket: ws://localhost:{args.port}"
          f"/v1/vacuum-pressure/stream"
          f"?symbol={args.symbol}&dt={args.dt}")
    print(f"  Frontend:  http://localhost:5174/vacuum-pressure.html")
    print("=" * 60)
    print()

    uvicorn.run(app, host=args.host, port=args.port,
                log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
