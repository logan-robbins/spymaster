"""Run the canonical vacuum-pressure dense-grid websocket server.

Canonical runtime:
    PRE-PROD .dbn ingest adapter -> in-memory EventDrivenVPEngine -> dense grid stream
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

# Ensure backend is on import path
backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vacuum Pressure Canonical Dense-Grid Stream (PRE-PROD DBN source)",
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
        help=(
            "Emit start time HH:MM in ET. Warmup is processed in-memory before emit."
        ),
    )
    parser.add_argument(
        "--throttle-ms",
        type=float,
        default=25.0,
        help="Minimum event-time ms between emitted grid updates. Default 25.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    if args.throttle_ms < 0:
        raise ValueError(f"--throttle-ms must be >= 0, got {args.throttle_ms}")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"

    from src.vacuum_pressure.config import resolve_config
    from src.vacuum_pressure.server import create_app
    from src.vacuum_pressure.stream_pipeline import DEFAULT_GRID_TICKS

    base_config = resolve_config(args.product_type, args.symbol, products_yaml_path)
    if DEFAULT_GRID_TICKS > base_config.grid_max_ticks:
        raise ValueError(
            f"canonical default grid K={DEFAULT_GRID_TICKS} exceeds configured max "
            f"{base_config.grid_max_ticks} for {base_config.product_type}/{base_config.symbol}"
        )
    config = replace(
        base_config,
        grid_max_ticks=DEFAULT_GRID_TICKS,
        config_version=f"{base_config.config_version}:k{DEFAULT_GRID_TICKS}",
    )

    print()
    print("=" * 64)
    print("  VACUUM PRESSURE CANONICAL DENSE-GRID STREAM")
    print("  PRE-PROD SOURCE ADAPTER: DATABENTO .DBN FILES")
    print("=" * 64)
    print(json.dumps(config.to_dict(), indent=2))
    print("-" * 64)
    print(
        json.dumps(
            {
                "dt": args.dt,
                "start_time": args.start_time,
                "grid_ticks": 50,
                "throttle_ms": args.throttle_ms,
            },
            indent=2,
        )
    )
    print("=" * 64)
    print()

    app = create_app(
        lake_root=backend_root / "lake",
        products_yaml_path=products_yaml_path,
    )

    qs_parts = [
        f"product_type={args.product_type}",
        f"symbol={args.symbol}",
        f"dt={args.dt}",
        f"throttle_ms={args.throttle_ms}",
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
