"""Run the vacuum-pressure WebSocket server (serving parity mode)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vacuum Pressure stream server (serving alias/ID required per client)",
    )
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--perf-latency-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL output path for producer-latency telemetry.",
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
        "--example-serving",
        type=str,
        default="",
        help="Optional serving alias/ID to print example stream URLs.",
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

    from src.vacuum_pressure.app import create_app

    app = create_app(
        lake_root=backend_root / "lake",
        perf_latency_jsonl=args.perf_latency_jsonl,
        perf_window_start_et=args.perf_window_start_et,
        perf_window_end_et=args.perf_window_end_et,
        perf_summary_every_bins=args.perf_summary_every_bins,
    )

    print()
    print("=" * 64)
    print("  VACUUM PRESSURE STREAM SERVER")
    print("  RUNTIME CONTRACT: ?serving=<alias_or_id>")
    print("=" * 64)
    print(f"  Health:    http://localhost:{args.port}/health")
    if args.example_serving.strip():
        serving = args.example_serving.strip()
        print(
            "  WebSocket: "
            f"ws://localhost:{args.port}/v1/vacuum-pressure/stream?serving={serving}"
        )
        print(
            "  Frontend:  "
            f"http://localhost:5174/vacuum-pressure.html?serving={serving}"
        )
    else:
        print(
            "  WebSocket: "
            f"ws://localhost:{args.port}/v1/vacuum-pressure/stream?serving=<alias_or_id>"
        )
        print(
            "  Frontend:  "
            "http://localhost:5174/vacuum-pressure.html?serving=<alias_or_id>"
        )
    print("=" * 64)
    print()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
