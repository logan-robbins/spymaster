"""Benchmark pressure-core full-grid throughput (no radius filtering)."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VP pressure-core full-grid replay throughput.",
    )
    parser.add_argument("--product-type", default="future_mbo", choices=["future_mbo", "equity_mbo"])
    parser.add_argument("--symbol", default="MNQH6")
    parser.add_argument("--dt", default="2026-02-06")
    parser.add_argument("--start-time", default="09:00")
    parser.add_argument(
        "--max-bins",
        type=int,
        default=0,
        help="Optional cap on emitted bins (0 means no cap).",
    )
    parser.add_argument(
        "--fail-on-out-of-range",
        action="store_true",
        help="Fail fast when price maps outside configured absolute grid.",
    )
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

    from src.qmachina.config import resolve_config
    from src.models.vacuum_pressure.core_pipeline import stream_core_events

    config = resolve_config(args.product_type, args.symbol)

    t0 = time.monotonic()
    bins = 0
    last_event_id = 0
    first_ts_ns = 0
    last_ts_ns = 0

    for grid in stream_core_events(
        lake_root=backend_root / "lake",
        config=config,
        dt=args.dt,
        start_time=args.start_time,
        fail_on_out_of_range=args.fail_on_out_of_range,
    ):
        bins += 1
        last_event_id = int(grid["event_id"])
        if first_ts_ns == 0:
            first_ts_ns = int(grid["bin_start_ns"])
        last_ts_ns = int(grid["bin_end_ns"])

        if args.max_bins > 0 and bins >= args.max_bins:
            break

    wall_s = time.monotonic() - t0
    evt_per_s = last_event_id / wall_s if wall_s > 0 else 0.0
    bins_per_s = bins / wall_s if wall_s > 0 else 0.0
    replay_span_s = (last_ts_ns - first_ts_ns) / 1e9 if first_ts_ns and last_ts_ns else 0.0

    print("Pressure Core Benchmark")
    print("-" * 64)
    print(f"Instrument:        {args.product_type}:{args.symbol}")
    print(f"Date:              {args.dt}")
    print(f"Start time (ET):   {args.start_time}")
    print(f"Rows per grid:     {config.n_absolute_ticks}")
    print(f"Cell width (ms):   {config.cell_width_ms}")
    print(f"Bins emitted:      {bins}")
    print(f"Last event_id:     {last_event_id}")
    print(f"Replay span (sec): {replay_span_s:.2f}")
    print(f"Wall time (sec):   {wall_s:.2f}")
    print(f"Bins/sec:          {bins_per_s:.2f}")
    print(f"Events/sec:        {evt_per_s:.0f}")


if __name__ == "__main__":
    main()
