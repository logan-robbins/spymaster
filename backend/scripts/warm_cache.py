"""Pre-build book state cache without starting the full VP server.

Usage:
    uv run scripts/warm_cache.py --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --start-time 09:00
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure backend is on import path
backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-build VP book state cache for fast server startup.",
    )
    parser.add_argument(
        "--product-type",
        required=True,
        choices=["equity_mbo", "future_mbo"],
        help="Product type",
    )
    parser.add_argument("--symbol", required=True, help="Single instrument symbol")

    parser.add_argument("--dt", required=True, help="Date YYYY-MM-DD")
    parser.add_argument(
        "--start-time",
        required=True,
        help="Emit start time HH:MM in ET (determines warmup boundary for cache key)",
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
    logger = logging.getLogger("warm_cache")

    lake_root = backend_root / "lake"

    from src.qmachina.config import resolve_config
    from src.qmachina.stream_time_utils import compute_time_boundaries
    from src.qmachina.engine_factory import create_absolute_tick_engine
    from src.qmachina.book_cache import ensure_book_cache

    total_start = time.monotonic()
    logger.info(
        "Warming cache: %s/%s %s start_time=%s",
        args.product_type, args.symbol, args.dt, args.start_time,
    )

    base_config = resolve_config(args.product_type, args.symbol)

    warmup_start_ns, _emit_after_ns = compute_time_boundaries(
        args.product_type, args.dt, args.start_time,
    )
    if warmup_start_ns == 0:
        raise RuntimeError(
            f"No warmup boundary for {args.symbol} (start_time={args.start_time})."
        )

    engine = create_absolute_tick_engine(base_config)

    t_start = time.monotonic()
    cache_path = ensure_book_cache(
        engine=engine,
        lake_root=lake_root,
        product_type=args.product_type,
        symbol=args.symbol,
        dt=args.dt,
        warmup_start_ns=warmup_start_ns,
    )
    elapsed = time.monotonic() - t_start

    total_elapsed = time.monotonic() - total_start

    print()
    print("=" * 64)
    print("  CACHE WARMING RESULTS")
    print("=" * 64)
    if cache_path:
        print(f"  {args.symbol}: {cache_path} ({elapsed:.2f}s)")
    else:
        print(f"  {args.symbol}: FAILED ({elapsed:.2f}s)")
    print(f"  Total: {total_elapsed:.2f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
