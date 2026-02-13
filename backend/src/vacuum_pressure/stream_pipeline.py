"""Live event-stream pipeline for vacuum-pressure dense-grid streaming.

Canonical path only:
    DBN event source -> EventDrivenVPEngine -> dense grid stream

No window-based aggregation, no silver replay path, and no alternate math
engines. All computation remains in-memory and updates on every MBO event.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, Tuple

from ..data_eng.config import PRICE_SCALE
from .config import VPRuntimeConfig
from .event_engine import EventDrivenVPEngine
from .replay_source import iter_mbo_events

logger = logging.getLogger(__name__)

# Warmup horizon before emit start when start_time is provided.
FUTURES_WARMUP_HOURS = 0.5
EQUITY_WARMUP_HOURS = 0.5

# Canonical live default: +/- 50 ticks around spot.
DEFAULT_GRID_TICKS = 50


def _compute_time_boundaries(
    product_type: str,
    dt: str,
    start_time: str | None,
) -> tuple[int, int]:
    """Compute (skip_to_ns, emit_after_ns) from ET start_time."""
    if not start_time:
        return 0, 0

    import pandas as pdt

    warmup_hours = (
        FUTURES_WARMUP_HOURS if "future" in product_type
        else EQUITY_WARMUP_HOURS
    )

    # start_time is HH:MM in Eastern (fixed EST)
    et_start = pdt.Timestamp(
        f"{dt} {start_time}:00", tz="Etc/GMT+5"
    ).tz_convert("UTC")

    warmup_start = et_start - pdt.Timedelta(hours=warmup_hours)

    skip_to_ns = int(warmup_start.value)
    emit_after_ns = int(et_start.value)
    return skip_to_ns, emit_after_ns


def _resolve_tick_int(config: VPRuntimeConfig) -> int:
    """Resolve integer price increment for event-engine bucket mapping."""
    if config.product_type == "future_mbo":
        tick_int = int(round(config.tick_size / PRICE_SCALE))
    elif config.product_type == "equity_mbo":
        # Equities use configured bucket size as the spatial increment.
        tick_int = int(round(config.bucket_size_dollars / PRICE_SCALE))
    else:
        raise ValueError(f"Unsupported product_type: {config.product_type}")

    if tick_int <= 0:
        raise ValueError(
            f"Resolved tick_int must be > 0, got {tick_int} for {config.product_type}/{config.symbol}"
        )
    return tick_int


def _create_event_engine(
    config: VPRuntimeConfig,
) -> EventDrivenVPEngine:
    """Create the canonical event-driven VP engine for live streaming."""
    if config.grid_max_ticks < DEFAULT_GRID_TICKS:
        raise ValueError(
            f"Configured grid_max_ticks={config.grid_max_ticks} is below required "
            f"live default {DEFAULT_GRID_TICKS} for {config.product_type}/{config.symbol}"
        )

    tick_int = _resolve_tick_int(config)
    return EventDrivenVPEngine(
        K=DEFAULT_GRID_TICKS,
        tick_int=tick_int,
        bucket_size_dollars=config.bucket_size_dollars,
    )


def stream_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
    throttle_ms: float = 25.0,
) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    """Synchronous live pipeline: DBN events -> EventDrivenVPEngine grids.

    Args:
        lake_root: Path to the lake directory.
        config: Resolved runtime config.
        dt: Date string (YYYY-MM-DD).
        start_time: Emit start in ET HH:MM. Warmup processed in-memory.
        throttle_ms: Minimum event-time spacing between emitted updates.
            All events are still processed; throttle only reduces yields.
    """
    if throttle_ms < 0:
        raise ValueError(f"throttle_ms must be >= 0, got {throttle_ms}")

    skip_to_ns, emit_after_ns = _compute_time_boundaries(
        config.product_type, dt, start_time,
    )

    engine = _create_event_engine(config)
    throttle_ns = int(throttle_ms * 1_000_000)

    event_count = 0
    yielded_count = 0
    warmup_count = 0
    last_yield_ts_ns = 0

    t_wall_start = time.monotonic()

    for event in iter_mbo_events(
        lake_root,
        config.product_type,
        config.symbol,
        dt,
        skip_to_ns=skip_to_ns,
    ):
        ts_ns, action, side, price, size, order_id, flags = event
        event_count += 1

        grid = engine.update(
            ts_ns=ts_ns,
            action=action,
            side=side,
            price_int=price,
            size=size,
            order_id=order_id,
            flags=flags,
        )

        if emit_after_ns > 0 and ts_ns < emit_after_ns:
            warmup_count += 1
            continue

        if throttle_ns > 0 and last_yield_ts_ns > 0:
            if (ts_ns - last_yield_ts_ns) < throttle_ns:
                continue

        last_yield_ts_ns = ts_ns
        yielded_count += 1
        yield grid["event_id"], grid

    elapsed = time.monotonic() - t_wall_start
    rate = event_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "live stream complete: %d events processed, %d grids emitted, %d warmup, %.2fs wall (%.0f evt/s)",
        event_count,
        yielded_count,
        warmup_count,
        elapsed,
        rate,
    )


async def async_stream_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    speed: float = 1.0,
    start_time: str | None = None,
    throttle_ms: float = 25.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async live pipeline wrapper with optional replay pacing."""
    if speed < 0:
        raise ValueError(f"speed must be >= 0, got {speed}")

    import concurrent.futures
    import queue as thread_queue

    q: thread_queue.Queue = thread_queue.Queue(maxsize=256)
    _SENTINEL = object()

    def _producer() -> None:
        """Run synchronous pipeline in a background thread."""
        try:
            produced = 0
            for _event_id, grid in stream_events(
                lake_root=lake_root,
                config=config,
                dt=dt,
                start_time=start_time,
                throttle_ms=throttle_ms,
            ):
                q.put(grid)
                produced += 1
            logger.info("async live producer done: %d grids produced", produced)
        except Exception as exc:
            logger.error("async live producer error: %s", exc, exc_info=True)
        finally:
            q.put(_SENTINEL)

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = loop.run_in_executor(executor, _producer)

    last_emitted_ts_ns: int | None = None
    pacing = speed > 0

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break

            grid = item
            ts_ns = grid["ts_ns"]

            if pacing and last_emitted_ts_ns is not None:
                delta_s = (ts_ns - last_emitted_ts_ns) / 1e9
                wait = delta_s / speed
                if wait > 0:
                    await asyncio.sleep(wait)

            last_emitted_ts_ns = ts_ns
            yield grid
    finally:
        await future
        executor.shutdown(wait=False)
