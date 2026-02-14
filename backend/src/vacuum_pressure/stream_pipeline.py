"""Live event-stream pipeline for vacuum-pressure dense-grid streaming.

Canonical path only:
    DBN event source -> EventDrivenVPEngine -> dense grid stream

Three processing phases when ``start_time`` is provided:
    1. **Book-only fast-forward** (pre-warmup): All events from session start
       through ``warmup_start_ns`` are processed via the lightweight
       ``apply_book_event()`` path.  This builds correct order book state
       without computing grid mechanics / derivatives / forces (~10-50x
       faster than full VP).  **Cached** after first run — subsequent
       launches with the same symbol/date/start_time skip Phase 1 entirely.
    2. **VP warmup**: Events from ``warmup_start_ns`` through ``emit_after_ns``
       are processed through the full VP engine to populate derivative chains,
       but grid snapshots are not emitted to the consumer.
    3. **Live emit**: Events after ``emit_after_ns`` are processed and emitted.

When ``start_time`` is None, all events go through the full VP engine and
are emitted immediately (subject to throttle).

Book state cache:
    Stored at ``lake/cache/vp_book/{symbol}_{dt}_{hash}.pkl``.
    Cache key includes the .dbn file's mtime + size so it auto-invalidates
    when raw data is re-downloaded.  Book state is independent of VP formula
    parameters — formula changes do NOT require cache regeneration.
    To force regeneration: ``rm -rf backend/lake/cache/vp_book/``
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, Tuple

from ..data_eng.config import PRICE_SCALE
from .config import VPRuntimeConfig
from .event_engine import EventDrivenVPEngine
from .replay_source import _resolve_dbn_path, iter_mbo_events

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
    """Compute (warmup_start_ns, emit_after_ns) from ET start_time.

    Uses ``America/New_York`` for correct EST/EDT handling year-round.

    Returns:
        (warmup_start_ns, emit_after_ns).  Both 0 when start_time is None.
    """
    if not start_time:
        return 0, 0

    import pandas as pdt

    warmup_hours = (
        FUTURES_WARMUP_HOURS if "future" in product_type
        else EQUITY_WARMUP_HOURS
    )

    # start_time is HH:MM in Eastern (America/New_York handles EST/EDT)
    et_start = pdt.Timestamp(
        f"{dt} {start_time}:00", tz="America/New_York"
    ).tz_convert("UTC")

    warmup_start = et_start - pdt.Timedelta(hours=warmup_hours)

    warmup_start_ns = int(warmup_start.value)
    emit_after_ns = int(et_start.value)
    return warmup_start_ns, emit_after_ns


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


def _resolve_book_cache_path(
    lake_root: Path,
    product_type: str,
    symbol: str,
    dt: str,
    warmup_start_ns: int,
) -> Path | None:
    """Compute deterministic cache path for book state checkpoint.

    Cache key includes the .dbn file's mtime + size so it auto-invalidates
    when raw data is re-downloaded.  Returns None if the .dbn file cannot
    be resolved (e.g. not yet downloaded).
    """
    try:
        dbn_path = _resolve_dbn_path(lake_root, product_type, symbol, dt)
    except FileNotFoundError:
        return None

    stat = dbn_path.stat()
    raw = f"{product_type}:{symbol}:{dt}:{warmup_start_ns}:{stat.st_mtime_ns}:{stat.st_size}"
    key_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    cache_dir = lake_root / "cache" / "vp_book"
    return cache_dir / f"{symbol}_{dt}_{key_hash}.pkl"


def ensure_book_cache(
    engine: EventDrivenVPEngine,
    lake_root: Path,
    product_type: str,
    symbol: str,
    dt: str,
    warmup_start_ns: int,
) -> Path | None:
    """Build book-state cache if not present. Returns cache path.

    Processes all events from session start through ``warmup_start_ns``
    via lightweight ``apply_book_event()`` and saves the resulting book
    state to a deterministic cache file.  Subsequent calls with the same
    inputs return immediately after loading from cache.

    Args:
        engine: Event engine to populate with book state.
        lake_root: Path to the lake directory.
        product_type: Product type (equity_mbo or future_mbo).
        symbol: Contract symbol.
        dt: Date string (YYYY-MM-DD).
        warmup_start_ns: Nanosecond boundary for book-only phase.

    Returns:
        Cache file path, or None if warmup_start_ns is 0.
    """
    if warmup_start_ns == 0:
        return None

    cache_path = _resolve_book_cache_path(
        lake_root, product_type, symbol, dt, warmup_start_ns,
    )
    if cache_path is None:
        return None

    t_wall_start = time.monotonic()

    if cache_path.exists():
        logger.info("Loading cached book state: %s", cache_path.name)
        engine.import_book_state(cache_path.read_bytes())
        engine.sync_rest_depth_from_book()
        logger.info(
            "Book cache loaded in %.2fs: %d orders, spot=%d, book_valid=%s",
            time.monotonic() - t_wall_start,
            engine.order_count,
            engine.spot_ref_price_int,
            engine.book_valid,
        )
        return cache_path

    # Build book state from scratch
    book_only_count = 0
    for event in iter_mbo_events(lake_root, product_type, symbol, dt):
        ts_ns, action, side, price, size, order_id, flags = event

        if ts_ns >= warmup_start_ns:
            break

        engine.apply_book_event(
            ts_ns=ts_ns,
            action=action,
            side=side,
            price_int=price,
            size=size,
            order_id=order_id,
            flags=flags,
        )
        book_only_count += 1
        if book_only_count % 1_000_000 == 0:
            elapsed_so_far = time.monotonic() - t_wall_start
            logger.info(
                "book-only fast-forward: %dM events (%.1fs, %d orders, spot=%d)",
                book_only_count // 1_000_000,
                elapsed_so_far,
                engine.order_count,
                engine.spot_ref_price_int,
            )

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(engine.export_book_state())
    engine.sync_rest_depth_from_book()

    elapsed_ff = time.monotonic() - t_wall_start
    logger.info(
        "Book cache built: %d events in %.2fs (%d orders, spot=%d, book_valid=%s) -> %s",
        book_only_count,
        elapsed_ff,
        engine.order_count,
        engine.spot_ref_price_int,
        engine.book_valid,
        cache_path.name,
    )
    return cache_path


def stream_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
    throttle_ms: float = 25.0,
) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    """Synchronous live pipeline: DBN events -> EventDrivenVPEngine grids.

    Three-phase processing ensures correct book state without replaying
    the full session through the expensive VP engine:

    1. **Book-only** (ts < warmup_start_ns): Lightweight ``apply_book_event``
       builds correct order book / spot from all pre-warmup events.
       **Cached** after first run for instant restart.
    2. **VP warmup** (warmup_start_ns <= ts < emit_after_ns): Full engine
       processes events to populate derivative chains; grids not emitted.
    3. **Live emit** (ts >= emit_after_ns): Full engine + grid emission.

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

    warmup_start_ns, emit_after_ns = _compute_time_boundaries(
        config.product_type, dt, start_time,
    )

    engine = _create_event_engine(config)
    throttle_ns = int(throttle_ms * 1_000_000)

    event_count = 0
    yielded_count = 0
    warmup_count = 0
    last_yield_ts_ns = 0

    t_wall_start = time.monotonic()

    # --- Phase 1: Book state cache (delegated to ensure_book_cache) ---
    cache_loaded = False
    if warmup_start_ns > 0:
        cache_path = ensure_book_cache(
            engine, lake_root, config.product_type, config.symbol, dt, warmup_start_ns,
        )
        cache_loaded = cache_path is not None

    # When cache is loaded, skip_to_ns jumps past pre-warmup events.
    for event in iter_mbo_events(
        lake_root,
        config.product_type,
        config.symbol,
        dt,
        skip_to_ns=warmup_start_ns if cache_loaded else 0,
    ):
        ts_ns, action, side, price, size, order_id, flags = event
        event_count += 1

        # Skip any residual pre-warmup records
        if ts_ns < warmup_start_ns:
            continue

        # Phases 2 & 3: Full VP engine processing
        grid = engine.update(
            ts_ns=ts_ns,
            action=action,
            side=side,
            price_int=price,
            size=size,
            order_id=order_id,
            flags=flags,
        )

        # Phase 2: Warmup -- process but don't emit
        if emit_after_ns > 0 and ts_ns < emit_after_ns:
            warmup_count += 1
            continue

        # Phase 3: Live emit (with throttle)
        if throttle_ns > 0 and last_yield_ts_ns > 0:
            if (ts_ns - last_yield_ts_ns) < throttle_ns:
                continue

        last_yield_ts_ns = ts_ns
        yielded_count += 1
        yield grid["event_id"], grid

    elapsed = time.monotonic() - t_wall_start
    rate = event_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "live stream complete: %d events (%d warmup, %d emitted), "
        "%.2fs wall (%.0f evt/s)",
        event_count,
        warmup_count,
        yielded_count,
        elapsed,
        rate,
    )


async def async_stream_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
    throttle_ms: float = 25.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async live pipeline wrapper with real-time replay pacing."""
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

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break

            grid = item
            ts_ns = grid["ts_ns"]

            if last_emitted_ts_ns is not None:
                delta_s = (ts_ns - last_emitted_ts_ns) / 1e9
                if delta_s > 0:
                    await asyncio.sleep(delta_s)

            last_emitted_ts_ns = ts_ns
            yield grid
    finally:
        await future
        executor.shutdown(wait=False)
