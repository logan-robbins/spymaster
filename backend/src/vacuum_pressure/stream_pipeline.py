"""Streaming orchestrator for live vacuum-pressure pipeline.

Two streaming architectures coexist in this module:

**Window-based (legacy, deprecated):**
    Wires together:
        1. DBN replay source (event-by-event from .dbn files)
        2. Book engine (FuturesBookEngine / EquityBookEngine)
        3. Incremental signal engine (per-window signal computation)
        4. Arrow IPC serialization for WebSocket broadcast
    Functions: ``stream_windows()``, ``async_stream_windows()``

**Event-driven (canonical):**
    Wires together:
        1. DBN replay source (event-by-event from .dbn files)
        2. EventDrivenVPEngine (combined book + physics in one pass)
        3. Dense grid output (2K+1 buckets per event)
    Functions: ``stream_events()``, ``async_stream_events()``

The event-driven path replaces the window-based path. Both
StreamingBookAdapter and IncrementalSignalEngine are superseded by
EventDrivenVPEngine, which processes each MBO event individually with
no 1-second window aggregation.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd

from ..data_eng.config import PRICE_SCALE
from ..data_eng.stages.silver.equity_mbo.book_engine import (
    DEPTH_FLOW_COLUMNS as EQUITY_DEPTH_FLOW_COLUMNS,
)
from ..data_eng.stages.silver.equity_mbo.book_engine import (
    SNAP_COLUMNS as EQUITY_SNAP_COLUMNS,
)
from ..data_eng.stages.silver.equity_mbo.book_engine import EquityBookEngine
from ..data_eng.stages.silver.future_mbo.book_engine import (
    DEPTH_FLOW_COLUMNS as FUTURES_DEPTH_FLOW_COLUMNS,
)
from ..data_eng.stages.silver.future_mbo.book_engine import (
    SNAP_COLUMNS as FUTURES_SNAP_COLUMNS,
)
from ..data_eng.stages.silver.future_mbo.book_engine import FuturesBookEngine
from .config import VPRuntimeConfig
from .event_engine import EventDrivenVPEngine
from .formulas import GoldSignalConfig
from .incremental import IncrementalSignalEngine
from .replay_source import iter_mbo_events

logger = logging.getLogger(__name__)

# Price scale imported from data_eng for config -> engine parameter mapping
DATA_PRICE_SCALE: float = PRICE_SCALE


# Type alias for a window yield: (window_end_ts_ns, snap_dict, flow_df)
WindowData = Tuple[int, Dict[str, Any], pd.DataFrame]


class StreamingBookAdapter:
    """Wraps FuturesBookEngine/EquityBookEngine to yield windows on-the-fly.

    The existing book engines append to internal snap_rows/depth_flow_rows
    lists. This adapter tracks the list lengths and extracts new rows
    whenever the engine emits a window (detected by list growth).

    This approach avoids monkey-patching or modifying the engine code,
    keeping the batch pipeline completely untouched.
    """

    def __init__(
        self,
        engine: Union[FuturesBookEngine, EquityBookEngine],
        product_type: str,
    ) -> None:
        self._engine = engine
        self._product_type = product_type
        self._snap_cursor: int = 0
        self._flow_cursor: int = 0

        # Column lists for DataFrame construction
        if product_type == "future_mbo":
            self._snap_cols = FUTURES_SNAP_COLUMNS
            self._flow_cols = FUTURES_DEPTH_FLOW_COLUMNS
        else:
            self._snap_cols = EQUITY_SNAP_COLUMNS
            self._flow_cols = EQUITY_DEPTH_FLOW_COLUMNS

    # Maximum gap windows to emit. Larger gaps are fast-forwarded:
    # the engine skips to the target window without generating all
    # intermediate empty windows. 3600 = 1 hour of gap tolerance.
    MAX_GAP_WINDOWS = 3600

    def feed_event(
        self,
        ts: int,
        action: str,
        side: str,
        price: int,
        size: int,
        order_id: int,
        flags: int,
    ) -> Optional[WindowData]:
        """Feed one MBO event to the book engine.

        Returns:
            (window_end_ts_ns, snap_dict, flow_df) if a window was emitted,
            None otherwise.
        """
        # Detect large time gaps BEFORE feeding to the engine.
        # The book engine's _flush_until generates ALL intermediate windows
        # which is prohibitively slow for gaps of 100K+ seconds (snapshot
        # to first real event in GLBX futures data).
        self._maybe_fast_forward(ts)

        self._engine.apply_event(ts, action, side, price, size, order_id, flags)
        results = self._extract_new_windows()
        return results[-1] if results else None

    def feed_event_all(
        self,
        ts: int,
        action: str,
        side: str,
        price: int,
        size: int,
        order_id: int,
        flags: int,
    ) -> List[WindowData]:
        """Feed one event and return ALL newly emitted windows."""
        self._maybe_fast_forward(ts)
        self._engine.apply_event(ts, action, side, price, size, order_id, flags)
        return self._extract_new_windows()

    def _maybe_fast_forward(self, ts: int) -> None:
        """Fast-forward the book engine's window state if a large gap is detected.

        For GLBX futures, the midnight snapshot has ts_event from ~10 hours
        prior. When the first real event arrives at midnight UTC, there's a
        382K-second gap that would generate hundreds of thousands of empty
        windows via _flush_until. We skip these by:
        1. Emitting the current window
        2. Resetting the engine's window state to just before the new event
        """
        if self._engine.curr_window_id is None:
            return

        new_window_id = ts // self._engine.window_ns
        gap = new_window_id - self._engine.curr_window_id
        if gap <= self.MAX_GAP_WINDOWS:
            return

        logger.info(
            "Fast-forwarding book engine: %d -> %d (%d gap windows skipped)",
            self._engine.curr_window_id, new_window_id, gap - 1,
        )

        # Emit the current window (it has the snapshot data)
        self._engine.flush_final()
        self._extract_new_windows()  # consume and discard

        # Reset engine window state to just before the new event.
        # This avoids generating hundreds of thousands of empty gap windows.
        # The engine's book state (orders, depth) is preserved -- only the
        # window tracking is advanced.
        self._engine.curr_window_id = None  # Will be set by next _start_window

    def _extract_new_windows(self) -> List[WindowData]:
        """Extract any newly emitted windows from the engine's row buffers."""
        new_snap_count = len(self._engine.snap_rows) - self._snap_cursor
        if new_snap_count <= 0:
            return []

        results: List[WindowData] = []
        while self._snap_cursor < len(self._engine.snap_rows):
            snap_dict = self._engine.snap_rows[self._snap_cursor]
            wid = snap_dict["window_end_ts_ns"]

            flow_rows = []
            while self._flow_cursor < len(self._engine.depth_flow_rows):
                row = self._engine.depth_flow_rows[self._flow_cursor]
                if row["window_end_ts_ns"] == wid:
                    flow_rows.append(row)
                    self._flow_cursor += 1
                elif row["window_end_ts_ns"] < wid:
                    self._flow_cursor += 1
                else:
                    break

            flow_df = pd.DataFrame(flow_rows, columns=self._flow_cols) if flow_rows else pd.DataFrame(columns=self._flow_cols)
            results.append((wid, snap_dict, flow_df))
            self._snap_cursor += 1

        # Memory management: trim consumed rows
        if self._snap_cursor > 200:
            trim = self._snap_cursor - 10
            del self._engine.snap_rows[:trim]
            old_flow = self._flow_cursor
            self._flow_cursor = min(10, old_flow)
            if old_flow > 10:
                del self._engine.depth_flow_rows[:old_flow - 10]
            self._snap_cursor = 10

        return results

    def flush(self) -> List[WindowData]:
        """Flush the final window from the book engine."""
        self._engine.flush_final()
        # Collect any remaining windows
        results = []
        while self._snap_cursor < len(self._engine.snap_rows):
            snap_dict = self._engine.snap_rows[self._snap_cursor]
            wid = snap_dict["window_end_ts_ns"]
            flow_rows = []
            while self._flow_cursor < len(self._engine.depth_flow_rows):
                row = self._engine.depth_flow_rows[self._flow_cursor]
                if row["window_end_ts_ns"] == wid:
                    flow_rows.append(row)
                    self._flow_cursor += 1
                elif row["window_end_ts_ns"] < wid:
                    self._flow_cursor += 1
                else:
                    break
            flow_df = pd.DataFrame(flow_rows, columns=self._flow_cols) if flow_rows else pd.DataFrame(columns=self._flow_cols)
            results.append((wid, snap_dict, flow_df))
            self._snap_cursor += 1
        return results


FUTURES_WARMUP_HOURS = 0.5
"""Hours of warmup before emit start for futures.

The midnight snapshot seeds the full book state. Warmup just needs
enough time for resting order state + EMA warmup (longest window is
120s for z-score). 30 minutes is sufficient.
"""

EQUITY_WARMUP_HOURS = 0.5
"""Hours of warmup before emit start for equities."""


def _compute_time_boundaries(
    product_type: str,
    dt: str,
    start_time: str | None,
) -> tuple[int, int]:
    """Compute skip and emit timestamps from a start_time.

    Args:
        product_type: Product type.
        dt: Date string (YYYY-MM-DD).
        start_time: When to start emitting, e.g. "09:30" (ET).
            If None, no skipping (start from beginning).

    Returns:
        (skip_to_ns, emit_after_ns):
            skip_to_ns: Skip non-snapshot events before this timestamp.
            emit_after_ns: Only emit windows at or after this timestamp.
            Both are 0 if start_time is None.
    """
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

    # Warmup: go back N hours from the requested emit start
    warmup_start = et_start - pdt.Timedelta(hours=warmup_hours)

    skip_to_ns = int(warmup_start.value)
    emit_after_ns = int(et_start.value)

    return skip_to_ns, emit_after_ns


def _create_book_engine(
    config: VPRuntimeConfig,
) -> Union[FuturesBookEngine, EquityBookEngine]:
    """Create the appropriate book engine for the product type."""
    if config.product_type == "future_mbo":
        tick_int = int(round(config.tick_size / PRICE_SCALE))
        return FuturesBookEngine(
            tick_int=tick_int,
            grid_max_ticks=config.grid_max_ticks,
        )
    elif config.product_type == "equity_mbo":
        bucket_int = int(round(config.bucket_size_dollars / PRICE_SCALE))
        return EquityBookEngine(
            bucket_int=bucket_int,
            grid_max_buckets=config.grid_max_ticks,
        )
    else:
        raise ValueError(f"Unsupported product_type: {config.product_type}")


def stream_windows(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    gold_config: GoldSignalConfig | None = None,
    start_time: str | None = None,
) -> "Generator[Tuple[int, Dict[str, Any]], None, None]":
    """Synchronous generator: replay .dbn -> book engine -> signals.

    Yields (window_end_ts_ns, signals_dict) for each 1-second window.
    This is the core hot-path pipeline for live streaming.

    Args:
        lake_root: Path to the lake directory.
        config: Resolved VP runtime config.
        dt: Date string (YYYY-MM-DD).
        gold_config: Optional gold-layer tuning config.
        start_time: When to start emitting, as "HH:MM" in ET.
            Events before (start_time - warmup) are skipped at the
            source level. None = start from beginning.

    Yields:
        (window_end_ts_ns, signals_dict) per window.
    """
    skip_to_ns, emit_after_ns = _compute_time_boundaries(config.product_type, dt, start_time)

    engine = _create_book_engine(config)
    adapter = StreamingBookAdapter(engine, config.product_type)
    signal_engine = IncrementalSignalEngine(
        bucket_size_dollars=config.bucket_size_dollars,
        gold_config=gold_config,
    )

    event_count = 0
    window_count = 0
    warmup_count = 0

    for event in iter_mbo_events(lake_root, config.product_type, config.symbol, dt, skip_to_ns=skip_to_ns):
        ts, action, side, price, size, order_id, flags = event
        event_count += 1

        windows = adapter.feed_event_all(ts, action, side, price, size, order_id, flags)
        for wid, snap_dict, flow_df in windows:
            signals = signal_engine.process_window(snap_dict, flow_df)
            # Warmup: process through signal engine but don't yield
            if emit_after_ns > 0 and wid < emit_after_ns:
                warmup_count += 1
                continue
            window_count += 1
            yield wid, signals

    # Flush final window
    for wid, snap_dict, flow_df in adapter.flush():
        signals = signal_engine.process_window(snap_dict, flow_df)
        if emit_after_ns > 0 and wid < emit_after_ns:
            warmup_count += 1
            continue
        window_count += 1
        yield wid, signals

    logger.info(
        "Stream pipeline complete: %d events -> %d windows emitted (%d warmup)",
        event_count, window_count, warmup_count,
    )


async def async_stream_windows(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    gold_config: GoldSignalConfig | None = None,
    speed: float = 1.0,
    skip_minutes: int = 0,
    start_time: str | None = None,
) -> AsyncGenerator[Tuple[int, Dict[str, Any], Dict[str, Any], pd.DataFrame], None]:
    """Async generator: replay .dbn -> book engine -> signals with pacing.

    The heavy synchronous work (DBN iteration, book engine, signal
    computation) runs in a background thread. Results are communicated
    via an asyncio.Queue so the event loop stays responsive for WebSocket
    I/O and new connections.

    Args:
        lake_root: Path to the lake directory.
        config: Resolved VP runtime config.
        dt: Date string (YYYY-MM-DD).
        gold_config: Optional gold-layer tuning config.
        speed: Replay speed multiplier (0 = fire-hose).
        skip_minutes: Minutes to skip from emit start (warmup windows
            still processed through signal engine).
        start_time: When to start emitting, as "HH:MM" in ET.
            Skips non-snapshot events before (start_time - warmup)
            at the source level. None = start from beginning.

    Yields:
        (window_end_ts_ns, signals_dict, snap_dict, flow_df) per window.
    """
    import queue as thread_queue
    import concurrent.futures

    skip_to_ns, emit_after_ns = _compute_time_boundaries(config.product_type, dt, start_time)

    q: thread_queue.Queue = thread_queue.Queue(maxsize=128)
    _SENTINEL = object()

    def _producer() -> None:
        """Run the synchronous pipeline in a background thread."""
        engine = _create_book_engine(config)
        adapter = StreamingBookAdapter(engine, config.product_type)
        signal_engine = IncrementalSignalEngine(
            bucket_size_dollars=config.bucket_size_dollars,
            gold_config=gold_config,
        )

        # skip_minutes skips additional windows AFTER the start_time
        skip_count = skip_minutes * 60
        window_count = 0
        warmup_count = 0
        emitted_count = 0

        try:
            for event in iter_mbo_events(lake_root, config.product_type, config.symbol, dt, skip_to_ns=skip_to_ns):
                ts, action, side, price, size, order_id, flags = event

                windows = adapter.feed_event_all(ts, action, side, price, size, order_id, flags)
                for wid, snap_dict, flow_df in windows:
                    window_count += 1

                    # Warmup: before emit_after_ns, process but don't emit
                    if emit_after_ns > 0 and wid < emit_after_ns:
                        signal_engine.process_window(snap_dict, flow_df)
                        warmup_count += 1
                        continue

                    signals = signal_engine.process_window(snap_dict, flow_df)

                    # Additional skip after start_time (if requested)
                    emitted_count += 1
                    if emitted_count <= skip_count:
                        continue

                    # Blocking put -- provides natural backpressure
                    q.put((wid, signals, snap_dict, flow_df))

            # Flush final window
            for wid, snap_dict, flow_df in adapter.flush():
                window_count += 1
                if emit_after_ns > 0 and wid < emit_after_ns:
                    signal_engine.process_window(snap_dict, flow_df)
                    warmup_count += 1
                    continue
                signals = signal_engine.process_window(snap_dict, flow_df)
                emitted_count += 1
                if emitted_count <= skip_count:
                    continue
                q.put((wid, signals, snap_dict, flow_df))

            logger.info(
                "Stream pipeline producer done: %d total windows, %d warmup, %d emitted",
                window_count, warmup_count, emitted_count - skip_count,
            )
        except Exception as exc:
            logger.error("Stream pipeline producer error: %s", exc, exc_info=True)
        finally:
            q.put(_SENTINEL)

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = loop.run_in_executor(executor, _producer)

    last_emitted_ts: int | None = None
    pacing = speed > 0

    try:
        while True:
            # Non-blocking poll from the thread-safe queue,
            # yielding to the event loop between checks.
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break

            wid, signals, snap_dict, flow_df = item

            # Pacing: sleep between windows
            if pacing and last_emitted_ts is not None:
                delta_s = (wid - last_emitted_ts) / 1_000_000_000.0
                wait = delta_s / speed
                if wait > 0:
                    await asyncio.sleep(wait)

            last_emitted_ts = wid
            yield wid, signals, snap_dict, flow_df
    finally:
        await future
        executor.shutdown(wait=False)


# ======================================================================
# Event-driven streaming (canonical replacement for window-based above)
# ======================================================================


def _create_event_engine(config: VPRuntimeConfig) -> EventDrivenVPEngine:
    """Create an EventDrivenVPEngine from resolved runtime config.

    Maps VPRuntimeConfig parameters to EventDrivenVPEngine constructor
    arguments. For futures, tick_int is derived from tick_size. For
    equities, tick_int is derived from bucket_size_dollars.

    Args:
        config: Resolved vacuum-pressure runtime configuration.

    Returns:
        Configured EventDrivenVPEngine instance.
    """
    if config.product_type == "future_mbo":
        tick_int = int(round(config.tick_size / DATA_PRICE_SCALE))
    else:  # equity_mbo
        tick_int = int(round(config.bucket_size_dollars / DATA_PRICE_SCALE))

    return EventDrivenVPEngine(
        K=config.grid_max_ticks,
        tick_int=tick_int,
        bucket_size_dollars=config.bucket_size_dollars,
    )


def stream_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
    throttle_ms: float = 0,
) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    """Synchronous generator: replay .dbn -> EventDrivenVPEngine -> dense grid.

    Feeds each MBO event from the raw .dbn file directly to the canonical
    EventDrivenVPEngine. No intermediate book adapter or window aggregation.

    The engine processes every event and maintains internal order book state,
    spot tracking, per-bucket derivative chains, and force variants. Each
    yield produces the full dense grid (2K+1 buckets).

    If ``throttle_ms > 0``, only yields when at least ``throttle_ms``
    milliseconds of event-time have elapsed since the last yield. This
    reduces output volume for transport-limited consumers (e.g. WebSocket)
    without losing any engine state (all events are still processed).

    Args:
        lake_root: Path to the lake directory.
        config: Resolved VP runtime config.
        dt: Date string (YYYY-MM-DD).
        start_time: When to start emitting, as "HH:MM" in ET.
            Events before (start_time - warmup) are skipped at the
            source level. Snapshot/Clear records always processed.
            None = start from beginning of session.
        throttle_ms: Minimum milliseconds between yields (event-time).
            0 = yield after every event. Positive values throttle output
            by skipping intermediate yields (engine still processes all
            events for correct state).

    Yields:
        (event_id, grid_dict) where grid_dict contains:
            ts_ns, event_id, spot_ref_price_int, mid_price,
            best_bid_price_int, best_ask_price_int, book_valid,
            buckets (list of 2K+1 dicts), touched_k.
    """
    skip_to_ns, emit_after_ns = _compute_time_boundaries(
        config.product_type, dt, start_time,
    )

    engine = _create_event_engine(config)
    throttle_ns = int(throttle_ms * 1_000_000)

    event_count = 0
    yielded_count = 0
    warmup_count = 0
    last_yield_ts_ns: int = 0

    t_wall_start = time.monotonic()

    for event in iter_mbo_events(
        lake_root, config.product_type, config.symbol, dt,
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

        # Warmup: process through engine but don't yield
        if emit_after_ns > 0 and ts_ns < emit_after_ns:
            warmup_count += 1
            continue

        # Throttle: only yield when enough event-time has elapsed
        if throttle_ns > 0 and last_yield_ts_ns > 0:
            if (ts_ns - last_yield_ts_ns) < throttle_ns:
                continue

        last_yield_ts_ns = ts_ns
        yielded_count += 1
        yield grid["event_id"], grid

    elapsed = time.monotonic() - t_wall_start
    rate = event_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "stream_events complete: %d events, %d yielded, %d warmup, "
        "%.2fs wall (%.0f evt/s)",
        event_count, yielded_count, warmup_count, elapsed, rate,
    )


async def async_stream_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    speed: float = 1.0,
    start_time: str | None = None,
    throttle_ms: float = 100,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async generator: replay .dbn -> EventDrivenVPEngine -> dense grid with pacing.

    Heavy synchronous work (DBN iteration + engine updates) runs in a
    background thread. Results are communicated via ``asyncio.Queue`` so
    the event loop stays responsive for WebSocket I/O.

    Pacing is applied based on event-time deltas scaled by ``1/speed``.
    The ``throttle_ms`` parameter controls how frequently updates are
    emitted (in event-time). Default 100ms means max ~10 updates/sec
    over WebSocket, which is sufficient for frontend rendering.

    Args:
        lake_root: Path to the lake directory.
        config: Resolved VP runtime config.
        dt: Date string (YYYY-MM-DD).
        speed: Replay speed multiplier. 0 = fire-hose (no delays).
        start_time: When to start emitting, as "HH:MM" in ET.
            None = start from beginning of session.
        throttle_ms: Minimum milliseconds between yields (event-time).
            Default 100 = max ~10 updates/sec over WebSocket.

    Yields:
        grid_dict containing: ts_ns, event_id, spot_ref_price_int,
        mid_price, best_bid_price_int, best_ask_price_int, book_valid,
        buckets (list of 2K+1 dicts), touched_k.
    """
    import concurrent.futures
    import queue as thread_queue

    q: thread_queue.Queue = thread_queue.Queue(maxsize=256)
    _SENTINEL = object()

    def _producer() -> None:
        """Run synchronous event pipeline in background thread."""
        try:
            produced = 0
            for _event_id, grid in stream_events(
                lake_root=lake_root,
                config=config,
                dt=dt,
                start_time=start_time,
                throttle_ms=throttle_ms,
            ):
                # Blocking put provides natural backpressure
                q.put(grid)
                produced += 1

            logger.info(
                "async_stream_events producer done: %d grids produced",
                produced,
            )
        except Exception as exc:
            logger.error(
                "async_stream_events producer error: %s", exc, exc_info=True,
            )
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

            # Pacing: sleep between events based on event-time delta
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
