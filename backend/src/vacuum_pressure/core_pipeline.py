"""Pressure-core full-grid pipeline (no radius filtering).

This module focuses on force-field correctness and throughput by emitting
full-grid columnar snapshots for each fixed-width time bin. It intentionally
does not perform serve-time window extraction.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator

from ..data_eng.config import PRICE_SCALE
from .config import VPRuntimeConfig
from .event_engine import AbsoluteTickEngine
from .replay_source import iter_mbo_events

logger = logging.getLogger(__name__)

FUTURES_WARMUP_HOURS = 0.5
EQUITY_WARMUP_HOURS = 0.5

_ANCHORABLE_ACTIONS = frozenset({"A", "M", "F", "C"})
_SOFT_REANCHOR_AFTER_ET_HHMM = "09:30"
_SOFT_REANCHOR_AFTER_EVENT_COUNT = 10_000


def _compute_time_boundaries(
    product_type: str,
    dt: str,
    start_time: str | None,
) -> tuple[int, int]:
    """Compute (warmup_start_ns, emit_after_ns) from ET start_time."""
    if not start_time:
        return 0, 0

    import pandas as pdt

    warmup_hours = FUTURES_WARMUP_HOURS if "future" in product_type else EQUITY_WARMUP_HOURS
    et_start = pdt.Timestamp(f"{dt} {start_time}:00", tz="America/New_York").tz_convert("UTC")
    warmup_start = et_start - pdt.Timedelta(hours=warmup_hours)
    return int(warmup_start.value), int(et_start.value)


def _resolve_tick_int(config: VPRuntimeConfig) -> int:
    """Resolve integer tick increment used by the pressure core."""
    if config.product_type == "future_mbo":
        tick_int = int(round(config.tick_size / PRICE_SCALE))
    elif config.product_type == "equity_mbo":
        tick_int = int(round(config.bucket_size_dollars / PRICE_SCALE))
    else:
        raise ValueError(f"Unsupported product_type: {config.product_type}")
    if tick_int <= 0:
        raise ValueError(
            f"Resolved tick_int must be > 0, got {tick_int} for {config.product_type}/{config.symbol}"
        )
    return tick_int


def _soft_reanchor_after_utc_ns(dt: str) -> int:
    """Resolve the one-time soft re-anchor ET clock boundary into UTC ns."""
    import pandas as pdt

    ts_utc = pdt.Timestamp(
        f"{dt} {_SOFT_REANCHOR_AFTER_ET_HHMM}:00",
        tz="America/New_York",
    ).tz_convert("UTC")
    return int(ts_utc.value)


def _create_core_engine(
    config: VPRuntimeConfig,
    *,
    fail_on_out_of_range: bool,
) -> AbsoluteTickEngine:
    """Create core engine with BBO-independent anchor behavior."""
    tick_int = _resolve_tick_int(config)
    return AbsoluteTickEngine(
        n_ticks=config.n_absolute_ticks,
        tick_int=tick_int,
        bucket_size_dollars=config.bucket_size_dollars,
        tau_velocity=config.tau_velocity,
        tau_acceleration=config.tau_acceleration,
        tau_jerk=config.tau_jerk,
        tau_rest_decay=config.tau_rest_decay,
        c1_v_add=config.c1_v_add,
        c2_v_rest_pos=config.c2_v_rest_pos,
        c3_a_add=config.c3_a_add,
        c4_v_pull=config.c4_v_pull,
        c5_v_fill=config.c5_v_fill,
        c6_v_rest_neg=config.c6_v_rest_neg,
        c7_a_pull=config.c7_a_pull,
        auto_anchor_from_bbo=False,
        fail_on_out_of_range=fail_on_out_of_range,
    )


def _price_to_abs_tick(price_int: int, tick_int: int) -> int:
    """Map integer price to absolute tick using nearest-tick rounding."""
    return int(math.floor(price_int / tick_int + 0.5))


def _ensure_anchor_from_event(engine: AbsoluteTickEngine, action: str, price_int: int) -> bool:
    """Set anchor once from first anchorable priced event when unset."""
    if engine.anchor_tick_idx >= 0:
        return False
    if action not in _ANCHORABLE_ACTIONS:
        return False
    if price_int <= 0:
        return False

    anchor_tick_idx = _price_to_abs_tick(price_int, engine.tick_int)
    engine.set_anchor_tick_idx(anchor_tick_idx)
    engine.sync_rest_depth_from_book()
    logger.info(
        "Core anchor set from event price: anchor_tick_idx=%d price_int=%d",
        anchor_tick_idx,
        price_int,
    )
    return True


def _snapshot_core_grid(
    engine: AbsoluteTickEngine,
    *,
    bin_seq: int,
    bin_start_ns: int,
    bin_end_ns: int,
    bin_event_count: int,
) -> Dict[str, Any]:
    """Snapshot the entire core grid as columnar numpy arrays."""
    full = engine.grid_snapshot_arrays()
    columns = {name: arr.copy() for name, arr in full.items()}

    anchor_tick_idx = engine.anchor_tick_idx
    if anchor_tick_idx >= 0:
        tick_abs_start = anchor_tick_idx - (engine.n_ticks // 2)
    else:
        tick_abs_start = 0

    return {
        "ts_ns": bin_end_ns,
        "bin_seq": bin_seq,
        "bin_start_ns": bin_start_ns,
        "bin_end_ns": bin_end_ns,
        "bin_event_count": bin_event_count,
        "event_id": engine.event_count,
        "anchor_tick_idx": anchor_tick_idx,
        "tick_int": engine.tick_int,
        "tick_abs_start": tick_abs_start,
        "n_rows": engine.n_ticks,
        "book_valid": engine.book_valid,
        "columns": columns,
    }


def stream_core_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
    *,
    fail_on_out_of_range: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """Synchronous pressure-core stream: DBN events -> full-grid fixed bins."""
    warmup_start_ns, emit_after_ns = _compute_time_boundaries(config.product_type, dt, start_time)
    engine = _create_core_engine(config, fail_on_out_of_range=fail_on_out_of_range)
    soft_reanchor_after_ns = _soft_reanchor_after_utc_ns(dt)
    soft_reanchor_applied = False

    cell_width_ns = int(config.cell_width_ms * 1_000_000)
    if cell_width_ns <= 0:
        raise ValueError(f"cell_width_ms must resolve to positive ns, got {config.cell_width_ms}")

    event_count = 0
    warmup_count = 0
    yielded_count = 0

    bin_initialized = False
    bin_seq = 0
    bin_start_ns = 0
    bin_end_ns = 0
    bin_event_count = 0

    t_wall_start = time.monotonic()

    for ts_ns, action, side, price, size, order_id, flags in iter_mbo_events(
        lake_root,
        config.product_type,
        config.symbol,
        dt,
    ):
        if ts_ns < warmup_start_ns:
            continue

        _ensure_anchor_from_event(engine, action, price)
        if not soft_reanchor_applied and engine.anchor_tick_idx >= 0:
            trigger_by_time = ts_ns >= soft_reanchor_after_ns
            trigger_by_events = event_count >= _SOFT_REANCHOR_AFTER_EVENT_COUNT
            if trigger_by_time or trigger_by_events:
                old_anchor_tick_idx = engine.anchor_tick_idx
                if engine.soft_reanchor_to_order_book_bbo():
                    soft_reanchor_applied = True
                    logger.info(
                        "Core soft re-anchor applied: old_anchor_tick_idx=%d new_anchor_tick_idx=%d "
                        "trigger_by_time=%s trigger_by_events=%s event_count=%d ts_ns=%d",
                        old_anchor_tick_idx,
                        engine.anchor_tick_idx,
                        trigger_by_time,
                        trigger_by_events,
                        event_count,
                        ts_ns,
                    )

        if emit_after_ns > 0 and ts_ns < emit_after_ns:
            engine.update(
                ts_ns=ts_ns,
                action=action,
                side=side,
                price_int=price,
                size=size,
                order_id=order_id,
                flags=flags,
            )
            warmup_count += 1
            event_count += 1
            continue

        if not bin_initialized:
            bin_start_ns = emit_after_ns if emit_after_ns > 0 else ts_ns
            bin_end_ns = bin_start_ns + cell_width_ns
            bin_seq = 0
            bin_event_count = 0
            bin_initialized = True

        while ts_ns >= bin_end_ns:
            engine.advance_time(bin_end_ns)
            yield _snapshot_core_grid(
                engine,
                bin_seq=bin_seq,
                bin_start_ns=bin_start_ns,
                bin_end_ns=bin_end_ns,
                bin_event_count=bin_event_count,
            )
            yielded_count += 1
            bin_seq += 1
            bin_start_ns = bin_end_ns
            bin_end_ns += cell_width_ns
            bin_event_count = 0

        engine.update(
            ts_ns=ts_ns,
            action=action,
            side=side,
            price_int=price,
            size=size,
            order_id=order_id,
            flags=flags,
        )
        bin_event_count += 1
        event_count += 1

    if bin_initialized and bin_event_count > 0:
        engine.advance_time(bin_end_ns)
        yield _snapshot_core_grid(
            engine,
            bin_seq=bin_seq,
            bin_start_ns=bin_start_ns,
            bin_end_ns=bin_end_ns,
            bin_event_count=bin_event_count,
        )
        yielded_count += 1

    elapsed = time.monotonic() - t_wall_start
    rate = event_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "core-grid stream complete: %d events (%d warmup, %d emitted bins), %.2fs wall (%.0f evt/s)",
        event_count,
        warmup_count,
        yielded_count,
        elapsed,
        rate,
    )


async def async_stream_core_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
    *,
    fail_on_out_of_range: bool = False,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async wrapper with fixed wall-clock pacing by cell width."""
    import concurrent.futures
    import queue as thread_queue

    q: thread_queue.Queue = thread_queue.Queue(maxsize=256)
    _SENTINEL = object()

    def _producer() -> None:
        try:
            produced = 0
            for grid in stream_core_events(
                lake_root=lake_root,
                config=config,
                dt=dt,
                start_time=start_time,
                fail_on_out_of_range=fail_on_out_of_range,
            ):
                q.put(grid)
                produced += 1
            logger.info("async core-grid producer done: %d bins produced", produced)
        except Exception as exc:  # pragma: no cover - passthrough logging
            logger.error("async core-grid producer error: %s", exc, exc_info=True)
        finally:
            q.put(_SENTINEL)

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = loop.run_in_executor(executor, _producer)

    first = True
    interval_s = float(config.cell_width_ms) / 1000.0

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break
            if not first:
                await asyncio.sleep(interval_s)
            first = False
            yield item
    finally:
        await future
        executor.shutdown(wait=False)
