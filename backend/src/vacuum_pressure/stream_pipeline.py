"""Live event-stream pipeline for canonical fixed-bin VP streaming.

Uses AbsoluteTickEngine with per-tick independent state arrays.
Spectrum operates on ALL N_TICKS absolute ticks.
At serve time, a window of ±grid_radius_ticks around spot is extracted
and emitted with relative k values for frontend compatibility.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator

import numpy as np

from ..data_eng.config import PRICE_SCALE
from .config import VPRuntimeConfig
from .event_engine import AbsoluteTickEngine
from .replay_source import _resolve_dbn_path, iter_mbo_events
from .spectrum import IndependentCellSpectrum

logger = logging.getLogger(__name__)

FUTURES_WARMUP_HOURS = 0.5
EQUITY_WARMUP_HOURS = 0.5


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
    """Resolve integer price increment for event-engine bucket mapping."""
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


def _create_engine(config: VPRuntimeConfig) -> AbsoluteTickEngine:
    """Create the absolute-tick VP engine for fixed-bin streaming."""
    tick_int = _resolve_tick_int(config)
    return AbsoluteTickEngine(
        n_ticks=config.n_absolute_ticks,
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
    """Compute deterministic cache path for book state checkpoint."""
    try:
        dbn_path = _resolve_dbn_path(lake_root, product_type, symbol, dt)
    except FileNotFoundError:
        return None

    stat = dbn_path.stat()
    raw = f"v2:{product_type}:{symbol}:{dt}:{warmup_start_ns}:{stat.st_mtime_ns}:{stat.st_size}"
    key_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    cache_dir = lake_root / "cache" / "vp_book"
    return cache_dir / f"{symbol}_{dt}_{key_hash}.pkl"


def ensure_book_cache(
    engine: AbsoluteTickEngine,
    lake_root: Path,
    product_type: str,
    symbol: str,
    dt: str,
    warmup_start_ns: int,
) -> Path | None:
    """Build book-state cache if not present. Returns cache path."""
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
            "Book cache loaded in %.2fs: %d orders, anchor=%d, book_valid=%s",
            time.monotonic() - t_wall_start,
            engine.order_count,
            engine.anchor_tick_idx,
            engine.book_valid,
        )
        return cache_path

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
                "book-only fast-forward: %dM events (%.1fs, %d orders, anchor=%d)",
                book_only_count // 1_000_000,
                elapsed_so_far,
                engine.order_count,
                engine.anchor_tick_idx,
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(engine.export_book_state())
    engine.sync_rest_depth_from_book()

    elapsed_ff = time.monotonic() - t_wall_start
    logger.info(
        "Book cache built: %d events in %.2fs (%d orders, anchor=%d, book_valid=%s) -> %s",
        book_only_count,
        elapsed_ff,
        engine.order_count,
        engine.anchor_tick_idx,
        engine.book_valid,
        cache_path.name,
    )
    return cache_path


def _build_bin_grid(
    engine: AbsoluteTickEngine,
    spectrum: IndependentCellSpectrum,
    *,
    bin_seq: int,
    bin_start_ns: int,
    bin_end_ns: int,
    bin_event_count: int,
    window_radius: int,
) -> Dict[str, Any]:
    """Snapshot engine + spectrum, extract serve-time window around spot.

    1. Feed ALL N_TICKS pressure/vacuum to spectrum.
    2. Compute spot, find array center index.
    3. Slice ±window_radius ticks around spot.
    4. Emit with relative k values for frontend compatibility.
    """
    full = engine.grid_snapshot_arrays()
    pressure_full = full["pressure_variant"]
    vacuum_full = full["vacuum_variant"]

    # Spectrum on all absolute ticks
    spectrum_out = spectrum.update(
        ts_ns=bin_end_ns,
        pressure=pressure_full,
        vacuum=vacuum_full,
    )

    projection_horizons = spectrum.projection_horizons_ms
    projected = spectrum_out.projected_score_by_horizon

    # Compute spot and window center
    spot_int = engine.spot_ref_price_int
    center_idx = engine.spot_to_idx(spot_int) if spot_int > 0 else None

    if center_idx is None:
        # No valid spot — emit empty grid
        n_out = 2 * window_radius + 1
        return {
            "ts_ns": bin_end_ns,
            "bin_seq": bin_seq,
            "bin_start_ns": bin_start_ns,
            "bin_end_ns": bin_end_ns,
            "bin_event_count": bin_event_count,
            "event_id": engine.event_count,
            "spot_ref_price_int": 0,
            "mid_price": 0.0,
            "best_bid_price_int": engine.best_bid_price_int,
            "best_ask_price_int": engine.best_ask_price_int,
            "book_valid": engine.book_valid,
            "buckets": [
                _empty_bucket_row(k, projection_horizons)
                for k in range(-window_radius, window_radius + 1)
            ],
        }

    # Extract window with padding for edges
    n_ticks = engine.n_ticks
    w_start = center_idx - window_radius
    w_end = center_idx + window_radius + 1

    # Clamp to valid array bounds
    arr_start = max(0, w_start)
    arr_end = min(n_ticks, w_end)

    # How many zeros to pad on each side
    pad_left = arr_start - w_start
    pad_right = w_end - arr_end

    # Build output buckets
    buckets: list[Dict[str, Any]] = []
    out_idx = 0

    # Left padding (out-of-range ticks)
    for i in range(pad_left):
        k = -window_radius + i
        buckets.append(_empty_bucket_row(k, projection_horizons))
        out_idx += 1

    # Actual data slice
    for abs_idx in range(arr_start, arr_end):
        k = abs_idx - center_idx
        row: Dict[str, Any] = {
            "k": int(k),
            "add_mass": float(full["add_mass"][abs_idx]),
            "pull_mass": float(full["pull_mass"][abs_idx]),
            "fill_mass": float(full["fill_mass"][abs_idx]),
            "rest_depth": float(full["rest_depth"][abs_idx]),
            "v_add": float(full["v_add"][abs_idx]),
            "v_pull": float(full["v_pull"][abs_idx]),
            "v_fill": float(full["v_fill"][abs_idx]),
            "v_rest_depth": float(full["v_rest_depth"][abs_idx]),
            "a_add": float(full["a_add"][abs_idx]),
            "a_pull": float(full["a_pull"][abs_idx]),
            "a_fill": float(full["a_fill"][abs_idx]),
            "a_rest_depth": float(full["a_rest_depth"][abs_idx]),
            "j_add": float(full["j_add"][abs_idx]),
            "j_pull": float(full["j_pull"][abs_idx]),
            "j_fill": float(full["j_fill"][abs_idx]),
            "j_rest_depth": float(full["j_rest_depth"][abs_idx]),
            "pressure_variant": float(full["pressure_variant"][abs_idx]),
            "vacuum_variant": float(full["vacuum_variant"][abs_idx]),
            "last_event_id": int(full["last_event_id"][abs_idx]),
            "spectrum_score": float(spectrum_out.score[abs_idx]),
            "spectrum_state_code": int(spectrum_out.state_code[abs_idx]),
        }
        for horizon_ms in projection_horizons:
            row[f"proj_score_h{horizon_ms}"] = float(projected[horizon_ms][abs_idx])
        buckets.append(row)
        out_idx += 1

    # Right padding
    for i in range(pad_right):
        k = window_radius - pad_right + 1 + i
        buckets.append(_empty_bucket_row(k, projection_horizons))

    return {
        "ts_ns": bin_end_ns,
        "bin_seq": bin_seq,
        "bin_start_ns": bin_start_ns,
        "bin_end_ns": bin_end_ns,
        "bin_event_count": bin_event_count,
        "event_id": engine.event_count,
        "spot_ref_price_int": spot_int,
        "mid_price": engine.mid_price,
        "best_bid_price_int": engine.best_bid_price_int,
        "best_ask_price_int": engine.best_ask_price_int,
        "book_valid": engine.book_valid,
        "buckets": buckets,
    }


def _empty_bucket_row(k: int, projection_horizons: tuple[int, ...]) -> Dict[str, Any]:
    """Create a zero-valued bucket row for padding."""
    row: Dict[str, Any] = {
        "k": k,
        "add_mass": 0.0,
        "pull_mass": 0.0,
        "fill_mass": 0.0,
        "rest_depth": 0.0,
        "v_add": 0.0, "v_pull": 0.0, "v_fill": 0.0, "v_rest_depth": 0.0,
        "a_add": 0.0, "a_pull": 0.0, "a_fill": 0.0, "a_rest_depth": 0.0,
        "j_add": 0.0, "j_pull": 0.0, "j_fill": 0.0, "j_rest_depth": 0.0,
        "pressure_variant": 0.0,
        "vacuum_variant": 0.0,
        "last_event_id": 0,
        "spectrum_score": 0.0,
        "spectrum_state_code": 0,
    }
    for horizon_ms in projection_horizons:
        row[f"proj_score_h{horizon_ms}"] = 0.0
    return row


def stream_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
) -> Generator[Dict[str, Any], None, None]:
    """Synchronous pipeline: DBN events -> fixed-width cell bins."""
    warmup_start_ns, emit_after_ns = _compute_time_boundaries(
        config.product_type, dt, start_time,
    )

    engine = _create_engine(config)
    window_radius = config.grid_radius_ticks

    cell_width_ns = int(config.cell_width_ms * 1_000_000)
    if cell_width_ns <= 0:
        raise ValueError(f"cell_width_ms must resolve to positive ns, got {config.cell_width_ms}")

    # Spectrum operates on ALL N_TICKS absolute ticks
    spectrum = IndependentCellSpectrum(
        n_cells=config.n_absolute_ticks,
        windows=config.spectrum_windows,
        rollup_weights=config.spectrum_rollup_weights,
        derivative_weights=config.spectrum_derivative_weights,
        tanh_scale=config.spectrum_tanh_scale,
        neutral_threshold=config.spectrum_threshold_neutral,
        zscore_window_bins=config.zscore_window_bins,
        zscore_min_periods=config.zscore_min_periods,
        projection_horizons_ms=config.projection_horizons_ms,
        default_dt_s=float(config.cell_width_ms) / 1000.0,
    )

    event_count = 0
    yielded_count = 0
    warmup_count = 0

    bin_initialized = False
    bin_seq = 0
    bin_start_ns = 0
    bin_end_ns = 0
    bin_event_count = 0

    t_wall_start = time.monotonic()

    cache_loaded = False
    if warmup_start_ns > 0:
        cache_path = ensure_book_cache(
            engine, lake_root, config.product_type, config.symbol, dt, warmup_start_ns,
        )
        cache_loaded = cache_path is not None

    for event in iter_mbo_events(
        lake_root,
        config.product_type,
        config.symbol,
        dt,
        skip_to_ns=warmup_start_ns if cache_loaded else 0,
    ):
        ts_ns, action, side, price, size, order_id, flags = event

        if ts_ns < warmup_start_ns:
            continue

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
            grid = _build_bin_grid(
                engine,
                spectrum,
                bin_seq=bin_seq,
                bin_start_ns=bin_start_ns,
                bin_end_ns=bin_end_ns,
                bin_event_count=bin_event_count,
                window_radius=window_radius,
            )
            yield grid
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
        yield _build_bin_grid(
            engine,
            spectrum,
            bin_seq=bin_seq,
            bin_start_ns=bin_start_ns,
            bin_end_ns=bin_end_ns,
            bin_event_count=bin_event_count,
            window_radius=window_radius,
        )
        yielded_count += 1

    elapsed = time.monotonic() - t_wall_start
    rate = event_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "fixed-bin stream complete: %d events (%d warmup, %d emitted bins), %.2fs wall (%.0f evt/s)",
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
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async wrapper with fixed wall-clock pacing by cell width."""
    import concurrent.futures
    import queue as thread_queue

    q: thread_queue.Queue = thread_queue.Queue(maxsize=256)
    _SENTINEL = object()

    def _producer() -> None:
        try:
            produced = 0
            for grid in stream_events(
                lake_root=lake_root,
                config=config,
                dt=dt,
                start_time=start_time,
            ):
                q.put(grid)
                produced += 1
            logger.info("async fixed-bin producer done: %d bins produced", produced)
        except Exception as exc:
            logger.error("async fixed-bin producer error: %s", exc, exc_info=True)
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
