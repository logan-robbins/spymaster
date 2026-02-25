"""EMA ensemble stream pipeline for qMachina.

Emits 1-row Arrow IPC per bin (k=0 only):
  k, fill_mass, rest_depth, last_event_id

BBO tracking and grid_update JSON messages are identical to the VP pipeline.
Gold features (EMA values, net signal) are computed in-browser from close prices.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np

from ...qmachina.config import RuntimeConfig
from ...qmachina.async_stream_wrapper import ProducerLatencyConfig, make_async_stream_events
from ...qmachina.book_cache import ensure_book_cache
from ...qmachina.engine_factory import create_absolute_tick_engine
from ...qmachina.stream_time_utils import compute_time_boundaries
from qm_engine import AbsoluteTickEngine
from qm_engine import iter_mbo_events

logger = logging.getLogger(__name__)

MODEL_ID = "ema_ensemble"


def build_model_config(config: "RuntimeConfig") -> dict:
    """Build EMA ensemble model_config payload for runtime_config wire message."""
    return {
        "model": MODEL_ID,
    }


def _build_ema_bin_grid(
    engine: AbsoluteTickEngine,
    *,
    bin_seq: int,
    bin_start_ns: int,
    bin_end_ns: int,
    bin_event_count: int,
) -> Dict[str, Any]:
    """Snapshot engine and emit a single k=0 row for EMA model."""
    spot_int = engine.spot_ref_price_int
    center_idx = engine.spot_to_idx(spot_int) if spot_int > 0 else None

    if center_idx is None or center_idx < 0 or center_idx >= engine.n_ticks:
        fill_mass = 0.0
        rest_depth = 0.0
        last_event_id = 0
    else:
        full = engine.grid_snapshot_arrays()
        fill_mass = float(full["fill_mass"][center_idx]) if "fill_mass" in full else 0.0
        rest_depth = float(full["rest_depth"][center_idx]) if "rest_depth" in full else 0.0
        last_event_id = int(full["last_event_id"][center_idx]) if "last_event_id" in full else 0

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
        "grid_cols": {
            "k": np.array([0], dtype=np.int32),
            "fill_mass": np.array([fill_mass], dtype=np.float64),
            "rest_depth": np.array([rest_depth], dtype=np.float64),
            "last_event_id": np.array([last_event_id], dtype=np.int64),
        },
    }


def stream_events(
    lake_root: Path,
    config: RuntimeConfig,
    dt: str,
    start_time: str | None = None,
) -> Generator[Dict[str, Any], None, None]:
    """Synchronous EMA pipeline: DBN events -> fixed-width single-row bins."""
    warmup_start_ns, emit_after_ns = compute_time_boundaries(
        config.product_type, dt, start_time,
    )

    engine = create_absolute_tick_engine(config)
    cell_width_ns = int(config.cell_width_ms * 1_000_000)
    if cell_width_ns <= 0:
        raise ValueError(f"cell_width_ms must resolve to positive ns, got {config.cell_width_ms}")

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
            engine.advance_time(bin_end_ns)
            grid = _build_ema_bin_grid(
                engine,
                bin_seq=bin_seq,
                bin_start_ns=bin_start_ns,
                bin_end_ns=bin_end_ns,
                bin_event_count=bin_event_count,
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
        engine.advance_time(bin_end_ns)
        grid = _build_ema_bin_grid(
            engine,
            bin_seq=bin_seq,
            bin_start_ns=bin_start_ns,
            bin_end_ns=bin_end_ns,
            bin_event_count=bin_event_count,
        )
        yield grid
        yielded_count += 1

    elapsed = time.monotonic() - t_wall_start
    rate = event_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "ema_ensemble stream complete: %d events (%d warmup, %d bins), %.2fs (%.0f evt/s)",
        event_count,
        warmup_count,
        yielded_count,
        elapsed,
        rate,
    )


async_stream_events = make_async_stream_events(
    stream_events,
    label="ema_ensemble",
    supports_latency=False,
)
