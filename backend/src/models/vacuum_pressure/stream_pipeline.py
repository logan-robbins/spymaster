"""Live event-stream pipeline for canonical fixed-bin silver streaming.

Uses AbsoluteTickEngine with per-tick independent state arrays.
Emits a window of ±grid_radius_ticks around spot with relative k values.

Silver output: base cell tensors (mechanics + EMA derivatives) + BBO metadata only.
Gold features (pressure_variant, vacuum_variant, composite*, state5_code,
flow_score, flow_state_code) are computed downstream by gold_builder (offline)
or GoldFeatureRuntime (frontend in-memory).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np

from ...qmachina.config import RuntimeConfig
from ...qmachina.stage_schema import SILVER_FLOAT_COLS, SILVER_INT_COL_DTYPES
from ...qmachina.async_stream_wrapper import PRODUCER_PERF_KEY, ProducerLatencyConfig, make_async_stream_events
from ...qmachina.book_cache import ensure_book_cache
from ...qmachina.engine_factory import create_absolute_tick_engine
from ...qmachina.stream_time_utils import compute_time_boundaries
from qm_engine import AbsoluteTickEngine
from qm_engine import iter_mbo_events

logger = logging.getLogger(__name__)


def build_model_config(config: "RuntimeConfig") -> dict:
    """Build VP-specific model_config payload for runtime_config wire message."""
    return {
        "model": config.model_id,
        "state_model": {
            "name": "derivative",
            "center_exclusion_radius": config.state_model_center_exclusion_radius,
            "spatial_decay_power": config.state_model_spatial_decay_power,
            "zscore_window_bins": config.state_model_zscore_window_bins,
            "zscore_min_periods": config.state_model_zscore_min_periods,
            "tanh_scale": config.state_model_tanh_scale,
            "d1_weight": config.state_model_d1_weight,
            "d2_weight": config.state_model_d2_weight,
            "d3_weight": config.state_model_d3_weight,
            "bull_pressure_weight": config.state_model_bull_pressure_weight,
            "bull_vacuum_weight": config.state_model_bull_vacuum_weight,
            "bear_pressure_weight": config.state_model_bear_pressure_weight,
            "bear_vacuum_weight": config.state_model_bear_vacuum_weight,
            "mixed_weight": config.state_model_mixed_weight,
        },
    }


def _sign_from_ticks(delta_ticks: int) -> int:
    if delta_ticks > 0:
        return 1
    if delta_ticks < 0:
        return -1
    return 0


def _best_move_ticks(
    current_price_int: int,
    previous_price_int: int | None,
    *,
    tick_int: int,
) -> int:
    """Compute integer best-price move in ticks between emitted bins."""
    if tick_int <= 0:
        return 0
    if previous_price_int is None:
        return 0
    if current_price_int <= 0 or previous_price_int <= 0:
        return 0
    return int(round((current_price_int - previous_price_int) / tick_int))


def _microstate_id(ask_sign: int, bid_sign: int) -> int:
    return int((ask_sign + 1) * 3 + (bid_sign + 1))


def _annotate_permutation_labels(
    grid: Dict[str, Any],
    *,
    ask_move_ticks: int,
    bid_move_ticks: int,
) -> None:
    """Annotate each emitted row with BBO movement microstate labels (silver only)."""
    ask_sign = _sign_from_ticks(ask_move_ticks)
    bid_sign = _sign_from_ticks(bid_move_ticks)
    micro_id = _microstate_id(ask_sign, bid_sign)
    chase_up_flag = int(ask_sign > 0 and bid_sign > 0)
    chase_down_flag = int(ask_sign < 0 and bid_sign < 0)

    grid["best_ask_move_ticks"] = ask_move_ticks
    grid["best_bid_move_ticks"] = bid_move_ticks
    grid["ask_reprice_sign"] = ask_sign
    grid["bid_reprice_sign"] = bid_sign
    grid["microstate_id"] = micro_id
    grid["chase_up_flag"] = chase_up_flag
    grid["chase_down_flag"] = chase_down_flag

    cols: Dict[str, np.ndarray] = grid["grid_cols"]
    cols["best_ask_move_ticks"].fill(ask_move_ticks)
    cols["best_bid_move_ticks"].fill(bid_move_ticks)
    cols["ask_reprice_sign"].fill(ask_sign)
    cols["bid_reprice_sign"].fill(bid_sign)
    cols["microstate_id"].fill(micro_id)
    cols["chase_up_flag"].fill(chase_up_flag)
    cols["chase_down_flag"].fill(chase_down_flag)


def _build_bin_grid(
    engine: AbsoluteTickEngine,
    *,
    bin_seq: int,
    bin_start_ns: int,
    bin_end_ns: int,
    bin_event_count: int,
    window_radius: int,
) -> Dict[str, Any]:
    """Snapshot engine and emit a dense silver columnar window around spot."""
    full = engine.grid_snapshot_arrays()

    spot_int = engine.spot_ref_price_int
    center_idx = engine.spot_to_idx(spot_int) if spot_int > 0 else None

    n_rows = 2 * window_radius + 1
    k_values = np.arange(-window_radius, window_radius + 1, dtype=np.int32)
    grid_cols: Dict[str, np.ndarray] = {"k": k_values}
    for col in SILVER_FLOAT_COLS:
        grid_cols[col] = np.zeros(n_rows, dtype=np.float64)
    for col, dtype in SILVER_INT_COL_DTYPES.items():
        if col == "k":
            continue  # k is set above as a range; do not overwrite
        grid_cols[col] = np.zeros(n_rows, dtype=dtype)

    if center_idx is None:
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
            "grid_cols": grid_cols,
        }

    n_ticks = engine.n_ticks
    w_start = center_idx - window_radius
    w_end = center_idx + window_radius + 1

    arr_start = max(0, w_start)
    arr_end = min(n_ticks, w_end)

    pad_left = arr_start - w_start
    if arr_end > arr_start:
        dst_slice = slice(pad_left, pad_left + (arr_end - arr_start))
        src_slice = slice(arr_start, arr_end)

        for col in SILVER_FLOAT_COLS:
            if col in full:
                grid_cols[col][dst_slice] = full[col][src_slice]
        grid_cols["last_event_id"][dst_slice] = np.asarray(
            full["last_event_id"][src_slice], dtype=np.int64
        )

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
        "grid_cols": grid_cols,
    }


def stream_events(
    lake_root: Path,
    config: RuntimeConfig,
    dt: str,
    start_time: str | None = None,
    *,
    capture_producer_timing: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """Synchronous pipeline: DBN events -> fixed-width silver cell bins."""
    warmup_start_ns, emit_after_ns = compute_time_boundaries(
        config.product_type, dt, start_time,
    )

    engine = create_absolute_tick_engine(config)
    window_radius = config.grid_radius_ticks

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
    bin_first_ingest_wall_ns: int | None = None
    bin_last_ingest_wall_ns: int | None = None
    prev_best_bid_price_int: int | None = None
    prev_best_ask_price_int: int | None = None

    t_wall_start = time.monotonic()

    cache_loaded = False
    if warmup_start_ns > 0:
        cache_path = ensure_book_cache(
            engine, lake_root, config.product_type, config.symbol, dt, warmup_start_ns,
        )
        cache_loaded = cache_path is not None

    for event in iter_mbo_events(
        str(lake_root),
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
            grid = _build_bin_grid(
                engine,
                bin_seq=bin_seq,
                bin_start_ns=bin_start_ns,
                bin_end_ns=bin_end_ns,
                bin_event_count=bin_event_count,
                window_radius=window_radius,
            )
            ask_move_ticks = _best_move_ticks(
                int(grid["best_ask_price_int"]),
                prev_best_ask_price_int,
                tick_int=engine.tick_int,
            )
            bid_move_ticks = _best_move_ticks(
                int(grid["best_bid_price_int"]),
                prev_best_bid_price_int,
                tick_int=engine.tick_int,
            )
            _annotate_permutation_labels(
                grid,
                ask_move_ticks=ask_move_ticks,
                bid_move_ticks=bid_move_ticks,
            )
            prev_best_ask_price_int = int(grid["best_ask_price_int"])
            prev_best_bid_price_int = int(grid["best_bid_price_int"])
            if capture_producer_timing:
                grid[PRODUCER_PERF_KEY] = {
                    "bin_first_ingest_wall_ns": bin_first_ingest_wall_ns,
                    "bin_last_ingest_wall_ns": bin_last_ingest_wall_ns,
                    "grid_ready_wall_ns": time.monotonic_ns(),
                }
            yield grid
            yielded_count += 1

            bin_seq += 1
            bin_start_ns = bin_end_ns
            bin_end_ns += cell_width_ns
            bin_event_count = 0
            bin_first_ingest_wall_ns = None
            bin_last_ingest_wall_ns = None

        ingest_wall_ns = time.monotonic_ns() if capture_producer_timing else 0
        engine.update(
            ts_ns=ts_ns,
            action=action,
            side=side,
            price_int=price,
            size=size,
            order_id=order_id,
            flags=flags,
        )
        if capture_producer_timing:
            if bin_event_count == 0:
                bin_first_ingest_wall_ns = ingest_wall_ns
            bin_last_ingest_wall_ns = ingest_wall_ns
        bin_event_count += 1
        event_count += 1

    if bin_initialized and bin_event_count > 0:
        engine.advance_time(bin_end_ns)
        grid = _build_bin_grid(
            engine,
            bin_seq=bin_seq,
            bin_start_ns=bin_start_ns,
            bin_end_ns=bin_end_ns,
            bin_event_count=bin_event_count,
            window_radius=window_radius,
        )
        ask_move_ticks = _best_move_ticks(
            int(grid["best_ask_price_int"]),
            prev_best_ask_price_int,
            tick_int=engine.tick_int,
        )
        bid_move_ticks = _best_move_ticks(
            int(grid["best_bid_price_int"]),
            prev_best_bid_price_int,
            tick_int=engine.tick_int,
        )
        _annotate_permutation_labels(
            grid,
            ask_move_ticks=ask_move_ticks,
            bid_move_ticks=bid_move_ticks,
        )
        if capture_producer_timing:
            grid[PRODUCER_PERF_KEY] = {
                "bin_first_ingest_wall_ns": bin_first_ingest_wall_ns,
                "bin_last_ingest_wall_ns": bin_last_ingest_wall_ns,
                "grid_ready_wall_ns": time.monotonic_ns(),
            }
        yield grid
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


async_stream_events = make_async_stream_events(
    stream_events,
    label="vacuum_pressure",
    supports_latency=True,
)
