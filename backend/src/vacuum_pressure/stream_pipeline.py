"""Live event-stream pipeline for canonical fixed-bin VP streaming.

Uses AbsoluteTickEngine with per-tick independent state arrays.
Spectrum operates on ALL N_TICKS absolute ticks.
At serve time, a window of ±grid_radius_ticks around spot is extracted
and emitted with relative k values for frontend compatibility.
"""
from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator

import numpy as np

from ..data_eng.config import PRICE_SCALE
from .config import VPRuntimeConfig
from .event_engine import AbsoluteTickEngine
from .replay_source import _resolve_dbn_path, iter_mbo_events
from .spectrum import IndependentCellSpectrum, ProjectionModelConfig

logger = logging.getLogger(__name__)

FUTURES_WARMUP_HOURS = 0.5
EQUITY_WARMUP_HOURS = 0.5
_PRODUCER_PERF_KEY = "_producer_perf"

STATE5_BEAR_VACUUM = -2
STATE5_BEAR_PRESSURE = -1
STATE5_MIXED = 0
STATE5_BULL_PRESSURE = 1
STATE5_BULL_VACUUM = 2

_ABOVE_STATE5_BY_SIGNS: dict[tuple[int, int], int] = {
    (1, 1): STATE5_BULL_VACUUM,
    (1, 0): STATE5_BEAR_PRESSURE,
    (1, -1): STATE5_BEAR_PRESSURE,
    (0, 1): STATE5_BULL_VACUUM,
    (0, 0): STATE5_MIXED,
    (0, -1): STATE5_BEAR_PRESSURE,
    (-1, 1): STATE5_MIXED,
    (-1, 0): STATE5_BEAR_PRESSURE,
    (-1, -1): STATE5_BEAR_PRESSURE,
}

_BELOW_STATE5_BY_SIGNS: dict[tuple[int, int], int] = {
    (1, 1): STATE5_BULL_PRESSURE,
    (1, 0): STATE5_BEAR_VACUUM,
    (1, -1): STATE5_BEAR_VACUUM,
    (0, 1): STATE5_BULL_PRESSURE,
    (0, 0): STATE5_MIXED,
    (0, -1): STATE5_BEAR_VACUUM,
    (-1, 1): STATE5_MIXED,
    (-1, 0): STATE5_BEAR_VACUUM,
    (-1, -1): STATE5_BEAR_VACUUM,
}

_GRID_FLOAT_COLS: tuple[str, ...] = (
    "pressure_variant",
    "vacuum_variant",
    "add_mass",
    "pull_mass",
    "fill_mass",
    "rest_depth",
    "bid_depth",
    "ask_depth",
    "v_add",
    "v_pull",
    "v_fill",
    "v_rest_depth",
    "v_bid_depth",
    "v_ask_depth",
    "a_add",
    "a_pull",
    "a_fill",
    "a_rest_depth",
    "a_bid_depth",
    "a_ask_depth",
    "j_add",
    "j_pull",
    "j_fill",
    "j_rest_depth",
    "j_bid_depth",
    "j_ask_depth",
    "composite",
    "composite_d1",
    "composite_d2",
    "composite_d3",
)

_GRID_INT_COL_DTYPES: dict[str, Any] = {
    "best_ask_move_ticks": np.int32,
    "best_bid_move_ticks": np.int32,
    "ask_reprice_sign": np.int8,
    "bid_reprice_sign": np.int8,
    "microstate_id": np.int8,
    "state5_code": np.int8,
    "chase_up_flag": np.int8,
    "chase_down_flag": np.int8,
    "last_event_id": np.int64,
}


@dataclass(frozen=True)
class ProducerLatencyConfig:
    """Configuration for producer-latency telemetry output."""

    output_path: Path
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    summary_every_bins: int = 200


def _latency_us(start_ns: int | None, end_ns: int | None) -> float | None:
    """Return latency in microseconds, or None when inputs are missing."""
    if start_ns is None or end_ns is None:
        return None
    delta_ns = end_ns - start_ns
    if delta_ns < 0:
        return 0.0
    return float(delta_ns) / 1_000.0


def _sign_from_ticks(delta_ticks: int) -> int:
    """Map tick move into {-1, 0, +1}."""
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
    """Encode (ask_sign, bid_sign) into stable microstate id [0, 8]."""
    return int((ask_sign + 1) * 3 + (bid_sign + 1))


def _state5_code(k: int, ask_sign: int, bid_sign: int) -> int:
    """Map signs + bucket location to 5-state directional permutation code."""
    if k == 0:
        return STATE5_MIXED
    key = (ask_sign, bid_sign)
    if k > 0:
        return _ABOVE_STATE5_BY_SIGNS[key]
    return _BELOW_STATE5_BY_SIGNS[key]


def _annotate_permutation_labels(
    grid: Dict[str, Any],
    *,
    ask_move_ticks: int,
    bid_move_ticks: int,
) -> np.ndarray:
    """Annotate each emitted row with permutation microstate + directional labels."""
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
    k_values = np.asarray(cols["k"], dtype=np.int32)
    state_above = _ABOVE_STATE5_BY_SIGNS[(ask_sign, bid_sign)]
    state_below = _BELOW_STATE5_BY_SIGNS[(ask_sign, bid_sign)]

    state5_series = np.full(k_values.shape[0], STATE5_MIXED, dtype=np.int8)
    state5_series[k_values > 0] = np.int8(state_above)
    state5_series[k_values < 0] = np.int8(state_below)

    cols["best_ask_move_ticks"].fill(ask_move_ticks)
    cols["best_bid_move_ticks"].fill(bid_move_ticks)
    cols["ask_reprice_sign"].fill(ask_sign)
    cols["bid_reprice_sign"].fill(bid_sign)
    cols["microstate_id"].fill(micro_id)
    cols["state5_code"][:] = state5_series
    cols["chase_up_flag"].fill(chase_up_flag)
    cols["chase_down_flag"].fill(chase_down_flag)

    return state5_series


class _ProducerLatencyWriter:
    """Append per-bin producer latency metrics to a JSONL file."""

    def __init__(
        self,
        config: ProducerLatencyConfig,
        *,
        product_type: str,
        symbol: str,
        dt: str,
    ) -> None:
        if config.summary_every_bins <= 0:
            raise ValueError(
                f"summary_every_bins must be > 0, got {config.summary_every_bins}"
            )
        if (
            config.window_start_ns is not None
            and config.window_end_ns is not None
            and config.window_end_ns <= config.window_start_ns
        ):
            raise ValueError(
                "window_end_ns must be greater than window_start_ns for producer latency capture"
            )

        self._config = config
        self._product_type = product_type
        self._symbol = symbol
        self._dt = dt
        self._records_written = 0
        self._last_to_queue_us = deque(maxlen=config.summary_every_bins)
        self._queue_block_us = deque(maxlen=config.summary_every_bins)

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = config.output_path.open("a", encoding="utf-8")
        logger.info(
            "producer latency capture enabled: output=%s window_start_ns=%s window_end_ns=%s summary_every_bins=%d",
            config.output_path,
            config.window_start_ns,
            config.window_end_ns,
            config.summary_every_bins,
        )

    def close(self) -> None:
        self._emit_summary(force=True)
        self._fh.flush()
        self._fh.close()
        logger.info(
            "producer latency capture complete: %d records written to %s",
            self._records_written,
            self._config.output_path,
        )

    def _in_window(self, bin_start_ns: int, bin_end_ns: int) -> bool:
        if self._config.window_start_ns is not None and bin_end_ns <= self._config.window_start_ns:
            return False
        if self._config.window_end_ns is not None and bin_start_ns >= self._config.window_end_ns:
            return False
        return True

    @staticmethod
    def _pct(values: deque[float]) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype=np.float64)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        return float(p50), float(p95), float(p99)

    def _emit_summary(self, *, force: bool = False) -> None:
        if not self._last_to_queue_us:
            return
        if not force and self._records_written % self._config.summary_every_bins != 0:
            return

        lag_p50, lag_p95, lag_p99 = self._pct(self._last_to_queue_us)
        q_p50, q_p95, q_p99 = self._pct(self._queue_block_us)
        logger.info(
            "producer latency summary: records=%d sample=%d last_ingest_to_queue_put_done_us[p50=%.1f p95=%.1f p99=%.1f] queue_block_us[p50=%.1f p95=%.1f p99=%.1f]",
            self._records_written,
            len(self._last_to_queue_us),
            lag_p50,
            lag_p95,
            lag_p99,
            q_p50,
            q_p95,
            q_p99,
        )

    def record(
        self,
        grid: Dict[str, Any],
        perf_meta: Dict[str, int | None] | None,
        *,
        queue_put_start_wall_ns: int,
        queue_put_done_wall_ns: int,
    ) -> None:
        bin_start_ns = int(grid["bin_start_ns"])
        bin_end_ns = int(grid["bin_end_ns"])
        if not self._in_window(bin_start_ns, bin_end_ns):
            return

        first_ingest = None if perf_meta is None else perf_meta.get("bin_first_ingest_wall_ns")
        last_ingest = None if perf_meta is None else perf_meta.get("bin_last_ingest_wall_ns")
        grid_ready = None if perf_meta is None else perf_meta.get("grid_ready_wall_ns")

        first_to_ready = _latency_us(first_ingest, grid_ready)
        last_to_ready = _latency_us(last_ingest, grid_ready)
        ready_to_queue_done = _latency_us(grid_ready, queue_put_done_wall_ns)
        first_to_queue_done = _latency_us(first_ingest, queue_put_done_wall_ns)
        last_to_queue_done = _latency_us(last_ingest, queue_put_done_wall_ns)
        queue_block = _latency_us(queue_put_start_wall_ns, queue_put_done_wall_ns)

        record = {
            "ts_wall_ns": queue_put_done_wall_ns,
            "product_type": self._product_type,
            "symbol": self._symbol,
            "dt": self._dt,
            "bin_seq": int(grid["bin_seq"]),
            "bin_start_ns": bin_start_ns,
            "bin_end_ns": bin_end_ns,
            "bin_event_count": int(grid["bin_event_count"]),
            "event_id": int(grid["event_id"]),
            "bin_first_ingest_wall_ns": first_ingest,
            "bin_last_ingest_wall_ns": last_ingest,
            "grid_ready_wall_ns": grid_ready,
            "queue_put_start_wall_ns": queue_put_start_wall_ns,
            "queue_put_done_wall_ns": queue_put_done_wall_ns,
            "first_ingest_to_grid_ready_us": first_to_ready,
            "last_ingest_to_grid_ready_us": last_to_ready,
            "grid_ready_to_queue_put_done_us": ready_to_queue_done,
            "first_ingest_to_queue_put_done_us": first_to_queue_done,
            "last_ingest_to_queue_put_done_us": last_to_queue_done,
            "queue_block_us": queue_block,
        }
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

        if last_to_queue_done is not None:
            self._last_to_queue_us.append(last_to_queue_done)
        if queue_block is not None:
            self._queue_block_us.append(queue_block)

        self._records_written += 1
        self._emit_summary()


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
        engine.reanchor_to_bbo()
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
    engine.reanchor_to_bbo()
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
    """Snapshot engine + spectrum and emit a dense columnar window around spot."""
    full = engine.grid_snapshot_arrays()
    pressure_full = full["pressure_variant"]
    vacuum_full = full["vacuum_variant"]

    # Spectrum on all absolute ticks
    spectrum_out = spectrum.update(
        ts_ns=bin_end_ns,
        pressure=pressure_full,
        vacuum=vacuum_full,
    )

    # Compute spot and window center
    spot_int = engine.spot_ref_price_int
    center_idx = engine.spot_to_idx(spot_int) if spot_int > 0 else None

    n_rows = 2 * window_radius + 1
    k_values = np.arange(-window_radius, window_radius + 1, dtype=np.int32)
    grid_cols: Dict[str, np.ndarray] = {"k": k_values}
    for col in _GRID_FLOAT_COLS:
        grid_cols[col] = np.zeros(n_rows, dtype=np.float64)
    for col, dtype in _GRID_INT_COL_DTYPES.items():
        grid_cols[col] = np.zeros(n_rows, dtype=dtype)

    if center_idx is None:
        # No valid spot — emit empty grid columns.
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

    # Extract window with padding for edges
    n_ticks = engine.n_ticks
    w_start = center_idx - window_radius
    w_end = center_idx + window_radius + 1

    # Clamp to valid array bounds
    arr_start = max(0, w_start)
    arr_end = min(n_ticks, w_end)

    # How many zeros to pad on each side.
    pad_left = arr_start - w_start
    if arr_end > arr_start:
        dst_slice = slice(pad_left, pad_left + (arr_end - arr_start))
        src_slice = slice(arr_start, arr_end)

        grid_cols["pressure_variant"][dst_slice] = full["pressure_variant"][src_slice]
        grid_cols["vacuum_variant"][dst_slice] = full["vacuum_variant"][src_slice]
        grid_cols["add_mass"][dst_slice] = full["add_mass"][src_slice]
        grid_cols["pull_mass"][dst_slice] = full["pull_mass"][src_slice]
        grid_cols["fill_mass"][dst_slice] = full["fill_mass"][src_slice]
        grid_cols["rest_depth"][dst_slice] = full["rest_depth"][src_slice]
        grid_cols["bid_depth"][dst_slice] = full["bid_depth"][src_slice]
        grid_cols["ask_depth"][dst_slice] = full["ask_depth"][src_slice]
        grid_cols["v_add"][dst_slice] = full["v_add"][src_slice]
        grid_cols["v_pull"][dst_slice] = full["v_pull"][src_slice]
        grid_cols["v_fill"][dst_slice] = full["v_fill"][src_slice]
        grid_cols["v_rest_depth"][dst_slice] = full["v_rest_depth"][src_slice]
        grid_cols["v_bid_depth"][dst_slice] = full["v_bid_depth"][src_slice]
        grid_cols["v_ask_depth"][dst_slice] = full["v_ask_depth"][src_slice]
        grid_cols["a_add"][dst_slice] = full["a_add"][src_slice]
        grid_cols["a_pull"][dst_slice] = full["a_pull"][src_slice]
        grid_cols["a_fill"][dst_slice] = full["a_fill"][src_slice]
        grid_cols["a_rest_depth"][dst_slice] = full["a_rest_depth"][src_slice]
        grid_cols["a_bid_depth"][dst_slice] = full["a_bid_depth"][src_slice]
        grid_cols["a_ask_depth"][dst_slice] = full["a_ask_depth"][src_slice]
        grid_cols["j_add"][dst_slice] = full["j_add"][src_slice]
        grid_cols["j_pull"][dst_slice] = full["j_pull"][src_slice]
        grid_cols["j_fill"][dst_slice] = full["j_fill"][src_slice]
        grid_cols["j_rest_depth"][dst_slice] = full["j_rest_depth"][src_slice]
        grid_cols["j_bid_depth"][dst_slice] = full["j_bid_depth"][src_slice]
        grid_cols["j_ask_depth"][dst_slice] = full["j_ask_depth"][src_slice]
        grid_cols["composite"][dst_slice] = spectrum_out.composite[src_slice]
        grid_cols["composite_d1"][dst_slice] = spectrum_out.composite_d1[src_slice]
        grid_cols["composite_d2"][dst_slice] = spectrum_out.composite_d2[src_slice]
        grid_cols["composite_d3"][dst_slice] = spectrum_out.composite_d3[src_slice]
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
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
    *,
    capture_producer_timing: bool = False,
    projection_use_cubic: bool = False,
    projection_cubic_scale: float = 1.0 / 6.0,
    projection_damping_lambda: float = 0.0,
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
        windows=config.flow_windows,
        rollup_weights=config.flow_rollup_weights,
        derivative_weights=config.flow_derivative_weights,
        tanh_scale=config.flow_tanh_scale,
        neutral_threshold=config.flow_neutral_threshold,
        zscore_window_bins=config.flow_zscore_window_bins,
        zscore_min_periods=config.flow_zscore_min_periods,
        projection_horizons_ms=config.projection_horizons_ms,
        default_dt_s=float(config.cell_width_ms) / 1000.0,
        projection_model=ProjectionModelConfig(
            use_cubic=projection_use_cubic,
            cubic_scale=projection_cubic_scale,
            damping_lambda=projection_damping_lambda,
        ),
    )
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
            grid = _build_bin_grid(
                engine,
                spectrum,
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
                grid[_PRODUCER_PERF_KEY] = {
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
            spectrum,
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
            grid[_PRODUCER_PERF_KEY] = {
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


async def async_stream_events(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    start_time: str | None = None,
    *,
    producer_latency_config: ProducerLatencyConfig | None = None,
    projection_use_cubic: bool = False,
    projection_cubic_scale: float = 1.0 / 6.0,
    projection_damping_lambda: float = 0.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async wrapper with fixed wall-clock pacing by cell width."""
    import concurrent.futures
    import queue as thread_queue

    q: thread_queue.Queue = thread_queue.Queue(maxsize=256)
    _SENTINEL = object()
    producer_errors: list[Exception] = []

    def _producer() -> None:
        latency_writer = None
        try:
            if producer_latency_config is not None:
                latency_writer = _ProducerLatencyWriter(
                    producer_latency_config,
                    product_type=config.product_type,
                    symbol=config.symbol,
                    dt=dt,
                )
            produced = 0
            for grid in stream_events(
                lake_root=lake_root,
                config=config,
                dt=dt,
                start_time=start_time,
                capture_producer_timing=latency_writer is not None,
                projection_use_cubic=projection_use_cubic,
                projection_cubic_scale=projection_cubic_scale,
                projection_damping_lambda=projection_damping_lambda,
            ):
                perf_meta = grid.pop(_PRODUCER_PERF_KEY, None) if latency_writer is not None else None
                queue_put_start_wall_ns = time.monotonic_ns()
                q.put(grid)
                queue_put_done_wall_ns = time.monotonic_ns()
                if latency_writer is not None:
                    latency_writer.record(
                        grid,
                        perf_meta,
                        queue_put_start_wall_ns=queue_put_start_wall_ns,
                        queue_put_done_wall_ns=queue_put_done_wall_ns,
                    )
                produced += 1
            logger.info("async fixed-bin producer done: %d bins produced", produced)
        except Exception as exc:
            logger.error("async fixed-bin producer error: %s", exc, exc_info=True)
            producer_errors.append(exc)
        finally:
            if latency_writer is not None:
                latency_writer.close()
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
                if producer_errors:
                    raise RuntimeError("async fixed-bin producer failed") from producer_errors[0]
                break
            if not first:
                await asyncio.sleep(interval_s)
            first = False
            yield item
    finally:
        await future
        executor.shutdown(wait=False)
