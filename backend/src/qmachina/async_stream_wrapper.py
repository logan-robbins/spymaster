"""Shared async stream wrapper and producer latency infrastructure.

Encapsulates the thread-pool + queue + sentinel + wall-clock pacing pattern
used by every stream pipeline. Models call make_async_stream_events() to get
an async generator function that wraps their synchronous stream_events().

ProducerLatencyConfig lives here because it is relevant only to this wrapper's
optional timing output, not to any specific model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Generator

import numpy as np

logger = logging.getLogger(__name__)

# Key used by VP's stream_events to embed timing metadata in a grid dict.
# The async wrapper pops this key before enqueueing.
PRODUCER_PERF_KEY = "_producer_perf"


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
        self._last_to_queue_us: deque[float] = deque(maxlen=config.summary_every_bins)
        self._queue_block_us: deque[float] = deque(maxlen=config.summary_every_bins)

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
    def _pct(values: deque) -> tuple[float, float, float]:
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
            "producer latency summary: records=%d sample=%d "
            "last_ingest_to_queue_put_done_us[p50=%.1f p95=%.1f p99=%.1f] "
            "queue_block_us[p50=%.1f p95=%.1f p99=%.1f]",
            self._records_written,
            len(self._last_to_queue_us),
            lag_p50, lag_p95, lag_p99,
            q_p50, q_p95, q_p99,
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


def make_async_stream_events(
    sync_fn: Callable[..., Generator[Dict[str, Any], None, None]],
    *,
    label: str,
    supports_latency: bool = False,
) -> Callable:
    """Return an async generator function that wraps a synchronous stream_events function.

    Args:
        sync_fn: The synchronous generator function (stream_events).
        label: Human-readable model name for log messages.
        supports_latency: If True, passes capture_producer_timing to sync_fn and
                          handles ProducerLatencyConfig telemetry in the wrapper.

    Returns:
        An async generator function with the standard signature:
        async_stream_events(lake_root, config, dt, start_time, *, producer_latency_config)
    """

    async def async_stream_events(
        lake_root,
        config,
        dt: str,
        start_time: str | None = None,
        *,
        producer_latency_config: "ProducerLatencyConfig | None" = None,
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
                if supports_latency and producer_latency_config is not None:
                    latency_writer = _ProducerLatencyWriter(
                        producer_latency_config,
                        product_type=config.product_type,
                        symbol=config.symbol,
                        dt=dt,
                    )
                produced = 0
                kwargs: dict = {}
                if supports_latency:
                    kwargs["capture_producer_timing"] = latency_writer is not None
                for grid in sync_fn(
                    lake_root=lake_root,
                    config=config,
                    dt=dt,
                    start_time=start_time,
                    **kwargs,
                ):
                    perf_meta = (
                        grid.pop(PRODUCER_PERF_KEY, None)
                        if latency_writer is not None
                        else None
                    )
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
                logger.info("%s async producer done: %d bins produced", label, produced)
            except Exception as exc:
                logger.error("%s async producer error: %s", label, exc, exc_info=True)
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
                        raise RuntimeError(f"{label} producer failed") from producer_errors[0]
                    break
                if not first:
                    await asyncio.sleep(interval_s)
                first = False
                yield item
        finally:
            await future
            executor.shutdown(wait=False)

    return async_stream_events
