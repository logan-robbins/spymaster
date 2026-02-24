"""Shared pipeline utilities for core and stream pipelines.

Extracted to eliminate duplication between core_pipeline.py and
stream_pipeline.py.
"""
from __future__ import annotations

from ...data_eng.config import PRICE_SCALE
from ...qmachina.config import RuntimeConfig

FUTURES_WARMUP_HOURS: float = 0.5
EQUITY_WARMUP_HOURS: float = 0.5


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


def _resolve_tick_int(config: RuntimeConfig) -> int:
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
