"""Book-state cache build and load utilities for stream pipelines."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_book_cache_path(
    lake_root: Path,
    product_type: str,
    symbol: str,
    dt: str,
    warmup_start_ns: int,
) -> Path | None:
    """Compute deterministic cache path for book state checkpoint."""
    from qm_engine import resolve_dbn_path as _resolve_dbn_path

    try:
        dbn_path = Path(_resolve_dbn_path(str(lake_root), product_type, symbol, dt))
    except FileNotFoundError:
        return None

    stat = dbn_path.stat()
    raw = f"v2:{product_type}:{symbol}:{dt}:{warmup_start_ns}:{stat.st_mtime_ns}:{stat.st_size}"
    key_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    cache_dir = lake_root / "cache" / "book_engine"
    return cache_dir / f"{symbol}_{dt}_{key_hash}.pkl"


def ensure_book_cache(
    engine,
    lake_root: Path,
    product_type: str,
    symbol: str,
    dt: str,
    warmup_start_ns: int,
) -> Path | None:
    """Build book-state cache if not present. Returns cache path."""
    from qm_engine import iter_mbo_events

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
    for event in iter_mbo_events(str(lake_root), product_type, symbol, dt):
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
    cache_path.write_bytes(bytes(engine.export_book_state()))
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
