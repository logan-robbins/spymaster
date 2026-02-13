"""DBN file replay source for vacuum-pressure live streaming.

Reads raw Databento .dbn files and yields MBO events one-by-one,
with optional real-time pacing based on ts_event deltas.

The generator produces tuples matching the book engine's apply_event()
signature: (ts, action, side, price, size, order_id, flags).

Two pacing modes:
    speed > 0: Pace events by ts_event deltas scaled by 1/speed.
               speed=1.0 is wall-clock real-time; speed=10.0 is 10x.
    speed = 0: Fire-hose mode (as fast as possible, for backtest).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Generator, Tuple

import numpy as np

from ..data_eng.utils import session_window_ns

logger = logging.getLogger(__name__)

F_SNAPSHOT = 32
NULL_PRICE = np.iinfo("int64").max

# Event tuple: (ts_event, action, side, price, size, order_id, flags)
MBOEvent = Tuple[int, str, str, int, int, int, int]


def _resolve_dbn_path(
    lake_root: Path,
    product_type: str,
    symbol: str,
    dt: str,
) -> Path:
    """Resolve the raw .dbn file path for a given symbol and date.

    For futures, the lake stores raw data under the parent symbol
    (e.g. MNQ, not MNQH6). We search for .dbn files matching the date.

    Args:
        lake_root: Path to the lake directory.
        product_type: Product type (equity_mbo or future_mbo).
        symbol: Instrument symbol (e.g. MNQH6 or QQQ).
        dt: Date string (YYYY-MM-DD).

    Returns:
        Path to the .dbn file.

    Raises:
        FileNotFoundError: If no .dbn file found.
    """
    date_compact = dt.replace("-", "")

    # For futures, extract parent symbol (MNQH6 -> MNQ, SIH6 -> SI)
    # For equities, symbol is already the ticker (QQQ)
    if product_type == "future_mbo":
        # Try progressively shorter prefixes
        parent = symbol
        for i in range(len(symbol), 0, -1):
            candidate = symbol[:i]
            raw_dir = (
                lake_root / "raw" / "source=databento"
                / f"product_type={product_type}"
                / f"symbol={candidate}"
                / "table=market_by_order_dbn"
            )
            if raw_dir.exists():
                parent = candidate
                break
    else:
        parent = symbol

    raw_dir = (
        lake_root / "raw" / "source=databento"
        / f"product_type={product_type}"
        / f"symbol={parent}"
        / "table=market_by_order_dbn"
    )

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw DBN directory not found: {raw_dir}\n"
            f"Download raw data first with batch_download scripts."
        )

    dbn_files = list(raw_dir.glob(f"*{date_compact}*.dbn*"))
    if not dbn_files:
        raise FileNotFoundError(
            f"No .dbn files found for date {dt} in {raw_dir}/\n"
            f"Available files: {[f.name for f in raw_dir.iterdir()]}"
        )

    # Prefer uncompressed .dbn over .dbn.zst
    for f in dbn_files:
        if f.suffix == ".dbn":
            return f
    return dbn_files[0]


def iter_mbo_events(
    lake_root: Path,
    product_type: str,
    symbol: str,
    dt: str,
    skip_to_ns: int = 0,
) -> Generator[MBOEvent, None, None]:
    """Iterate MBO events from a raw .dbn file.

    Applies the same filtering as the bronze ingestion stage:
    - rtype == 160 (MBO only)
    - Drops spread symbols (futures: symbols containing '-')
    - Session window filter (snapshot/clear records exempt)
    - Null price handling

    Events are yielded in file order (already sorted by ts_event, sequence
    within a Databento .dbn file).

    Args:
        lake_root: Path to the lake directory.
        product_type: Product type (equity_mbo or future_mbo).
        symbol: Contract symbol to filter (e.g. MNQH6 or QQQ).
        dt: Date string (YYYY-MM-DD).
        skip_to_ns: Skip non-snapshot events before this nanosecond
            timestamp. Snapshot/Clear records are always emitted
            regardless (they build the initial book state). This
            allows fast startup by jumping to a warmup point.

    Yields:
        Tuples of (ts_event, action, side, price, size, order_id, flags).
    """
    import databento as db
    from databento_dbn import MBOMsg

    dbn_path = _resolve_dbn_path(lake_root, product_type, symbol, dt)
    logger.info(
        "Opening DBN replay: %s (%s, %s, %s)",
        dbn_path.name, product_type, symbol, dt,
    )

    # Check metadata for symbol filtering
    store = db.DBNStore.from_file(str(dbn_path))
    metadata_symbols = set(store.metadata.symbols) if store.metadata.symbols else set()
    logger.info(
        "DBN metadata: symbols=%s, schema=%s",
        metadata_symbols, store.metadata.schema,
    )

    # Build instrument_id -> symbol mapping from metadata
    iid_to_symbol: dict[int, str] = {}
    if store.metadata.mappings:
        for sym_name, mappings in store.metadata.mappings.items():
            for m in mappings:
                # m is a dict or object with 'symbol' = instrument_id string
                iid_str = m.get("symbol", "") if isinstance(m, dict) else getattr(m, "symbol", "")
                if iid_str:
                    try:
                        iid_to_symbol[int(iid_str)] = sym_name
                    except (ValueError, TypeError):
                        pass
    logger.info("Instrument map: %s", iid_to_symbol)

    session_start_ns, session_end_ns = session_window_ns(dt, product_type)

    event_count = 0
    yielded_count = 0

    # Snapshot timestamp normalization:
    # Databento MBO snapshot records carry the ORIGINAL order placement
    # timestamp as ts_event (spanning hours/days of history). If fed
    # directly to the book engine, each ts_event change triggers
    # _flush_until which generates hundreds of thousands of empty
    # gap-fill windows (382K+ for GLBX futures).
    #
    # Fix: normalize all snapshot record timestamps to the Clear
    # record's ts_event. This puts the entire snapshot into a single
    # 1-second window, matching the book engine's expectations.
    snapshot_active = False
    snapshot_anchor_ts: int = 0

    for record in store:
        event_count += 1

        # Filter: MBO records only
        if not isinstance(record, MBOMsg):
            continue

        # Extract fields -- enum values need string conversion
        ts_event = int(record.ts_event)
        action = str(record.action)       # Action.ADD -> "A", Action.CLEAR -> "R"
        side = str(record.side)           # Side.BID -> "B", Side.ASK -> "A"
        price = int(record.price)
        size = int(record.size)
        order_id = int(record.order_id)
        flags = int(record.flags)

        is_snapshot = (flags & F_SNAPSHOT) != 0
        is_last = (flags & 128) != 0

        # Track snapshot state for timestamp normalization
        if action == "R" and is_snapshot:
            snapshot_active = True
            snapshot_anchor_ts = ts_event
        elif snapshot_active and is_last:
            # F_LAST ends the snapshot sequence
            snapshot_active = False

        # Normalize snapshot timestamps to prevent gap-fill explosion
        if is_snapshot and snapshot_anchor_ts > 0:
            ts_event = snapshot_anchor_ts

        # Filter: symbol match via instrument_id mapping
        iid = record.instrument_id
        rec_symbol = iid_to_symbol.get(iid, "")

        # For futures, drop spread symbols (contain '-')
        if product_type == "future_mbo" and "-" in rec_symbol:
            continue

        # Filter to requested contract symbol
        if rec_symbol and rec_symbol != symbol:
            continue

        # Session window filter
        is_clear = action == "R"
        in_window = session_start_ns <= ts_event < session_end_ns

        if not (in_window or is_snapshot or is_clear):
            continue

        # Null price handling
        if price == NULL_PRICE:
            if action in ("A", "M"):
                logger.warning(
                    "Null price on Add/Modify at ts=%d, skipping", ts_event
                )
                continue
            price = 0

        # Fast-forward: skip non-snapshot events before skip_to_ns.
        # Snapshot and Clear records are always emitted (they build
        # the initial book state). This makes startup fast by jumping
        # to a warmup point without re-processing the entire session.
        if skip_to_ns > 0 and ts_event < skip_to_ns:
            if not (is_snapshot or is_clear):
                continue

        yielded_count += 1
        yield (ts_event, action, side, price, size, order_id, flags)

    logger.info(
        "DBN replay complete: %d total records, %d MBO events yielded (%d skipped by skip_to_ns)",
        event_count, yielded_count, event_count - yielded_count,
    )


async def async_iter_mbo_events(
    lake_root: Path,
    product_type: str,
    symbol: str,
    dt: str,
    speed: float = 1.0,
    skip_ns: int = 0,
) -> "AsyncIterator":
    """Async generator that yields MBO events with optional pacing.

    Wraps iter_mbo_events with asyncio.sleep-based pacing so the event
    loop remains responsive during replay.

    Args:
        lake_root: Path to the lake directory.
        product_type: Product type.
        symbol: Contract symbol.
        dt: Date string.
        speed: Replay speed multiplier. 0 = fire-hose (no delays).
        skip_ns: Skip events before this timestamp (nanoseconds).
            Events before skip_ns are yielded without pacing.

    Yields:
        Tuples of (ts_event, action, side, price, size, order_id, flags).
    """
    prev_ts: int | None = None
    pacing = speed > 0

    # Batch size for yielding events between sleeps.
    # We don't sleep after every single event (millions per second);
    # instead we batch by time boundary.
    last_window_id: int | None = None

    for event in iter_mbo_events(lake_root, product_type, symbol, dt):
        ts = event[0]

        # Fast-forward mode: yield without pacing until skip_ns
        if ts < skip_ns:
            yield event
            prev_ts = ts
            continue

        # Pacing: sleep at window boundaries (1-second boundaries)
        if pacing and prev_ts is not None:
            window_id = ts // 1_000_000_000
            if last_window_id is not None and window_id > last_window_id:
                # Sleep for the inter-window gap
                delta_ns = ts - prev_ts
                if delta_ns > 0:
                    await asyncio.sleep(delta_ns / (1e9 * speed))
            last_window_id = window_id

        prev_ts = ts
        if last_window_id is None:
            last_window_id = ts // 1_000_000_000

        yield event
