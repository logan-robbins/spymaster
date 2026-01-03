"""Load Bronze data stage - first stage in all pipelines."""
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.duckdb_reader import DuckDBReader
from src.common.event_types import FuturesTrade, MBP10, BidAskLevel, EventSource, Aggressor
from src.common.config import CONFIG

logger = logging.getLogger(__name__)


def futures_trades_from_df(trades_df: pd.DataFrame) -> List[FuturesTrade]:
    """Convert futures trades DataFrame to FuturesTrade objects.
    
    Supports both legacy trades schema and MBP-10 extracted trades.
    MBP-10 trades have: ts_event_ns, price, size, aggressor, symbol
    Legacy trades have: ts_event_ns, ts_recv_ns, price, size, aggressor, symbol, exchange, seq
    """
    if trades_df.empty:
        return []

    df = trades_df
    if not df["ts_event_ns"].is_monotonic_increasing:
        df = df.sort_values("ts_event_ns")

    ts_event = df["ts_event_ns"].to_numpy()
    ts_recv = df["ts_recv_ns"].to_numpy() if "ts_recv_ns" in df.columns else ts_event
    prices = df["price"].to_numpy()
    sizes = df["size"].to_numpy()
    symbols = df["symbol"].to_numpy() if "symbol" in df.columns else np.array(["ES"] * len(df))

    if "aggressor" in df.columns:
        aggressors = pd.to_numeric(df["aggressor"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        aggressors = np.zeros(len(df), dtype=int)

    agg_map = {1: Aggressor.BUY, -1: Aggressor.SELL, 0: Aggressor.MID}

    # Vectorized construction for performance
    trades: List[FuturesTrade] = [
        FuturesTrade(
            ts_event_ns=int(ts_event[i]),
            ts_recv_ns=int(ts_recv[i]),
            source=EventSource.DIRECT_FEED,
            symbol=str(symbols[i]),
            price=float(prices[i]),
            size=int(sizes[i]),
            aggressor=agg_map.get(int(aggressors[i]), Aggressor.MID),
            exchange=None,
            conditions=None,
            seq=None
        )
        for i in range(len(df))
    ]

    return trades


def mbp10_from_df(mbp_df: pd.DataFrame) -> List[MBP10]:
    """Convert Bronze MBP-10 DataFrame to MBP10 objects.
    
    Includes action/side/price/size for true OFI computation.
    """
    if mbp_df.empty:
        return []

    df = mbp_df
    if not df["ts_event_ns"].is_monotonic_increasing:
        df = df.sort_values("ts_event_ns")

    # Check if OFI fields are present (new schema)
    has_ofi_fields = "action" in df.columns

    mbp_list: List[MBP10] = []
    for row in df.itertuples(index=False):
        levels = [
            BidAskLevel(
                bid_px=getattr(row, f"bid_px_{i}"),
                bid_sz=getattr(row, f"bid_sz_{i}"),
                ask_px=getattr(row, f"ask_px_{i}"),
                ask_sz=getattr(row, f"ask_sz_{i}")
            )
            for i in range(1, 11)
        ]
        symbol = getattr(row, "symbol", "ES")
        is_snapshot = bool(getattr(row, "is_snapshot", False))
        seq = getattr(row, "seq", None)
        seq_val = None if pd.isna(seq) else int(seq)
        
        # OFI fields (new schema)
        action = getattr(row, "action", None) if has_ofi_fields else None
        side = getattr(row, "side", None) if has_ofi_fields else None
        action_price = getattr(row, "action_price", None) if has_ofi_fields else None
        action_size = getattr(row, "action_size", None) if has_ofi_fields else None

        mbp_list.append(MBP10(
            ts_event_ns=int(row.ts_event_ns),
            ts_recv_ns=int(getattr(row, "ts_recv_ns", row.ts_event_ns)),
            source=EventSource.DIRECT_FEED,
            symbol=str(symbol),
            levels=levels,
            is_snapshot=is_snapshot,
            seq=seq_val,
            action=action,
            side=side,
            action_price=float(action_price) if action_price is not None and not pd.isna(action_price) else None,
            action_size=int(action_size) if action_size is not None and not pd.isna(action_size) else None
        ))

    return mbp_list


class LoadBronzeStage(BaseStage):
    """Load Bronze data using DuckDB for efficient Parquet queries.

    MBP-10 is the single source of truth for ES futures data:
    - Book snapshots (action = A/C/M) for liquidity/OFI features
    - Trades (action = T) for OHLCV/tape features
    
    This eliminates redundant trades schema ingestion.

    Outputs:
        trades: List[FuturesTrade]
        trades_df: pd.DataFrame (extracted from MBP-10 action='T')
        mbp10_snapshots: List[MBP10]
        option_trades_df: pd.DataFrame
    """

    @property
    def name(self) -> str:
        return "load_bronze"

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        reader = DuckDBReader()
        logger.info(f"  Loading Bronze data for {ctx.date}...")

        # Compute session bounds
        # Load full premarket + RTH: 04:00-16:00 ET (12 hours)
        # Premarket (04:00-09:30 ET) needed for PM_HIGH/PM_LOW calculation
        # RTH (09:30-16:00 ET) for all level types
        session_start = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(hours=4, minutes=0)
        session_end = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(hours=16, minutes=0)
        session_start_ns = int(session_start.tz_convert("UTC").value)
        session_end_ns = int(session_end.tz_convert("UTC").value)

        # Add buffer for barrier window lookback (CONFIG.W_b seconds before/after)
        buffer_ns = int(CONFIG.W_b * 1e9)
        ts_start = session_start_ns - buffer_ns
        ts_end = session_end_ns + buffer_ns

        # Load trades from MBP-10 action='T' events
        # This is more efficient than separate trades schema (same underlying data)
        logger.debug(f"    Reading ES trades from MBP-10...")
        trades_df = reader.read_futures_trades_from_mbp10(
            date=ctx.date,
            start_ns=ts_start,
            end_ns=ts_end
        )
        if trades_df.empty:
            raise ValueError(f"No ES trades found in MBP-10 for {ctx.date}")

        trades = futures_trades_from_df(trades_df)
        logger.info(f"    ES trades: {len(trades):,} records (from MBP-10 action='T')")

        # Load MBP-10 downsampled (book snapshots for liquidity features)
        logger.debug(f"    Reading ES MBP-10 (downsampled)...")
        mbp_df = reader.read_futures_mbp10_downsampled(
            date=ctx.date,
            start_ns=ts_start,
            end_ns=ts_end
        )
        if mbp_df.empty:
            raise ValueError(f"No MBP-10 data after downsampling for {ctx.date}")

        mbp10_snapshots = mbp10_from_df(mbp_df)
        logger.info(f"    MBP-10 snapshots: {len(mbp10_snapshots):,} records")

        # Load ES options
        logger.debug(f"    Reading ES option trades...")
        option_trades_df = reader.read_option_trades(underlying='ES', date=ctx.date)
        opt_count = len(option_trades_df) if option_trades_df is not None else 0
        logger.info(f"    ES options: {opt_count:,} records")

        return {
            'trades': trades,
            'trades_df': trades_df,
            'mbp10_snapshots': mbp10_snapshots,
            'option_trades_df': option_trades_df,
            '_reader': reader,  # Keep for warmup stages
        }
