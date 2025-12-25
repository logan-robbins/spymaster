"""Load Bronze data stage - first stage in all pipelines."""
from typing import Any, Dict, List
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.duckdb_reader import DuckDBReader
from src.pipeline.utils.vectorized_ops import futures_trades_from_df, mbp10_from_df
from src.common.config import CONFIG


class LoadBronzeStage(BaseStage):
    """Load Bronze data using DuckDB for efficient Parquet queries.

    Loads:
    - ES futures trades
    - ES MBP-10 order book snapshots (downsampled)
    - SPY option trades

    Outputs:
        trades: List[FuturesTrade]
        trades_df: pd.DataFrame (raw)
        mbp10_snapshots: List[MBP10]
        option_trades_df: pd.DataFrame
    """

    @property
    def name(self) -> str:
        return "load_bronze"

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        reader = DuckDBReader()

        # Load ES trades
        trades_df = reader.read_futures_trades(symbol='ES', date=ctx.date)
        if trades_df.empty:
            raise ValueError(f"No ES trades found for {ctx.date}")

        trades = futures_trades_from_df(trades_df)

        # Compute session bounds for MBP-10 loading
        session_start = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        session_end = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(hours=16)
        session_start_ns = int(session_start.tz_convert("UTC").value)
        session_end_ns = int(session_end.tz_convert("UTC").value)

        # Add buffer for barrier window lookback
        buffer_ns = int(CONFIG.W_b * 1e9)
        ts_start = session_start_ns - buffer_ns
        ts_end = session_end_ns + buffer_ns

        # Load MBP-10 downsampled
        mbp_df = reader.read_futures_mbp10_downsampled(
            date=ctx.date,
            start_ns=ts_start,
            end_ns=ts_end
        )
        if mbp_df.empty:
            raise ValueError(f"No MBP-10 data after downsampling for {ctx.date}")

        mbp10_snapshots = mbp10_from_df(mbp_df)

        # Load SPY options
        option_trades_df = reader.read_option_trades(underlying='SPY', date=ctx.date)

        return {
            'trades': trades,
            'trades_df': trades_df,
            'mbp10_snapshots': mbp10_snapshots,
            'option_trades_df': option_trades_df,
            '_reader': reader,  # Keep for warmup stages
        }
