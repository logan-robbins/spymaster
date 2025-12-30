"""DuckDB-based Bronze data reader optimized for pipeline stages.

Provides efficient Parquet querying with column pruning and predicate pushdown.
Wraps BronzeReader with additional pipeline-specific functionality.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from src.io.bronze import BronzeReader
from src.common.config import CONFIG


class DuckDBReader:
    """Efficient Bronze data reader for pipeline stages.

    Uses DuckDB for Parquet querying with:
    - Column pruning
    - Predicate pushdown
    - Time-based downsampling for MBP-10 data
    """

    def __init__(self, data_root: Optional[str] = None):
        """Initialize reader.

        Args:
            data_root: Root data directory (defaults to CONFIG.DATA_ROOT)
        """
        self._bronze_reader = BronzeReader(data_root=data_root)

    @property
    def bronze_root(self) -> str:
        return self._bronze_reader.bronze_root

    @property
    def duckdb(self):
        return self._bronze_reader.duckdb

    def read_futures_trades(
        self,
        symbol: str = 'ES',
        date: str = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        front_month_only: bool = True,
        specific_contract: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read futures trades from Bronze.
        
        Args:
            symbol: Symbol prefix (e.g., 'ES')
            date: Date string (YYYY-MM-DD)
            start_ns: Start timestamp (nanoseconds)
            end_ns: End timestamp (nanoseconds)
            front_month_only: If True, filter to front-month contract (default)
            specific_contract: If provided, filter to specific contract (e.g., 'ESZ5')
        
        Returns:
            DataFrame with trades
        """
        return self._bronze_reader.read_futures_trades(
            symbol=symbol, date=date, start_ns=start_ns, end_ns=end_ns,
            front_month_only=front_month_only, specific_contract=specific_contract
        )

    def read_futures_mbp10(
        self,
        symbol: str = 'ES',
        date: str = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        front_month_only: bool = True,
        specific_contract: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read futures MBP-10 from Bronze.
        
        Args:
            symbol: Symbol prefix (e.g., 'ES')
            date: Date string (YYYY-MM-DD)
            start_ns: Start timestamp (nanoseconds)
            end_ns: End timestamp (nanoseconds)
            front_month_only: If True, filter to front-month contract (default)
            specific_contract: If provided, filter to specific contract (e.g., 'ESZ5')
        
        Returns:
            DataFrame with MBP-10 snapshots
        """
        return self._bronze_reader.read_futures_mbp10(
            symbol=symbol, date=date, start_ns=start_ns, end_ns=end_ns,
            front_month_only=front_month_only, specific_contract=specific_contract
        )

    def read_futures_mbp10_downsampled(
        self,
        date: str,
        start_ns: int,
        end_ns: int,
        bucket_ms: Optional[int] = None
    ) -> pd.DataFrame:
        """Read MBP-10 snapshots downsampled to bucket intervals.

        Keeps the latest snapshot per bucket to preserve order book state.
        This is more efficient than loading all snapshots.

        Args:
            date: Date string (YYYY-MM-DD)
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            bucket_ms: Bucket size in milliseconds (defaults to CONFIG.SNAP_INTERVAL_MS)

        Returns:
            DataFrame with one snapshot per bucket, ordered by timestamp
        """
        if bucket_ms is None:
            bucket_ms = CONFIG.SNAP_INTERVAL_MS

        base = Path(self.bronze_root) / "futures" / "mbp10" / f"symbol=ES" / f"date={date}"
        if not base.exists():
            return pd.DataFrame()

        bucket_ns = int(bucket_ms * 1e6)
        glob_pattern = str(base / "**" / "*.parquet")

        query = f"""
            SELECT * EXCLUDE(bucket, rn)
            FROM (
                SELECT *,
                    CAST((ts_event_ns - {start_ns}) / {bucket_ns} AS BIGINT) AS bucket,
                    row_number() OVER (PARTITION BY bucket ORDER BY ts_event_ns DESC) AS rn
                FROM read_parquet('{glob_pattern}', hive_partitioning=true, union_by_name=true)
                WHERE ts_event_ns BETWEEN {start_ns} AND {end_ns}
            )
            WHERE rn = 1
            ORDER BY ts_event_ns
        """
        return self.duckdb.execute(query).fetchdf()

    def read_option_trades(
        self,
        underlying: str = 'ES',
        date: str = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read option trades from Bronze."""
        return self._bronze_reader.read_option_trades(
            underlying=underlying, date=date, start_ns=start_ns, end_ns=end_ns
        )

    def get_available_dates(
        self,
        schema_path: str,
        partition_key: str
    ) -> List[str]:
        """Get list of available dates for a schema/partition."""
        return self._bronze_reader.get_available_dates(schema_path, partition_key)

    def get_warmup_dates(self, date: str, warmup_days: int) -> List[str]:
        """Get prior trading dates for warmup period.

        Args:
            date: Target date (YYYY-MM-DD)
            warmup_days: Number of prior days needed

        Returns:
            List of prior weekday dates, oldest first
        """
        from datetime import datetime

        if warmup_days <= 0:
            return []

        available = self.get_available_dates('futures/trades', 'symbol=ES')
        weekday_dates = [
            d for d in available
            if datetime.strptime(d, '%Y-%m-%d').weekday() < 5
        ]

        if date not in weekday_dates:
            return []

        idx = weekday_dates.index(date)
        start_idx = max(0, idx - warmup_days)
        return weekday_dates[start_idx:idx]
