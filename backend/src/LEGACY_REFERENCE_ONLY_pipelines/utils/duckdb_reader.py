"""DuckDB-based Bronze data reader optimized for pipeline stages.

Provides efficient Parquet querying with column pruning and predicate pushdown.
Wraps BronzeReader with additional pipeline-specific functionality.

Front-Month Contract Filtering:
    Bronze ES data contains multiple contracts (front-month, back-month, spreads)
    under a single 'symbol=ES' partition. This reader implements price-based
    filtering to isolate front-month data:
    
    - Front-month trades at ~X (e.g., 6060)
    - Back-month trades at ~X+160 (calendar spread premium)
    - Spreads/other instruments have non-ES price ranges
    
    The filter uses BBO mid-price clustering to select only front-month data,
    preventing "phantom events" from back-month trades crossing levels.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import pandas as pd

from src.io.bronze import BronzeReader
from src.common.config import CONFIG
from src.common.utils.es_contract_calendar import get_front_month_contract_code

logger = logging.getLogger(__name__)

# ES contract spread premium: back-month typically trades ~120-180 pts higher
ES_BACK_MONTH_PREMIUM_MIN = 100  # Minimum pts difference to be back-month
ES_VALID_PRICE_RANGE = (3000, 10000)  # Valid ES outright price range


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

    def _get_front_month_price_anchor(
        self,
        glob_pattern: str,
        start_ns: int,
        end_ns: int
    ) -> Optional[float]:
        """Determine the front-month price anchor using BBO clustering.
        
        The front-month contract has the highest trading volume. We find its
        price cluster by computing the mode of BBO mid-prices in 10-point buckets.
        
        Returns:
            Anchor price (center of front-month cluster), or None if no valid data
        """
        # Sample BBO mid-prices and find the dominant price cluster
        query = f"""
            WITH mid_prices AS (
                SELECT 
                    ROUND((bid_px_1 + ask_px_1) / 2 / 10) * 10 AS price_bucket,
                    COUNT(*) AS cnt
                FROM read_parquet('{glob_pattern}', hive_partitioning=true, union_by_name=true)
                WHERE ts_event_ns BETWEEN {start_ns} AND {end_ns}
                  AND bid_px_1 >= {ES_VALID_PRICE_RANGE[0]} 
                  AND bid_px_1 <= {ES_VALID_PRICE_RANGE[1]}
                  AND ask_px_1 >= {ES_VALID_PRICE_RANGE[0]} 
                  AND ask_px_1 <= {ES_VALID_PRICE_RANGE[1]}
                GROUP BY price_bucket
            )
            SELECT price_bucket, cnt
            FROM mid_prices
            ORDER BY cnt DESC
            LIMIT 1
        """
        result = self.duckdb.execute(query).fetchdf()
        if result.empty:
            return None
        return float(result['price_bucket'].iloc[0])

    def read_futures_trades_from_mbp10(
        self,
        date: str,
        start_ns: int,
        end_ns: int,
        front_month_only: bool = True
    ) -> pd.DataFrame:
        """Extract trades from MBP-10 action='T' events.
        
        MBP-10 includes trades as action='T' events, eliminating need for
        separate trades schema. This is more efficient than dual ingestion.
        
        IMPORTANT: By default, filters to front-month contract using price-based
        clustering. This prevents back-month trade contamination (which causes
        phantom level touches when back-month prices cross front-month levels).
        
        Args:
            date: Date string (YYYY-MM-DD)
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            front_month_only: If True (default), filter to front-month price cluster
            
        Returns:
            DataFrame with trades (ts_event_ns, price, size, aggressor, symbol)
        """
        base = Path(self.bronze_root) / "futures" / "mbp10" / f"symbol=ES" / f"date={date}"
        if not base.exists():
            return pd.DataFrame()
        
        glob_pattern = str(base / "**" / "*.parquet")
        
        # Determine front-month price filter using BBO clustering
        price_filter = ""
        if front_month_only:
            anchor = self._get_front_month_price_anchor(glob_pattern, start_ns, end_ns)
            if anchor:
                # Front-month trades should be within ±50 pts of anchor
                # Back-month is typically 120-180 pts higher
                price_min = anchor - 50
                price_max = anchor + 50
                price_filter = f"AND action_price >= {price_min} AND action_price <= {price_max}"
                logger.info(f"    Front-month price filter: {price_min:.0f}-{price_max:.0f} (anchor={anchor:.0f})")
        
        # Extract trades: action='T', map side (A=buy lifted ask, B=sell hit bid)
        query = f"""
            SELECT 
                ts_event_ns,
                action_price AS price,
                action_size AS size,
                CASE side 
                    WHEN 'A' THEN 1   -- Buyer lifted ask (aggressive buy)
                    WHEN 'B' THEN -1  -- Seller hit bid (aggressive sell)
                    ELSE 0 
                END AS aggressor,
                symbol
            FROM read_parquet('{glob_pattern}', hive_partitioning=true, union_by_name=true)
            WHERE ts_event_ns BETWEEN {start_ns} AND {end_ns}
              AND action = 'T'
              AND action_price >= {ES_VALID_PRICE_RANGE[0]} 
              AND action_price <= {ES_VALID_PRICE_RANGE[1]}
              {price_filter}
            ORDER BY ts_event_ns
        """
        return self.duckdb.execute(query).fetchdf()

    def read_futures_mbp10_downsampled(
        self,
        date: str,
        start_ns: int,
        end_ns: int,
        bucket_ms: Optional[int] = None,
        front_month_only: bool = True
    ) -> pd.DataFrame:
        """Read MBP-10 snapshots downsampled to bucket intervals.

        Keeps the latest snapshot per bucket to preserve order book state.
        This is more efficient than loading all snapshots.
        
        IMPORTANT: By default, filters to front-month contract using price-based
        clustering. This prevents back-month order book contamination.

        Args:
            date: Date string (YYYY-MM-DD)
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            bucket_ms: Bucket size in milliseconds (defaults to CONFIG.SNAP_INTERVAL_MS)
            front_month_only: If True (default), filter to front-month price cluster

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
        
        # Determine front-month price filter using BBO clustering
        price_filter = ""
        if front_month_only:
            anchor = self._get_front_month_price_anchor(glob_pattern, start_ns, end_ns)
            if anchor:
                price_min = anchor - 50
                price_max = anchor + 50
                price_filter = f"AND bid_px_1 >= {price_min} AND bid_px_1 <= {price_max}"
                price_filter += f" AND ask_px_1 >= {price_min} AND ask_px_1 <= {price_max}"
                logger.debug(f"    MBP-10 front-month filter: {price_min:.0f}-{price_max:.0f}")

        # Filter ES outrights by front-month price cluster
        # This excludes back-month contracts, spreads, and invalid data
        query = f"""
            SELECT * EXCLUDE(bucket, rn)
            FROM (
                SELECT *,
                    CAST((ts_event_ns - {start_ns}) / {bucket_ns} AS BIGINT) AS bucket,
                    row_number() OVER (PARTITION BY bucket ORDER BY ts_event_ns DESC) AS rn
                FROM read_parquet('{glob_pattern}', hive_partitioning=true, union_by_name=true)
                WHERE ts_event_ns BETWEEN {start_ns} AND {end_ns}
                  AND bid_px_1 >= {ES_VALID_PRICE_RANGE[0]} 
                  AND bid_px_1 <= {ES_VALID_PRICE_RANGE[1]}
                  AND ask_px_1 >= {ES_VALID_PRICE_RANGE[0]} 
                  AND ask_px_1 <= {ES_VALID_PRICE_RANGE[1]}
                  {price_filter}
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
