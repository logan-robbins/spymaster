"""
Silver layer compaction job per PLAN.md §2.2 Phase 1.

Transforms Bronze (raw, append-only) into Silver (clean, deduped, sorted):
- Deterministic dedup on event_id = hash(source, ts_event_ns, symbol, price, size, ...)
- Sort by ts_event_ns within each partition
- ZSTD compression

Future enhancements (Phase 1+):
- As-of joins: attach best-known quote to trades
- As-of joins: attach greeks snapshot to option trades
"""

import os
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import duckdb
except ImportError:
    duckdb = None


class SilverCompactor:
    """
    Compacts Bronze Parquet partitions into Silver.

    Usage:
        compactor = SilverCompactor()
        compactor.compact_date('2025-12-16', schema='futures.trades')
        compactor.compact_all_schemas('2025-12-16')
    """

    # Schema configurations: (bronze_path, silver_path, key_columns_for_dedup)
    SCHEMA_CONFIG = {
        'stocks.trades': {
            'bronze_path': 'stocks/trades',
            'silver_path': 'stocks/trades',
            'partition_key': 'symbol',
            'dedup_cols': ['source', 'ts_event_ns', 'symbol', 'price', 'size', 'exchange', 'seq'],
        },
        'stocks.quotes': {
            'bronze_path': 'stocks/quotes',
            'silver_path': 'stocks/quotes',
            'partition_key': 'symbol',
            'dedup_cols': ['source', 'ts_event_ns', 'symbol', 'bid_px', 'ask_px', 'bid_sz', 'ask_sz', 'seq'],
        },
        'futures.trades': {
            'bronze_path': 'futures/trades',
            'silver_path': 'futures/trades',
            'partition_key': 'symbol',
            'dedup_cols': ['source', 'ts_event_ns', 'symbol', 'price', 'size', 'exchange', 'seq'],
        },
        'futures.mbp10': {
            'bronze_path': 'futures/mbp10',
            'silver_path': 'futures/mbp10',
            'partition_key': 'symbol',
            # MBP10 dedup: use ts_event_ns + symbol (levels are complex nested)
            'dedup_cols': ['source', 'ts_event_ns', 'symbol', 'seq'],
        },
        'options.trades': {
            'bronze_path': 'options/trades',
            'silver_path': 'options/trades_enriched',  # Silver has enriched schema
            'partition_key': 'underlying',
            'dedup_cols': ['source', 'ts_event_ns', 'option_symbol', 'price', 'size', 'seq'],
        },
        'options.greeks_snapshots': {
            'bronze_path': 'options/greeks_snapshots',
            'silver_path': 'options/greeks_snapshots',
            'partition_key': 'underlying',
            'dedup_cols': ['source', 'ts_event_ns', 'option_symbol', 'snapshot_id'],
        },
    }

    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize Silver compactor.

        Args:
            data_root: Root directory for data lake (defaults to backend/data/lake)
        """
        if duckdb is None:
            raise ImportError("DuckDB is required for Silver compaction. Run: uv add duckdb")

        self.data_root = data_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'lake'
        )
        self.bronze_root = os.path.join(self.data_root, 'bronze')
        self.silver_root = os.path.join(self.data_root, 'silver')

        # Ensure silver directory exists
        os.makedirs(self.silver_root, exist_ok=True)

    def compact_date(
        self,
        date: str,
        schema: str,
        partition_value: str = 'ES',
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Compact a single date partition from Bronze to Silver.

        Args:
            date: Date string (YYYY-MM-DD)
            schema: Schema name (e.g., 'futures.trades', 'stocks.quotes')
            partition_value: Partition key value (e.g., 'ES', 'SPY')
            force: Overwrite existing Silver data if True

        Returns:
            Dict with stats: rows_read, rows_written, duplicates_removed
        """
        if schema not in self.SCHEMA_CONFIG:
            raise ValueError(f"Unknown schema: {schema}. Valid: {list(self.SCHEMA_CONFIG.keys())}")

        config = self.SCHEMA_CONFIG[schema]
        partition_key = config['partition_key']

        # Build Bronze path
        bronze_base = os.path.join(
            self.bronze_root,
            config['bronze_path'],
            f'{partition_key}={partition_value}',
            f'date={date}'
        )

        if not os.path.exists(bronze_base):
            return {
                'status': 'skipped',
                'reason': f'Bronze path not found: {bronze_base}',
                'rows_read': 0,
                'rows_written': 0,
                'duplicates_removed': 0
            }

        # Build Silver output path
        silver_base = os.path.join(
            self.silver_root,
            config['silver_path'],
            f'{partition_key}={partition_value}',
            f'date={date}'
        )

        if os.path.exists(silver_base) and not force:
            return {
                'status': 'skipped',
                'reason': f'Silver already exists: {silver_base}. Use force=True to overwrite.',
                'rows_read': 0,
                'rows_written': 0,
                'duplicates_removed': 0
            }

        # Read all Bronze files for this date
        glob_pattern = os.path.join(bronze_base, '**', '*.parquet')

        try:
            # Use DuckDB to read and process
            df = self._read_and_dedup(glob_pattern, config['dedup_cols'])

            if df.empty:
                return {
                    'status': 'empty',
                    'reason': 'No data found in Bronze',
                    'rows_read': 0,
                    'rows_written': 0,
                    'duplicates_removed': 0
                }

            rows_read = len(df) + df.get('_duplicates_count', pd.Series([0])).sum()
            rows_written = len(df)
            duplicates_removed = int(rows_read - rows_written)

            # Write to Silver, partitioned by hour
            self._write_silver_partitioned(df, silver_base)

            return {
                'status': 'success',
                'bronze_path': bronze_base,
                'silver_path': silver_base,
                'rows_read': int(rows_read),
                'rows_written': rows_written,
                'duplicates_removed': duplicates_removed
            }

        except Exception as e:
            return {
                'status': 'error',
                'reason': str(e),
                'rows_read': 0,
                'rows_written': 0,
                'duplicates_removed': 0
            }

    def _read_and_dedup(
        self,
        glob_pattern: str,
        dedup_cols: List[str]
    ) -> pd.DataFrame:
        """
        Read Bronze Parquet files and deduplicate.

        Uses DuckDB for efficient processing:
        1. Compute event_id hash from dedup columns
        2. Use ROW_NUMBER() to keep first occurrence
        3. Sort by ts_event_ns
        """
        # Filter to columns that exist (some may be optional)
        # Build the hash expression for event_id
        hash_expr = self._build_hash_expr(dedup_cols)

        query = f"""
        WITH bronze_data AS (
            SELECT *, {hash_expr} as event_id
            FROM read_parquet('{glob_pattern}', hive_partitioning=true)
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ts_recv_ns) as rn
            FROM bronze_data
        )
        SELECT * EXCLUDE (rn, event_id)
        FROM ranked
        WHERE rn = 1
        ORDER BY ts_event_ns
        """

        try:
            result = duckdb.execute(query).fetchdf()
            return result
        except duckdb.IOException:
            # No files found
            return pd.DataFrame()
        except duckdb.CatalogException as e:
            # Column doesn't exist - fall back to simpler dedup
            return self._read_and_dedup_simple(glob_pattern)

    def _build_hash_expr(self, dedup_cols: List[str]) -> str:
        """
        Build DuckDB hash expression for event_id.

        Handles NULL values gracefully with COALESCE.
        """
        # Concatenate columns with separator, coalescing NULLs
        parts = []
        for col in dedup_cols:
            parts.append(f"COALESCE(CAST({col} AS VARCHAR), '')")

        concat_expr = " || '|' || ".join(parts)
        return f"md5({concat_expr})"

    def _read_and_dedup_simple(self, glob_pattern: str) -> pd.DataFrame:
        """
        Fallback dedup using just ts_event_ns + ts_recv_ns.
        """
        query = f"""
        WITH bronze_data AS (
            SELECT *
            FROM read_parquet('{glob_pattern}', hive_partitioning=true)
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY ts_event_ns
                    ORDER BY ts_recv_ns
                ) as rn
            FROM bronze_data
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
        ORDER BY ts_event_ns
        """

        try:
            return duckdb.execute(query).fetchdf()
        except duckdb.IOException:
            return pd.DataFrame()

    def _write_silver_partitioned(
        self,
        df: pd.DataFrame,
        silver_base: str
    ) -> None:
        """
        Write Silver data partitioned by hour.

        Uses ZSTD compression per PLAN.md §2.2.
        """
        if df.empty:
            return

        # Extract hour from ts_event_ns
        df['_hour'] = pd.to_datetime(df['ts_event_ns'], unit='ns', utc=True).dt.strftime('%H')

        # Write each hour partition
        for hour, hour_df in df.groupby('_hour'):
            hour_path = os.path.join(silver_base, f'hour={hour}')
            os.makedirs(hour_path, exist_ok=True)

            # Remove temporary column
            hour_df = hour_df.drop(columns=['_hour'])

            # Generate filename
            file_path = os.path.join(hour_path, 'part-0000.parquet')

            # Write with ZSTD compression
            table = pa.Table.from_pandas(hour_df, preserve_index=False)
            pq.write_table(
                table,
                file_path,
                compression='zstd',
                compression_level=3
            )

            print(f"  Silver: {len(hour_df)} rows -> {file_path}")

    def compact_all_schemas(
        self,
        date: str,
        force: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compact all schemas for a given date.

        Args:
            date: Date string (YYYY-MM-DD)
            force: Overwrite existing Silver data

        Returns:
            Dict mapping schema -> compaction result
        """
        results = {}

        # Define default partition values per schema
        partition_defaults = {
            'stocks.trades': 'SPY',
            'stocks.quotes': 'SPY',
            'futures.trades': 'ES',
            'futures.mbp10': 'ES',
            'options.trades': 'SPY',
            'options.greeks_snapshots': 'SPY',
        }

        for schema, partition_value in partition_defaults.items():
            print(f"Compacting {schema} for {date}...")
            result = self.compact_date(date, schema, partition_value, force)
            results[schema] = result
            print(f"  -> {result['status']}: {result.get('rows_written', 0)} rows")

        return results

    def get_available_bronze_dates(
        self,
        schema: str,
        partition_value: str
    ) -> List[str]:
        """
        Get list of dates available in Bronze for a schema.

        Returns:
            List of date strings (YYYY-MM-DD)
        """
        if schema not in self.SCHEMA_CONFIG:
            return []

        config = self.SCHEMA_CONFIG[schema]
        partition_key = config['partition_key']

        base_path = os.path.join(
            self.bronze_root,
            config['bronze_path'],
            f'{partition_key}={partition_value}'
        )

        if not os.path.exists(base_path):
            return []

        dates = []
        for item in os.listdir(base_path):
            if item.startswith('date='):
                dates.append(item.replace('date=', ''))

        return sorted(dates)

    def compact_all_dates(
        self,
        schema: str,
        partition_value: str,
        force: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compact all available dates for a schema.

        Args:
            schema: Schema name
            partition_value: Partition key value
            force: Overwrite existing Silver data

        Returns:
            Dict mapping date -> compaction result
        """
        dates = self.get_available_bronze_dates(schema, partition_value)
        results = {}

        for date in dates:
            print(f"Compacting {schema}/{partition_value} for {date}...")
            result = self.compact_date(date, schema, partition_value, force)
            results[date] = result

        return results


class SilverReader:
    """
    Silver layer Parquet reader.

    Similar to BronzeReader but reads from Silver directory.
    """

    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize Silver reader.

        Args:
            data_root: Root directory for data lake
        """
        if duckdb is None:
            raise ImportError("DuckDB is required. Run: uv add duckdb")

        self.data_root = data_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'lake'
        )
        self.silver_root = os.path.join(self.data_root, 'silver')

    def read_futures_trades(
        self,
        symbol: str = 'ES',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read futures trades from Silver."""
        return self._read_schema(
            'futures/trades',
            f'symbol={symbol}',
            date,
            start_ns,
            end_ns
        )

    def read_futures_mbp10(
        self,
        symbol: str = 'ES',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read futures MBP-10 from Silver."""
        return self._read_schema(
            'futures/mbp10',
            f'symbol={symbol}',
            date,
            start_ns,
            end_ns
        )

    def read_stock_trades(
        self,
        symbol: str = 'SPY',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read stock trades from Silver."""
        return self._read_schema(
            'stocks/trades',
            f'symbol={symbol}',
            date,
            start_ns,
            end_ns
        )

    def read_option_trades(
        self,
        underlying: str = 'SPY',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read enriched option trades from Silver."""
        return self._read_schema(
            'options/trades_enriched',
            f'underlying={underlying}',
            date,
            start_ns,
            end_ns
        )

    def _read_schema(
        self,
        schema_path: str,
        partition_key: str,
        date: Optional[str],
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> pd.DataFrame:
        """Read Parquet files from Silver with optional filtering."""
        base_path = os.path.join(self.silver_root, schema_path, partition_key)

        if date:
            glob_pattern = os.path.join(base_path, f'date={date}', '**', '*.parquet')
        else:
            glob_pattern = os.path.join(base_path, '**', '*.parquet')

        if not os.path.exists(base_path):
            return pd.DataFrame()

        try:
            query = f"SELECT * FROM read_parquet('{glob_pattern}', hive_partitioning=true)"

            conditions = []
            if start_ns is not None:
                conditions.append(f'ts_event_ns >= {start_ns}')
            if end_ns is not None:
                conditions.append(f'ts_event_ns <= {end_ns}')

            if conditions:
                query = f"SELECT * FROM ({query}) WHERE {' AND '.join(conditions)}"

            query += ' ORDER BY ts_event_ns'

            return duckdb.execute(query).fetchdf()

        except Exception as e:
            print(f"Silver READ ERROR ({schema_path}): {e}")
            return pd.DataFrame()

    def get_available_dates(self, schema_path: str, partition_key: str) -> List[str]:
        """Get list of available dates in Silver."""
        base_path = os.path.join(self.silver_root, schema_path, partition_key)

        if not os.path.exists(base_path):
            return []

        dates = []
        for item in os.listdir(base_path):
            if item.startswith('date='):
                dates.append(item.replace('date=', ''))

        return sorted(dates)
