"""
Bronze layer Parquet writer per PLAN.md §2.2-§2.4.

Writes append-only, replayable, schema-versioned raw captures:
- stocks/trades/symbol=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
- stocks/quotes/symbol=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
- options/trades/underlying=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
- options/greeks_snapshots/underlying=SPY/date=YYYY-MM-DD/part-*.parquet

Agent I deliverable per §12 of PLAN.md.
"""

import os
import asyncio
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Type, TypeVar
from enum import Enum

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .event_types import (
    StockTrade, StockQuote, OptionTrade, GreeksSnapshot,
    FuturesTrade, MBP10, EventSource, Aggressor
)
from .config import CONFIG


# Type variable for event types
T = TypeVar('T', StockTrade, StockQuote, OptionTrade, GreeksSnapshot, FuturesTrade, MBP10)


def _enum_to_str(obj: Any) -> Any:
    """Convert Enum values to strings for serialization."""
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list):
        return [_enum_to_str(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _enum_to_str(v) for k, v in obj.items()}
    return obj


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dict with enum handling."""
    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        result[field.name] = _enum_to_str(value)
    return result


class BronzeWriter:
    """
    High-throughput Bronze layer Parquet writer.

    Micro-batches events to Parquet files partitioned by:
    - date (YYYY-MM-DD)
    - hour (HH)
    - symbol/underlying

    Uses ZSTD compression per PLAN.md §2.2.
    """

    # Schema name to path prefix mapping
    SCHEMA_PATHS = {
        'stocks.trades': 'stocks/trades',
        'stocks.quotes': 'stocks/quotes',
        'options.trades': 'options/trades',
        'options.greeks_snapshots': 'options/greeks_snapshots',
        'futures.trades': 'futures/trades',
        'futures.mbp10': 'futures/mbp10',
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        buffer_limit: int = 1000,
        flush_interval_seconds: float = 5.0
    ):
        """
        Initialize Bronze writer.

        Args:
            data_root: Root directory for data lake (defaults to backend/data/lake)
            buffer_limit: Max events to buffer before flush
            flush_interval_seconds: Max time between flushes
        """
        self.data_root = data_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'lake'
        )
        self.buffer_limit = buffer_limit
        self.flush_interval = flush_interval_seconds

        # Separate buffers per schema
        self._buffers: Dict[str, List[Dict[str, Any]]] = {
            schema: [] for schema in self.SCHEMA_PATHS
        }
        self._last_flush: Dict[str, float] = {
            schema: time.time() for schema in self.SCHEMA_PATHS
        }
        self._lock = asyncio.Lock()

        # Ensure bronze directory exists
        self.bronze_root = os.path.join(self.data_root, 'bronze')
        os.makedirs(self.bronze_root, exist_ok=True)

    def _get_partition_path(
        self,
        schema_name: str,
        ts_event_ns: int,
        partition_key: str  # symbol or underlying
    ) -> str:
        """
        Build Hive-style partition path.

        Returns path like:
        bronze/stocks/trades/symbol=SPY/date=2025-12-22/hour=14/
        """
        dt = datetime.fromtimestamp(ts_event_ns / 1e9, tz=timezone.utc)
        date_str = dt.strftime('%Y-%m-%d')
        hour_str = dt.strftime('%H')

        schema_path = self.SCHEMA_PATHS[schema_name]

        # Determine partition key name
        if schema_name.startswith('stocks') or schema_name.startswith('futures'):
            key_name = 'symbol'
        else:
            key_name = 'underlying'

        # Greeks snapshots don't partition by hour (per PLAN.md §2.3)
        if schema_name == 'options.greeks_snapshots':
            return os.path.join(
                self.bronze_root,
                schema_path,
                f'{key_name}={partition_key}',
                f'date={date_str}'
            )
        else:
            return os.path.join(
                self.bronze_root,
                schema_path,
                f'{key_name}={partition_key}',
                f'date={date_str}',
                f'hour={hour_str}'
            )

    async def write_stock_trade(self, event: StockTrade) -> None:
        """Buffer a stock trade event."""
        await self._buffer_event('stocks.trades', event, event.symbol)

    async def write_stock_quote(self, event: StockQuote) -> None:
        """Buffer a stock quote event."""
        await self._buffer_event('stocks.quotes', event, event.symbol)

    async def write_option_trade(self, event: OptionTrade) -> None:
        """Buffer an option trade event."""
        await self._buffer_event('options.trades', event, event.underlying)

    async def write_greeks_snapshot(self, event: GreeksSnapshot) -> None:
        """Buffer a greeks snapshot event."""
        await self._buffer_event('options.greeks_snapshots', event, event.underlying)

    async def write_futures_trade(self, event: FuturesTrade) -> None:
        """Buffer a futures trade event."""
        await self._buffer_event('futures.trades', event, event.symbol)

    async def write_mbp10(self, event: MBP10) -> None:
        """Buffer an MBP-10 event."""
        await self._buffer_event('futures.mbp10', event, event.symbol)

    async def _buffer_event(
        self,
        schema_name: str,
        event: T,
        partition_key: str
    ) -> None:
        """
        Buffer an event and check flush triggers.
        """
        event_dict = dataclass_to_dict(event)
        event_dict['_partition_key'] = partition_key

        async with self._lock:
            self._buffers[schema_name].append(event_dict)

        # Check triggers
        should_flush = (
            len(self._buffers[schema_name]) >= self.buffer_limit or
            (time.time() - self._last_flush[schema_name]) > self.flush_interval
        )

        if should_flush:
            await self.flush_schema(schema_name)

    async def flush_schema(self, schema_name: str) -> None:
        """Flush a specific schema's buffer to Parquet."""
        async with self._lock:
            if not self._buffers[schema_name]:
                return

            data_to_write = self._buffers[schema_name]
            self._buffers[schema_name] = []
            self._last_flush[schema_name] = time.time()

        # Write in executor to avoid blocking
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._write_parquet,
            schema_name,
            data_to_write
        )

    async def flush_all(self) -> None:
        """Flush all schema buffers."""
        for schema_name in self.SCHEMA_PATHS:
            await self.flush_schema(schema_name)

    def _write_parquet(
        self,
        schema_name: str,
        data: List[Dict[str, Any]]
    ) -> None:
        """
        Write data to Parquet files, grouped by partition.

        Uses ZSTD compression level 3 per PLAN.md §2.2.
        """
        if not data:
            return

        # Group by partition
        partitions: Dict[str, List[Dict[str, Any]]] = {}
        for record in data:
            partition_key = record.pop('_partition_key')
            ts_event_ns = record['ts_event_ns']
            partition_path = self._get_partition_path(
                schema_name, ts_event_ns, partition_key
            )

            if partition_path not in partitions:
                partitions[partition_path] = []
            partitions[partition_path].append(record)

        # Write each partition
        for partition_path, records in partitions.items():
            try:
                os.makedirs(partition_path, exist_ok=True)

                # Generate unique filename
                timestamp_str = datetime.utcnow().strftime('%H%M%S_%f')
                file_path = os.path.join(
                    partition_path,
                    f'part-{timestamp_str}.parquet'
                )

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Sort by event time within file
                df = df.sort_values('ts_event_ns')

                # Write with ZSTD compression
                table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_table(
                    table,
                    file_path,
                    compression='zstd',
                    compression_level=3
                )

                print(f"  Bronze: {len(df)} rows -> {file_path}")

            except Exception as e:
                print(f"  Bronze ERROR ({schema_name}): {e}")

    def get_bronze_path(self) -> str:
        """Return the bronze root path."""
        return self.bronze_root


class BronzeReader:
    """
    Bronze layer Parquet reader using DuckDB for efficient querying.

    Supports:
    - Time-range queries by ts_event_ns
    - Partition pruning by date/hour
    - Multi-schema loading for replay
    """

    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize Bronze reader.

        Args:
            data_root: Root directory for data lake
        """
        self.data_root = data_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'lake'
        )
        self.bronze_root = os.path.join(self.data_root, 'bronze')

        # Lazy import to avoid startup cost
        self._duckdb = None

    @property
    def duckdb(self):
        if self._duckdb is None:
            import duckdb
            self._duckdb = duckdb
        return self._duckdb

    def read_stock_trades(
        self,
        symbol: str = 'SPY',
        date: str = None,  # YYYY-MM-DD
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read stock trades from Bronze."""
        return self._read_schema(
            'stocks/trades',
            f'symbol={symbol}',
            date,
            start_ns,
            end_ns
        )

    def read_stock_quotes(
        self,
        symbol: str = 'SPY',
        date: str = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read stock quotes from Bronze."""
        return self._read_schema(
            'stocks/quotes',
            f'symbol={symbol}',
            date,
            start_ns,
            end_ns
        )

    def read_option_trades(
        self,
        underlying: str = 'SPY',
        date: str = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read option trades from Bronze."""
        return self._read_schema(
            'options/trades',
            f'underlying={underlying}',
            date,
            start_ns,
            end_ns
        )

    def read_greeks_snapshots(
        self,
        underlying: str = 'SPY',
        date: str = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read greeks snapshots from Bronze."""
        return self._read_schema(
            'options/greeks_snapshots',
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
        """
        Read Parquet files for a schema with optional filtering.
        """
        # Build glob pattern
        base_path = os.path.join(self.bronze_root, schema_path, partition_key)

        if date:
            glob_pattern = os.path.join(base_path, f'date={date}', '**', '*.parquet')
        else:
            glob_pattern = os.path.join(base_path, '**', '*.parquet')

        # Check if path exists
        if not os.path.exists(base_path):
            return pd.DataFrame()

        try:
            # Use DuckDB for efficient reading
            query = f"SELECT * FROM read_parquet('{glob_pattern}', hive_partitioning=true)"

            # Add time filters
            conditions = []
            if start_ns is not None:
                conditions.append(f'ts_event_ns >= {start_ns}')
            if end_ns is not None:
                conditions.append(f'ts_event_ns <= {end_ns}')

            if conditions:
                query = f"SELECT * FROM ({query}) WHERE {' AND '.join(conditions)}"

            query += ' ORDER BY ts_event_ns'

            result = self.duckdb.execute(query).fetchdf()
            return result

        except Exception as e:
            print(f"  Bronze READ ERROR ({schema_path}): {e}")
            return pd.DataFrame()

    def get_available_dates(self, schema_path: str, partition_key: str) -> List[str]:
        """
        Get list of available dates for a schema/partition.

        Returns list of date strings in YYYY-MM-DD format.
        """
        base_path = os.path.join(self.bronze_root, schema_path, partition_key)

        if not os.path.exists(base_path):
            return []

        dates = []
        for item in os.listdir(base_path):
            if item.startswith('date='):
                dates.append(item.replace('date=', ''))

        return sorted(dates)

    def get_latest_date(self, schema_path: str, partition_key: str) -> Optional[str]:
        """Get the most recent available date."""
        dates = self.get_available_dates(schema_path, partition_key)
        return dates[-1] if dates else None
