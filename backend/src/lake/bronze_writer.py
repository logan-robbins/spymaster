"""
Bronze layer Parquet writer per PLAN.md §2.2-§2.4 (Phase 2: NATS + S3).

Writes append-only, replayable, schema-versioned raw captures:
- stocks/trades/symbol=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
- stocks/quotes/symbol=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
- options/trades/underlying=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
- options/greeks_snapshots/underlying=SPY/date=YYYY-MM-DD/part-*.parquet
- futures/trades/symbol=ES/date=YYYY-MM-DD/hour=HH/part-*.parquet
- futures/mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/part-*.parquet

Phase 2 changes:
- Removed WAL (NATS JetStream is the WAL)
- Subscribes to market.* subjects via NATS
- Supports S3/MinIO storage (via s3fs)

Agent B deliverable per NEXT.md.
"""

import os
import asyncio
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Type, TypeVar, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from src.common.event_types import (
    StockTrade, StockQuote, OptionTrade, GreeksSnapshot,
    FuturesTrade, MBP10, EventSource, Aggressor
)
from src.common.config import CONFIG


# Type variable for event types
T = TypeVar('T', StockTrade, StockQuote, OptionTrade, GreeksSnapshot, FuturesTrade, MBP10)


def _flatten_dict(obj: Any) -> Any:
    """Flatten nested structures for Parquet serialization."""
    if isinstance(obj, dict):
        return {k: _flatten_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Convert lists to JSON strings to avoid nested array issues
        return str(obj) if obj else None
    elif hasattr(obj, 'value'):  # Enum
        return obj.value
    return obj


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass/dict to flat dict for Parquet."""
    if isinstance(obj, dict):
        return {k: _flatten_dict(v) for k, v in obj.items()}
    
    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        result[field.name] = _flatten_dict(value)
    return result


class BronzeWriter:
    """
    High-throughput Bronze layer Parquet writer (Phase 2: NATS + S3).

    Subscribes to market.* subjects from NATS and writes to:
    - Local filesystem (if USE_S3=False)
    - S3/MinIO (if USE_S3=True)

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
    
    # Subject to schema mapping for NATS subscriptions
    SUBJECT_TO_SCHEMA = {
        'market.stocks.trades': 'stocks.trades',
        'market.stocks.quotes': 'stocks.quotes',
        'market.options.trades': 'options.trades',
        'market.options.greeks': 'options.greeks_snapshots',
        'market.futures.trades': 'futures.trades',
        'market.futures.mbp10': 'futures.mbp10',
    }

    def __init__(
        self,
        bus=None,
        data_root: Optional[str] = None,
        buffer_limit: int = 1000,
        flush_interval_seconds: float = 5.0,
        use_s3: Optional[bool] = None
    ):
        """
        Initialize Bronze writer.

        Args:
            bus: NATSBus instance (if None, will be set via start())
            data_root: Root directory for data lake (local or S3 prefix)
            buffer_limit: Max events to buffer before flush
            flush_interval_seconds: Max time between flushes
            use_s3: Use S3 storage (defaults to CONFIG.USE_S3)
        """
        self.bus = bus
        self.data_root = data_root or CONFIG.DATA_ROOT
        self.buffer_limit = buffer_limit
        self.flush_interval = flush_interval_seconds
        self.use_s3 = use_s3 if use_s3 is not None else False  # Default to local filesystem

        # Separate buffers per schema
        self._buffers: Dict[str, List[Dict[str, Any]]] = {
            schema: [] for schema in self.SCHEMA_PATHS
        }
        self._last_flush: Dict[str, float] = {
            schema: time.time() for schema in self.SCHEMA_PATHS
        }
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Storage backend
        if self.use_s3:
            self.bronze_root = f"{CONFIG.S3_BUCKET}/bronze"
            self.fs = s3fs.S3FileSystem(
                endpoint_url=CONFIG.S3_ENDPOINT,
                key=CONFIG.S3_ACCESS_KEY,
                secret=CONFIG.S3_SECRET_KEY
            )
            print(f"  Bronze writer: S3 storage at {CONFIG.S3_ENDPOINT}/{self.bronze_root}")
        else:
            self.bronze_root = os.path.join(self.data_root, 'bronze')
            os.makedirs(self.bronze_root, exist_ok=True)
            self.fs = None
            print(f"  Bronze writer: Local storage at {self.bronze_root}")

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

    async def start(self, bus=None):
        """
        Start Bronze writer and subscribe to NATS market.* subjects.
        
        Args:
            bus: NATSBus instance (overrides constructor value)
        """
        if bus:
            self.bus = bus
        
        if not self.bus:
            raise ValueError("NATSBus not provided")
        
        self._running = True
        
        # Subscribe to all market data subjects
        for subject in self.SUBJECT_TO_SCHEMA.keys():
            await self.bus.subscribe(
                subject,
                self._handle_message,
                durable_name=f"bronze_writer_{subject.replace('.', '_')}"
            )
        
        # Start periodic flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())
        print("  Bronze writer: started")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """
        Handle incoming NATS message and buffer it.
        
        Message format (from NATS):
        {
            "schema_name": "stocks.trades",
            "ts_event_ns": 1234567890000000000,
            "ts_recv_ns": 1234567890000000000,
            "symbol": "SPY",
            ... (schema-specific fields)
        }
        """
        try:
            # Extract schema and partition key
            # Messages from NATS already have schema_name if published correctly
            # But we can also infer from subject via msg.subject (not available in callback)
            # For now, assume message has schema info or map based on fields
            
            schema_name = None
            partition_key = None
            
            # Infer schema from message structure
            if 'symbol' in message:
                if 'bid_px' in message or 'ask_px' in message:
                    schema_name = 'stocks.quotes'
                    partition_key = message['symbol']
                elif 'size' in message and 'price' in message:
                    if message.get('symbol', '').startswith('ES'):
                        schema_name = 'futures.trades'
                        partition_key = message['symbol']
                    else:
                        schema_name = 'stocks.trades'
                        partition_key = message['symbol']
                elif 'levels' in message:
                    schema_name = 'futures.mbp10'
                    partition_key = message['symbol']
            elif 'underlying' in message:
                if 'option_symbol' in message:
                    if 'delta' in message or 'gamma' in message:
                        schema_name = 'options.greeks_snapshots'
                    else:
                        schema_name = 'options.trades'
                    partition_key = message['underlying']
            
            if schema_name and partition_key:
                await self._buffer_event(schema_name, message, partition_key)
            else:
                print(f"  Bronze: Cannot infer schema from message: {list(message.keys())[:5]}")
                
        except Exception as e:
            print(f"  Bronze ERROR handling message: {e}")
    
    async def _periodic_flush(self):
        """Periodic flush task to ensure timely writes."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                
                # Check each schema for time-based flush
                for schema_name in self.SCHEMA_PATHS:
                    if (time.time() - self._last_flush[schema_name]) > self.flush_interval:
                        await self.flush_schema(schema_name)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  Bronze ERROR in periodic flush: {e}")

    async def _buffer_event(
        self,
        schema_name: str,
        event: Any,  # Can be dict or dataclass
        partition_key: str
    ) -> None:
        """
        Buffer an event and check flush triggers.
        """
        # Convert to dict if needed
        if isinstance(event, dict):
            event_dict = dataclass_to_dict(event)
        else:
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
        Supports both local filesystem and S3 storage.
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
                # Generate unique filename
                timestamp_str = datetime.now(timezone.utc).strftime('%H%M%S_%f')
                file_name = f'part-{timestamp_str}.parquet'
                
                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Sort by event time within file
                df = df.sort_values('ts_event_ns')

                # Write with ZSTD compression
                table = pa.Table.from_pandas(df, preserve_index=False)
                
                if self.use_s3:
                    # S3 path
                    file_path = f"{partition_path}/{file_name}"
                    
                    # Ensure directory exists (S3FS handles this)
                    self.fs.makedirs(partition_path, exist_ok=True)
                    
                    # Write to S3
                    with self.fs.open(file_path, 'wb') as f:
                        pq.write_table(
                            table,
                            f,
                            compression='zstd',
                            compression_level=3
                        )
                    print(f"  Bronze: {len(df)} rows -> s3://{file_path}")
                else:
                    # Local filesystem
                    os.makedirs(partition_path, exist_ok=True)
                    file_path = os.path.join(partition_path, file_name)
                    
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
    
    async def stop(self) -> None:
        """Stop Bronze writer, flush remaining data, and cleanup."""
        self._running = False
        
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush_all()
        print("  Bronze writer: stopped")


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
