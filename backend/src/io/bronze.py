"""
Bronze layer Parquet writer per PLAN.md §2.2-§2.4 (Phase 2: NATS + S3).

Writes append-only, replayable, schema-versioned raw captures:
- options/trades/underlying=ES/date=YYYY-MM-DD/hour=HH/part-*.parquet
- futures/trades/symbol=ES/date=YYYY-MM-DD/hour=HH/part-*.parquet
- futures/mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/part-*.parquet
"""

import os
import asyncio
import time
from dataclasses import fields
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from src.common.event_types import OptionTrade, FuturesTrade, MBP10
from src.common.config import CONFIG


# Type variable for event types
T = TypeVar('T', OptionTrade, FuturesTrade, MBP10)


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


def _normalize_enum(value: Any) -> Any:
    """Normalize Enum values for serialization."""
    if hasattr(value, "value"):
        return value.value
    return value


def _flatten_mbp10_event(event: Any) -> Dict[str, Any]:
    """Flatten MBP10 event levels into bid/ask fields.
    
    Includes action/side/price/size for true OFI computation per Cont et al. (2014).
    """
    if isinstance(event, dict):
        base = {
            "ts_event_ns": event["ts_event_ns"],
            "ts_recv_ns": event.get("ts_recv_ns", event["ts_event_ns"]),
            "source": _normalize_enum(event.get("source")),
            "symbol": event.get("symbol"),
            "is_snapshot": event.get("is_snapshot", False),
            "seq": event.get("seq"),
            # OFI fields (action/side/price/size for true event-based OFI)
            "action": _normalize_enum(event.get("action")),
            "side": _normalize_enum(event.get("side")),
            "action_price": float(event.get("price", 0.0)),
            "action_size": int(event.get("size", 0)),
        }
        levels = event.get("levels", [])
    else:
        base = {
            "ts_event_ns": event.ts_event_ns,
            "ts_recv_ns": getattr(event, "ts_recv_ns", event.ts_event_ns),
            "source": _normalize_enum(event.source),
            "symbol": event.symbol,
            "is_snapshot": getattr(event, "is_snapshot", False),
            "seq": getattr(event, "seq", None),
            # OFI fields (action/side/price/size for true event-based OFI)
            "action": _normalize_enum(getattr(event, "action", None)),
            "side": _normalize_enum(getattr(event, "side", None)),
            "action_price": float(getattr(event, "action_price", 0.0) or 0.0),
            "action_size": int(getattr(event, "action_size", 0) or 0),
        }
        levels = getattr(event, "levels", [])

    # Pad/truncate to 10 levels
    for idx in range(10):
        level = levels[idx] if idx < len(levels) else None
        if level is None:
            bid_px = 0.0
            bid_sz = 0
            ask_px = 0.0
            ask_sz = 0
        elif isinstance(level, dict):
            bid_px = float(level.get("bid_px", 0.0))
            bid_sz = int(level.get("bid_sz", 0))
            ask_px = float(level.get("ask_px", 0.0))
            ask_sz = int(level.get("ask_sz", 0))
        else:
            bid_px = float(getattr(level, "bid_px", 0.0))
            bid_sz = int(getattr(level, "bid_sz", 0))
            ask_px = float(getattr(level, "ask_px", 0.0))
            ask_sz = int(getattr(level, "ask_sz", 0))

        level_idx = idx + 1
        base[f"bid_px_{level_idx}"] = bid_px
        base[f"bid_sz_{level_idx}"] = bid_sz
        base[f"ask_px_{level_idx}"] = ask_px
        base[f"ask_sz_{level_idx}"] = ask_sz

    return base


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
        'options.trades': 'options/trades',
        'futures.trades': 'futures/trades',
        'futures.mbp10': 'futures/mbp10',
    }
    
    # Subject to schema mapping for NATS subscriptions
    SUBJECT_TO_SCHEMA = {
        'market.options.trades': 'options.trades',
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
        bronze/futures/trades/symbol=ES/date=2025-12-22/hour=14/
        """
        dt = datetime.fromtimestamp(ts_event_ns / 1e9, tz=timezone.utc)
        date_str = dt.strftime('%Y-%m-%d')
        hour_str = dt.strftime('%H')

        schema_path = self.SCHEMA_PATHS[schema_name]

        # Determine partition key name
        key_name = 'symbol' if schema_name.startswith('futures') else 'underlying'

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
            "schema_name": "futures.trades",
            "ts_event_ns": 1234567890000000000,
            "ts_recv_ns": 1234567890000000000,
            "symbol": "ES",
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
            if 'levels' in message:
                schema_name = 'futures.mbp10'
                partition_key = message['symbol']
            elif 'symbol' in message and 'size' in message and 'price' in message:
                schema_name = 'futures.trades'
                partition_key = message['symbol']
            elif 'underlying' in message and 'option_symbol' in message:
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
        if schema_name == 'futures.mbp10':
            event_dict = _flatten_mbp10_event(event)
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

                # Get canonical schema for this schema_name
                arrow_schema = self._get_arrow_schema(schema_name)

                # Write with ZSTD compression (enforce schema to prevent type inconsistencies)
                table = pa.Table.from_pandas(df, schema=arrow_schema, preserve_index=False)
                
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

    def _get_arrow_schema(self, schema_name: str) -> pa.Schema:
        """Get canonical Arrow schema for a schema name."""
        from src.common.schemas import OptionTradeV1, FuturesTradeV1, MBP10V1
        
        schema_map = {
            'options.trades': OptionTradeV1._arrow_schema,
            'futures.trades': FuturesTradeV1._arrow_schema,
            'futures.mbp10': MBP10V1._arrow_schema,
        }
        
        if schema_name not in schema_map:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        return schema_map[schema_name]
    
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
        # Use CONFIG.DATA_ROOT which points to backend/data
        from src.common.config import CONFIG
        self.data_root = data_root or CONFIG.DATA_ROOT
        self.bronze_root = os.path.join(self.data_root, 'bronze')

        # Lazy import to avoid startup cost
        self._duckdb = None

    @property
    def duckdb(self):
        if self._duckdb is None:
            import duckdb
            self._duckdb = duckdb
        return self._duckdb

    def read_option_trades(
        self,
        underlying: str = 'ES',
        date: str = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """Read option trades from Bronze."""
        trades = self._read_schema(
            'options/trades',
            f'underlying={underlying}',
            date,
            start_ns,
            end_ns
        )
        
        if trades.empty:
            return trades
            
        # Try to read Open Interest from statistics
        try:
            stats = self._read_schema(
                'options/statistics',
                f'underlying={underlying}',
                date,
                None, # Read all stats for the day
                None
            )
            
            if not stats.empty and 'open_interest' in stats.columns:
                # Deduplicate stats: Keep latest OI per option symbol
                # Sort by timestamp to ensure we get the latest update
                latest_stats = (
                    stats.sort_values('ts_event_ns')
                    .groupby('option_symbol', as_index=False)
                    .agg({'open_interest': 'last'})
                )
                
                # Merge efficiently
                # Left join to preserve all trades
                trades = trades.merge(
                    latest_stats[['option_symbol', 'open_interest']], 
                    on='option_symbol', 
                    how='left'
                )
                
                # Fill missing OI with 0 (or could try to infer, but 0 is safer)
                trades['open_interest'] = trades['open_interest'].fillna(0)
            else:
                trades['open_interest'] = 0.0
                
        except Exception as e:
            print(f"WARNING: Failed to merge Open Interest: {e}")
            trades['open_interest'] = 0.0
            
        return trades

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
            front_month_only: If True, automatically select front-month contract (default)
            specific_contract: If provided, filter to exact contract (e.g., 'ESZ5')
        
        Returns:
            DataFrame with trades, filtered to single contract if requested
        """
        df = self._read_schema(
            'futures/trades',
            f'symbol={symbol}',
            date,
            start_ns,
            end_ns
        )
        
        # Apply contract filtering
        if not df.empty and (front_month_only or specific_contract):
            if specific_contract:
                # Explicit contract specified
                df = df[df['symbol'] == specific_contract].copy()
            elif front_month_only and date:
                # Auto-select front month
                from src.common.utils.contract_selector import ContractSelector
                selector = ContractSelector(self.bronze_root)
                try:
                    selection = selector.select_front_month(date)
                    df = df[df['symbol'] == selection.front_month_symbol].copy()
                except Exception as e:
                    print(f"WARNING: Front-month selection failed for {date}: {e}")
                    # Fall back to returning all contracts
        
        return df

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
            front_month_only: If True, automatically select front-month contract (default)
            specific_contract: If provided, filter to exact contract (e.g., 'ESZ5')
        
        Returns:
            DataFrame with MBP-10 snapshots, filtered to single contract if requested
        """
        df = self._read_schema(
            'futures/mbp10',
            f'symbol={symbol}',
            date,
            start_ns,
            end_ns
        )
        
        # Apply contract filtering
        if not df.empty and (front_month_only or specific_contract):
            if specific_contract:
                # Explicit contract specified
                df = df[df['symbol'] == specific_contract].copy()
            elif front_month_only and date:
                # Auto-select front month (use same selection as trades)
                from src.common.utils.contract_selector import ContractSelector
                selector = ContractSelector(self.bronze_root)
                try:
                    selection = selector.select_front_month(date)
                    df = df[df['symbol'] == selection.front_month_symbol].copy()
                except Exception as e:
                    print(f"WARNING: Front-month selection failed for {date}: {e}")
                    # Fall back to returning all contracts
        
        return df

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
            # union_by_name=true handles schema variations (e.g., dictionary-encoded vs plain strings)
            query = f"SELECT * FROM read_parquet('{glob_pattern}', hive_partitioning=true, union_by_name=true)"

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
