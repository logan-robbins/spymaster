"""
Gold layer Parquet writer for levels.signals per PLAN.md §2.2-§2.4 (Phase 2: NATS + S3).

Writes derived analytics:
- gold/levels/signals/underlying=ES/date=YYYY-MM-DD/hour=HH/part-*.parquet

Gold represents computed/derived data that can be regenerated from Bronze.

Phase 2 changes:
- Subscribes to levels.signals subject via NATS
- Supports S3/MinIO storage (via s3fs)

Agent B deliverable per NEXT.md.
"""

import os
import asyncio
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from src.common.config import CONFIG


class GoldWriter:
    """
    Gold layer Parquet writer for level signals (Phase 2: NATS + S3).

    Subscribes to levels.signals subject from NATS and writes to:
    - Local filesystem (if USE_S3=False)
    - S3/MinIO (if USE_S3=True)

    Writes snap tick level signals to Parquet files partitioned by:
    - underlying (ES)
    - date (YYYY-MM-DD)
    - hour (HH)

    Uses ZSTD compression per PLAN.md §2.2.
    """

    def __init__(
        self,
        bus=None,
        data_root: Optional[str] = None,
        buffer_limit: int = 500,
        flush_interval_seconds: float = 10.0,
        use_s3: Optional[bool] = None
    ):
        """
        Initialize Gold writer.

        Args:
            bus: NATSBus instance (if None, will be set via start())
            data_root: Root directory for data lake (local or S3 prefix)
            buffer_limit: Max records to buffer before flush
            flush_interval_seconds: Max time between flushes
            use_s3: Use S3 storage (defaults to CONFIG.USE_S3)
        """
        self.bus = bus
        self.data_root = data_root or CONFIG.DATA_ROOT
        self.buffer_limit = buffer_limit
        self.flush_interval = flush_interval_seconds
        self.use_s3 = use_s3 if use_s3 is not None else False  # Default to local filesystem

        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Storage backend
        if self.use_s3:
            self.gold_root = f"{CONFIG.S3_BUCKET}/gold"
            self.fs = s3fs.S3FileSystem(
                endpoint_url=CONFIG.S3_ENDPOINT,
                key=CONFIG.S3_ACCESS_KEY,
                secret=CONFIG.S3_SECRET_KEY
            )
            print(f"  Gold writer: S3 storage at {CONFIG.S3_ENDPOINT}/{self.gold_root}")
        else:
            self.gold_root = os.path.join(self.data_root, 'gold')
            os.makedirs(self.gold_root, exist_ok=True)
            self.fs = None
            print(f"  Gold writer: Local storage at {self.gold_root}")

    def _get_partition_path(
        self,
        ts_ms: int,
        underlying: str
    ) -> str:
        """
        Build Hive-style partition path.

        Returns path like:
        gold/levels/signals/underlying=ES/date=2025-12-22/hour=14/
        """
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        date_str = dt.strftime('%Y-%m-%d')
        hour_str = dt.strftime('%H')

        return os.path.join(
            self.gold_root,
            'levels',
            'signals',
            f'underlying={underlying}',
            f'date={date_str}',
            f'hour={hour_str}'
        )

    async def start(self, bus=None):
        """
        Start Gold writer and subscribe to NATS levels.signals subject.
        
        Args:
            bus: NATSBus instance (overrides constructor value)
        """
        if bus:
            self.bus = bus
        
        if not self.bus:
            raise ValueError("NATSBus not provided")
        
        self._running = True
        
        # Subscribe to level signals
        await self.bus.subscribe(
            'levels.signals',
            self._handle_level_signals,
            durable_name='gold_writer_levels_signals'
        )
        
        # Start periodic flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())
        print("  Gold writer: started")
    
    async def _handle_level_signals(self, payload: Dict[str, Any]):
        """
        Handle incoming level signals from NATS and buffer them.
        
        Payload format matches §6.4 WS payload.
        """
        try:
            await self.write_level_signals(payload)
        except Exception as e:
            print(f"  Gold ERROR handling level signals: {e}")
    
    async def _periodic_flush(self):
        """Periodic flush task to ensure timely writes."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                
                if (time.time() - self._last_flush) > self.flush_interval:
                    await self.flush()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  Gold ERROR in periodic flush: {e}")
    
    async def write_level_signals(
        self,
        payload: Dict[str, Any]
    ) -> None:
        """
        Buffer a level signals payload.

        The payload should match §6.4 WS payload format:
        {
            "ts": Unix ms,
            "spy": {"spot": float, "bid": float, "ask": float},
            "levels": [LevelSignal, ...]
        }
        """
        ts_ms = payload.get('ts', int(time.time() * 1000))
        spy_data = payload.get('spy', {})
        levels = payload.get('levels', [])

        # Flatten each level signal into a record
        for level in levels:
            record = {
                # Market context
                'ts_event_ns': ts_ms * 1_000_000,  # Convert ms to ns
                'underlying': 'ES',
                'spot': spy_data.get('spot'),
                'bid': spy_data.get('bid'),
                'ask': spy_data.get('ask'),

                # Level identity
                'level_id': level.get('id'),
                'level_kind': level.get('kind'),
                'level_price': level.get('price'),
                'direction': level.get('direction'),
                'distance': level.get('distance'),

                # Scores
                'break_score_raw': level.get('break_score_raw'),
                'break_score_smooth': level.get('break_score_smooth'),
                'signal': level.get('signal'),
                'confidence': level.get('confidence'),

                # Barrier metrics (flattened)
                'barrier_state': level.get('barrier', {}).get('state'),
                'barrier_delta_liq': level.get('barrier', {}).get('delta_liq'),
                'barrier_replenishment_ratio': level.get('barrier', {}).get('replenishment_ratio'),
                'barrier_added': level.get('barrier', {}).get('added'),
                'barrier_canceled': level.get('barrier', {}).get('canceled'),
                'barrier_filled': level.get('barrier', {}).get('filled'),

                # Tape metrics (flattened)
                'tape_imbalance': level.get('tape', {}).get('imbalance'),
                'tape_buy_vol': level.get('tape', {}).get('buy_vol'),
                'tape_sell_vol': level.get('tape', {}).get('sell_vol'),
                'tape_velocity': level.get('tape', {}).get('velocity'),
                'tape_sweep_detected': level.get('tape', {}).get('sweep', {}).get('detected', False),
                'tape_sweep_direction': level.get('tape', {}).get('sweep', {}).get('direction'),
                'tape_sweep_notional': level.get('tape', {}).get('sweep', {}).get('notional'),

                # Fuel metrics (flattened)
                'fuel_effect': level.get('fuel', {}).get('effect'),
                'fuel_net_dealer_gamma': level.get('fuel', {}).get('net_dealer_gamma'),
                'fuel_call_wall': level.get('fuel', {}).get('call_wall'),
                'fuel_put_wall': level.get('fuel', {}).get('put_wall'),
                'fuel_hvl': level.get('fuel', {}).get('hvl'),

                # Runway (flattened)
                'runway_direction': level.get('runway', {}).get('direction'),
                'runway_next_level_id': level.get('runway', {}).get('next_obstacle', {}).get('id') if level.get('runway', {}).get('next_obstacle') else None,
                'runway_next_level_price': level.get('runway', {}).get('next_obstacle', {}).get('price') if level.get('runway', {}).get('next_obstacle') else None,
                'runway_distance': level.get('runway', {}).get('distance'),
                'runway_quality': level.get('runway', {}).get('quality'),

                # Note
                'note': level.get('note'),

                # Partition key (for writing)
                '_ts_ms': ts_ms,
            }
            async with self._lock:
                self._buffer.append(record)

        # Check flush triggers
        should_flush = (
            len(self._buffer) >= self.buffer_limit or
            (time.time() - self._last_flush) > self.flush_interval
        )

        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        """Flush buffer to Parquet."""
        async with self._lock:
            if not self._buffer:
                return

            data_to_write = self._buffer
            self._buffer = []
            self._last_flush = time.time()

        # Write in executor to avoid blocking
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._write_parquet,
            data_to_write
        )

    def _write_parquet(self, data: List[Dict[str, Any]]) -> None:
        """
        Write data to Parquet files, grouped by partition.
        Supports both local filesystem and S3 storage.
        """
        if not data:
            return

        # Group by partition
        partitions: Dict[str, List[Dict[str, Any]]] = {}
        for record in data:
            ts_ms = record.pop('_ts_ms')
            underlying = record.get('underlying', 'ES')
            partition_path = self._get_partition_path(ts_ms, underlying)

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

                # Sort by event time
                df = df.sort_values('ts_event_ns')

                # Write with ZSTD compression
                table = pa.Table.from_pandas(df, preserve_index=False)
                
                if self.use_s3:
                    # S3 path
                    file_path = f"{partition_path}/{file_name}"
                    
                    # Ensure directory exists
                    self.fs.makedirs(partition_path, exist_ok=True)
                    
                    # Write to S3
                    with self.fs.open(file_path, 'wb') as f:
                        pq.write_table(
                            table,
                            f,
                            compression='zstd',
                            compression_level=3
                        )
                    print(f"  Gold: {len(df)} rows -> s3://{file_path}")
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
                    print(f"  Gold: {len(df)} rows -> {file_path}")

            except Exception as e:
                print(f"  Gold ERROR: {e}")

    def get_gold_path(self) -> str:
        """Return the gold root path."""
        return self.gold_root
    
    async def stop(self) -> None:
        """Stop Gold writer, flush remaining data, and cleanup."""
        self._running = False
        
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush()
        print("  Gold writer: stopped")


class GoldReader:
    """
    Gold layer Parquet reader using DuckDB.

    Reads levels.signals data for analysis and replay validation.
    """

    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize Gold reader.

        Args:
            data_root: Root directory for data lake
        """
        self.data_root = data_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'lake'
        )
        self.gold_root = os.path.join(self.data_root, 'gold')
        self._duckdb = None

    @property
    def duckdb(self):
        if self._duckdb is None:
            import duckdb
            self._duckdb = duckdb
        return self._duckdb

    def read_level_signals(
        self,
        underlying: str = 'ES',
        date: str = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read level signals from Gold.

        Args:
            underlying: Underlying symbol
            date: Specific date (YYYY-MM-DD)
            start_ns: Filter by ts_event_ns >= start_ns
            end_ns: Filter by ts_event_ns <= end_ns

        Returns:
            DataFrame with level signals
        """
        base_path = os.path.join(
            self.gold_root,
            'levels',
            'signals',
            f'underlying={underlying}'
        )

        if not os.path.exists(base_path):
            return pd.DataFrame()

        if date:
            glob_pattern = os.path.join(base_path, f'date={date}', '**', '*.parquet')
        else:
            glob_pattern = os.path.join(base_path, '**', '*.parquet')

        try:
            query = f"SELECT * FROM read_parquet('{glob_pattern}', hive_partitioning=true)"

            conditions = []
            if start_ns is not None:
                conditions.append(f'ts_event_ns >= {start_ns}')
            if end_ns is not None:
                conditions.append(f'ts_event_ns <= {end_ns}')

            if conditions:
                query = f"SELECT * FROM ({query}) WHERE {' AND '.join(conditions)}"

            query += ' ORDER BY ts_event_ns, level_id'

            result = self.duckdb.execute(query).fetchdf()
            return result

        except Exception as e:
            print(f"  Gold READ ERROR: {e}")
            return pd.DataFrame()

    def get_available_dates(self, underlying: str = 'ES') -> List[str]:
        """Get list of available dates."""
        base_path = os.path.join(
            self.gold_root,
            'levels',
            'signals',
            f'underlying={underlying}'
        )

        if not os.path.exists(base_path):
            return []

        dates = []
        for item in os.listdir(base_path):
            if item.startswith('date='):
                dates.append(item.replace('date=', ''))

        return sorted(dates)
