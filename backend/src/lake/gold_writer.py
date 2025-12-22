"""
Gold layer Parquet writer for levels.signals per PLAN.md §2.2-§2.4.

Writes derived analytics:
- gold/levels/signals/underlying=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet

Gold represents computed/derived data that can be regenerated from Bronze.

Agent I deliverable per §12 of PLAN.md.
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


class GoldWriter:
    """
    Gold layer Parquet writer for level signals.

    Writes snap tick level signals to Parquet files partitioned by:
    - underlying (SPY)
    - date (YYYY-MM-DD)
    - hour (HH)

    Uses ZSTD compression per PLAN.md §2.2.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        buffer_limit: int = 500,
        flush_interval_seconds: float = 10.0
    ):
        """
        Initialize Gold writer.

        Args:
            data_root: Root directory for data lake
            buffer_limit: Max records to buffer before flush
            flush_interval_seconds: Max time between flushes
        """
        self.data_root = data_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'lake'
        )
        self.buffer_limit = buffer_limit
        self.flush_interval = flush_interval_seconds

        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()

        # Ensure gold directory exists
        self.gold_root = os.path.join(self.data_root, 'gold')
        os.makedirs(self.gold_root, exist_ok=True)

    def _get_partition_path(
        self,
        ts_ms: int,
        underlying: str
    ) -> str:
        """
        Build Hive-style partition path.

        Returns path like:
        gold/levels/signals/underlying=SPY/date=2025-12-22/hour=14/
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
                'underlying': 'SPY',
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
        """
        if not data:
            return

        # Group by partition
        partitions: Dict[str, List[Dict[str, Any]]] = {}
        for record in data:
            ts_ms = record.pop('_ts_ms')
            underlying = record.get('underlying', 'SPY')
            partition_path = self._get_partition_path(ts_ms, underlying)

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

                # Sort by event time
                df = df.sort_values('ts_event_ns')

                # Write with ZSTD compression
                table = pa.Table.from_pandas(df, preserve_index=False)
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
        underlying: str = 'SPY',
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

    def get_available_dates(self, underlying: str = 'SPY') -> List[str]:
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
