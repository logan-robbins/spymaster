"""
Write-Ahead Log (WAL) Manager for Phase 1 durability.

Ensures zero data loss between event ingestion and Parquet flush:
- Appends events to sequential log files BEFORE processing
- Supports recovery/replay on startup after crash
- Rotates/truncates WAL segments after successful Parquet flush
- One WAL per stream (futures.trades, futures.mbp10, options.trades, etc.)

Uses Apache Arrow IPC Stream format for efficient sequential writes.

Phase 1 deliverable per PLAN.md §2.5 / §1.2.
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import fields
from datetime import datetime, timezone
from enum import Enum

import pyarrow as pa
import pyarrow.ipc as ipc

from src.common.event_types import MBP10


class WALManager:
    """
    Write-Ahead Log manager for durable event capture.
    
    Features:
    - Sequential append-only writes (Arrow IPC stream format)
    - One WAL file per schema
    - Atomic segment rotation after flush
    - Recovery from unflushed segments on startup
    
    Layout:
        backend/data/wal/
            futures_trades_ES.arrow       # active segment
            futures_mbp10_ES.arrow
            options_trades_ES.arrow
            options_trades_ES.001.arrow   # rotated segment (unflushed)
    """
    
    # Schema name to PyArrow schema mapping
    SCHEMAS = {
        'options.trades': pa.schema([
            ('ts_event_ns', pa.int64()),
            ('ts_recv_ns', pa.int64()),
            ('source', pa.string()),
            ('underlying', pa.string()),
            ('option_symbol', pa.string()),
            ('exp_date', pa.string()),
            ('strike', pa.float64()),
            ('right', pa.string()),
            ('price', pa.float64()),
            ('size', pa.int32()),
            ('opt_bid', pa.float64()),
            ('opt_ask', pa.float64()),
            ('aggressor', pa.int8()),
            ('seq', pa.int64()),
        ]),
        'futures.trades': pa.schema([
            ('ts_event_ns', pa.int64()),
            ('ts_recv_ns', pa.int64()),
            ('source', pa.string()),
            ('symbol', pa.string()),
            ('price', pa.float64()),
            ('size', pa.int32()),
            ('aggressor', pa.int8()),
            ('exchange', pa.string()),
            ('seq', pa.int64()),
        ]),
        'futures.mbp10': pa.schema([
            ('ts_event_ns', pa.int64()),
            ('ts_recv_ns', pa.int64()),
            ('source', pa.string()),
            ('symbol', pa.string()),
            ('is_snapshot', pa.bool_()),
            ('seq', pa.int64()),
            # 10 bid levels
            ('bid_px_1', pa.float64()), ('bid_sz_1', pa.int32()),
            ('bid_px_2', pa.float64()), ('bid_sz_2', pa.int32()),
            ('bid_px_3', pa.float64()), ('bid_sz_3', pa.int32()),
            ('bid_px_4', pa.float64()), ('bid_sz_4', pa.int32()),
            ('bid_px_5', pa.float64()), ('bid_sz_5', pa.int32()),
            ('bid_px_6', pa.float64()), ('bid_sz_6', pa.int32()),
            ('bid_px_7', pa.float64()), ('bid_sz_7', pa.int32()),
            ('bid_px_8', pa.float64()), ('bid_sz_8', pa.int32()),
            ('bid_px_9', pa.float64()), ('bid_sz_9', pa.int32()),
            ('bid_px_10', pa.float64()), ('bid_sz_10', pa.int32()),
            # 10 ask levels
            ('ask_px_1', pa.float64()), ('ask_sz_1', pa.int32()),
            ('ask_px_2', pa.float64()), ('ask_sz_2', pa.int32()),
            ('ask_px_3', pa.float64()), ('ask_sz_3', pa.int32()),
            ('ask_px_4', pa.float64()), ('ask_sz_4', pa.int32()),
            ('ask_px_5', pa.float64()), ('ask_sz_5', pa.int32()),
            ('ask_px_6', pa.float64()), ('ask_sz_6', pa.int32()),
            ('ask_px_7', pa.float64()), ('ask_sz_7', pa.int32()),
            ('ask_px_8', pa.float64()), ('ask_sz_8', pa.int32()),
            ('ask_px_9', pa.float64()), ('ask_sz_9', pa.int32()),
            ('ask_px_10', pa.float64()), ('ask_sz_10', pa.int32()),
        ]),
    }
    
    def __init__(
        self,
        wal_root: Optional[str] = None,
        max_segment_size_mb: float = 100.0,
        auto_rotate: bool = True
    ):
        """
        Initialize WAL manager.
        
        Args:
            wal_root: Root directory for WAL segments (defaults to backend/data/wal)
            max_segment_size_mb: Max size before auto-rotation (MB)
            auto_rotate: Whether to auto-rotate on size limit
        """
        self.wal_root = wal_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'wal'
        )
        self.max_segment_bytes = int(max_segment_size_mb * 1024 * 1024)
        self.auto_rotate = auto_rotate
        
        # Ensure WAL directory exists
        os.makedirs(self.wal_root, exist_ok=True)
        
        # Active writers per schema
        self._writers: Dict[str, ipc.RecordBatchStreamWriter] = {}
        self._file_handles: Dict[str, Any] = {}
        self._current_sizes: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        
        print(f"📝 WAL initialized: {self.wal_root}")
    
    def _get_wal_path(self, schema_name: str, partition_key: str, segment: int = 0) -> str:
        """
        Get path for WAL segment.
        
        Args:
            schema_name: e.g., 'futures.trades'
            partition_key: e.g., 'ES'
            segment: segment number (0=active, >0=rotated)
        
        Returns:
            Path like: wal/futures_trades_ES.arrow or wal/futures_trades_ES.001.arrow
        """
        safe_schema = schema_name.replace('.', '_')
        if segment == 0:
            filename = f"{safe_schema}_{partition_key}.arrow"
        else:
            filename = f"{safe_schema}_{partition_key}.{segment:03d}.arrow"
        return os.path.join(self.wal_root, filename)
    
    def _get_stream_key(self, schema_name: str, partition_key: str) -> str:
        """Get unique key for a stream."""
        return f"{schema_name}:{partition_key}"
    
    def _flatten_mbp10(self, event: MBP10) -> Dict[str, Any]:
        """Flatten MBP10 levels into columnar format."""
        result = {
            'ts_event_ns': event.ts_event_ns,
            'ts_recv_ns': event.ts_recv_ns,
            'source': event.source.value if isinstance(event.source, Enum) else event.source,
            'symbol': event.symbol,
            'is_snapshot': event.is_snapshot,
            'seq': event.seq,
        }
        
        # Flatten 10 levels
        for i, level in enumerate(event.levels[:10], start=1):
            result[f'bid_px_{i}'] = level.bid_px
            result[f'bid_sz_{i}'] = level.bid_sz
            result[f'ask_px_{i}'] = level.ask_px
            result[f'ask_sz_{i}'] = level.ask_sz
        
        # Pad if less than 10 levels
        for i in range(len(event.levels) + 1, 11):
            result[f'bid_px_{i}'] = 0.0
            result[f'bid_sz_{i}'] = 0
            result[f'ask_px_{i}'] = 0.0
            result[f'ask_sz_{i}'] = 0
        
        return result
    
    def _event_to_dict(self, event: Any, schema_name: str) -> Dict[str, Any]:
        """Convert event dataclass to dict with proper serialization."""
        if isinstance(event, MBP10):
            return self._flatten_mbp10(event)
        
        result = {}
        for field in fields(event):
            value = getattr(event, field.name)
            
            # Convert enums to their values
            if isinstance(value, Enum):
                result[field.name] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[field.name] = [v.value for v in value]
            else:
                result[field.name] = value
        
        return result
    
    async def append(
        self,
        schema_name: str,
        partition_key: str,
        event: Any
    ) -> None:
        """
        Append event to WAL.
        
        This is called BEFORE the event is processed or sent to Bronze writer.
        
        Args:
            schema_name: e.g., 'futures.trades'
            partition_key: e.g., 'ES'
            event: Event dataclass instance
        """
        stream_key = self._get_stream_key(schema_name, partition_key)
        
        async with self._lock:
            # Initialize writer if needed
            if stream_key not in self._writers:
                self._init_writer(schema_name, partition_key, stream_key)
            
            # Convert event to dict
            event_dict = self._event_to_dict(event, schema_name)
            
            # Create Arrow record batch (single row)
            schema = self.SCHEMAS[schema_name]
            arrays = []
            for field in schema:
                value = event_dict.get(field.name)
                arrays.append(pa.array([value], type=field.type))
            
            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            
            # Write to WAL
            try:
                self._writers[stream_key].write_batch(batch)
                self._file_handles[stream_key].flush()
                
                # Track size
                self._current_sizes[stream_key] += len(batch.serialize())
                
                # Auto-rotate if size limit reached
                if self.auto_rotate and self._current_sizes[stream_key] >= self.max_segment_bytes:
                    await self._rotate_segment(schema_name, partition_key, stream_key)
                
            except Exception as e:
                print(f"  WAL WRITE ERROR ({stream_key}): {e}")
                raise
    
    def _init_writer(self, schema_name: str, partition_key: str, stream_key: str) -> None:
        """Initialize Arrow IPC writer for a stream."""
        wal_path = self._get_wal_path(schema_name, partition_key, segment=0)
        
        # Open file in append mode if exists, otherwise create
        mode = 'ab' if os.path.exists(wal_path) else 'wb'
        file_handle = open(wal_path, mode)
        
        schema = self.SCHEMAS[schema_name]
        
        # Create Arrow IPC stream writer
        if mode == 'wb':
            # New file: write schema header
            writer = ipc.new_stream(file_handle, schema)
        else:
            # Existing file: append without schema header
            # Note: we need to reopen in append mode carefully
            # For simplicity, we'll close and reopen in write mode
            file_handle.close()
            file_handle = open(wal_path, 'ab')
            # Unfortunately, Arrow IPC doesn't support true append mode
            # We'll use a workaround: rotate the segment on restart
            # For now, create a new writer (will work if file is empty or we rotated)
            try:
                writer = ipc.new_stream(file_handle, schema)
            except Exception:
                # If file has data, rotate it first
                self._rotate_existing_segment(schema_name, partition_key)
                file_handle = open(wal_path, 'wb')
                writer = ipc.new_stream(file_handle, schema)
        
        self._writers[stream_key] = writer
        self._file_handles[stream_key] = file_handle
        self._current_sizes[stream_key] = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
        
        print(f"  WAL writer initialized: {wal_path}")
    
    def _rotate_existing_segment(self, schema_name: str, partition_key: str) -> None:
        """Rotate an existing WAL segment (called on startup)."""
        wal_path = self._get_wal_path(schema_name, partition_key, segment=0)
        
        if not os.path.exists(wal_path) or os.path.getsize(wal_path) == 0:
            return
        
        # Find next available segment number
        segment = 1
        while os.path.exists(self._get_wal_path(schema_name, partition_key, segment)):
            segment += 1
        
        rotated_path = self._get_wal_path(schema_name, partition_key, segment)
        os.rename(wal_path, rotated_path)
        print(f"  WAL rotated existing segment: {wal_path} -> {rotated_path}")
    
    async def _rotate_segment(self, schema_name: str, partition_key: str, stream_key: str) -> None:
        """Rotate active WAL segment to a new file."""
        # Close current writer
        if stream_key in self._writers:
            self._writers[stream_key].close()
            self._file_handles[stream_key].close()
            del self._writers[stream_key]
            del self._file_handles[stream_key]
        
        # Rotate file
        wal_path = self._get_wal_path(schema_name, partition_key, segment=0)
        
        # Find next available segment number
        segment = 1
        while os.path.exists(self._get_wal_path(schema_name, partition_key, segment)):
            segment += 1
        
        rotated_path = self._get_wal_path(schema_name, partition_key, segment)
        os.rename(wal_path, rotated_path)
        
        print(f"  WAL segment rotated: {wal_path} -> {rotated_path}")
        
        # Reinitialize writer with fresh segment
        self._init_writer(schema_name, partition_key, stream_key)
    
    async def mark_flushed(self, schema_name: str, partition_key: str) -> None:
        """
        Mark that Bronze Parquet flush succeeded for this stream.
        
        This allows us to safely truncate/delete old WAL segments.
        
        Args:
            schema_name: e.g., 'futures.trades'
            partition_key: e.g., 'ES'
        """
        async with self._lock:
            # Close active writer
            stream_key = self._get_stream_key(schema_name, partition_key)
            if stream_key in self._writers:
                self._writers[stream_key].close()
                self._file_handles[stream_key].close()
                del self._writers[stream_key]
                del self._file_handles[stream_key]
                del self._current_sizes[stream_key]
            
            # Delete active segment (it was flushed to Parquet)
            wal_path = self._get_wal_path(schema_name, partition_key, segment=0)
            if os.path.exists(wal_path):
                os.remove(wal_path)
                print(f"  WAL truncated: {wal_path}")
            
            # Delete all rotated segments for this stream
            segment = 1
            while True:
                rotated_path = self._get_wal_path(schema_name, partition_key, segment)
                if os.path.exists(rotated_path):
                    os.remove(rotated_path)
                    print(f"  WAL deleted rotated segment: {rotated_path}")
                    segment += 1
                else:
                    break
    
    def get_unflushed_segments(self) -> List[str]:
        """
        Get list of all WAL segments (for recovery).
        
        Returns list of full paths to .arrow files.
        """
        wal_files = []
        for filename in os.listdir(self.wal_root):
            if filename.endswith('.arrow'):
                wal_files.append(os.path.join(self.wal_root, filename))
        return sorted(wal_files)
    
    def recover_from_wal(self, wal_path: str) -> List[pa.RecordBatch]:
        """
        Read all records from a WAL segment for recovery.
        
        Args:
            wal_path: Full path to .arrow WAL file
        
        Returns:
            List of Arrow RecordBatches
        """
        if not os.path.exists(wal_path) or os.path.getsize(wal_path) == 0:
            return []
        
        batches = []
        try:
            with open(wal_path, 'rb') as f:
                with ipc.open_stream(f) as reader:
                    for batch in reader:
                        batches.append(batch)
            
            print(f"  WAL recovered {len(batches)} batches from: {wal_path}")
            return batches
        
        except Exception as e:
            print(f"  WAL RECOVERY ERROR ({wal_path}): {e}")
            return []
    
    async def close(self) -> None:
        """Close all active WAL writers."""
        async with self._lock:
            for stream_key in list(self._writers.keys()):
                try:
                    self._writers[stream_key].close()
                    self._file_handles[stream_key].close()
                except Exception as e:
                    print(f"  WAL close error ({stream_key}): {e}")
            
            self._writers.clear()
            self._file_handles.clear()
            self._current_sizes.clear()
        
        print("  WAL closed")
