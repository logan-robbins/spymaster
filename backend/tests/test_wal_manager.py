"""
Tests for WAL (Write-Ahead Log) Manager.

Verifies Phase 1 durability guarantees per PLAN.md §2.5 / §1.2:
- Events are written to WAL before processing
- WAL segments can be recovered after crash
- WAL is truncated after successful Parquet flush
- Automatic segment rotation on size limit
"""

import os
import asyncio
import tempfile
import shutil
import time
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.ipc as ipc

from src.io.wal import WALManager
from src.common.event_types import (
    FuturesTrade, MBP10, OptionTrade, StockTrade, StockQuote,
    EventSource, Aggressor, BidAskLevel
)


@pytest.fixture
def temp_wal_dir():
    """Create a temporary directory for WAL testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def wal_manager(temp_wal_dir):
    """Create WAL manager with temp directory."""
    return WALManager(
        wal_root=temp_wal_dir,
        max_segment_size_mb=0.01,  # 10KB for testing rotation
        auto_rotate=True
    )


@pytest.mark.asyncio
async def test_wal_append_futures_trade(wal_manager, temp_wal_dir):
    """Test appending futures trade to WAL."""
    event = FuturesTrade(
        ts_event_ns=1700000000000000000,
        ts_recv_ns=1700000000100000000,
        source=EventSource.REPLAY,
        symbol='ES',
        price=6870.25,
        size=10,
        aggressor=Aggressor.BUY,
        exchange='CME',
        seq=1
    )
    
    await wal_manager.append('futures.trades', 'ES', event)
    
    # Verify WAL file exists
    wal_path = os.path.join(temp_wal_dir, 'futures_trades_ES.arrow')
    assert os.path.exists(wal_path)
    assert os.path.getsize(wal_path) > 0
    
    # Verify we can read it back
    batches = wal_manager.recover_from_wal(wal_path)
    assert len(batches) == 1
    assert batches[0].num_rows == 1
    
    # Verify data
    table = pa.Table.from_batches(batches)
    df = table.to_pandas()
    assert df['symbol'].iloc[0] == 'ES'
    assert df['price'].iloc[0] == 6870.25
    assert df['size'].iloc[0] == 10


@pytest.mark.asyncio
async def test_wal_append_mbp10(wal_manager, temp_wal_dir):
    """Test appending MBP-10 to WAL."""
    levels = [
        BidAskLevel(bid_px=6870.00, bid_sz=100, ask_px=6870.25, ask_sz=50),
        BidAskLevel(bid_px=6869.75, bid_sz=200, ask_px=6870.50, ask_sz=75),
    ]
    
    event = MBP10(
        ts_event_ns=1700000000000000000,
        ts_recv_ns=1700000000100000000,
        source=EventSource.REPLAY,
        symbol='ES',
        levels=levels,
        is_snapshot=True,
        seq=1
    )
    
    await wal_manager.append('futures.mbp10', 'ES', event)
    
    # Verify WAL file exists
    wal_path = os.path.join(temp_wal_dir, 'futures_mbp10_ES.arrow')
    assert os.path.exists(wal_path)
    
    # Verify we can read it back
    batches = wal_manager.recover_from_wal(wal_path)
    assert len(batches) == 1
    
    # Verify flattened MBP data
    table = pa.Table.from_batches(batches)
    df = table.to_pandas()
    assert df['bid_px_1'].iloc[0] == 6870.00
    assert df['bid_sz_1'].iloc[0] == 100
    assert df['ask_px_1'].iloc[0] == 6870.25
    assert df['ask_sz_1'].iloc[0] == 50


@pytest.mark.asyncio
async def test_wal_append_option_trade(wal_manager, temp_wal_dir):
    """Test appending option trade to WAL."""
    event = OptionTrade(
        ts_event_ns=1700000000000000000,
        ts_recv_ns=1700000000100000000,
        source=EventSource.POLYGON_WS,
        underlying='SPY',
        option_symbol='O:SPY251216C00676000',
        exp_date='2025-12-16',
        strike=676.0,
        right='C',
        price=5.50,
        size=10,
        opt_bid=5.45,
        opt_ask=5.55,
        aggressor=Aggressor.BUY,
        seq=1
    )
    
    await wal_manager.append('options.trades', 'SPY', event)
    
    # Verify WAL file exists
    wal_path = os.path.join(temp_wal_dir, 'options_trades_SPY.arrow')
    assert os.path.exists(wal_path)
    
    # Verify we can read it back
    batches = wal_manager.recover_from_wal(wal_path)
    assert len(batches) == 1
    
    table = pa.Table.from_batches(batches)
    df = table.to_pandas()
    assert df['underlying'].iloc[0] == 'SPY'
    assert df['strike'].iloc[0] == 676.0
    assert df['price'].iloc[0] == 5.50


@pytest.mark.asyncio
async def test_wal_multiple_events(wal_manager, temp_wal_dir):
    """Test appending multiple events to same WAL."""
    for i in range(5):
        event = FuturesTrade(
            ts_event_ns=1700000000000000000 + i * 1000000,
            ts_recv_ns=1700000000100000000 + i * 1000000,
            source=EventSource.REPLAY,
            symbol='ES',
            price=6870.25 + i * 0.25,
            size=10 + i,
            aggressor=Aggressor.BUY,
            seq=i
        )
        await wal_manager.append('futures.trades', 'ES', event)
    
    # Verify all events in WAL
    wal_path = os.path.join(temp_wal_dir, 'futures_trades_ES.arrow')
    batches = wal_manager.recover_from_wal(wal_path)
    
    # Should have 5 batches (one per append)
    assert len(batches) == 5
    
    # Verify prices
    all_prices = []
    for batch in batches:
        table = pa.Table.from_batches([batch])
        df = table.to_pandas()
        all_prices.append(df['price'].iloc[0])
    
    expected_prices = [6870.25, 6870.50, 6870.75, 6871.00, 6871.25]
    assert all_prices == expected_prices


@pytest.mark.asyncio
async def test_wal_mark_flushed(wal_manager, temp_wal_dir):
    """Test WAL truncation after successful Parquet flush."""
    # Write some events
    for i in range(3):
        event = FuturesTrade(
            ts_event_ns=1700000000000000000 + i * 1000000,
            ts_recv_ns=1700000000100000000 + i * 1000000,
            source=EventSource.REPLAY,
            symbol='ES',
            price=6870.25,
            size=10,
            aggressor=Aggressor.BUY,
            seq=i
        )
        await wal_manager.append('futures.trades', 'ES', event)
    
    wal_path = os.path.join(temp_wal_dir, 'futures_trades_ES.arrow')
    assert os.path.exists(wal_path)
    
    # Mark as flushed
    await wal_manager.mark_flushed('futures.trades', 'ES')
    
    # WAL should be deleted
    assert not os.path.exists(wal_path)


@pytest.mark.asyncio
async def test_wal_segment_rotation(wal_manager, temp_wal_dir):
    """Test automatic WAL segment rotation on size limit."""
    # Write many events to trigger rotation (max_segment_size_mb=0.01 = 10KB)
    for i in range(200):
        event = FuturesTrade(
            ts_event_ns=1700000000000000000 + i * 1000000,
            ts_recv_ns=1700000000100000000 + i * 1000000,
            source=EventSource.REPLAY,
            symbol='ES',
            price=6870.25 + i * 0.01,
            size=10 + i,
            aggressor=Aggressor.BUY,
            seq=i
        )
        await wal_manager.append('futures.trades', 'ES', event)
    
    # Should have created rotated segments
    wal_files = os.listdir(temp_wal_dir)
    wal_files = [f for f in wal_files if f.startswith('futures_trades_ES')]
    
    # Should have active segment + rotated segments
    assert len(wal_files) >= 2
    assert 'futures_trades_ES.arrow' in wal_files  # active
    
    # Should have .001, .002, etc.
    rotated = [f for f in wal_files if '.001.arrow' in f or '.002.arrow' in f]
    assert len(rotated) >= 1


@pytest.mark.asyncio
async def test_wal_recovery_multiple_segments(wal_manager, temp_wal_dir):
    """Test recovery from multiple WAL segments."""
    # Write events to create multiple segments
    for i in range(150):
        event = FuturesTrade(
            ts_event_ns=1700000000000000000 + i * 1000000,
            ts_recv_ns=1700000000100000000 + i * 1000000,
            source=EventSource.REPLAY,
            symbol='ES',
            price=6870.25,
            size=10,
            aggressor=Aggressor.BUY,
            seq=i
        )
        await wal_manager.append('futures.trades', 'ES', event)
    
    # Get all segments
    segments = wal_manager.get_unflushed_segments()
    assert len(segments) >= 2
    
    # Recover from all segments
    total_events = 0
    for segment_path in segments:
        batches = wal_manager.recover_from_wal(segment_path)
        for batch in batches:
            total_events += batch.num_rows
    
    # Should have recovered all 150 events
    assert total_events == 150


@pytest.mark.asyncio
async def test_wal_mark_flushed_deletes_rotated_segments(wal_manager, temp_wal_dir):
    """Test that mark_flushed deletes all segments including rotated."""
    # Write enough to create rotated segments
    for i in range(150):
        event = FuturesTrade(
            ts_event_ns=1700000000000000000 + i * 1000000,
            ts_recv_ns=1700000000100000000 + i * 1000000,
            source=EventSource.REPLAY,
            symbol='ES',
            price=6870.25,
            size=10,
            aggressor=Aggressor.BUY,
            seq=i
        )
        await wal_manager.append('futures.trades', 'ES', event)
    
    # Verify multiple segments exist
    wal_files_before = [f for f in os.listdir(temp_wal_dir) if f.startswith('futures_trades_ES')]
    assert len(wal_files_before) >= 2
    
    # Mark as flushed
    await wal_manager.mark_flushed('futures.trades', 'ES')
    
    # All segments should be deleted
    wal_files_after = [f for f in os.listdir(temp_wal_dir) if f.startswith('futures_trades_ES')]
    assert len(wal_files_after) == 0


@pytest.mark.asyncio
async def test_wal_separate_streams(wal_manager, temp_wal_dir):
    """Test that different streams have separate WAL files."""
    # Write to futures.trades
    futures_event = FuturesTrade(
        ts_event_ns=1700000000000000000,
        ts_recv_ns=1700000000100000000,
        source=EventSource.REPLAY,
        symbol='ES',
        price=6870.25,
        size=10,
        aggressor=Aggressor.BUY,
        seq=1
    )
    await wal_manager.append('futures.trades', 'ES', futures_event)
    
    # Write to options.trades
    options_event = OptionTrade(
        ts_event_ns=1700000000000000000,
        ts_recv_ns=1700000000100000000,
        source=EventSource.POLYGON_WS,
        underlying='SPY',
        option_symbol='O:SPY251216C00676000',
        exp_date='2025-12-16',
        strike=676.0,
        right='C',
        price=5.50,
        size=10,
        aggressor=Aggressor.BUY,
        seq=1
    )
    await wal_manager.append('options.trades', 'SPY', options_event)
    
    # Should have separate WAL files
    wal_files = os.listdir(temp_wal_dir)
    assert 'futures_trades_ES.arrow' in wal_files
    assert 'options_trades_SPY.arrow' in wal_files


@pytest.mark.asyncio
async def test_wal_close(wal_manager, temp_wal_dir):
    """Test graceful WAL shutdown."""
    # Write some events
    for i in range(5):
        event = FuturesTrade(
            ts_event_ns=1700000000000000000 + i * 1000000,
            ts_recv_ns=1700000000100000000 + i * 1000000,
            source=EventSource.REPLAY,
            symbol='ES',
            price=6870.25,
            size=10,
            aggressor=Aggressor.BUY,
            seq=i
        )
        await wal_manager.append('futures.trades', 'ES', event)
    
    # Close WAL
    await wal_manager.close()
    
    # Should still be able to read WAL files
    wal_path = os.path.join(temp_wal_dir, 'futures_trades_ES.arrow')
    assert os.path.exists(wal_path)
    
    batches = wal_manager.recover_from_wal(wal_path)
    assert len(batches) == 5


@pytest.mark.asyncio
async def test_wal_empty_recovery(wal_manager, temp_wal_dir):
    """Test recovery from empty/nonexistent WAL."""
    nonexistent = os.path.join(temp_wal_dir, 'nonexistent.arrow')
    batches = wal_manager.recover_from_wal(nonexistent)
    assert batches == []


@pytest.mark.asyncio
async def test_wal_concurrent_writes(wal_manager, temp_wal_dir):
    """Test concurrent writes to same WAL stream."""
    async def write_events(start_seq, count):
        for i in range(count):
            event = FuturesTrade(
                ts_event_ns=1700000000000000000 + (start_seq + i) * 1000000,
                ts_recv_ns=1700000000100000000 + (start_seq + i) * 1000000,
                source=EventSource.REPLAY,
                symbol='ES',
                price=6870.25,
                size=10,
                aggressor=Aggressor.BUY,
                seq=start_seq + i
            )
            await wal_manager.append('futures.trades', 'ES', event)
    
    # Launch concurrent writes
    await asyncio.gather(
        write_events(0, 10),
        write_events(10, 10),
        write_events(20, 10)
    )
    
    # Verify all events written (may be across multiple segments due to rotation)
    segments = wal_manager.get_unflushed_segments()
    assert len(segments) >= 1
    
    total_rows = 0
    for segment_path in segments:
        batches = wal_manager.recover_from_wal(segment_path)
        for batch in batches:
            total_rows += batch.num_rows
    
    assert total_rows == 30

