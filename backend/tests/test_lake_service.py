"""
Tests for Lake Service (Phase 2: NATS + S3).

Tests BronzeWriter and GoldWriter with local filesystem (not S3/NATS).
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest
import pandas as pd

from src.io.bronze import BronzeWriter, dataclass_to_dict
from src.io.gold import GoldWriter
from src.common.event_types import StockTrade, StockQuote, EventSource, Aggressor


@pytest.mark.asyncio
async def test_bronze_writer_local_filesystem():
    """Test BronzeWriter writes to local filesystem correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = BronzeWriter(
            data_root=tmpdir,
            buffer_limit=2,  # Small buffer for testing
            flush_interval_seconds=1.0,
            use_s3=False
        )
        
        # Create test events
        ts_base = int(time.time() * 1e9)
        
        trade1 = {
            'ts_event_ns': ts_base,
            'ts_recv_ns': ts_base + 1000,
            'symbol': 'SPY',
            'price': 600.50,
            'size': 100,
            'source': 'test'
        }
        
        trade2 = {
            'ts_event_ns': ts_base + 1_000_000_000,  # +1 second
            'ts_recv_ns': ts_base + 1_001_000_000,
            'symbol': 'SPY',
            'price': 600.55,
            'size': 200,
            'source': 'test'
        }
        
        # Buffer events
        await writer._buffer_event('stocks.trades', trade1, 'SPY')
        await writer._buffer_event('stocks.trades', trade2, 'SPY')
        
        # Should auto-flush due to buffer_limit=2
        await asyncio.sleep(0.1)
        
        # Verify files were written
        bronze_root = Path(tmpdir) / 'bronze' / 'stocks' / 'trades' / 'symbol=SPY'
        assert bronze_root.exists()
        
        # Find parquet files
        parquet_files = list(bronze_root.rglob('*.parquet'))
        assert len(parquet_files) > 0
        
        # Read back data
        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 2
        assert df['symbol'].iloc[0] == 'SPY'
        assert df['price'].iloc[0] == 600.50
        
        print(f"✅ BronzeWriter test passed: {len(parquet_files)} files written")


@pytest.mark.asyncio
async def test_gold_writer_local_filesystem():
    """Test GoldWriter writes level signals to local filesystem correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = GoldWriter(
            data_root=tmpdir,
            buffer_limit=2,
            flush_interval_seconds=1.0,
            use_s3=False
        )
        
        # Create test level signals payload
        ts_ms = int(time.time() * 1000)
        
        payload1 = {
            'ts': ts_ms,
            'spy': {'spot': 600.50, 'bid': 600.49, 'ask': 600.51},
            'levels': [
                {
                    'id': 'STRIKE_600',
                    'kind': 'STRIKE',
                    'price': 600.0,
                    'direction': 'SUPPORT',
                    'distance': 0.50,
                    'break_score_raw': 75,
                    'break_score_smooth': 72,
                    'signal': 'BREAK',
                    'confidence': 'HIGH',
                    'barrier': {
                        'state': 'VACUUM',
                        'delta_liq': -500,
                        'replenishment_ratio': 0.2,
                        'added': 100,
                        'canceled': 400,
                        'filled': 200
                    },
                    'tape': {
                        'imbalance': -0.6,
                        'buy_vol': 10000,
                        'sell_vol': 25000,
                        'velocity': -0.05,
                        'sweep': {
                            'detected': True,
                            'direction': 'DOWN',
                            'notional': 1_000_000
                        }
                    },
                    'fuel': {
                        'effect': 'AMPLIFY',
                        'net_dealer_gamma': -50000,
                        'call_wall': 605,
                        'put_wall': 595,
                        'hvl': 600
                    },
                    'runway': {
                        'direction': 'DOWN',
                        'next_obstacle': {'id': 'PUT_WALL', 'price': 595.0},
                        'distance': 5.0,
                        'quality': 'CLEAR'
                    },
                    'note': 'Test signal'
                }
            ]
        }
        
        # Write signals
        await writer.write_level_signals(payload1)
        
        # Manually flush since we're not using start()
        await writer.flush()
        
        # Verify files were written
        gold_root = Path(tmpdir) / 'gold' / 'levels' / 'signals' / 'underlying=SPY'
        assert gold_root.exists()
        
        # Find parquet files
        parquet_files = list(gold_root.rglob('*.parquet'))
        assert len(parquet_files) > 0
        
        # Read back data
        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 1
        assert df['level_id'].iloc[0] == 'STRIKE_600'
        assert df['break_score_raw'].iloc[0] == 75
        assert df['barrier_state'].iloc[0] == 'VACUUM'
        
        print(f"✅ GoldWriter test passed: {len(parquet_files)} files written")


@pytest.mark.asyncio
async def test_dataclass_to_dict_conversion():
    """Test dataclass to dict conversion with enum handling."""
    trade = StockTrade(
        ts_event_ns=int(time.time() * 1e9),
        ts_recv_ns=int(time.time() * 1e9),
        source=EventSource.POLYGON_WS,
        symbol='SPY',
        price=600.50,
        size=100,
        exchange=1,
        conditions=['@', 'F'],
        seq=12345
    )
    
    result = dataclass_to_dict(trade)
    
    assert result['symbol'] == 'SPY'
    assert result['price'] == 600.50
    assert result['source'] == 'polygon_ws'  # Enum converted to value
    assert isinstance(result['conditions'], str)  # List converted to string
    
    print("✅ Dataclass conversion test passed")


if __name__ == '__main__':
    # Run tests manually
    asyncio.run(test_bronze_writer_local_filesystem())
    asyncio.run(test_gold_writer_local_filesystem())
    asyncio.run(test_dataclass_to_dict_conversion())
    print("\n✅ All Lake service tests passed!")

