"""
Test Core Service functionality.

Tests:
- Core Service initialization
- Message handling
- Snap loop execution
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from src.common.bus import NATSBus
from src.common.config import CONFIG
from src.common.event_types import (
    FuturesTrade, MBP10, OptionTrade,
    EventSource, Aggressor, BidAskLevel
)
from src.core.service import CoreService
from src.core.greek_enricher import GreekEnricher


@pytest.fixture
def mock_bus():
    """Create a mock NATS bus."""
    bus = MagicMock(spec=NATSBus)
    bus.connect = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    bus.close = AsyncMock()
    return bus


@pytest.fixture
def mock_greek_enricher():
    """Create a mock greek enricher."""
    enricher = MagicMock(spec=GreekEnricher)
    enricher.get_greeks = MagicMock(return_value={
        "delta": 0.5,
        "gamma": 0.02,
        "theta": -0.1,
        "vega": 0.15
    })
    return enricher


@pytest.fixture
def core_service(mock_bus, mock_greek_enricher):
    """Create a CoreService instance with mocks."""
    return CoreService(
        bus=mock_bus,
        greek_enricher=mock_greek_enricher,
        config=CONFIG,
        user_hotzones=[680.0, 685.0]
    )


def test_core_service_init(core_service, mock_bus):
    """Test CoreService initializes correctly."""
    assert core_service.bus == mock_bus
    assert core_service.market_state is not None
    assert core_service.level_signal_service is not None
    assert core_service.running is False


@pytest.mark.asyncio
async def test_core_service_subscribe(core_service, mock_bus):
    """Test CoreService subscribes to correct subjects."""
    # Start service (but don't wait for snap loop)
    core_service.running = True
    await core_service._subscribe_to_market_data()
    
    # Verify subscriptions
    assert mock_bus.subscribe.call_count == 3
    
    # Check subjects
    calls = mock_bus.subscribe.call_args_list
    subjects = [call[1]["subject"] for call in calls]
    
    assert "market.futures.trades" in subjects
    assert "market.futures.mbp10" in subjects
    assert "market.options.trades" in subjects


@pytest.mark.asyncio
async def test_handle_futures_trade(core_service):
    """Test handling of futures trade message."""
    # Create a test trade message
    trade_msg = {
        "ts_event_ns": 1700000000000000000,
        "ts_recv_ns": 1700000000100000000,
        "source": "replay",
        "symbol": "ES",
        "price": 5450.25,
        "size": 10,
        "aggressor": 1,  # BUY
        "exchange": None,
        "conditions": None,
        "seq": 1
    }
    
    # Handle the message
    await core_service._handle_futures_trade(trade_msg)
    
    # Verify market state was updated
    assert core_service.market_state.last_es_trade is not None
    assert core_service.market_state.last_es_trade.price == 5450.25
    assert core_service.market_state.last_es_trade.size == 10


@pytest.mark.asyncio
async def test_handle_futures_mbp10(core_service):
    """Test handling of MBP-10 message."""
    # Create a test MBP-10 message
    mbp_msg = {
        "ts_event_ns": 1700000000000000000,
        "ts_recv_ns": 1700000000100000000,
        "source": "replay",
        "symbol": "ES",
        "levels": [
            {
                "bid_px": 5450.00,
                "bid_sz": 50,
                "ask_px": 5450.25,
                "ask_sz": 75
            }
        ],
        "is_snapshot": False,
        "seq": 1
    }
    
    # Handle the message
    await core_service._handle_futures_mbp10(mbp_msg)
    
    # Verify market state was updated
    assert core_service.market_state.es_mbp10_snapshot is not None
    assert len(core_service.market_state.es_mbp10_snapshot.levels) == 1
    assert core_service.market_state.es_mbp10_snapshot.levels[0].bid_px == 5450.00


@pytest.mark.asyncio
async def test_handle_option_trade(core_service, mock_greek_enricher):
    """Test handling of option trade message."""
    # Create a test option trade message
    option_msg = {
        "ts_event_ns": 1700000000000000000,
        "ts_recv_ns": 1700000000100000000,
        "source": "replay",
        "underlying": "SPY",
        "option_symbol": "O:SPY251216C00680000",
        "exp_date": "2025-12-16",
        "strike": 680.0,
        "right": "C",
        "price": 5.50,
        "size": 10,
        "opt_bid": 5.45,
        "opt_ask": 5.55,
        "aggressor": 1,  # BUY
        "conditions": None,
        "seq": 1
    }
    
    # Handle the message
    await core_service._handle_option_trade(option_msg)
    
    # Verify greek enricher was called
    mock_greek_enricher.get_greeks.assert_called_once()
    
    # Verify market state was updated
    key = (680.0, "C", "2025-12-16")
    assert key in core_service.market_state.option_flows


@pytest.mark.asyncio
async def test_snap_loop_publishes(core_service, mock_bus):
    """Test that snap loop publishes level signals."""
    # Set up some market data first
    trade_msg = {
        "ts_event_ns": 1700000000000000000,
        "ts_recv_ns": 1700000000100000000,
        "source": "replay",
        "symbol": "ES",
        "price": 6800.0,  # ES price ~= SPY * 10
        "size": 10,
        "aggressor": 1,
        "exchange": None,
        "conditions": None,
        "seq": 1
    }
    await core_service._handle_futures_trade(trade_msg)
    
    mbp_msg = {
        "ts_event_ns": 1700000000000000000,
        "ts_recv_ns": 1700000000100000000,
        "source": "replay",
        "symbol": "ES",
        "levels": [
            {
                "bid_px": 6799.75,
                "bid_sz": 50,
                "ask_px": 6800.25,
                "ask_sz": 75
            }
        ],
        "is_snapshot": False,
        "seq": 1
    }
    await core_service._handle_futures_mbp10(mbp_msg)
    
    # Run snap loop once
    core_service.running = True
    
    # Create a task for snap loop
    snap_task = asyncio.create_task(core_service._snap_loop())
    
    # Wait a bit for one iteration
    await asyncio.sleep(0.3)
    
    # Stop the loop
    core_service.running = False
    snap_task.cancel()
    
    try:
        await snap_task
    except asyncio.CancelledError:
        pass
    
    # Verify publish was called
    assert mock_bus.publish.called
    
    # Check that we published to levels.signals
    call_args = mock_bus.publish.call_args_list
    subjects = [call[1]["subject"] for call in call_args]
    assert "levels.signals" in subjects


@pytest.mark.asyncio
async def test_core_service_stop(core_service):
    """Test CoreService stops gracefully."""
    core_service.running = True
    core_service.snap_task = asyncio.create_task(asyncio.sleep(10))
    
    await core_service.stop()
    
    assert core_service.running is False
    assert core_service.snap_task.cancelled()


def test_core_service_without_greek_enricher(mock_bus):
    """Test CoreService can initialize without greek enricher."""
    service = CoreService(
        bus=mock_bus,
        greek_enricher=None,
        config=CONFIG
    )
    
    assert service.greek_enricher is None
    assert service.market_state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

