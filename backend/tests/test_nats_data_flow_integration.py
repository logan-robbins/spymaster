"""
Integration Test: End-to-End NATS Data Flow

Tests the complete pipeline:
Core Service → NATS (levels.signals) → Gateway → WebSocket → Frontend Schema

This test creates a mock NATS message bus and verifies data transformations.
"""

import pytest
import json
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import modules to test
from src.gateway.socket_broadcaster import SocketBroadcaster
from tests.test_confluence_and_ml_integration import generate_mock_levels_signals_payload


class MockNATSBus:
    """Mock NATS bus for testing without actual NATS infrastructure."""
    
    def __init__(self):
        self.published_messages: List[tuple[str, Dict[str, Any]]] = []
        self.subscribers: Dict[str, callable] = {}
    
    async def connect(self):
        """Mock connect."""
        pass
    
    async def publish(self, subject: str, payload: Dict[str, Any]):
        """Mock publish - store message."""
        self.published_messages.append((subject, payload))
    
    async def subscribe(self, subject: str, callback: callable, durable_name: Optional[str] = None):
        """Mock subscribe - register callback."""
        self.subscribers[subject] = callback
    
    async def simulate_message(self, subject: str, payload: Dict[str, Any]):
        """Simulate receiving a message on a subject."""
        if subject in self.subscribers:
            await self.subscribers[subject](payload)


@pytest.mark.asyncio
async def test_core_to_gateway_data_flow():
    """
    Test data flow from Core Service through Gateway normalization.
    
    Simulates:
    1. Core publishes to levels.signals
    2. Gateway receives and normalizes
    3. Gateway broadcasts to WebSocket clients
    """
    # Create mock NATS bus
    mock_bus = MockNATSBus()
    
    # Create Gateway with mock bus
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    # Generate mock Core Service payload
    core_payload = generate_mock_levels_signals_payload(
        num_levels=2,
        include_confluence=True,
        include_viewport=True
    )
    
    # Simulate Core publishing to NATS
    await mock_bus.simulate_message("levels.signals", core_payload)
    
    # Give time for processing
    await asyncio.sleep(0.01)
    
    # Verify Gateway received and normalized
    assert broadcaster._latest_levels is not None
    normalized = broadcaster._latest_levels
    
    # Verify structure
    assert "ts" in normalized
    assert "spy" in normalized
    assert "levels" in normalized
    
    # Verify levels were normalized
    levels = normalized["levels"]
    assert len(levels) == 2
    
    # Verify first level has all expected fields
    level = levels[0]
    
    # Core physics fields
    assert "id" in level
    assert "level_price" in level
    assert "barrier_state" in level
    assert "tape_imbalance" in level
    assert "gamma_exposure" in level
    
    # Confluence fields
    assert "confluence_count" in level
    assert "confluence_pressure" in level
    assert "confluence_alignment" in level
    assert "confluence_level" in level
    assert "confluence_level_name" in level
    
    # ML predictions merged
    assert "ml_predictions" in level
    ml_pred = level["ml_predictions"]
    assert "p_tradeable_2" in ml_pred
    assert "p_break" in ml_pred
    assert "utility_score" in ml_pred


@pytest.mark.asyncio
async def test_gateway_normalizes_direction():
    """Test Gateway normalizes SUPPORT/RESISTANCE to DOWN/UP."""
    mock_bus = MockNATSBus()
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    # Generate payload with SUPPORT direction
    payload = generate_mock_levels_signals_payload(num_levels=1, include_confluence=True, include_viewport=False)
    payload["levels"][0]["direction"] = "SUPPORT"
    
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    
    # Check normalization
    normalized_level = broadcaster._latest_levels["levels"][0]
    assert normalized_level["direction"] == "DOWN"  # SUPPORT → DOWN
    
    # Test RESISTANCE
    payload["levels"][0]["direction"] = "RESISTANCE"
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    
    normalized_level = broadcaster._latest_levels["levels"][0]
    assert normalized_level["direction"] == "UP"  # RESISTANCE → UP


@pytest.mark.asyncio
async def test_gateway_normalizes_signal():
    """Test Gateway normalizes REJECT/CONTESTED/NEUTRAL to BOUNCE/NO_TRADE."""
    mock_bus = MockNATSBus()
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    payload = generate_mock_levels_signals_payload(num_levels=1, include_confluence=True, include_viewport=False)
    
    # Test REJECT → BOUNCE
    payload["levels"][0]["signal"] = "REJECT"
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    assert broadcaster._latest_levels["levels"][0]["signal"] == "BOUNCE"
    
    # Test CONTESTED → NO_TRADE
    payload["levels"][0]["signal"] = "CONTESTED"
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    assert broadcaster._latest_levels["levels"][0]["signal"] == "NO_TRADE"
    
    # Test NEUTRAL → NO_TRADE
    payload["levels"][0]["signal"] = "NEUTRAL"
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    assert broadcaster._latest_levels["levels"][0]["signal"] == "NO_TRADE"
    
    # Test BREAK unchanged
    payload["levels"][0]["signal"] = "BREAK"
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    assert broadcaster._latest_levels["levels"][0]["signal"] == "BREAK"


@pytest.mark.asyncio
async def test_viewport_predictions_merged_per_level():
    """Test that viewport predictions are correctly merged into each level."""
    mock_bus = MockNATSBus()
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    # Generate payload with multiple levels and viewport predictions
    payload = generate_mock_levels_signals_payload(
        num_levels=3,
        include_confluence=True,
        include_viewport=True
    )
    
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    
    levels = broadcaster._latest_levels["levels"]
    
    # Verify all levels have ML predictions
    for i, level in enumerate(levels):
        assert "ml_predictions" in level, f"Level {i} missing ml_predictions"
        
        ml_pred = level["ml_predictions"]
        assert "p_tradeable_2" in ml_pred
        assert "p_break" in ml_pred
        assert "p_bounce" in ml_pred
        assert "strength_signed" in ml_pred
        assert "utility_score" in ml_pred
        assert "stage" in ml_pred
        assert "time_to_threshold" in ml_pred
        
        # Verify values are reasonable
        assert 0.0 <= ml_pred["p_tradeable_2"] <= 1.0
        assert 0.0 <= ml_pred["p_break"] <= 1.0
        assert ml_pred["stage"] in ["stage_a", "stage_b"]


@pytest.mark.asyncio
async def test_missing_viewport_predictions_gracefully_handled():
    """Test that levels without viewport predictions don't crash Gateway."""
    mock_bus = MockNATSBus()
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    # Generate payload WITHOUT viewport predictions
    payload = generate_mock_levels_signals_payload(
        num_levels=2,
        include_confluence=True,
        include_viewport=False
    )
    
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    
    # Verify levels processed successfully
    assert broadcaster._latest_levels is not None
    levels = broadcaster._latest_levels["levels"]
    assert len(levels) == 2
    
    # Verify ml_predictions not added when viewport absent
    for level in levels:
        assert "ml_predictions" not in level or level["ml_predictions"] is None


@pytest.mark.asyncio
async def test_confluence_features_preserved_in_normalization():
    """Test that all confluence features survive Gateway normalization."""
    mock_bus = MockNATSBus()
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    payload = generate_mock_levels_signals_payload(
        num_levels=1,
        include_confluence=True,
        include_viewport=False
    )
    
    # Set specific confluence values
    payload["levels"][0]["confluence_count"] = 4
    payload["levels"][0]["confluence_pressure"] = 0.75
    payload["levels"][0]["confluence_alignment"] = 1  # ALIGNED
    payload["levels"][0]["confluence_level"] = 2  # ULTRA_PREMIUM
    payload["levels"][0]["confluence_level_name"] = "ULTRA_PREMIUM"
    
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    
    # Verify all confluence fields preserved
    level = broadcaster._latest_levels["levels"][0]
    assert level["confluence_count"] == 4
    assert level["confluence_pressure"] == 0.75
    assert level["confluence_alignment"] == 1
    assert level["confluence_level"] == 2
    assert level["confluence_level_name"] == "ULTRA_PREMIUM"


@pytest.mark.asyncio
async def test_json_serialization_of_normalized_payload():
    """Test that normalized payload can be JSON serialized for WebSocket."""
    mock_bus = MockNATSBus()
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    payload = generate_mock_levels_signals_payload(
        num_levels=2,
        include_confluence=True,
        include_viewport=True
    )
    
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    
    # Get normalized payload
    normalized = broadcaster._build_payload()
    
    # Serialize to JSON
    json_str = json.dumps(normalized)
    
    # Deserialize and verify structure
    deserialized = json.loads(json_str)
    assert "levels" in deserialized
    assert len(deserialized["levels"]["levels"]) == 2
    
    # Verify nested structures survived serialization
    level = deserialized["levels"]["levels"][0]
    assert "ml_predictions" in level
    assert "time_to_threshold" in level["ml_predictions"]


@pytest.mark.asyncio
async def test_session_context_computation():
    """Test that session context (is_first_15m, bars_since_open) computed correctly."""
    mock_bus = MockNATSBus()
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    # Use specific timestamp (9:35 AM ET on a trading day)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    # Create timestamp for 9:35 AM ET
    dt = datetime(2025, 12, 26, 9, 35, 0, tzinfo=ZoneInfo("America/New_York"))
    ts_ms = int(dt.timestamp() * 1000)
    
    payload = generate_mock_levels_signals_payload(num_levels=1, include_confluence=False, include_viewport=False)
    payload["ts"] = ts_ms
    
    await mock_bus.simulate_message("levels.signals", payload)
    await asyncio.sleep(0.01)
    
    level = broadcaster._latest_levels["levels"][0]
    
    # 9:35 AM is 5 minutes after 9:30 AM open
    assert level["is_first_15m"] is True
    assert level["bars_since_open"] == 5


@pytest.mark.asyncio
async def test_multiple_concurrent_updates():
    """Test Gateway handles rapid message updates correctly."""
    mock_bus = MockNATSBus()
    broadcaster = SocketBroadcaster(bus=mock_bus)
    await broadcaster.start()
    
    # Send 10 rapid updates
    for i in range(10):
        payload = generate_mock_levels_signals_payload(
            num_levels=1,
            include_confluence=True,
            include_viewport=True
        )
        # Modify spot price to track updates
        payload["spy"]["spot"] = 687.0 + i * 0.10
        
        await mock_bus.simulate_message("levels.signals", payload)
        await asyncio.sleep(0.001)  # 1ms between messages
    
    # Verify latest state
    assert broadcaster._latest_levels is not None
    
    # Should have last update (spot = 687.90)
    final_spot = broadcaster._latest_levels["spy"]["spot"]
    assert final_spot == 687.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
