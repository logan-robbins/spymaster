"""
Integration test for Gateway Service (Agent D).

Tests:
1. Gateway connects to NATS
2. Gateway receives messages on levels.signals
3. Gateway broadcasts to WebSocket clients
4. Multiple clients can connect simultaneously
5. New clients receive cached state
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.gateway.socket_broadcaster import SocketBroadcaster
from src.common.bus import NATSBus


@pytest.mark.asyncio
async def test_gateway_nats_subscription():
    """Test that Gateway subscribes to NATS correctly."""
    # Create mock bus
    mock_bus = MagicMock(spec=NATSBus)
    mock_bus.connect = AsyncMock()
    mock_bus.subscribe = AsyncMock()
    
    # Create broadcaster
    broadcaster = SocketBroadcaster(bus=mock_bus)
    
    # Start (this calls bus.connect and bus.subscribe)
    await broadcaster.start()
    
    # Verify NATS connection was established
    mock_bus.connect.assert_called_once()
    
    # Verify subscription to levels.signals
    mock_bus.subscribe.assert_called_once()
    call_args = mock_bus.subscribe.call_args
    assert call_args[1]["subject"] == "levels.signals"
    assert call_args[1]["durable_name"] == "gateway_levels"


@pytest.mark.asyncio
async def test_gateway_broadcasts_to_multiple_clients():
    """Test that Gateway broadcasts to multiple WebSocket clients."""
    # Create broadcaster (no bus needed for this test)
    broadcaster = SocketBroadcaster()
    
    # Create mock WebSocket clients
    mock_ws1 = AsyncMock()
    mock_ws2 = AsyncMock()
    mock_ws3 = AsyncMock()
    
    # Add to connections
    broadcaster.active_connections = [mock_ws1, mock_ws2, mock_ws3]
    
    # Broadcast message
    test_message = {"test": "payload", "value": 123}
    await broadcaster.broadcast(test_message)
    
    # Verify all clients received the message
    expected_payload = json.dumps(test_message)
    mock_ws1.send_text.assert_called_once_with(expected_payload)
    mock_ws2.send_text.assert_called_once_with(expected_payload)
    mock_ws3.send_text.assert_called_once_with(expected_payload)


@pytest.mark.asyncio
async def test_gateway_removes_failed_connections():
    """Test that Gateway removes clients that fail to receive messages."""
    broadcaster = SocketBroadcaster()
    
    # Create mock WebSocket clients
    mock_ws_good = AsyncMock()
    mock_ws_bad = AsyncMock()
    mock_ws_bad.send_text.side_effect = Exception("Connection broken")
    
    # Add to connections
    broadcaster.active_connections = [mock_ws_good, mock_ws_bad]
    
    # Broadcast message
    await broadcaster.broadcast({"test": "message"})
    
    # Verify good client received message
    assert mock_ws_good.send_text.called
    
    # Verify bad client was removed
    assert mock_ws_bad not in broadcaster.active_connections
    assert len(broadcaster.active_connections) == 1


@pytest.mark.asyncio
async def test_gateway_disconnect():
    """Test that Gateway properly removes disconnected clients."""
    broadcaster = SocketBroadcaster()
    
    # Create and connect mock WebSocket
    mock_ws = AsyncMock()
    await broadcaster.connect(mock_ws)
    
    assert mock_ws in broadcaster.active_connections
    
    # Disconnect
    await broadcaster.disconnect(mock_ws)
    
    # Verify client was removed
    assert mock_ws not in broadcaster.active_connections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

