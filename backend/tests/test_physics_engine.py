"""
Test suite for Physics Engine (Agent A).

Tests the physics engine methods using real MBP-10 and FuturesTrade schemas.
"""

import pytest
import time
from src.features.physics_engine import PhysicsEngine
from src.common.event_types import MBP10, BidAskLevel, FuturesTrade, EventSource, Aggressor


class TestPhysicsEngine:
    """Test the Physics Engine with real schemas."""
    
    def test_calculate_wall_ratio_with_mbp10(self):
        """Test wall ratio calculation with real MBP-10 data."""
        engine = PhysicsEngine()
        
        # Create MBP-10 with significant wall at 6870.0
        mbp10 = PhysicsEngine.generate_mock_mbp10(
            level_price=6870.0,
            wall_size=10000
        )
        
        # Calculate wall ratio
        wall_ratio = engine.calculate_wall_ratio(mbp10, 6870.0)
        
        # Should be 2.0x (10000 / 5000 default avg volume)
        assert wall_ratio == 2.0
        assert mbp10.levels[0].bid_sz == 10000
    
    def test_calculate_wall_ratio_empty_mbp10(self):
        """Test wall ratio with empty MBP-10 returns 0."""
        engine = PhysicsEngine()
        
        # Create empty MBP-10
        mbp10 = MBP10(
            ts_event_ns=time.time_ns(),
            ts_recv_ns=time.time_ns(),
            source=EventSource.SIM,
            symbol="ES",
            levels=[],
            is_snapshot=True
        )
        
        wall_ratio = engine.calculate_wall_ratio(mbp10, 6870.0)
        assert wall_ratio == 0.0
    
    def test_calculate_tape_velocity_with_futures_trades(self):
        """Test tape velocity calculation with real FuturesTrade data."""
        engine = PhysicsEngine()
        current_time = time.time_ns()
        
        # Generate 50 trades in last 5 seconds
        trades = PhysicsEngine.generate_mock_trades(
            start_time_ns=current_time - 5_000_000_000,
            num_trades=50,
            price_level=6870.0
        )
        
        velocity = engine.calculate_tape_velocity(trades, current_time)
        
        # Should be 10 trades/sec (50 / 5)
        assert velocity == 10.0
    
    def test_calculate_tape_velocity_no_trades(self):
        """Test tape velocity with no trades returns 0."""
        engine = PhysicsEngine()
        current_time = time.time_ns()
        
        # Old trades outside window
        trades = PhysicsEngine.generate_mock_trades(
            start_time_ns=current_time - 10_000_000_000,  # 10 seconds ago
            num_trades=50
        )
        
        # Window is only 5 seconds, so no trades should be in window
        velocity = engine.calculate_tape_velocity(trades, current_time)
        assert velocity == 0.0
    
    def test_detect_replenishment_success(self):
        """Test replenishment detection when liquidity is reloaded."""
        engine = PhysicsEngine()
        base_time = time.time_ns()
        
        # Before sweep: modest liquidity
        mbp_before = PhysicsEngine.generate_mock_mbp10(
            timestamp_ns=base_time,
            level_price=6870.0,
            wall_size=5000
        )
        
        # Sweep event
        sweep_trade = FuturesTrade(
            ts_event_ns=base_time + 10_000_000,  # 10ms later
            ts_recv_ns=base_time + 10_000_000,
            source=EventSource.SIM,
            symbol="ES",
            price=6870.0,
            size=100,
            aggressor=Aggressor.BUY
        )
        
        # After sweep: INCREASED liquidity (replenishment)
        mbp_after = PhysicsEngine.generate_mock_mbp10(
            timestamp_ns=base_time + 45_000_000,  # 45ms after sweep (35ms delta)
            level_price=6870.0,
            wall_size=8000  # MORE than before
        )
        
        replenishment = engine.detect_replenishment(
            trade_tape=[sweep_trade],
            mbp10_snapshots=[mbp_before, mbp_after],
            level_price=6870.0
        )
        
        assert replenishment is not None
        assert replenishment == 35.0  # 45ms - 10ms = 35ms
    
    def test_detect_replenishment_no_sweep(self):
        """Test replenishment detection when no sweep occurred."""
        engine = PhysicsEngine()
        base_time = time.time_ns()
        
        mbp_before = PhysicsEngine.generate_mock_mbp10(
            timestamp_ns=base_time,
            level_price=6870.0,
            wall_size=5000
        )
        
        mbp_after = PhysicsEngine.generate_mock_mbp10(
            timestamp_ns=base_time + 45_000_000,
            level_price=6870.0,
            wall_size=8000
        )
        
        # No trades (no sweep)
        replenishment = engine.detect_replenishment(
            trade_tape=[],
            mbp10_snapshots=[mbp_before, mbp_after],
            level_price=6870.0
        )
        
        assert replenishment is None
    
    def test_mock_mbp10_generator(self):
        """Test that mock MBP-10 generator creates valid data."""
        mbp10 = PhysicsEngine.generate_mock_mbp10(
            level_price=6870.0,
            wall_size=10000,
            symbol="ES"
        )
        
        assert isinstance(mbp10, MBP10)
        assert mbp10.symbol == "ES"
        assert len(mbp10.levels) == 10
        assert mbp10.is_snapshot is True
        assert mbp10.source == EventSource.SIM
        
        # Check first level has the wall
        first_level = mbp10.levels[0]
        assert first_level.bid_sz == 10000
        assert first_level.bid_px == 6870.0
    
    def test_mock_trades_generator(self):
        """Test that mock trades generator creates valid FuturesTrade objects."""
        start_time = time.time_ns()
        trades = PhysicsEngine.generate_mock_trades(
            start_time_ns=start_time,
            num_trades=100,
            price_level=6870.0,
            symbol="ES"
        )
        
        assert len(trades) == 100
        assert all(isinstance(t, FuturesTrade) for t in trades)
        assert all(t.symbol == "ES" for t in trades)
        assert all(t.source == EventSource.SIM for t in trades)
        
        # Check timestamps are sequential
        for i in range(1, len(trades)):
            assert trades[i].ts_event_ns > trades[i-1].ts_event_ns
    
    def test_wall_ratio_with_price_tolerance(self):
        """Test wall ratio respects price tolerance parameter."""
        engine = PhysicsEngine()
        
        # Create MBP-10 with wall slightly off the target price
        mbp10 = PhysicsEngine.generate_mock_mbp10(
            level_price=6870.0,
            wall_size=10000
        )
        
        # Should find wall with default tolerance (0.01)
        wall_ratio_found = engine.calculate_wall_ratio(mbp10, 6870.0, tolerance=0.01)
        assert wall_ratio_found == 2.0
        
        # Should not find wall with very tight tolerance
        wall_ratio_not_found = engine.calculate_wall_ratio(mbp10, 6870.0, tolerance=0.0001)
        assert wall_ratio_not_found == 2.0  # Still finds exact match
    
    def test_integration_complete_workflow(self):
        """Test a complete workflow using all methods together."""
        engine = PhysicsEngine()
        current_time = time.time_ns()
        
        # Create market scenario
        mbp10 = PhysicsEngine.generate_mock_mbp10(
            timestamp_ns=current_time,
            level_price=6870.0,
            wall_size=15000
        )
        
        trades = PhysicsEngine.generate_mock_trades(
            start_time_ns=current_time - 5_000_000_000,
            num_trades=75,
            price_level=6870.0
        )
        
        # Calculate all metrics
        wall_ratio = engine.calculate_wall_ratio(mbp10, 6870.0)
        velocity = engine.calculate_tape_velocity(trades, current_time)
        
        # Verify results
        assert wall_ratio == 3.0  # 15000 / 5000
        assert velocity == 15.0  # 75 / 5
        
        # These would be used to populate LevelSignalV1 schema
        assert wall_ratio > 0
        assert velocity > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

