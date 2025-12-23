"""
Integration test for Agent A (Physics Engine) and Agent B (Context Engine).

This test verifies that both engines can work together to create a complete
level signal with physics metrics and context features.
"""

import pytest
import pandas as pd
import time
from src.features.physics_engine import PhysicsEngine
from src.features.context_engine import ContextEngine
from src.common.schemas.levels_signals import LevelSignalV1, LevelKind, OutcomeLabel


class TestAgentABIntegration:
    """Integration tests for Physics Engine + Context Engine."""
    
    def test_combined_signal_creation(self):
        """Test creating a complete LevelSignalV1 using both engines."""
        # Setup: Generate mock data for both engines
        
        # 1. Context Engine: Create OHLCV data
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=500,
            base_price=687.0
        )
        context_engine = ContextEngine(ohlcv_df=ohlcv)
        
        # 2. Physics Engine: Create mock order book data
        physics_engine = PhysicsEngine()
        current_time = time.time_ns()
        
        mbp10 = PhysicsEngine.generate_mock_mbp10(
            timestamp_ns=current_time,
            level_price=6870.0,  # ES price (SPY * 10)
            wall_size=10000
        )
        
        trades = PhysicsEngine.generate_mock_trades(
            start_time_ns=current_time - 5_000_000_000,
            num_trades=50,
            price_level=6870.0
        )
        
        # 3. Get context features
        sample_row = ohlcv.iloc[200]
        ts_ns = int(sample_row['timestamp'].value)
        current_price = sample_row['close']
        
        is_opening = context_engine.is_first_15m(ts_ns)
        active_levels = context_engine.get_active_levels(current_price, ts_ns)
        pm_high = context_engine.get_premarket_high()
        sma_200 = context_engine.get_sma_200_at_time(ts_ns)
        
        # 4. Get physics features
        wall_ratio = physics_engine.calculate_wall_ratio(mbp10, 6870.0)
        tape_velocity = physics_engine.calculate_tape_velocity(trades, current_time)
        
        # 5. Create LevelSignalV1 (combining both engines)
        if active_levels:
            level = active_levels[0]
            
            signal = LevelSignalV1(
                event_id=f"TEST_{ts_ns}",
                ts_event_ns=ts_ns,
                symbol="SPY",
                spot=current_price,
                level_price=level['level_price'],
                level_kind=level['level_kind'],
                distance=level['distance'],
                is_first_15m=is_opening,
                dist_to_sma_200=abs(current_price - sma_200) if sma_200 else None,
                wall_ratio=wall_ratio,
                tape_velocity=tape_velocity,
                gamma_exposure=0.0,  # Placeholder for now
                outcome=OutcomeLabel.UNDEFINED
            )
            
            # Verify signal was created successfully
            assert signal.event_id.startswith("TEST_")
            assert signal.symbol == "SPY"
            assert signal.level_kind in [LevelKind.PM_HIGH, LevelKind.PM_LOW, LevelKind.SMA_200]
            assert signal.wall_ratio == 2.0  # 10000 / 5000
            assert signal.tape_velocity == 10.0  # 50 trades / 5 seconds
            assert isinstance(signal.is_first_15m, bool)
    
    def test_level_detection_with_physics(self):
        """Test detecting multiple levels and getting physics for each."""
        # Create context engine
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=500,
            base_price=687.0
        )
        context_engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Create physics engine
        physics_engine = PhysicsEngine()
        
        # Get pre-market high
        pm_high = context_engine.get_premarket_high()
        
        if pm_high is not None:
            # Get timestamp
            sample_row = ohlcv.iloc[200]
            ts_ns = int(sample_row['timestamp'].value)
            
            # Detect levels near PM_HIGH
            levels = context_engine.get_active_levels(pm_high, ts_ns)
            
            # For each level, get physics
            signals = []
            for level in levels:
                # Create mock MBP-10 at this level (convert SPY to ES price)
                es_price = level['level_price'] * 10
                
                mbp10 = PhysicsEngine.generate_mock_mbp10(
                    timestamp_ns=ts_ns,
                    level_price=es_price,
                    wall_size=8000
                )
                
                wall_ratio = physics_engine.calculate_wall_ratio(mbp10, es_price)
                
                signal = LevelSignalV1(
                    event_id=f"LEVEL_{level['level_kind'].value}_{ts_ns}",
                    ts_event_ns=ts_ns,
                    symbol="SPY",
                    level_price=level['level_price'],
                    level_kind=level['level_kind'],
                    distance=level['distance'],
                    wall_ratio=wall_ratio,
                    gamma_exposure=0.0,
                    tape_velocity=0.0
                )
                
                signals.append(signal)
            
            # Verify we created signals
            assert len(signals) >= 1
            
            for signal in signals:
                assert signal.level_kind in [LevelKind.PM_HIGH, LevelKind.PM_LOW, LevelKind.SMA_200]
                assert signal.wall_ratio > 0.0
                assert signal.distance <= 0.10
    
    def test_time_context_affects_signal(self):
        """Test that time-of-day context (is_first_15m) varies correctly."""
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=500,
            base_price=687.0
        )
        context_engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Find a timestamp in opening range
        ohlcv_et = ohlcv.copy()
        ohlcv_et['time_et'] = ohlcv_et['timestamp'].dt.tz_convert('America/New_York').dt.time
        
        # Try to find opening range timestamp
        or_rows = ohlcv_et[
            (ohlcv_et['time_et'] >= pd.Timestamp('09:30:00').time()) &
            (ohlcv_et['time_et'] < pd.Timestamp('09:45:00').time())
        ]
        
        if len(or_rows) > 0:
            or_ts = int(or_rows.iloc[5]['timestamp'].value)
            is_opening = context_engine.is_first_15m(or_ts)
            assert is_opening is True
        
        # Find a timestamp after opening range
        post_or_rows = ohlcv_et[ohlcv_et['time_et'] >= pd.Timestamp('10:00:00').time()]
        
        if len(post_or_rows) > 0:
            post_ts = int(post_or_rows.iloc[5]['timestamp'].value)
            is_opening = context_engine.is_first_15m(post_ts)
            assert is_opening is False
    
    def test_workflow_simulation(self):
        """Simulate the complete workflow: scan price → detect level → get physics → create signal."""
        # Setup both engines
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=500,
            base_price=687.0
        )
        context_engine = ContextEngine(ohlcv_df=ohlcv)
        physics_engine = PhysicsEngine()
        
        # Simulate scanning through price data
        signals_created = []
        
        for idx in range(100, len(ohlcv), 50):
            row = ohlcv.iloc[idx]
            ts_ns = int(row['timestamp'].value)
            price = row['close']
            
            # Step 1: Check for levels near current price
            levels = context_engine.get_active_levels(price, ts_ns)
            
            if not levels:
                continue
            
            # Step 2: For each level, get physics
            for level in levels:
                # Create mock order book data
                es_price = level['level_price'] * 10
                
                mbp10 = PhysicsEngine.generate_mock_mbp10(
                    timestamp_ns=ts_ns,
                    level_price=es_price,
                    wall_size=5000 + (idx % 3) * 2000  # Vary the wall size
                )
                
                trades = PhysicsEngine.generate_mock_trades(
                    start_time_ns=ts_ns - 5_000_000_000,
                    num_trades=30 + (idx % 5) * 10,
                    price_level=es_price
                )
                
                # Get physics metrics
                wall_ratio = physics_engine.calculate_wall_ratio(mbp10, es_price)
                velocity = physics_engine.calculate_tape_velocity(trades, ts_ns)
                
                # Get context features
                is_opening = context_engine.is_first_15m(ts_ns)
                sma = context_engine.get_sma_200_at_time(ts_ns)
                
                # Create signal
                signal = LevelSignalV1(
                    event_id=f"SCAN_{idx}_{level['level_kind'].value}",
                    ts_event_ns=ts_ns,
                    symbol="SPY",
                    spot=price,
                    level_price=level['level_price'],
                    level_kind=level['level_kind'],
                    distance=level['distance'],
                    is_first_15m=is_opening,
                    dist_to_sma_200=abs(price - sma) if sma else None,
                    wall_ratio=wall_ratio,
                    tape_velocity=velocity,
                    gamma_exposure=0.0
                )
                
                signals_created.append(signal)
        
        # Verify we created multiple signals
        print(f"\n  Created {len(signals_created)} signals during price scan")
        
        # All signals should be valid
        for signal in signals_created:
            assert signal.symbol == "SPY"
            assert signal.wall_ratio >= 0.0
            assert signal.tape_velocity >= 0.0
            assert signal.distance <= 0.10
            assert isinstance(signal.is_first_15m, bool)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

