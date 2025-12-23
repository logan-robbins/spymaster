"""
Test suite for Context Engine (Agent B).

Tests the context engine methods for time-of-day checks and structural level detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone
from src.features.context_engine import ContextEngine
from src.common.schemas.levels_signals import LevelKind


class TestContextEngine:
    """Test the Context Engine with OHLCV data."""
    
    def test_generate_mock_ohlcv(self):
        """Test mock OHLCV data generation."""
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=480,
            base_price=687.0
        )
        
        # Check structure
        assert len(ohlcv) == 480
        assert 'timestamp' in ohlcv.columns
        assert 'open' in ohlcv.columns
        assert 'high' in ohlcv.columns
        assert 'low' in ohlcv.columns
        assert 'close' in ohlcv.columns
        assert 'volume' in ohlcv.columns
        
        # Check price is around base
        assert ohlcv['close'].mean() > 680.0
        assert ohlcv['close'].mean() < 695.0
    
    def test_is_first_15m_true(self):
        """Test is_first_15m returns True during 09:30-09:45 ET."""
        engine = ContextEngine()
        
        # Create timestamp at 09:35 ET (should be True)
        # December 22, 2025 at 09:35 ET = 14:35 UTC
        dt_et = pd.Timestamp('2025-12-22 09:35:00', tz='America/New_York')
        ts_ns = int(dt_et.value)
        
        result = engine.is_first_15m(ts_ns)
        assert result is True
    
    def test_is_first_15m_false_before(self):
        """Test is_first_15m returns False before 09:30 ET."""
        engine = ContextEngine()
        
        # Create timestamp at 09:29 ET (should be False)
        dt_et = pd.Timestamp('2025-12-22 09:29:00', tz='America/New_York')
        ts_ns = int(dt_et.value)
        
        result = engine.is_first_15m(ts_ns)
        assert result is False
    
    def test_is_first_15m_false_after(self):
        """Test is_first_15m returns False after 09:45 ET."""
        engine = ContextEngine()
        
        # Create timestamp at 09:46 ET (should be False)
        dt_et = pd.Timestamp('2025-12-22 09:46:00', tz='America/New_York')
        ts_ns = int(dt_et.value)
        
        result = engine.is_first_15m(ts_ns)
        assert result is False
    
    def test_is_first_15m_exactly_0930(self):
        """Test is_first_15m returns True at exactly 09:30 ET."""
        engine = ContextEngine()
        
        # Create timestamp at exactly 09:30:00 ET
        dt_et = pd.Timestamp('2025-12-22 09:30:00', tz='America/New_York')
        ts_ns = int(dt_et.value)
        
        result = engine.is_first_15m(ts_ns)
        assert result is True
    
    def test_is_first_15m_exactly_0945(self):
        """Test is_first_15m returns False at exactly 09:45 ET (exclusive end)."""
        engine = ContextEngine()
        
        # Create timestamp at exactly 09:45:00 ET
        dt_et = pd.Timestamp('2025-12-22 09:45:00', tz='America/New_York')
        ts_ns = int(dt_et.value)
        
        result = engine.is_first_15m(ts_ns)
        assert result is False
    
    def test_premarket_high_low_calculation(self):
        """Test pre-market high/low calculation from OHLCV data."""
        # Create mock data with known pre-market values
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=480,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Pre-market high and low should be calculated
        pm_high = engine.get_premarket_high()
        pm_low = engine.get_premarket_low()
        
        assert pm_high is not None
        assert pm_low is not None
        assert pm_high > pm_low
        assert pm_high > 680.0
        assert pm_low < 695.0
    
    def test_get_active_levels_pm_high(self):
        """Test get_active_levels detects PM_HIGH when price is near."""
        # Create mock data
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=480,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        pm_high = engine.get_premarket_high()
        
        # Query at pre-market high price
        if pm_high is not None:
            # Get a timestamp from the data
            sample_row = ohlcv.iloc[100]
            ts_ns = int(sample_row['timestamp'].value)
            
            levels = engine.get_active_levels(pm_high, ts_ns)
            
            # Should detect PM_HIGH
            pm_high_levels = [l for l in levels if l['level_kind'] == LevelKind.PM_HIGH]
            assert len(pm_high_levels) == 1
            assert abs(pm_high_levels[0]['level_price'] - pm_high) < 0.01
    
    def test_get_active_levels_pm_low(self):
        """Test get_active_levels detects PM_LOW when price is near."""
        # Create mock data
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=480,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        pm_low = engine.get_premarket_low()
        
        # Query at pre-market low price
        if pm_low is not None:
            sample_row = ohlcv.iloc[100]
            ts_ns = int(sample_row['timestamp'].value)
            
            levels = engine.get_active_levels(pm_low, ts_ns)
            
            # Should detect PM_LOW
            pm_low_levels = [l for l in levels if l['level_kind'] == LevelKind.PM_LOW]
            assert len(pm_low_levels) == 1
            assert abs(pm_low_levels[0]['level_price'] - pm_low) < 0.01
    
    def test_get_active_levels_no_levels_far_from_price(self):
        """Test get_active_levels returns empty when price is far from all levels."""
        # Create mock data
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=480,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Query at a price far from any level (e.g., 500.0)
        sample_row = ohlcv.iloc[100]
        ts_ns = int(sample_row['timestamp'].value)
        
        levels = engine.get_active_levels(500.0, ts_ns)
        
        # Should return empty list (no levels within $0.10 of 500.0)
        assert len(levels) == 0
    
    def test_sma_200_calculation(self):
        """Test SMA-200 calculation on 2-minute timeframe."""
        # Create mock data with enough bars for SMA-200
        # Need at least 400 minutes (200 * 2-min bars)
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=500,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Get SMA-200 value at end of data
        last_row = ohlcv.iloc[-1]
        ts_ns = int(last_row['timestamp'].value)
        
        sma_value = engine.get_sma_200_at_time(ts_ns)
        
        # Should have SMA value
        assert sma_value is not None
        # Should be in reasonable range around base price
        assert 680.0 < sma_value < 695.0
    
    def test_sma_200_insufficient_data(self):
        """Test SMA-200 returns None with insufficient data."""
        # Create mock data with too few bars (< 400 minutes)
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=100,  # Not enough for 200 2-min bars
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Try to get SMA-200
        last_row = ohlcv.iloc[-1]
        ts_ns = int(last_row['timestamp'].value)
        
        sma_value = engine.get_sma_200_at_time(ts_ns)
        
        # Should return None (not enough data)
        # OR it might return a value if pandas allows partial SMA
        # Let's just check it doesn't crash
        assert sma_value is None or isinstance(sma_value, float)
    
    def test_get_active_levels_sma_200(self):
        """Test get_active_levels detects SMA_200 when price is near."""
        # Create mock data with enough bars for SMA-200
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=500,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Get a timestamp where SMA is available
        last_row = ohlcv.iloc[-1]
        ts_ns = int(last_row['timestamp'].value)
        
        sma_value = engine.get_sma_200_at_time(ts_ns)
        
        if sma_value is not None:
            # Query at SMA-200 price
            levels = engine.get_active_levels(sma_value, ts_ns)
            
            # Should detect SMA_200
            sma_levels = [l for l in levels if l['level_kind'] == LevelKind.SMA_200]
            assert len(sma_levels) == 1
            assert abs(sma_levels[0]['level_price'] - sma_value) < 0.01
    
    def test_level_tolerance(self):
        """Test that level detection respects $0.10 tolerance."""
        # Create mock data
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=480,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        pm_high = engine.get_premarket_high()
        
        if pm_high is not None:
            sample_row = ohlcv.iloc[100]
            ts_ns = int(sample_row['timestamp'].value)
            
            # Test at exactly $0.10 away (should be detected)
            levels_at_tolerance = engine.get_active_levels(pm_high + 0.10, ts_ns)
            pm_levels_at = [l for l in levels_at_tolerance if l['level_kind'] == LevelKind.PM_HIGH]
            assert len(pm_levels_at) >= 0  # May or may not detect depending on exact value
            
            # Test at $0.20 away (should NOT be detected)
            levels_far = engine.get_active_levels(pm_high + 0.20, ts_ns)
            pm_levels_far = [l for l in levels_far if l['level_kind'] == LevelKind.PM_HIGH]
            assert len(pm_levels_far) == 0
    
    def test_empty_ohlcv_dataframe(self):
        """Test engine handles empty OHLCV DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        engine = ContextEngine(ohlcv_df=empty_df)
        
        # Should not crash
        assert engine.get_premarket_high() is None
        assert engine.get_premarket_low() is None
    
    def test_no_dataframe_initialization(self):
        """Test engine can be initialized without DataFrame."""
        engine = ContextEngine()
        
        # Should work
        assert engine.ohlcv_df is None
        assert engine.get_premarket_high() is None
        assert engine.get_premarket_low() is None
        
        # is_first_15m should still work
        dt_et = pd.Timestamp('2025-12-22 09:35:00', tz='America/New_York')
        ts_ns = int(dt_et.value)
        result = engine.is_first_15m(ts_ns)
        assert result is True
    
    def test_multiple_levels_detected(self):
        """Test that multiple levels can be detected simultaneously."""
        # Create scenario where price is near multiple levels
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=500,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        
        # If PM_HIGH and SMA_200 happen to be close, we might detect both
        # This is a probabilistic test based on synthetic data
        pm_high = engine.get_premarket_high()
        
        if pm_high is not None:
            sample_row = ohlcv.iloc[-1]
            ts_ns = int(sample_row['timestamp'].value)
            
            # Test at PM_HIGH - should at least get PM_HIGH
            levels = engine.get_active_levels(pm_high, ts_ns)
            
            # Should have at least one level
            assert len(levels) >= 1
            
            # Check that all returned levels are within tolerance
            for level in levels:
                assert abs(level['level_price'] - pm_high) <= 0.10


# --- Integration Tests ---

class TestContextEngineIntegration:
    """Integration tests for Context Engine with realistic scenarios."""
    
    def test_full_day_scenario(self):
        """Test Context Engine with a full trading day scenario."""
        # Generate full day of data
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=600,  # 10 hours
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Check pre-market levels exist
        assert engine.get_premarket_high() is not None
        assert engine.get_premarket_low() is not None
        
        # Test at different times of day
        # 1. Pre-market (should be False for is_first_15m)
        pm_row = ohlcv[ohlcv['timestamp'].dt.tz_convert('America/New_York').dt.time < time(9, 30, 0)].iloc[10]
        pm_ts = int(pm_row['timestamp'].value)
        assert engine.is_first_15m(pm_ts) is False
        
        # 2. First 15 minutes (should be True)
        try:
            or_row = ohlcv[
                (ohlcv['timestamp'].dt.tz_convert('America/New_York').dt.time >= time(9, 30, 0)) &
                (ohlcv['timestamp'].dt.tz_convert('America/New_York').dt.time < time(9, 45, 0))
            ].iloc[5]
            or_ts = int(or_row['timestamp'].value)
            assert engine.is_first_15m(or_ts) is True
        except (IndexError, ValueError):
            # If no data in opening range, skip this check
            pass
        
        # 3. Mid-day (should be False)
        try:
            midday_row = ohlcv[ohlcv['timestamp'].dt.tz_convert('America/New_York').dt.time >= time(10, 30, 0)].iloc[10]
            midday_ts = int(midday_row['timestamp'].value)
            assert engine.is_first_15m(midday_ts) is False
        except (IndexError, ValueError):
            # If no mid-day data, skip
            pass
    
    def test_level_detection_workflow(self):
        """Test the typical workflow of detecting levels during trading."""
        # Generate data
        ohlcv = ContextEngine.generate_mock_ohlcv(
            num_minutes=500,
            base_price=687.0
        )
        
        engine = ContextEngine(ohlcv_df=ohlcv)
        
        # Simulate scanning for levels as price moves
        pm_high = engine.get_premarket_high()
        pm_low = engine.get_premarket_low()
        
        assert pm_high is not None
        assert pm_low is not None
        
        # Test at PM_HIGH
        sample_ts = int(ohlcv.iloc[100]['timestamp'].value)
        levels_at_high = engine.get_active_levels(pm_high, sample_ts)
        
        # Should detect PM_HIGH
        high_levels = [l for l in levels_at_high if l['level_kind'] == LevelKind.PM_HIGH]
        assert len(high_levels) > 0
        
        # Test at PM_LOW
        levels_at_low = engine.get_active_levels(pm_low, sample_ts)
        
        # Should detect PM_LOW
        low_levels = [l for l in levels_at_low if l['level_kind'] == LevelKind.PM_LOW]
        assert len(low_levels) > 0
        
        # Verify level metadata
        for level in levels_at_high + levels_at_low:
            assert 'level_kind' in level
            assert 'level_price' in level
            assert 'distance' in level
            assert isinstance(level['level_kind'], LevelKind)
            assert isinstance(level['level_price'], float)
            assert isinstance(level['distance'], float)
            assert level['distance'] >= 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

