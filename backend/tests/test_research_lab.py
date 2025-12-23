"""
Tests for Agent C: Research Lab (Labeler + Experiment Runner)

Tests outcome classification logic and statistical analysis methods.
"""

import pytest
from src.research.labeler import get_outcome, label_signal_with_future_data
from src.research.experiment_runner import ExperimentRunner
from src.common.schemas.levels_signals import (
    LevelSignalV1,
    OutcomeLabel,
    LevelKind,
    Direction,
)


# ========== Tests for Labeler ==========

class TestLabeler:
    """Test suite for outcome classification logic."""
    
    def test_bounce_on_resistance_test(self):
        """Test BOUNCE classification when price rejects resistance."""
        # Price at 400.00, tries to go up, falls back down significantly
        signal_price = 400.00
        future_prices = [400.05, 400.03, 399.90, 399.75, 399.80]  # Falls 0.25
        
        outcome = get_outcome(signal_price, future_prices, direction="UP")
        assert outcome == OutcomeLabel.BOUNCE
    
    def test_break_on_resistance_test(self):
        """Test BREAK classification when price breaks through resistance."""
        # Price at 400.00, breaks up significantly
        signal_price = 400.00
        future_prices = [400.10, 400.15, 400.25, 400.35]  # Breaks up 0.35
        
        outcome = get_outcome(signal_price, future_prices, direction="UP")
        assert outcome == OutcomeLabel.BREAK
    
    def test_bounce_on_support_test(self):
        """Test BOUNCE classification when price rejects support (bounces up)."""
        # Price at 400.00, tests support, bounces back up
        signal_price = 400.00
        future_prices = [399.97, 399.95, 400.10, 400.25, 400.30]  # Bounces up 0.30
        
        outcome = get_outcome(signal_price, future_prices, direction="DOWN")
        assert outcome == OutcomeLabel.BOUNCE
    
    def test_break_on_support_test(self):
        """Test BREAK classification when price breaks through support."""
        # Price at 400.00, breaks down significantly
        signal_price = 400.00
        future_prices = [399.95, 399.85, 399.70, 399.75]  # Breaks down 0.30
        
        outcome = get_outcome(signal_price, future_prices, direction="DOWN")
        assert outcome == OutcomeLabel.BREAK
    
    def test_chop_no_clear_direction(self):
        """Test CHOP classification when price consolidates."""
        # Price at 400.00, oscillates without clear resolution
        signal_price = 400.00
        future_prices = [400.05, 399.95, 400.08, 399.92, 400.03]
        
        outcome = get_outcome(signal_price, future_prices, direction="UP")
        assert outcome == OutcomeLabel.CHOP
    
    def test_chop_small_movements(self):
        """Test CHOP when movements are below thresholds."""
        signal_price = 400.00
        future_prices = [400.10, 400.05, 399.90, 400.02]  # Max move 0.10
        
        outcome = get_outcome(signal_price, future_prices, direction="UP")
        assert outcome == OutcomeLabel.CHOP
    
    def test_empty_future_prices(self):
        """Test handling of empty future prices list."""
        outcome = get_outcome(400.00, [], direction="UP")
        assert outcome == OutcomeLabel.UNDEFINED
    
    def test_invalid_direction(self):
        """Test handling of invalid direction."""
        outcome = get_outcome(400.00, [401.00], direction="INVALID")
        assert outcome == OutcomeLabel.UNDEFINED
    
    def test_custom_thresholds(self):
        """Test with custom bounce/break thresholds."""
        signal_price = 400.00
        future_prices = [400.02, 399.85, 399.90]  # Falls 0.15
        
        # Default threshold (0.20) - should be CHOP
        outcome_default = get_outcome(signal_price, future_prices, direction="UP")
        assert outcome_default == OutcomeLabel.CHOP
        
        # Lower threshold (0.10) - should be BOUNCE
        outcome_custom = get_outcome(
            signal_price, 
            future_prices, 
            direction="UP",
            bounce_threshold=0.10
        )
        assert outcome_custom == OutcomeLabel.BOUNCE
    
    def test_label_signal_with_future_data(self):
        """Test convenience wrapper that returns outcome + future price."""
        signal_price = 400.00
        future_prices = [400.05, 400.03, 399.75, 399.80]
        
        outcome, future_5min = label_signal_with_future_data(
            signal_price, 
            future_prices, 
            direction="UP"
        )
        
        assert outcome == OutcomeLabel.BOUNCE
        assert future_5min == 399.80  # Last price
    
    def test_infer_direction_from_current_price(self):
        """Test automatic direction inference from current price."""
        signal_price = 400.00
        future_prices = [400.10, 400.25]
        
        # Current price above signal = resistance test (UP)
        outcome, _ = label_signal_with_future_data(
            signal_price,
            future_prices,
            direction=None,
            current_price=400.05
        )
        assert outcome == OutcomeLabel.BREAK


# ========== Tests for Experiment Runner ==========

class TestExperimentRunner:
    """Test suite for statistical analysis and backtesting."""
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for testing."""
        signals = []
        
        # 10 PM_HIGH signals: 7 bounces, 2 breaks, 1 chop
        for i in range(7):
            signals.append(LevelSignalV1(
                event_id=f"pm_high_bounce_{i}",
                ts_event_ns=1700000000000000000 + i * 1000000000,
                level_price=400.00,
                level_kind=LevelKind.PM_HIGH,
                wall_ratio=2.5,
                gamma_exposure=1000.0,
                tape_velocity=10.0,
                outcome=OutcomeLabel.BOUNCE,
            ))
        
        for i in range(2):
            signals.append(LevelSignalV1(
                event_id=f"pm_high_break_{i}",
                ts_event_ns=1700000000000000000 + (i+7) * 1000000000,
                level_price=400.00,
                level_kind=LevelKind.PM_HIGH,
                wall_ratio=0.5,
                gamma_exposure=-500.0,
                tape_velocity=20.0,
                outcome=OutcomeLabel.BREAK,
            ))
        
        signals.append(LevelSignalV1(
            event_id="pm_high_chop_0",
            ts_event_ns=1700000000000000000 + 9 * 1000000000,
            level_price=400.00,
            level_kind=LevelKind.PM_HIGH,
            wall_ratio=1.0,
            gamma_exposure=0.0,
            tape_velocity=5.0,
            outcome=OutcomeLabel.CHOP,
        ))
        
        # 5 SMA_200 signals: 2 bounces, 3 breaks
        for i in range(2):
            signals.append(LevelSignalV1(
                event_id=f"sma_bounce_{i}",
                ts_event_ns=1700000000000000000 + (i+10) * 1000000000,
                level_price=399.50,
                level_kind=LevelKind.SMA_200,
                wall_ratio=1.8,
                gamma_exposure=500.0,
                tape_velocity=8.0,
                outcome=OutcomeLabel.BOUNCE,
            ))
        
        for i in range(3):
            signals.append(LevelSignalV1(
                event_id=f"sma_break_{i}",
                ts_event_ns=1700000000000000000 + (i+12) * 1000000000,
                level_price=399.50,
                level_kind=LevelKind.SMA_200,
                wall_ratio=0.3,
                gamma_exposure=-200.0,
                tape_velocity=15.0,
                outcome=OutcomeLabel.BREAK,
            ))
        
        return signals
    
    @pytest.fixture
    def first_15m_signals(self):
        """Create signals with time-of-day variation."""
        signals = []
        
        # First 15 minutes signals (higher bounce rate)
        for i in range(5):
            signals.append(LevelSignalV1(
                event_id=f"first_15m_{i}",
                ts_event_ns=1700000000000000000 + i * 1000000000,
                level_price=400.00,
                level_kind=LevelKind.PM_HIGH,
                is_first_15m=True,
                wall_ratio=2.0,
                outcome=OutcomeLabel.BOUNCE if i < 4 else OutcomeLabel.BREAK,
            ))
        
        # Rest of day signals (lower bounce rate)
        for i in range(5):
            signals.append(LevelSignalV1(
                event_id=f"rest_of_day_{i}",
                ts_event_ns=1700000000000000000 + (i+5) * 1000000000,
                level_price=400.00,
                level_kind=LevelKind.SMA_200,
                is_first_15m=False,
                wall_ratio=1.0,
                outcome=OutcomeLabel.BOUNCE if i < 2 else OutcomeLabel.BREAK,
            ))
        
        return signals
    
    def test_runner_initialization(self):
        """Test ExperimentRunner initialization."""
        runner = ExperimentRunner()
        assert runner.signals == []
        
        signals = [LevelSignalV1(
            event_id="test",
            ts_event_ns=1700000000000000000,
            level_price=400.00,
            level_kind=LevelKind.STRIKE,
        )]
        runner = ExperimentRunner(signals=signals)
        assert len(runner.signals) == 1
    
    def test_simple_backtest(self, sample_signals):
        """Test simple backtest by level kind."""
        runner = ExperimentRunner(signals=sample_signals)
        results = runner.run_simple_backtest(print_report=False)
        
        # Check PM_HIGH stats (7 bounces, 2 breaks, 1 chop out of 10)
        pm_high = results["PM_HIGH"]
        assert pm_high["count"] == 10
        assert pm_high["bounce_rate"] == 0.7
        assert pm_high["break_rate"] == 0.2
        assert pm_high["chop_rate"] == 0.1
        
        # Check SMA_200 stats (2 bounces, 3 breaks out of 5)
        sma_200 = results["SMA_200"]
        assert sma_200["count"] == 5
        assert sma_200["bounce_rate"] == 0.4
        assert sma_200["break_rate"] == 0.6
    
    def test_simple_backtest_empty_signals(self):
        """Test backtest with no signals."""
        runner = ExperimentRunner()
        results = runner.run_simple_backtest(print_report=False)
        assert results == {}
    
    def test_physics_correlation(self, sample_signals):
        """Test correlation between physics metrics and outcomes."""
        runner = ExperimentRunner(signals=sample_signals)
        results = runner.run_physics_correlation(print_report=False)
        
        # Should have calculated correlations
        assert "wall_ratio_correlation" in results
        assert "gamma_exposure_correlation" in results
        assert "tape_velocity_correlation" in results
        
        # Higher wall_ratio should correlate with more bounces (positive correlation)
        assert results["wall_ratio_correlation"] > 0
        
        # Should have sample size info
        assert results["sample_size"] > 0
        assert results["bounce_count"] > 0
        assert results["break_count"] > 0
    
    def test_physics_correlation_filters_chop(self, sample_signals):
        """Test that correlation analysis excludes CHOP outcomes."""
        runner = ExperimentRunner(signals=sample_signals)
        results = runner.run_physics_correlation(print_report=False)
        
        # Should exclude the 1 CHOP signal from PM_HIGH
        # Total: 15 signals, 1 CHOP = 14 resolved
        assert results["sample_size"] == 14
    
    def test_physics_correlation_insufficient_data(self):
        """Test correlation with too few signals."""
        signals = [LevelSignalV1(
            event_id="single",
            ts_event_ns=1700000000000000000,
            level_price=400.00,
            level_kind=LevelKind.STRIKE,
            outcome=OutcomeLabel.BOUNCE,
        )]
        
        runner = ExperimentRunner(signals=signals)
        results = runner.run_physics_correlation(print_report=False)
        assert results == {}
    
    def test_time_based_analysis(self, first_15m_signals):
        """Test time-of-day analysis."""
        runner = ExperimentRunner(signals=first_15m_signals)
        results = runner.run_time_based_analysis(print_report=False)
        
        # First 15m should have higher bounce rate (4/5 = 80%)
        first_15m = results["first_15m"]
        assert first_15m["count"] == 5
        assert first_15m["bounce_rate"] == 0.8
        
        # Rest of day should have lower bounce rate (2/5 = 40%)
        rest = results["rest_of_day"]
        assert rest["count"] == 5
        assert rest["bounce_rate"] == 0.4
    
    def test_time_based_analysis_empty_signals(self):
        """Test time analysis with no signals."""
        runner = ExperimentRunner()
        results = runner.run_time_based_analysis(print_report=False)
        assert results == {}
    
    def test_pearson_correlation_calculation(self):
        """Test Pearson correlation helper method."""
        runner = ExperimentRunner()
        
        # Perfect positive correlation
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        corr = runner._pearson_correlation(x, y)
        assert abs(corr - 1.0) < 0.001
        
        # Perfect negative correlation
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        corr = runner._pearson_correlation(x, y)
        assert abs(corr - (-1.0)) < 0.001
        
        # No correlation
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 2, 5]
        corr = runner._pearson_correlation(x, y)
        assert abs(corr) <= 0.6  # Should be relatively low
    
    def test_pearson_correlation_edge_cases(self):
        """Test correlation edge cases."""
        runner = ExperimentRunner()
        
        # Insufficient data
        assert runner._pearson_correlation([1], [2]) == 0.0
        assert runner._pearson_correlation([], []) == 0.0
        
        # Mismatched lengths
        assert runner._pearson_correlation([1, 2], [1]) == 0.0
        
        # Zero variance
        assert runner._pearson_correlation([1, 1, 1], [2, 3, 4]) == 0.0


# ========== Integration Tests ==========

class TestResearchIntegration:
    """Integration tests combining labeler and experiment runner."""
    
    def test_end_to_end_research_workflow(self):
        """Test complete workflow from labeling to analysis."""
        # Step 1: Create raw signals without outcomes
        raw_signals = []
        for i in range(5):
            raw_signals.append(LevelSignalV1(
                event_id=f"signal_{i}",
                ts_event_ns=1700000000000000000 + i * 1000000000,
                level_price=400.00,
                level_kind=LevelKind.PM_HIGH,
                wall_ratio=2.0 if i < 3 else 0.5,  # First 3 have high wall ratio
            ))
        
        # Step 2: Label with outcomes
        future_prices_bounce = [400.05, 400.03, 399.75]  # Bounce pattern
        future_prices_break = [400.10, 400.25, 400.35]   # Break pattern
        
        labeled_signals = []
        for i, signal in enumerate(raw_signals):
            # First 3 bounce, last 2 break
            future_prices = future_prices_bounce if i < 3 else future_prices_break
            outcome = get_outcome(signal.level_price, future_prices, direction="UP")
            
            # Create new signal with outcome
            labeled_signal = signal.model_copy(update={"outcome": outcome})
            labeled_signals.append(labeled_signal)
        
        # Step 3: Run analysis
        runner = ExperimentRunner(signals=labeled_signals)
        
        backtest_results = runner.run_simple_backtest(print_report=False)
        assert backtest_results["PM_HIGH"]["bounce_rate"] == 0.6  # 3/5
        
        correlation_results = runner.run_physics_correlation(print_report=False)
        # Higher wall_ratio should correlate with bounces
        assert correlation_results["wall_ratio_correlation"] > 0
    
    def test_signals_with_all_fields_populated(self):
        """Test with fully populated signal objects."""
        signal = LevelSignalV1(
            event_id="complete_signal",
            ts_event_ns=1700000000000000000,
            symbol="SPY",
            spot=400.05,
            bid=400.04,
            ask=400.06,
            level_price=400.00,
            level_kind=LevelKind.STRIKE,
            level_id="STRIKE_400",
            direction=Direction.RESISTANCE,
            distance=0.05,
            is_first_15m=True,
            dist_to_sma_200=2.50,
            wall_ratio=2.5,
            replenishment_speed_ms=45.0,
            gamma_exposure=1500.0,
            tape_velocity=12.0,
            outcome=OutcomeLabel.BOUNCE,
            future_price_5min=399.80,
        )
        
        runner = ExperimentRunner(signals=[signal])
        results = runner.run_simple_backtest(print_report=False)
        
        assert results["STRIKE"]["count"] == 1
        assert results["STRIKE"]["bounce_rate"] == 1.0

