"""
Unit tests for Fuel Engine (Agent E deliverable)

Tests dealer gamma computation, wall identification, and effect classification
per PLAN.md §5.3.
"""

import pytest
from src.core.fuel_engine import FuelEngine, FuelEffect, GammaWall, FuelMetrics
from src.core.market_state import MarketState, OptionFlowAggregate
from src.common.config import Config


@pytest.fixture
def config():
    """Test configuration"""
    return Config(
        W_g=60.0,
        FUEL_STRIKE_RANGE=2.0,
        W_wall=300.0
    )


@pytest.fixture
def market_state():
    """Fresh market state for each test"""
    return MarketState(max_buffer_window_seconds=120.0)


@pytest.fixture
def fuel_engine(config):
    """Fuel engine instance"""
    return FuelEngine(config=config)


class TestBasicGammaComputation:
    """Test basic dealer gamma computation"""
    
    def test_amplify_effect_dealers_short_gamma(self, fuel_engine, market_state):
        """
        Test AMPLIFY classification when dealers are short gamma.
        
        Scenario: Customers bought calls at strike 680
        → Dealers sold calls
        → Dealers SHORT gamma (negative net_gamma_flow)
        → Effect should be AMPLIFY (dealers will chase moves)
        """
        # Setup: customers bought calls, dealers are short gamma
        market_state.option_flows[(680.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=680.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-50000.0,  # Negative = dealers SHORT gamma
            cumulative_volume=500
        )
        
        # Compute fuel state near level 680
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Assertions
        assert metrics.effect == FuelEffect.AMPLIFY
        assert metrics.net_dealer_gamma < 0
        assert metrics.confidence > 0
        print(f"✅ AMPLIFY: net_dealer_gamma={metrics.net_dealer_gamma}")
    
    def test_dampen_effect_dealers_long_gamma(self, fuel_engine, market_state):
        """
        Test DAMPEN classification when dealers are long gamma.
        
        Scenario: Customers sold calls at strike 680
        → Dealers bought calls
        → Dealers LONG gamma (positive net_gamma_flow)
        → Effect should be DAMPEN (dealers will fade moves)
        """
        # Setup: customers sold calls, dealers are long gamma
        market_state.option_flows[(680.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=680.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=+50000.0,  # Positive = dealers LONG gamma
            cumulative_volume=500
        )
        
        # Compute fuel state near level 680
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Assertions
        assert metrics.effect == FuelEffect.DAMPEN
        assert metrics.net_dealer_gamma > 0
        assert metrics.confidence > 0
        print(f"✅ DAMPEN: net_dealer_gamma={metrics.net_dealer_gamma}")
    
    def test_neutral_effect_minimal_gamma(self, fuel_engine, market_state):
        """
        Test NEUTRAL classification when gamma exposure is minimal.
        """
        # Setup: small gamma flow (below threshold)
        market_state.option_flows[(680.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=680.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=5000.0,  # Below threshold (10000)
            cumulative_volume=50
        )
        
        # Compute fuel state
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Assertions
        assert metrics.effect == FuelEffect.NEUTRAL
        assert abs(metrics.net_dealer_gamma) < 10000
        print(f"✅ NEUTRAL: net_dealer_gamma={metrics.net_dealer_gamma}")
    
    def test_no_option_flows_returns_neutral(self, fuel_engine, market_state):
        """Test that no option flows returns neutral metrics."""
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        assert metrics.effect == FuelEffect.NEUTRAL
        assert metrics.net_dealer_gamma == 0.0
        assert metrics.confidence == 0.0
        assert metrics.call_wall is None
        assert metrics.put_wall is None
        print("✅ No flows → NEUTRAL")


class TestStrikeRangeAggregation:
    """Test that fuel engine correctly aggregates gamma across strike range"""
    
    def test_aggregate_multiple_strikes_near_level(self, fuel_engine, market_state):
        """
        Test that engine sums gamma from multiple strikes near level.
        
        Setup: strikes 679, 680, 681 all have gamma flows
        Level: 680
        Strike range: ±2 (config default)
        Expected: should sum all three strikes
        """
        # Setup multiple strikes near level 680
        market_state.option_flows[(679.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=679.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-10000.0,
            cumulative_volume=100
        )
        market_state.option_flows[(680.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=680.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-20000.0,
            cumulative_volume=200
        )
        market_state.option_flows[(681.0, 'P', '2025-12-16')] = OptionFlowAggregate(
            strike=681.0,
            right='P',
            exp_date='2025-12-16',
            net_gamma_flow=-15000.0,
            cumulative_volume=150
        )
        
        # Compute fuel state
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Should sum all three: -10k + -20k + -15k = -45k
        expected_gamma = -45000.0
        assert metrics.net_dealer_gamma == expected_gamma
        assert metrics.effect == FuelEffect.AMPLIFY
        print(f"✅ Aggregated gamma: {metrics.net_dealer_gamma}")
    
    def test_exclude_strikes_outside_range(self, fuel_engine, market_state):
        """
        Test that strikes outside FUEL_STRIKE_RANGE are excluded.
        
        Strike range: ±2.0 (config default)
        Level: 680
        Valid range: 678-682
        """
        # Setup strikes: one inside, one outside range
        market_state.option_flows[(679.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=679.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-20000.0,
            cumulative_volume=200
        )
        market_state.option_flows[(685.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=685.0,  # Outside range (680 ± 2)
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-50000.0,  # Should be ignored
            cumulative_volume=500
        )
        
        # Compute fuel state
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Should only count 679 strike
        assert metrics.net_dealer_gamma == -20000.0
        print(f"✅ Excluded out-of-range strike: gamma={metrics.net_dealer_gamma}")


class TestWallIdentification:
    """Test call/put wall identification"""
    
    def test_call_wall_identification(self, fuel_engine, market_state):
        """
        Test call wall identification: strike with max POSITIVE gamma (dealers long).
        
        Call wall = resistance (dealers hedge by selling as price rises)
        """
        # Setup: calls at multiple strikes above spot
        market_state.option_flows[(681.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=681.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=+30000.0,  # Moderate positive
            cumulative_volume=300
        )
        market_state.option_flows[(682.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=682.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=+80000.0,  # Strong positive = WALL
            cumulative_volume=800
        )
        market_state.option_flows[(683.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=683.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=+10000.0,  # Weak positive
            cumulative_volume=100
        )
        
        # Compute fuel state
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Should identify 682 as call wall
        assert metrics.call_wall is not None
        assert metrics.call_wall.strike == 682.0
        assert metrics.call_wall.wall_type == 'CALL'
        assert metrics.call_wall.net_gamma > 0
        print(f"✅ Call wall at {metrics.call_wall.strike} with gamma {metrics.call_wall.net_gamma}")
    
    def test_put_wall_identification(self, fuel_engine, market_state):
        """
        Test put wall identification: strike with max POSITIVE gamma (dealers long).
        
        Put wall = support (dealers hedge by buying as price falls)
        """
        # Setup: puts at multiple strikes below spot
        market_state.option_flows[(679.0, 'P', '2025-12-16')] = OptionFlowAggregate(
            strike=679.0,
            right='P',
            exp_date='2025-12-16',
            net_gamma_flow=+20000.0,  # Moderate positive
            cumulative_volume=200
        )
        market_state.option_flows[(678.0, 'P', '2025-12-16')] = OptionFlowAggregate(
            strike=678.0,
            right='P',
            exp_date='2025-12-16',
            net_gamma_flow=+70000.0,  # Strong positive = WALL
            cumulative_volume=700
        )
        market_state.option_flows[(677.0, 'P', '2025-12-16')] = OptionFlowAggregate(
            strike=677.0,
            right='P',
            exp_date='2025-12-16',
            net_gamma_flow=+5000.0,  # Weak positive
            cumulative_volume=50
        )
        
        # Compute fuel state
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Should identify 678 as put wall
        assert metrics.put_wall is not None
        assert metrics.put_wall.strike == 678.0
        assert metrics.put_wall.wall_type == 'PUT'
        assert metrics.put_wall.net_gamma > 0
        print(f"✅ Put wall at {metrics.put_wall.strike} with gamma {metrics.put_wall.net_gamma}")
    
    def test_no_wall_when_below_threshold(self, fuel_engine, market_state):
        """Test that weak gamma concentrations don't create walls."""
        # Setup: small gamma flows (below WALL_STRENGTH_THRESHOLD = 50k)
        market_state.option_flows[(681.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=681.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=+20000.0,  # Below threshold
            cumulative_volume=200
        )
        
        # Compute fuel state
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Should not identify wall
        assert metrics.call_wall is None
        print("✅ No wall when gamma below threshold")


class TestHVLEstimation:
    """Test High Volatility Line (gamma flip) estimation"""
    
    def test_hvl_gamma_flip_detection(self, fuel_engine, market_state):
        """
        Test HVL detection when gamma flips sign across strikes.
        
        Setup: gamma negative below 680, positive above 680
        Expected: HVL near 680
        """
        # Below HVL: negative gamma (dealers short)
        market_state.option_flows[(678.0, 'P', '2025-12-16')] = OptionFlowAggregate(
            strike=678.0,
            right='P',
            exp_date='2025-12-16',
            net_gamma_flow=-30000.0,
            cumulative_volume=300
        )
        market_state.option_flows[(679.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=679.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-10000.0,
            cumulative_volume=100
        )
        # Above HVL: positive gamma (dealers long)
        market_state.option_flows[(681.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=681.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=+20000.0,
            cumulative_volume=200
        )
        market_state.option_flows[(682.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=682.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=+40000.0,
            cumulative_volume=400
        )
        
        # Compute fuel state
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Should detect HVL near 680
        assert metrics.hvl is not None
        assert 679.0 <= metrics.hvl <= 681.0
        print(f"✅ HVL detected at {metrics.hvl}")
    
    def test_no_hvl_when_gamma_single_sign(self, fuel_engine, market_state):
        """Test that HVL is None when gamma doesn't flip sign."""
        # All negative gamma (no flip)
        market_state.option_flows[(678.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=678.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-30000.0,
            cumulative_volume=300
        )
        market_state.option_flows[(680.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=680.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-20000.0,
            cumulative_volume=200
        )
        market_state.option_flows[(682.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=682.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-10000.0,
            cumulative_volume=100
        )
        
        # Compute fuel state
        metrics = fuel_engine.compute_fuel_state(
            level_price=680.0,
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Should not detect HVL
        assert metrics.hvl is None
        print("✅ No HVL when gamma doesn't flip")


class TestConfidenceScaling:
    """Test confidence scaling based on flow activity"""
    
    def test_confidence_scales_with_volume(self, fuel_engine, market_state):
        """Test that confidence increases with cumulative volume."""
        # Low volume case
        market_state_low = MarketState()
        market_state_low.option_flows[(680.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=680.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-20000.0,
            cumulative_volume=50  # Low volume
        )
        metrics_low = fuel_engine.compute_fuel_state(680.0, market_state_low, '2025-12-16')
        
        # High volume case
        market_state_high = MarketState()
        market_state_high.option_flows[(680.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=680.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=-20000.0,
            cumulative_volume=2000  # High volume
        )
        metrics_high = fuel_engine.compute_fuel_state(680.0, market_state_high, '2025-12-16')
        
        # High volume should have higher confidence
        assert metrics_high.confidence > metrics_low.confidence
        assert metrics_high.confidence >= 1.0  # Capped at 1.0
        print(f"✅ Confidence scaling: low={metrics_low.confidence:.2f}, high={metrics_high.confidence:.2f}")


class TestGlobalWallRetrieval:
    """Test get_all_walls helper for level universe"""
    
    def test_get_all_walls_across_strikes(self, fuel_engine, market_state):
        """Test retrieval of global call/put walls across entire range."""
        # Setup market state with spot price using ES trade (SPY 680 = ES 6800)
        from src.common.event_types import FuturesTrade, EventSource, Aggressor
        trade = FuturesTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol='ES',
            price=6800.0,  # ES price = SPY * 10
            size=1,
            aggressor=Aggressor.BUY
        )
        market_state.update_es_trade(trade)
        
        # Setup strong call wall at 685
        market_state.option_flows[(685.0, 'C', '2025-12-16')] = OptionFlowAggregate(
            strike=685.0,
            right='C',
            exp_date='2025-12-16',
            net_gamma_flow=+100000.0,
            cumulative_volume=1000
        )
        
        # Setup strong put wall at 675
        market_state.option_flows[(675.0, 'P', '2025-12-16')] = OptionFlowAggregate(
            strike=675.0,
            right='P',
            exp_date='2025-12-16',
            net_gamma_flow=+90000.0,
            cumulative_volume=900
        )
        
        # Get global walls
        call_wall, put_wall = fuel_engine.get_all_walls(
            market_state=market_state,
            exp_date_filter='2025-12-16'
        )
        
        # Assertions
        assert call_wall is not None
        assert call_wall.strike == 685.0
        assert put_wall is not None
        assert put_wall.strike == 675.0
        print(f"✅ Global walls: call={call_wall.strike}, put={put_wall.strike}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

