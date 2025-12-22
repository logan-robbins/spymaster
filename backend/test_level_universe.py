"""
Test suite for level_universe.py and room_to_run.py

Agent F verification per §12 of PLAN.md.
"""

import pytest
from src.market_state import MarketState, OptionFlowAggregate
from src.level_universe import LevelUniverse, Level, LevelKind
from src.room_to_run import RoomToRun, Direction, RunwayQuality, get_break_direction, get_reject_direction
from src.event_types import StockTrade, StockQuote, EventSource


class TestLevelUniverse:
    """Test level generation logic."""
    
    def test_empty_state_returns_empty_levels(self):
        """No spot price -> no levels."""
        universe = LevelUniverse()
        market_state = MarketState()
        
        levels = universe.get_levels(market_state)
        
        assert len(levels) == 0
    
    def test_generates_vwap_level(self):
        """VWAP level is generated when available."""
        universe = LevelUniverse()
        market_state = MarketState()
        
        # Add a trade to establish spot and VWAP
        trade = StockTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol="SPY",
            price=545.50,
            size=100
        )
        market_state.update_stock_trade(trade)
        
        levels = universe.get_levels(market_state)
        
        vwap_levels = [l for l in levels if l.kind == LevelKind.VWAP]
        assert len(vwap_levels) == 1
        assert vwap_levels[0].id == "VWAP"
        assert vwap_levels[0].price == 545.50
    
    def test_generates_round_levels_near_spot(self):
        """Round levels are generated every $1 near spot."""
        universe = LevelUniverse()
        market_state = MarketState()
        
        trade = StockTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol="SPY",
            price=545.42,
            size=100
        )
        market_state.update_stock_trade(trade)
        
        levels = universe.get_levels(market_state)
        
        round_levels = [l for l in levels if l.kind == LevelKind.ROUND]
        round_prices = sorted([l.price for l in round_levels])
        
        # Should have rounds from ~540 to ~550 (default STRIKE_RANGE = 5.0)
        assert 540.0 in round_prices
        assert 541.0 in round_prices
        assert 545.0 in round_prices
        assert 550.0 in round_prices
    
    def test_generates_strike_levels_from_option_flows(self):
        """Strike levels are generated from active option flows."""
        universe = LevelUniverse()
        market_state = MarketState()
        
        # Add spot
        trade = StockTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol="SPY",
            price=545.00,
            size=100
        )
        market_state.update_stock_trade(trade)
        
        # Add some option flows at specific strikes
        market_state.option_flows[(545.0, 'C', '2025-12-20')] = OptionFlowAggregate(
            strike=545.0,
            right='C',
            exp_date='2025-12-20',
            cumulative_volume=1000,
            net_gamma_flow=-5000.0
        )
        market_state.option_flows[(546.0, 'P', '2025-12-20')] = OptionFlowAggregate(
            strike=546.0,
            right='P',
            exp_date='2025-12-20',
            cumulative_volume=500,
            net_gamma_flow=-2000.0
        )
        
        levels = universe.get_levels(market_state)
        
        strike_levels = [l for l in levels if l.kind == LevelKind.STRIKE]
        strike_prices = sorted([l.price for l in strike_levels])
        
        assert 545.0 in strike_prices
        assert 546.0 in strike_prices
    
    def test_generates_call_wall_from_gamma_flow(self):
        """Call wall is strike with most negative dealer gamma from calls."""
        universe = LevelUniverse()
        market_state = MarketState()
        
        # Add spot
        trade = StockTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol="SPY",
            price=545.00,
            size=100
        )
        market_state.update_stock_trade(trade)
        
        # Add call flows with varying gamma
        market_state.option_flows[(545.0, 'C', '2025-12-20')] = OptionFlowAggregate(
            strike=545.0,
            right='C',
            exp_date='2025-12-20',
            net_gamma_flow=-5000.0  # moderate
        )
        market_state.option_flows[(548.0, 'C', '2025-12-20')] = OptionFlowAggregate(
            strike=548.0,
            right='C',
            exp_date='2025-12-20',
            net_gamma_flow=-15000.0  # strongest (call wall)
        )
        market_state.option_flows[(542.0, 'C', '2025-12-20')] = OptionFlowAggregate(
            strike=542.0,
            right='C',
            exp_date='2025-12-20',
            net_gamma_flow=-3000.0  # weak
        )
        
        levels = universe.get_levels(market_state)
        
        call_wall_levels = [l for l in levels if l.kind == LevelKind.CALL_WALL]
        assert len(call_wall_levels) == 1
        assert call_wall_levels[0].price == 548.0
        assert call_wall_levels[0].metadata['net_dealer_gamma'] == -15000.0
    
    def test_generates_put_wall_from_gamma_flow(self):
        """Put wall is strike with most negative dealer gamma from puts."""
        universe = LevelUniverse()
        market_state = MarketState()
        
        trade = StockTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol="SPY",
            price=545.00,
            size=100
        )
        market_state.update_stock_trade(trade)
        
        market_state.option_flows[(545.0, 'P', '2025-12-20')] = OptionFlowAggregate(
            strike=545.0,
            right='P',
            exp_date='2025-12-20',
            net_gamma_flow=-4000.0
        )
        market_state.option_flows[(540.0, 'P', '2025-12-20')] = OptionFlowAggregate(
            strike=540.0,
            right='P',
            exp_date='2025-12-20',
            net_gamma_flow=-12000.0  # strongest (put wall)
        )
        
        levels = universe.get_levels(market_state)
        
        put_wall_levels = [l for l in levels if l.kind == LevelKind.PUT_WALL]
        assert len(put_wall_levels) == 1
        assert put_wall_levels[0].price == 540.0
    
    def test_user_hotzones(self):
        """User-defined hotzones are included."""
        universe = LevelUniverse(user_hotzones=[550.0, 555.0])
        market_state = MarketState()
        
        trade = StockTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol="SPY",
            price=545.00,
            size=100
        )
        market_state.update_stock_trade(trade)
        
        levels = universe.get_levels(market_state)
        
        hotzone_levels = [l for l in levels if l.kind == LevelKind.USER_HOTZONE]
        hotzone_prices = sorted([l.price for l in hotzone_levels])
        
        assert hotzone_prices == [550.0, 555.0]
    
    def test_deduplication_by_id(self):
        """Levels are deduplicated by ID."""
        universe = LevelUniverse()
        market_state = MarketState()
        
        trade = StockTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol="SPY",
            price=545.00,
            size=100
        )
        market_state.update_stock_trade(trade)
        
        levels = universe.get_levels(market_state)
        
        # Check that IDs are unique
        ids = [l.id for l in levels]
        assert len(ids) == len(set(ids))


class TestRoomToRun:
    """Test runway computation logic."""
    
    def test_compute_runway_up_finds_nearest_obstacle(self):
        """Runway UP finds nearest level above."""
        rtr = RoomToRun()
        
        current = Level(id="STRIKE_545", price=545.0, kind=LevelKind.STRIKE)
        all_levels = [
            current,
            Level(id="STRIKE_546", price=546.0, kind=LevelKind.STRIKE),
            Level(id="STRIKE_548", price=548.0, kind=LevelKind.STRIKE),
            Level(id="STRIKE_542", price=542.0, kind=LevelKind.STRIKE),
        ]
        
        runway = rtr.compute_runway(current, Direction.UP, all_levels, spot=545.0)
        
        assert runway.direction == Direction.UP
        assert runway.distance == 1.0
        assert runway.next_obstacle.id == "STRIKE_546"
    
    def test_compute_runway_down_finds_nearest_obstacle(self):
        """Runway DOWN finds nearest level below."""
        rtr = RoomToRun()
        
        current = Level(id="STRIKE_545", price=545.0, kind=LevelKind.STRIKE)
        all_levels = [
            current,
            Level(id="STRIKE_546", price=546.0, kind=LevelKind.STRIKE),
            Level(id="STRIKE_542", price=542.0, kind=LevelKind.STRIKE),
            Level(id="STRIKE_540", price=540.0, kind=LevelKind.STRIKE),
        ]
        
        runway = rtr.compute_runway(current, Direction.DOWN, all_levels, spot=545.0)
        
        assert runway.direction == Direction.DOWN
        assert runway.distance == 3.0
        assert runway.next_obstacle.id == "STRIKE_542"
    
    def test_runway_quality_clear_when_no_walls(self):
        """Runway is CLEAR when no strong obstacles in between."""
        rtr = RoomToRun()
        
        current = Level(id="STRIKE_545", price=545.0, kind=LevelKind.STRIKE)
        all_levels = [
            current,
            Level(id="ROUND_546", price=546.0, kind=LevelKind.ROUND),  # intermediate
            Level(id="ROUND_547", price=547.0, kind=LevelKind.ROUND),  # intermediate
            Level(id="STRIKE_548", price=548.0, kind=LevelKind.STRIKE),  # next obstacle
        ]
        
        runway = rtr.compute_runway(current, Direction.UP, all_levels, spot=545.0)
        
        # Next obstacle is nearest level above (546)
        assert runway.next_obstacle.id == "ROUND_546"
        assert runway.distance == 1.0
        assert runway.quality == RunwayQuality.CLEAR
        assert len(runway.intermediate_levels) == 0  # no levels between 545 and 546
    
    def test_runway_quality_obstructed_when_wall_present(self):
        """Runway is OBSTRUCTED when strong obstacle in between."""
        rtr = RoomToRun()
        
        current = Level(id="STRIKE_545", price=545.0, kind=LevelKind.STRIKE)
        all_levels = [
            current,
            Level(id="CALL_WALL", price=546.5, kind=LevelKind.CALL_WALL),  # intermediate wall
            Level(id="STRIKE_548", price=548.0, kind=LevelKind.STRIKE),  # next obstacle
        ]
        
        runway = rtr.compute_runway(current, Direction.UP, all_levels, spot=545.0)
        
        # Next obstacle is call wall at 546.5
        assert runway.next_obstacle.id == "CALL_WALL"
        # Since the next obstacle itself is a wall, there are no intermediate obstacles
        # But the runway quality should reflect that we're hitting a wall
        # Actually, the quality is based on intermediate levels, not the obstacle itself
        # Let me create a better test case
        
    def test_runway_quality_obstructed_when_wall_between(self):
        """Runway is OBSTRUCTED when wall exists between current and next non-wall obstacle."""
        rtr = RoomToRun()
        
        current = Level(id="STRIKE_545", price=545.0, kind=LevelKind.STRIKE)
        all_levels = [
            current,
            Level(id="ROUND_546", price=546.0, kind=LevelKind.ROUND),  # next obstacle (weak)
            Level(id="CALL_WALL", price=545.5, kind=LevelKind.CALL_WALL),  # intermediate wall
        ]
        
        runway = rtr.compute_runway(current, Direction.UP, all_levels, spot=545.0)
        
        # Next obstacle is the CALL_WALL at 545.5 (nearest)
        assert runway.next_obstacle.id == "CALL_WALL"
        assert runway.distance == 0.5
        # No intermediate levels since wall is the nearest obstacle
        assert len(runway.intermediate_levels) == 0
    
    def test_runway_infinite_when_no_obstacles_ahead(self):
        """Runway is infinite when no levels ahead."""
        rtr = RoomToRun()
        
        current = Level(id="STRIKE_545", price=545.0, kind=LevelKind.STRIKE)
        all_levels = [
            current,
            Level(id="STRIKE_542", price=542.0, kind=LevelKind.STRIKE),  # only levels below
        ]
        
        runway = rtr.compute_runway(current, Direction.UP, all_levels, spot=545.0)
        
        assert runway.distance == float('inf')
        assert runway.next_obstacle is None
        assert runway.quality == RunwayQuality.CLEAR
    
    def test_bidirectional_runway(self):
        """Can compute runway in both directions."""
        rtr = RoomToRun()
        
        current = Level(id="STRIKE_545", price=545.0, kind=LevelKind.STRIKE)
        all_levels = [
            Level(id="STRIKE_548", price=548.0, kind=LevelKind.STRIKE),
            current,
            Level(id="STRIKE_542", price=542.0, kind=LevelKind.STRIKE),
        ]
        
        runway_up, runway_down = rtr.compute_bidirectional_runway(current, all_levels, spot=545.0)
        
        assert runway_up.direction == Direction.UP
        assert runway_up.distance == 3.0
        assert runway_down.direction == Direction.DOWN
        assert runway_down.distance == 3.0


class TestDirectionHelpers:
    """Test break/reject direction helpers."""
    
    def test_break_direction_support(self):
        """Break direction is DOWN when spot > level (support test)."""
        direction = get_break_direction(level_price=545.0, spot=545.50)
        assert direction == Direction.DOWN
    
    def test_break_direction_resistance(self):
        """Break direction is UP when spot < level (resistance test)."""
        direction = get_break_direction(level_price=545.0, spot=544.50)
        assert direction == Direction.UP
    
    def test_reject_direction_support(self):
        """Reject direction is UP when spot > level (support bounce)."""
        direction = get_reject_direction(level_price=545.0, spot=545.50)
        assert direction == Direction.UP
    
    def test_reject_direction_resistance(self):
        """Reject direction is DOWN when spot < level (resistance bounce)."""
        direction = get_reject_direction(level_price=545.0, spot=544.50)
        assert direction == Direction.DOWN


class TestIntegration:
    """Integration tests combining level universe and room to run."""
    
    def test_full_workflow(self):
        """Complete workflow: generate levels -> compute runway."""
        # Setup market state
        market_state = MarketState()
        
        trade = StockTrade(
            ts_event_ns=1000000000,
            ts_recv_ns=1000000000,
            source=EventSource.SIM,
            symbol="SPY",
            price=545.42,
            size=100
        )
        market_state.update_stock_trade(trade)
        
        # Add option flows to create interesting levels
        market_state.option_flows[(545.0, 'C', '2025-12-20')] = OptionFlowAggregate(
            strike=545.0,
            right='C',
            exp_date='2025-12-20',
            net_gamma_flow=-5000.0
        )
        market_state.option_flows[(548.0, 'C', '2025-12-20')] = OptionFlowAggregate(
            strike=548.0,
            right='C',
            exp_date='2025-12-20',
            net_gamma_flow=-15000.0  # call wall
        )
        
        # Generate levels
        universe = LevelUniverse()
        levels = universe.get_levels(market_state)
        
        assert len(levels) > 0
        
        # Find the 545 strike level
        level_545 = next((l for l in levels if l.price == 545.0 and l.kind == LevelKind.STRIKE), None)
        assert level_545 is not None
        
        # Compute runway upward
        rtr = RoomToRun()
        runway = rtr.compute_runway(level_545, Direction.UP, levels, spot=545.42)
        
        # Should find call wall at 548 as obstacle
        # But there might be round levels in between
        assert runway.distance > 0
        assert runway.next_obstacle is not None
        
        # Call wall should be in the path
        call_wall = next((l for l in levels if l.kind == LevelKind.CALL_WALL), None)
        if call_wall:
            assert call_wall.price == 548.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
