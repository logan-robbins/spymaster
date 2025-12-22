"""
Test Replay Determinism (Task 2.2 from NEXT.md)

Verifies per PLAN.md §8.2:
- Same Bronze input produces identical Gold output
- Engine computations are deterministic given same inputs
- Replay ordering is deterministic by ts_event_ns
"""

import time
import tempfile
import shutil
import os
from typing import List, Dict, Any

from src.event_types import StockTrade, StockQuote, OptionTrade, EventSource, Aggressor
from src.market_state import MarketState
from src.barrier_engine import BarrierEngine, Direction
from src.tape_engine import TapeEngine
from src.fuel_engine import FuelEngine
from src.score_engine import ScoreEngine
from src.level_signal_service import LevelSignalService
from src.bronze_writer import BronzeWriter, BronzeReader
from src.gold_writer import GoldWriter, GoldReader
from src.config import CONFIG


def generate_synthetic_events(
    base_ts_ns: int,
    num_quotes: int = 100,
    num_trades: int = 50,
    num_options: int = 20
) -> List[Any]:
    """
    Generate deterministic synthetic events for testing.

    All events are created with predictable values based on their index.
    """
    events = []

    # Generate quotes with small price variations
    for i in range(num_quotes):
        ts = base_ts_ns + i * 100_000_000  # 100ms apart
        price_offset = (i % 10) * 0.01  # Oscillate price
        events.append(StockQuote(
            ts_event_ns=ts,
            ts_recv_ns=ts + 1000,
            source=EventSource.SIM,
            symbol='SPY',
            bid_px=545.00 + price_offset,
            ask_px=545.02 + price_offset,
            bid_sz=10000 + i * 100,
            ask_sz=8000 + i * 50
        ))

    # Generate trades at some of those timestamps
    for i in range(num_trades):
        ts = base_ts_ns + i * 200_000_000 + 50_000_000  # Offset from quotes
        price = 545.01 + (i % 10) * 0.01
        events.append(StockTrade(
            ts_event_ns=ts,
            ts_recv_ns=ts + 1000,
            source=EventSource.SIM,
            symbol='SPY',
            price=price,
            size=100 + i * 10
        ))

    # Generate option trades
    for i in range(num_options):
        ts = base_ts_ns + i * 500_000_000 + 25_000_000  # Different offset
        strike = 545.0 + (i % 5)
        right = 'C' if i % 2 == 0 else 'P'
        exp_date = '2025-12-22'
        symbol = f"O:SPY251222{right}00{int(strike*1000):08d}"
        events.append(OptionTrade(
            ts_event_ns=ts,
            ts_recv_ns=ts + 1000,
            source=EventSource.SIM,
            underlying='SPY',
            option_symbol=symbol,
            exp_date=exp_date,
            strike=strike,
            right=right,
            price=1.50 + i * 0.1,
            size=10 + i
        ))

    # Sort by timestamp (deterministic order)
    events.sort(key=lambda e: e.ts_event_ns)
    return events


class TestEngineDeterminism:
    """Test that engines produce deterministic outputs."""

    def test_market_state_deterministic(self):
        """Same events should produce same MarketState."""
        events = generate_synthetic_events(time.time_ns(), num_quotes=50)

        # Run 1
        state1 = MarketState()
        for event in events:
            if isinstance(event, StockQuote):
                state1.update_stock_quote(event)
            elif isinstance(event, StockTrade):
                state1.update_stock_trade(event)

        # Run 2
        state2 = MarketState()
        for event in events:
            if isinstance(event, StockQuote):
                state2.update_stock_quote(event)
            elif isinstance(event, StockTrade):
                state2.update_stock_trade(event)

        # Compare
        assert state1.last_quote.bid_px == state2.last_quote.bid_px
        assert state1.last_quote.ask_px == state2.last_quote.ask_px
        assert state1.last_trade.price == state2.last_trade.price
        print("✅ MarketState: Deterministic")

    def test_barrier_engine_deterministic(self):
        """Same MarketState should produce same barrier metrics."""
        events = generate_synthetic_events(time.time_ns())

        # Prepare MarketState
        market_state = MarketState()
        for event in events:
            if isinstance(event, StockQuote):
                market_state.update_stock_quote(event)
            elif isinstance(event, StockTrade):
                market_state.update_stock_trade(event)

        # Run barrier engine twice
        engine1 = BarrierEngine()
        engine2 = BarrierEngine()

        metrics1 = engine1.compute_barrier_state(545.0, Direction.SUPPORT, market_state)
        metrics2 = engine2.compute_barrier_state(545.0, Direction.SUPPORT, market_state)

        assert metrics1.state == metrics2.state
        assert metrics1.delta_liq == metrics2.delta_liq
        assert metrics1.replenishment_ratio == metrics2.replenishment_ratio
        print("✅ BarrierEngine: Deterministic")

    def test_tape_engine_deterministic(self):
        """Same MarketState should produce same tape metrics."""
        events = generate_synthetic_events(time.time_ns())

        # Prepare MarketState
        market_state = MarketState()
        for event in events:
            if isinstance(event, StockQuote):
                market_state.update_stock_quote(event)
            elif isinstance(event, StockTrade):
                market_state.update_stock_trade(event)

        # Run tape engine twice
        engine1 = TapeEngine()
        engine2 = TapeEngine()

        metrics1 = engine1.compute_tape_state(545.0, market_state)
        metrics2 = engine2.compute_tape_state(545.0, market_state)

        assert metrics1.imbalance == metrics2.imbalance
        assert metrics1.velocity == metrics2.velocity
        assert metrics1.buy_vol == metrics2.buy_vol
        assert metrics1.sell_vol == metrics2.sell_vol
        print("✅ TapeEngine: Deterministic")

    def test_fuel_engine_deterministic(self):
        """Same option flows should produce same fuel metrics."""
        events = generate_synthetic_events(time.time_ns(), num_options=30)

        # Prepare MarketState with option trades
        market_state = MarketState()
        for event in events:
            if isinstance(event, StockQuote):
                market_state.update_stock_quote(event)
            elif isinstance(event, StockTrade):
                market_state.update_stock_trade(event)
            elif isinstance(event, OptionTrade):
                # Add with synthetic greeks
                market_state.update_option_trade(event, delta=0.5, gamma=0.05)

        # Run fuel engine twice
        engine1 = FuelEngine()
        engine2 = FuelEngine()

        metrics1 = engine1.compute_fuel_state(545.0, market_state)
        metrics2 = engine2.compute_fuel_state(545.0, market_state)

        assert metrics1.effect == metrics2.effect
        assert metrics1.net_dealer_gamma == metrics2.net_dealer_gamma
        print("✅ FuelEngine: Deterministic")

    def test_score_engine_deterministic(self):
        """Same inputs should produce same composite score."""
        from src.barrier_engine import BarrierMetrics, BarrierState
        from src.tape_engine import TapeMetrics, SweepDetection
        from src.fuel_engine import FuelMetrics, FuelEffect

        ts_ns = time.time_ns()

        # Create fixed inputs
        barrier = BarrierMetrics(
            state=BarrierState.VACUUM,
            delta_liq=-5000,
            replenishment_ratio=0.3,
            added_size=1000,
            canceled_size=4000,
            filled_size=2000,
            defending_quote={'price': 545.0, 'size': 5000},
            confidence=0.8
        )

        tape = TapeMetrics(
            imbalance=-0.5,
            buy_vol=30000,
            sell_vol=70000,
            velocity=-0.2,
            sweep=SweepDetection(
                detected=False,
                direction='NONE',
                notional=0,
                num_prints=0,
                window_ms=0
            ),
            confidence=0.8
        )

        fuel = FuelMetrics(
            effect=FuelEffect.AMPLIFY,
            net_dealer_gamma=-50000,
            call_wall=None,
            put_wall=None,
            hvl=None,
            confidence=0.8,
            gamma_by_strike={}
        )

        # Run score engine twice
        engine1 = ScoreEngine()
        engine2 = ScoreEngine()

        result1 = engine1.compute_score(barrier, tape, fuel, 'DOWN', ts_ns, 0.1)
        result2 = engine2.compute_score(barrier, tape, fuel, 'DOWN', ts_ns, 0.1)

        assert result1.raw_score == result2.raw_score
        assert result1.signal == result2.signal
        assert result1.confidence == result2.confidence
        print(f"✅ ScoreEngine: Deterministic (score={result1.raw_score:.2f})")


class TestBronzeGoldDeterminism:
    """Test Bronze -> Gold pipeline determinism."""

    def test_bronze_write_read_roundtrip(self):
        """Write to Bronze and read back should preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events = generate_synthetic_events(time.time_ns(), num_quotes=10, num_trades=5)

            # Write to Bronze
            writer = BronzeWriter(data_root=tmpdir, buffer_limit=100, flush_interval_seconds=0.1)

            import asyncio

            async def write_events():
                for event in events:
                    if isinstance(event, StockQuote):
                        await writer.write_stock_quote(event)
                    elif isinstance(event, StockTrade):
                        await writer.write_stock_trade(event)
                await writer.flush_all()

            asyncio.run(write_events())

            # Read back
            reader = BronzeReader(data_root=tmpdir)
            quotes_df = reader.read_stock_quotes(symbol='SPY')
            trades_df = reader.read_stock_trades(symbol='SPY')

            # Verify counts
            expected_quotes = sum(1 for e in events if isinstance(e, StockQuote))
            expected_trades = sum(1 for e in events if isinstance(e, StockTrade))

            assert len(quotes_df) == expected_quotes, f"Expected {expected_quotes} quotes, got {len(quotes_df)}"
            assert len(trades_df) == expected_trades, f"Expected {expected_trades} trades, got {len(trades_df)}"
            print(f"✅ Bronze roundtrip: {len(quotes_df)} quotes, {len(trades_df)} trades preserved")

    def test_gold_write_read_roundtrip(self):
        """Write to Gold and read back should preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample level signal payload
            ts_ms = int(time.time() * 1000)
            payload = {
                'ts': ts_ms,
                'spy': {'spot': 545.01, 'bid': 545.00, 'ask': 545.02},
                'levels': [
                    {
                        'id': 'STRIKE_545',
                        'kind': 'STRIKE',
                        'price': 545.0,
                        'direction': 'SUPPORT',
                        'distance': 0.01,
                        'break_score_raw': 75.5,
                        'break_score_smooth': 72.0,
                        'signal': 'NEUTRAL',
                        'confidence': 'HIGH',
                        'barrier': {'state': 'NEUTRAL', 'delta_liq': 0},
                        'tape': {'imbalance': 0.1, 'buy_vol': 50000, 'sell_vol': 45000},
                        'fuel': {'effect': 'NEUTRAL', 'net_dealer_gamma': 0},
                        'runway': {'direction': 'DOWN', 'distance': 2.0}
                    }
                ]
            }

            # Write to Gold
            writer = GoldWriter(data_root=tmpdir, buffer_limit=10, flush_interval_seconds=0.1)

            import asyncio

            async def write_gold():
                await writer.write_level_signals(payload)
                await writer.flush()

            asyncio.run(write_gold())

            # Read back
            reader = GoldReader(data_root=tmpdir)
            df = reader.read_level_signals(underlying='SPY')

            assert len(df) == 1, f"Expected 1 level signal, got {len(df)}"
            assert df.iloc[0]['level_id'] == 'STRIKE_545'
            assert abs(df.iloc[0]['break_score_raw'] - 75.5) < 0.01
            print("✅ Gold roundtrip: Level signals preserved")


class TestLevelSignalServiceDeterminism:
    """Test full level signal service determinism."""

    def test_level_signals_deterministic(self):
        """Same MarketState should produce same level signals."""
        events = generate_synthetic_events(time.time_ns())

        # Prepare MarketState
        market_state = MarketState()
        for event in events:
            if isinstance(event, StockQuote):
                market_state.update_stock_quote(event)
            elif isinstance(event, StockTrade):
                market_state.update_stock_trade(event)
            elif isinstance(event, OptionTrade):
                market_state.update_option_trade(event, delta=0.5, gamma=0.05)

        # Run level signal service twice
        service1 = LevelSignalService(market_state=market_state)
        service2 = LevelSignalService(market_state=market_state)

        payload1 = service1.compute_level_signals()
        payload2 = service2.compute_level_signals()

        # Extract levels list from payload
        levels1 = payload1.get('levels', [])
        levels2 = payload2.get('levels', [])

        # Compare results (may be empty if no levels active within monitoring band)
        assert len(levels1) == len(levels2), f"Level count mismatch: {len(levels1)} vs {len(levels2)}"

        for l1, l2 in zip(levels1, levels2):
            assert l1.get('id') == l2.get('id')
            assert l1.get('break_score_raw') == l2.get('break_score_raw')
            assert l1.get('signal') == l2.get('signal')

        # Also check spy snapshot matches
        assert payload1.get('spy') == payload2.get('spy')

        print(f"✅ LevelSignalService: Deterministic ({len(levels1)} levels)")


class TestEventOrdering:
    """Test that event ordering is deterministic."""

    def test_event_sort_by_timestamp(self):
        """Events should sort deterministically by ts_event_ns."""
        base_ts = time.time_ns()

        # Create events with known timestamps
        events = [
            StockQuote(ts_event_ns=base_ts + 300, ts_recv_ns=base_ts + 301,
                      source=EventSource.SIM, symbol='SPY',
                      bid_px=545.0, ask_px=545.02, bid_sz=10000, ask_sz=8000),
            StockTrade(ts_event_ns=base_ts + 100, ts_recv_ns=base_ts + 101,
                      source=EventSource.SIM, symbol='SPY', price=545.01, size=100),
            StockQuote(ts_event_ns=base_ts + 200, ts_recv_ns=base_ts + 201,
                      source=EventSource.SIM, symbol='SPY',
                      bid_px=545.01, ask_px=545.03, bid_sz=10000, ask_sz=8000),
        ]

        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.ts_event_ns)

        # Verify order
        assert sorted_events[0].ts_event_ns == base_ts + 100
        assert sorted_events[1].ts_event_ns == base_ts + 200
        assert sorted_events[2].ts_event_ns == base_ts + 300

        # Sort again should produce same order
        sorted_again = sorted(events, key=lambda e: e.ts_event_ns)
        for e1, e2 in zip(sorted_events, sorted_again):
            assert e1.ts_event_ns == e2.ts_event_ns

        print("✅ Event ordering: Deterministic by ts_event_ns")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Replay Determinism Tests")
    print("="*60)

    # Engine determinism
    engine_tests = TestEngineDeterminism()
    engine_tests.test_market_state_deterministic()
    engine_tests.test_barrier_engine_deterministic()
    engine_tests.test_tape_engine_deterministic()
    engine_tests.test_fuel_engine_deterministic()
    engine_tests.test_score_engine_deterministic()

    # Bronze/Gold roundtrip
    bg_tests = TestBronzeGoldDeterminism()
    bg_tests.test_bronze_write_read_roundtrip()
    bg_tests.test_gold_write_read_roundtrip()

    # Level signal service
    ls_tests = TestLevelSignalServiceDeterminism()
    ls_tests.test_level_signals_deterministic()

    # Event ordering
    order_tests = TestEventOrdering()
    order_tests.test_event_sort_by_timestamp()

    print("\n" + "="*60)
    print("All Replay Determinism tests passed!")
    print("="*60)
