"""
Tests for TapeEngine using ES trades.

Verifies:
- Imbalance calculation (buy/sell ratio)
- Velocity calculation (price slope)
- Sweep detection (clustered aggressive prints)
- ES trade analysis and metrics calculation
"""

import pytest
import time as _time

from src.core.tape_engine import TapeEngine, TapeMetrics, SweepDetection
from src.core.market_state import MarketState
from src.common.event_types import FuturesTrade, MBP10, BidAskLevel, EventSource, Aggressor
from src.common.config import Config


def create_es_trade(ts_ns: int, price: float, size: int, aggressor: Aggressor) -> FuturesTrade:
    """Helper to create ES futures trade."""
    return FuturesTrade(
        ts_event_ns=ts_ns,
        ts_recv_ns=ts_ns,
        source=EventSource.SIM,
        symbol="ES",
        price=price,
        size=size,
        aggressor=aggressor
    )


def create_mbp10(ts_ns: int, bid_price: float, ask_price: float) -> MBP10:
    """Helper to create minimal MBP10 snapshot."""
    levels = [BidAskLevel(bid_px=bid_price, bid_sz=100, ask_px=ask_price, ask_sz=100)]
    for _ in range(9):
        levels.append(BidAskLevel(bid_px=0.0, bid_sz=0, ask_px=0.0, ask_sz=0))

    return MBP10(
        ts_event_ns=ts_ns,
        ts_recv_ns=ts_ns,
        source=EventSource.SIM,
        symbol="ES",
        levels=levels,
        is_snapshot=True
    )


class TestTapeEngineImbalance:
    """Tests for buy/sell imbalance calculation."""

    def test_buy_imbalance_positive(self):
        """Positive imbalance when buy volume > sell volume."""
        config = Config()
        config.W_t = 10.0  # 10 second window
        config.TAPE_BAND = 1.0  # ES 1.0 point band

        engine = TapeEngine(config=config)
        market_state = MarketState()

        es_level = 6870.0
        ts_base = _time.time_ns()

        # Add MBP for price reference
        mbp = create_mbp10(ts_base, es_level - 0.25, es_level + 0.25)
        market_state.update_es_mbp10(mbp)

        # Add buy trades near level
        for i in range(10):
            trade = create_es_trade(
                ts_ns=ts_base + int(i * 0.5e9),
                price=es_level + 0.25,  # Lift ask
                size=50,
                aggressor=Aggressor.BUY
            )
            market_state.update_es_trade(trade)

        # Add fewer sell trades
        for i in range(3):
            trade = create_es_trade(
                ts_ns=ts_base + int((i + 0.2) * 0.5e9),
                price=es_level,  # Hit bid
                size=30,
                aggressor=Aggressor.SELL
            )
            market_state.update_es_trade(trade)

        metrics = engine.compute_tape_state(
            level_price=es_level,
            market_state=market_state
        )

        assert metrics.imbalance > 0, "Expected positive imbalance (buy > sell)"
        assert metrics.buy_vol > metrics.sell_vol
        assert metrics.buy_vol == 500  # 10 * 50
        assert metrics.sell_vol == 90  # 3 * 30

    def test_sell_imbalance_negative(self):
        """Negative imbalance when sell volume > buy volume."""
        config = Config()
        config.W_t = 10.0
        config.TAPE_BAND = 1.0

        engine = TapeEngine(config=config)
        market_state = MarketState()

        es_level = 6870.0
        ts_base = _time.time_ns()

        mbp = create_mbp10(ts_base, es_level - 0.25, es_level + 0.25)
        market_state.update_es_mbp10(mbp)

        # Add sell trades
        for i in range(15):
            trade = create_es_trade(
                ts_ns=ts_base + int(i * 0.3e9),
                price=es_level,
                size=40,
                aggressor=Aggressor.SELL
            )
            market_state.update_es_trade(trade)

        # Add fewer buy trades
        for i in range(5):
            trade = create_es_trade(
                ts_ns=ts_base + int((i + 0.1) * 0.3e9),
                price=es_level + 0.25,
                size=20,
                aggressor=Aggressor.BUY
            )
            market_state.update_es_trade(trade)

        metrics = engine.compute_tape_state(
            level_price=es_level,
            market_state=market_state
        )

        assert metrics.imbalance < 0, "Expected negative imbalance (sell > buy)"
        assert metrics.sell_vol > metrics.buy_vol

    def test_neutral_imbalance(self):
        """Near-zero imbalance when buy ≈ sell."""
        config = Config()
        config.W_t = 10.0
        config.TAPE_BAND = 1.0

        engine = TapeEngine(config=config)
        market_state = MarketState()

        es_level = 6870.0
        ts_base = _time.time_ns()

        mbp = create_mbp10(ts_base, es_level - 0.25, es_level + 0.25)
        market_state.update_es_mbp10(mbp)

        # Equal buy and sell
        for i in range(10):
            market_state.update_es_trade(create_es_trade(
                ts_ns=ts_base + int(i * 0.5e9),
                price=es_level + 0.25,
                size=50,
                aggressor=Aggressor.BUY
            ))
            market_state.update_es_trade(create_es_trade(
                ts_ns=ts_base + int((i + 0.2) * 0.5e9),
                price=es_level,
                size=50,
                aggressor=Aggressor.SELL
            ))

        metrics = engine.compute_tape_state(
            level_price=es_level,
            market_state=market_state
        )

        assert abs(metrics.imbalance) < 0.01, "Expected near-zero imbalance"


class TestTapeEngineVelocity:
    """Tests for price velocity calculation."""

    def test_positive_velocity_rising_prices(self):
        """Positive velocity when prices are rising."""
        config = Config()
        config.W_v = 5.0  # 5 second velocity window

        engine = TapeEngine(config=config)
        market_state = MarketState()

        ts_base = _time.time_ns()
        start_price = 6870.0

        # Rising prices over 3 seconds
        for i in range(10):
            trade = create_es_trade(
                ts_ns=ts_base + int(i * 0.3e9),
                price=start_price + i * 0.5,  # Rising by 0.5/trade
                size=10,
                aggressor=Aggressor.BUY
            )
            market_state.update_es_trade(trade)

        metrics = engine.compute_tape_state(
            level_price=6870.0,
            market_state=market_state
        )

        assert metrics.velocity > 0, "Expected positive velocity (rising prices)"

    def test_negative_velocity_falling_prices(self):
        """Negative velocity when prices are falling."""
        config = Config()
        config.W_v = 5.0

        engine = TapeEngine(config=config)
        market_state = MarketState()

        ts_base = _time.time_ns()
        start_price = 6870.0

        # Falling prices
        for i in range(10):
            trade = create_es_trade(
                ts_ns=ts_base + int(i * 0.3e9),
                price=start_price - i * 0.5,  # Falling by 0.5/trade
                size=10,
                aggressor=Aggressor.SELL
            )
            market_state.update_es_trade(trade)

        metrics = engine.compute_tape_state(
            level_price=6870.0,
            market_state=market_state
        )

        assert metrics.velocity < 0, "Expected negative velocity (falling prices)"

    def test_zero_velocity_no_trades(self):
        """Zero velocity when no trades."""
        engine = TapeEngine()
        market_state = MarketState()

        metrics = engine.compute_tape_state(
            level_price=6870.0,
            market_state=market_state
        )

        assert metrics.velocity == 0.0


class TestTapeEngineSweep:
    """Tests for sweep detection."""

    def test_sweep_detected_buy(self):
        """Detect BUY sweep with large clustered volume."""
        config = Config()
        config.SWEEP_MIN_NOTIONAL = 100_000.0  # Lower for testing
        config.SWEEP_MAX_GAP_MS = 100

        engine = TapeEngine(config=config)
        market_state = MarketState()

        es_level = 6870.0
        ts_base = _time.time_ns()

        # Create a BUY sweep
        # ES contract = $50/point, so 6870 * 100 * 50 = $34.35M per trade
        for i in range(5):
            trade = create_es_trade(
                ts_ns=ts_base + int(i * 50e6),  # 50ms gaps
                price=es_level + i * 0.25,
                size=100,  # Large size
                aggressor=Aggressor.BUY
            )
            market_state.update_es_trade(trade)

        metrics = engine.compute_tape_state(
            level_price=es_level,
            market_state=market_state
        )

        assert metrics.sweep.detected, "Expected sweep to be detected"
        assert metrics.sweep.direction == "UP", "Expected UP direction for BUY sweep"
        assert metrics.sweep.notional > config.SWEEP_MIN_NOTIONAL

    def test_sweep_detected_sell(self):
        """Detect SELL sweep with large clustered volume."""
        config = Config()
        config.SWEEP_MIN_NOTIONAL = 100_000.0
        config.SWEEP_MAX_GAP_MS = 100

        engine = TapeEngine(config=config)
        market_state = MarketState()

        es_level = 6870.0
        ts_base = _time.time_ns()

        # Create a SELL sweep
        for i in range(5):
            trade = create_es_trade(
                ts_ns=ts_base + int(i * 50e6),  # 50ms gaps
                price=es_level - i * 0.25,
                size=100,
                aggressor=Aggressor.SELL
            )
            market_state.update_es_trade(trade)

        metrics = engine.compute_tape_state(
            level_price=es_level,
            market_state=market_state
        )

        assert metrics.sweep.detected, "Expected sweep to be detected"
        assert metrics.sweep.direction == "DOWN", "Expected DOWN direction for SELL sweep"

    def test_sweep_not_detected_mixed_direction(self):
        """No sweep when trades are mixed direction."""
        config = Config()
        config.SWEEP_MIN_NOTIONAL = 100_000.0
        config.SWEEP_MAX_GAP_MS = 100

        engine = TapeEngine(config=config)
        market_state = MarketState()

        es_level = 6870.0
        ts_base = _time.time_ns()

        # Mixed BUY/SELL trades (not a sweep)
        for i in range(5):
            aggressor = Aggressor.BUY if i % 2 == 0 else Aggressor.SELL
            trade = create_es_trade(
                ts_ns=ts_base + int(i * 50e6),
                price=es_level,
                size=100,
                aggressor=aggressor
            )
            market_state.update_es_trade(trade)

        metrics = engine.compute_tape_state(
            level_price=es_level,
            market_state=market_state
        )

        assert not metrics.sweep.detected, "Expected no sweep with mixed direction"

    def test_sweep_not_detected_below_threshold(self):
        """No sweep when notional below threshold."""
        config = Config()
        config.SWEEP_MIN_NOTIONAL = 10_000_000.0  # Very high threshold
        config.SWEEP_MAX_GAP_MS = 100

        engine = TapeEngine(config=config)
        market_state = MarketState()

        es_level = 6870.0
        ts_base = _time.time_ns()

        # Small trades
        for i in range(3):
            trade = create_es_trade(
                ts_ns=ts_base + int(i * 50e6),
                price=es_level,
                size=1,  # Small size
                aggressor=Aggressor.BUY
            )
            market_state.update_es_trade(trade)

        metrics = engine.compute_tape_state(
            level_price=es_level,
            market_state=market_state
        )

        assert not metrics.sweep.detected, "Expected no sweep below threshold"


class TestTapeEngineNoData:
    """Tests for edge cases with no data."""

    def test_no_trades(self):
        """Handle case with no trades."""
        engine = TapeEngine()
        market_state = MarketState()

        metrics = engine.compute_tape_state(
            level_price=6870.0,
            market_state=market_state
        )

        assert metrics.imbalance == 0.0
        assert metrics.buy_vol == 0
        assert metrics.sell_vol == 0
        assert metrics.velocity == 0.0
        assert not metrics.sweep.detected
        assert metrics.confidence == 0.0





class TestTapeEngineConfidence:
    """Tests for confidence calculation."""

    def test_confidence_scales_with_volume(self):
        """Confidence should scale with total volume."""
        config = Config()

        engine = TapeEngine(config=config)
        market_state_low = MarketState()
        market_state_high = MarketState()

        ts_base = _time.time_ns()

        # Low volume
        market_state_low.update_es_trade(create_es_trade(
            ts_ns=ts_base,
            price=6870.0,
            size=10,
            aggressor=Aggressor.BUY
        ))

        # High volume
        for i in range(20):
            market_state_high.update_es_trade(create_es_trade(
                ts_ns=ts_base + int(i * 0.2e9),
                price=6870.0,
                size=50,
                aggressor=Aggressor.BUY
            ))

        metrics_low = engine.compute_tape_state(6870.0, market_state_low)
        metrics_high = engine.compute_tape_state(6870.0, market_state_high)

        assert metrics_high.confidence > metrics_low.confidence
