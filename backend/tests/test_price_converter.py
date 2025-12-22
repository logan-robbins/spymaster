"""
Tests for PriceConverter (ES <-> SPY price conversion).

Verifies:
- Default ratio conversion (ES/SPY = 10.0)
- Dynamic ratio when both prices available
- Round-trip accuracy (ES -> SPY -> ES)
- Tick-to-dollar conversions
"""

import pytest
from src.common.price_converter import PriceConverter


class TestPriceConverterDefaults:
    """Tests using default ratio (10.0)."""

    def test_default_ratio(self):
        """Default ES/SPY ratio should be 10.0."""
        pc = PriceConverter()
        assert pc.ratio == 10.0

    def test_es_to_spy_default(self):
        """Convert ES price to SPY using default ratio."""
        pc = PriceConverter()
        # ES 6870.0 / 10 = SPY 687.0
        assert pc.es_to_spy(6870.0) == 687.0
        assert pc.es_to_spy(6875.25) == 687.525
        assert pc.es_to_spy(6000.0) == 600.0

    def test_spy_to_es_default(self):
        """Convert SPY price to ES using default ratio."""
        pc = PriceConverter()
        # SPY 687.0 * 10 = ES 6870.0
        assert pc.spy_to_es(687.0) == 6870.0
        assert pc.spy_to_es(687.525) == 6875.25
        assert pc.spy_to_es(600.0) == 6000.0

    def test_roundtrip_default(self):
        """ES -> SPY -> ES should preserve value."""
        pc = PriceConverter()
        test_prices = [6870.0, 6875.25, 6000.0, 5999.50]
        for es_price in test_prices:
            spy_price = pc.es_to_spy(es_price)
            recovered_es = pc.spy_to_es(spy_price)
            assert abs(recovered_es - es_price) < 1e-9, f"Roundtrip failed for {es_price}"


class TestPriceConverterDynamic:
    """Tests with dynamic ratio calculation."""

    def test_dynamic_ratio_calculation(self):
        """Dynamic ratio from observed ES and SPY prices."""
        pc = PriceConverter()

        # Simulate observed prices: ES=6870, SPY=686.9 (ratio ~10.001)
        pc.update_es_price(6870.0)
        pc.update_spy_price(686.9)

        expected_ratio = 6870.0 / 686.9
        assert abs(pc.ratio - expected_ratio) < 1e-9

    def test_es_only_uses_default(self):
        """Only ES price -> still uses default ratio."""
        pc = PriceConverter()
        pc.update_es_price(6870.0)

        assert pc.ratio == 10.0  # Still default

    def test_spy_only_uses_default(self):
        """Only SPY price -> still uses default ratio."""
        pc = PriceConverter()
        pc.update_spy_price(687.0)

        assert pc.ratio == 10.0  # Still default

    def test_dynamic_ratio_affects_conversion(self):
        """Dynamic ratio should affect conversions."""
        pc = PriceConverter()

        # Set dynamic ratio: ES 6870 / SPY 687.5 = 9.9927...
        pc.update_es_price(6870.0)
        pc.update_spy_price(687.5)

        ratio = pc.ratio
        assert ratio != 10.0  # Dynamic ratio

        # Check conversion uses dynamic ratio
        spy_result = pc.es_to_spy(6870.0)
        assert spy_result == 6870.0 / ratio


class TestTickDollarConversion:
    """Tests for tick/dollar conversions."""

    def test_es_ticks_to_spy_dollars(self):
        """Convert ES tick count to SPY dollars."""
        pc = PriceConverter()

        # 4 ES ticks = 4 * $0.25 = $1.00 ES = $0.10 SPY
        assert abs(pc.es_ticks_to_spy_dollars(4) - 0.10) < 1e-9

        # 10 ES ticks = 10 * $0.25 = $2.50 ES = $0.25 SPY
        assert abs(pc.es_ticks_to_spy_dollars(10) - 0.25) < 1e-9

    def test_spy_dollars_to_es_ticks(self):
        """Convert SPY dollars to ES tick count."""
        pc = PriceConverter()

        # $0.10 SPY = $1.00 ES = 4 ES ticks
        assert abs(pc.spy_dollars_to_es_ticks(0.10) - 4.0) < 1e-9

        # $0.25 SPY = $2.50 ES = 10 ES ticks
        assert abs(pc.spy_dollars_to_es_ticks(0.25) - 10.0) < 1e-9

    def test_tick_dollar_roundtrip(self):
        """SPY dollars -> ES ticks -> SPY dollars should preserve value."""
        pc = PriceConverter()

        for spy_dollars in [0.05, 0.10, 0.25, 0.50, 1.00]:
            es_ticks = pc.spy_dollars_to_es_ticks(spy_dollars)
            recovered = pc.es_ticks_to_spy_dollars(es_ticks)
            assert abs(recovered - spy_dollars) < 1e-9


class TestGetState:
    """Tests for state inspection."""

    def test_get_state_initial(self):
        """Initial state should show no observed prices."""
        pc = PriceConverter()
        state = pc.get_state()

        assert state["ratio"] == 10.0
        assert state["dynamic_ratio"] is None
        assert state["last_es_price"] is None
        assert state["last_spy_price"] is None

    def test_get_state_after_updates(self):
        """State should reflect observed prices."""
        pc = PriceConverter()
        pc.update_es_price(6870.0)
        pc.update_spy_price(687.0)

        state = pc.get_state()
        assert state["last_es_price"] == 6870.0
        assert state["last_spy_price"] == 687.0
        assert state["dynamic_ratio"] == 10.0


class TestEdgeCases:
    """Edge case handling."""

    def test_zero_es_price(self):
        """Zero ES price should return zero SPY."""
        pc = PriceConverter()
        assert pc.es_to_spy(0.0) == 0.0

    def test_zero_spy_price(self):
        """Zero SPY price should return zero ES."""
        pc = PriceConverter()
        assert pc.spy_to_es(0.0) == 0.0

    def test_negative_prices(self):
        """Negative prices should work mathematically."""
        pc = PriceConverter()
        # Not realistic but mathematically valid
        assert pc.es_to_spy(-100.0) == -10.0
        assert pc.spy_to_es(-10.0) == -100.0
