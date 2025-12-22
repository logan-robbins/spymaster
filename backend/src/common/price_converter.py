"""
Price conversion between ES futures and SPY ETF.

ES ≈ SPY × 10 (approximate, varies slightly due to dividends/basis).
Use dynamic ratio when both prices are available, fallback to static.

This module enables:
- Levels defined in SPY terms (strikes, rounds) for trading SPY options
- Barrier physics queries in ES terms (MBP-10 depth)
- Output displayed in SPY terms for the trader
"""

from typing import Optional


class PriceConverter:
    """
    Converts prices between ES futures and SPY ETF.

    The ratio ES/SPY is approximately 10, but varies slightly due to:
    - Dividend expectations
    - Interest rate differential (cost of carry)
    - Fair value basis

    Usage:
        converter = PriceConverter()
        converter.update_es_price(6870.0)  # From ES trade
        spy_price = converter.es_to_spy(6870.0)  # Returns ~687.0
        es_level = converter.spy_to_es(687.0)    # Returns ~6870.0
    """

    DEFAULT_RATIO = 10.0  # ES/SPY baseline

    def __init__(self):
        self._dynamic_ratio: Optional[float] = None
        self._last_es_price: Optional[float] = None
        self._last_spy_price: Optional[float] = None

    def update_es_price(self, es_price: float):
        """
        Update with latest ES price.

        Args:
            es_price: ES futures price (e.g., 6870.0)
        """
        self._last_es_price = es_price
        self._update_ratio()

    def update_spy_price(self, spy_price: float):
        """
        Update with latest SPY price.

        Args:
            spy_price: SPY ETF price (e.g., 687.0)
        """
        self._last_spy_price = spy_price
        self._update_ratio()

    def _update_ratio(self):
        """Recalculate dynamic ratio if both prices are available."""
        if self._last_es_price and self._last_spy_price:
            self._dynamic_ratio = self._last_es_price / self._last_spy_price

    @property
    def ratio(self) -> float:
        """
        Get current ES/SPY ratio.

        Returns dynamic ratio if available, otherwise DEFAULT_RATIO.
        """
        return self._dynamic_ratio or self.DEFAULT_RATIO

    def es_to_spy(self, es_price: float) -> float:
        """
        Convert ES price to SPY equivalent.

        Args:
            es_price: ES futures price (e.g., 6870.0)

        Returns:
            SPY-equivalent price (e.g., 687.0)
        """
        return es_price / self.ratio

    def spy_to_es(self, spy_price: float) -> float:
        """
        Convert SPY price to ES equivalent.

        Args:
            spy_price: SPY price (e.g., 687.0)

        Returns:
            ES-equivalent price (e.g., 6870.0)
        """
        return spy_price * self.ratio

    def es_ticks_to_spy_dollars(self, es_ticks: int, es_tick_size: float = 0.25) -> float:
        """
        Convert ES tick count to SPY dollar amount.

        Args:
            es_ticks: Number of ES ticks
            es_tick_size: ES tick size (default $0.25)

        Returns:
            Equivalent SPY dollar amount
        """
        es_dollars = es_ticks * es_tick_size
        return es_dollars / self.ratio

    def spy_dollars_to_es_ticks(self, spy_dollars: float, es_tick_size: float = 0.25) -> float:
        """
        Convert SPY dollar amount to ES tick count.

        Args:
            spy_dollars: SPY dollar amount
            es_tick_size: ES tick size (default $0.25)

        Returns:
            Equivalent ES tick count (may be fractional)
        """
        es_dollars = spy_dollars * self.ratio
        return es_dollars / es_tick_size

    def get_state(self) -> dict:
        """Get converter state for debugging/logging."""
        return {
            "ratio": self.ratio,
            "dynamic_ratio": self._dynamic_ratio,
            "last_es_price": self._last_es_price,
            "last_spy_price": self._last_spy_price
        }
