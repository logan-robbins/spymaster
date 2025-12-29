"""
Price tracking for ES futures + ES options system.

ES futures and ES options are the SAME underlying instrument (E-mini S&P 500).
Both are quoted in S&P 500 index points (e.g., 6920.0).

NO CONVERSION: ES futures price = ES options price

This module:
- Tracks current ES price from futures trades
- Monitors basis spread between futures and options (<5 points typical)
- Provides pass-through methods (ES → ES, no-op)
"""

from typing import Optional


class PriceConverter:
    """
    Price tracker for ES futures + ES options system.

    ES futures and ES options use identical pricing (S&P 500 index points).
    All methods are pass-through (no conversion).

    Usage:
        converter = PriceConverter()
        converter.update_es_price(6920.0)  # From ES futures trade
        spot = converter.es_to_spx(6920.0)  # Returns 6920.0 (no conversion)
    """

    DEFAULT_RATIO = 1.0  # ES = ES (same underlying)

    def __init__(self):
        self._last_es_price: Optional[float] = None
        # Note: No basis tracking needed for ES options (same underlying!)

    def update_es_price(self, es_price: float):
        """
        Update with latest ES price.

        Args:
            es_price: ES futures price in index points (e.g., 5740.0)
        """
        self._last_es_price = es_price

    def update_spx_price(self, spx_price: float):
        """
        Update with index price (no-op for ES system - futures ARE the index proxy).

        Args:
            spx_price: Price value (not needed for ES futures/options)
        """
        pass

    @property
    def basis(self) -> float:
        """Get basis spread (always 0 - ES futures = ES options underlying)."""
        return 0.0
    
    @property
    def ratio(self) -> float:
        """Get price ratio (always 1.0 - ES futures = ES options)."""
        return 1.0

    def es_to_spx(self, es_price: float) -> float:
        """
        Pass-through for ES system (ES options = ES futures).

        Args:
            es_price: ES price in index points (e.g., 5740.0)

        Returns:
            Same price (no conversion needed!)
        """
        return es_price

    def spx_to_es(self, spx_price: float) -> float:
        """
        Pass-through for ES system (ES options = ES futures).

        Args:
            spx_price: ES price in index points (e.g., 5740.0)

        Returns:
            Same price (no conversion needed!)
        """
        return spx_price

    def es_ticks_to_spx_points(self, es_ticks: int, es_tick_size: float = 0.25) -> float:
        """
        Convert ES tick count to point amount (pass-through for ES system).

        Args:
            es_ticks: Number of ES ticks
            es_tick_size: ES tick size (default $0.25 per point)

        Returns:
            Point amount
        """
        return es_ticks * es_tick_size

    def spx_points_to_es_ticks(self, spx_points: float, es_tick_size: float = 0.25) -> float:
        """
        Convert point amount to ES tick count (pass-through for ES system).

        Args:
            spx_points: Point amount
            es_tick_size: ES tick size (default $0.25 per point)

        Returns:
            ES tick count (may be fractional)
        """
        return spx_points / es_tick_size
    

    def get_state(self) -> dict:
        """Get converter state for debugging/logging."""
        return {
            "basis": self.basis,
            "dynamic_basis": self._dynamic_basis,
            "last_es_price": self._last_es_price,
            "last_spx_price": self._last_spx_price
        }
