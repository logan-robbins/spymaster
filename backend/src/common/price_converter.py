"""
Price conversion for ES futures system.

v1 Final Call: ES futures + ES options (PERFECT alignment!)
- ES futures: Quoted in S&P 500 index points (e.g., 5740.00)
- ES options: SAME underlying, SAME units, SAME venue (CME)
- NO conversion needed!

This module provides:
- Pass-through methods for API consistency
- Optional basis tracking (for diagnostics)
- Legacy compatibility with existing code

Key evolution:
- v0 (SPY): ES/10 ≈ SPY (ES 6870 → SPY 687) - REQUIRED CONVERSION
- v1 (ES):  ES = ES (ES 5740 → ES 5740) - NO CONVERSION!

This is simpler, cleaner, and more accurate than SPY or SPX approaches.
"""

from typing import Optional


class PriceConverter:
    """
    Price converter for ES futures system (v1: ES futures + ES options).

    ES options and ES futures use identical pricing - both in S&P 500 index points.
    This class provides pass-through methods for API consistency with existing code.

    Usage:
        converter = PriceConverter()
        converter.update_es_price(5740.0)  # From ES futures trade
        # For ES options:
        spot = 5740.0  # Direct use, no conversion!
        es_level = converter.es_to_spx(5740.0)  # Returns 5740.0 (pass-through)
    """

    DEFAULT_RATIO = 1.0  # No conversion needed (ES = ES)

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
        Legacy compatibility method (no-op for ES options).
        
        For ES options, this is not needed since ES futures = ES options underlying.
        Kept for API compatibility with existing code.

        Args:
            spx_price: Price value (ignored for ES system)
        """
        pass  # No-op for ES options

    @property
    def basis(self) -> float:
        """
        Get basis spread (always 0 for ES options).
        
        ES options and ES futures are the same underlying - no basis spread!
        Kept for API compatibility.
        """
        return 0.0
    
    @property
    def ratio(self) -> float:
        """
        Get price ratio (always 1.0 for ES system).
        
        Legacy compatibility property. ES futures and ES options use same pricing.
        """
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
    
    # Legacy compatibility aliases (some code may still use these)
    def es_to_spy(self, es_price: float) -> float:
        """Legacy alias for es_to_spx (backward compatibility)."""
        return self.es_to_spx(es_price)
    
    def spy_to_es(self, spy_price: float) -> float:
        """Legacy alias for spx_to_es (backward compatibility)."""
        return self.spx_to_es(spy_price)

    def get_state(self) -> dict:
        """Get converter state for debugging/logging."""
        return {
            "basis": self.basis,
            "dynamic_basis": self._dynamic_basis,
            "last_es_price": self._last_es_price,
            "last_spx_price": self._last_spx_price
        }
