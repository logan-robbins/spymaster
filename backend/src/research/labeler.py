"""
Labeler - Agent C Implementation

Classifies level test outcomes based on forward price excursions anchored at t1.
"""

from typing import List, Optional
from src.common.schemas.levels_signals import OutcomeLabel


def get_outcome(
    anchor_price: float,
    future_prices: List[float],
    direction: str = "UP",
    threshold: float = 2.0
) -> OutcomeLabel:
    """
    Classify the outcome of a level test based on forward price action.

    Args:
        anchor_price: Anchor spot price at confirmation time t1
        future_prices: List of future prices after t1
        direction: Direction of test - "UP" for resistance test, "DOWN" for support test
        threshold: Price excursion required for BREAK/BOUNCE ($2.00)

    Returns:
        OutcomeLabel: BOUNCE, BREAK, or CHOP
    """
    if anchor_price is None or not future_prices:
        return OutcomeLabel.UNDEFINED

    max_price = max(future_prices)
    min_price = min(future_prices)

    max_excursion_up = max_price - anchor_price
    max_excursion_down = anchor_price - min_price

    if direction.upper() == "UP":
        if max_excursion_up >= threshold and max_excursion_up > max_excursion_down:
            return OutcomeLabel.BREAK
        if max_excursion_down >= threshold:
            return OutcomeLabel.BOUNCE
    elif direction.upper() == "DOWN":
        if max_excursion_down >= threshold and max_excursion_down > max_excursion_up:
            return OutcomeLabel.BREAK
        if max_excursion_up >= threshold:
            return OutcomeLabel.BOUNCE
    else:
        return OutcomeLabel.UNDEFINED

    return OutcomeLabel.CHOP


def label_signal_with_future_data(
    anchor_price: float,
    future_prices: List[float],
    direction: Optional[str] = None,
    current_price: Optional[float] = None
) -> tuple[OutcomeLabel, Optional[float]]:
    """
    Convenience wrapper that returns both outcome and future price (5 min forward).
    
    Args:
        signal_price: Price at signal generation
        future_prices: List of future prices
        direction: Optional explicit direction. If None, infers from current_price
        current_price: Current spot price (used to infer direction if not provided)
    
    Returns:
        Tuple of (OutcomeLabel, future_price_5min)
    """
    
    # Infer direction if not provided
    if direction is None and current_price is not None:
        direction = "UP" if current_price >= signal_price else "DOWN"
    elif direction is None:
        direction = "UP"  # Default
    
    outcome = get_outcome(anchor_price, future_prices, direction)
    
    # Get price at 5 minutes forward (or last available price)
    future_price_5min = future_prices[-1] if future_prices else None
    
    return outcome, future_price_5min
