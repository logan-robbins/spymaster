"""
Labeler - Agent C Implementation

Classifies level test outcomes based on forward price excursions anchored at t1.
"""

from typing import List, Optional
from src.common.schemas.levels_signals import OutcomeLabel


def get_outcome(
    level_price: float,
    future_prices: List[float],
    direction: str = "UP",
    threshold: float = 2.0,
    break_threshold: Optional[float] = None,
    bounce_threshold: Optional[float] = None
) -> OutcomeLabel:
    """
    Classify the outcome of a level test based on forward price action.

    Args:
        level_price: Price level being tested (anchor for break/bounce decisions)
        future_prices: List of future prices after confirmation time t1
        direction: Direction of test - "UP" for resistance test, "DOWN" for support test
        threshold: Symmetric price excursion required for BREAK/BOUNCE ($2.00)
        break_threshold: Optional threshold for BREAK (defaults to threshold)
        bounce_threshold: Optional threshold for BOUNCE (defaults to threshold)

    Returns:
        OutcomeLabel: BOUNCE, BREAK, or CHOP
    """
    if level_price is None or not future_prices:
        return OutcomeLabel.UNDEFINED

    break_threshold = threshold if break_threshold is None else break_threshold
    bounce_threshold = threshold if bounce_threshold is None else bounce_threshold

    break_idx = None
    bounce_idx = None

    if direction.upper() == "UP":
        break_level = level_price + break_threshold
        bounce_level = level_price - bounce_threshold
        for i, price in enumerate(future_prices):
            if break_idx is None and price >= break_level:
                break_idx = i
            if bounce_idx is None and price <= bounce_level:
                bounce_idx = i
            if break_idx is not None and bounce_idx is not None:
                break
    elif direction.upper() == "DOWN":
        break_level = level_price - break_threshold
        bounce_level = level_price + bounce_threshold
        for i, price in enumerate(future_prices):
            if break_idx is None and price <= break_level:
                break_idx = i
            if bounce_idx is None and price >= bounce_level:
                bounce_idx = i
            if break_idx is not None and bounce_idx is not None:
                break
    else:
        return OutcomeLabel.UNDEFINED

    if break_idx is not None and bounce_idx is not None:
        if break_idx < bounce_idx:
            return OutcomeLabel.BREAK
        if bounce_idx < break_idx:
            return OutcomeLabel.BOUNCE
        return OutcomeLabel.CHOP
    if break_idx is not None:
        return OutcomeLabel.BREAK
    if bounce_idx is not None:
        return OutcomeLabel.BOUNCE

    return OutcomeLabel.CHOP


def label_signal_with_future_data(
    level_price: float,
    future_prices: List[float],
    direction: Optional[str] = None,
    current_price: Optional[float] = None,
    threshold: float = 2.0
) -> tuple[OutcomeLabel, Optional[float]]:
    """
    Convenience wrapper that returns both outcome and future price (5 min forward).
    
    Args:
        level_price: Price level being tested
        future_prices: List of future prices
        direction: Optional explicit direction. If None, infers from current_price
        current_price: Current spot price (used to infer direction if not provided)
        threshold: Price excursion required for BREAK/BOUNCE ($2.00)
    
    Returns:
        Tuple of (OutcomeLabel, future_price_5min)
    """
    
    # Infer direction if not provided
    if direction is None and current_price is not None:
        direction = "UP" if current_price >= level_price else "DOWN"
    elif direction is None:
        direction = "UP"  # Default
    
    outcome = get_outcome(level_price, future_prices, direction, threshold=threshold)
    
    # Get price at 5 minutes forward (or last available price)
    future_price_5min = future_prices[-1] if future_prices else None
    
    return outcome, future_price_5min
