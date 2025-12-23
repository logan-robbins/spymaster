"""
Labeler - Agent C Implementation

Classifies level test outcomes based on forward price excursions.

Outcome Logic:
- BOUNCE: Price reverses > $0.20 in opposite direction AND breaks < $0.05 in test direction
- BREAK: Price continues > $0.20 in test direction
- CHOP: Everything else (no clear resolution)
"""

from typing import List, Optional
from src.common.schemas.levels_signals import OutcomeLabel


def get_outcome(
    signal_price: float,
    future_prices: List[float],
    direction: str = "UP",
    bounce_threshold: float = 0.20,
    break_threshold: float = 0.20,
    bounce_limit: float = 0.05,
    epsilon: float = 1e-6
) -> OutcomeLabel:
    """
    Classify the outcome of a level test based on future price action.
    
    Args:
        signal_price: The price at which the level was touched/tested
        future_prices: List of future prices (e.g., next 5 minutes of 1-second bars)
        direction: Direction of test - "UP" for resistance test, "DOWN" for support test
        bounce_threshold: Price excursion required in reverse direction for BOUNCE ($0.20)
        break_threshold: Price excursion required in test direction for BREAK ($0.20)
        bounce_limit: Max allowed excursion in test direction for BOUNCE ($0.05)
        epsilon: Small value to handle floating point precision errors (default 1e-6)
    
    Returns:
        OutcomeLabel: BOUNCE, BREAK, or CHOP
        
    Logic:
        BOUNCE: Price reverses significantly (rejects the level)
            - Reverse direction excursion > bounce_threshold ($0.20)
            - Test direction excursion < bounce_limit ($0.05)
            
        BREAK: Price continues significantly through the level
            - Test direction excursion > break_threshold ($0.20)
            
        CHOP: Neither condition met (consolidation, no clear outcome)
    
    Examples:
        # Resistance test at 400.00
        >>> get_outcome(400.00, [400.05, 400.03, 399.85, 399.75], direction="UP")
        OutcomeLabel.BOUNCE  # Rejected down by $0.25
        
        >>> get_outcome(400.00, [400.10, 400.25, 400.40], direction="UP")
        OutcomeLabel.BREAK   # Broke through up by $0.40
        
        >>> get_outcome(400.00, [400.05, 399.95, 400.08, 399.92], direction="UP")
        OutcomeLabel.CHOP    # No clear direction
    """
    
    if not future_prices:
        return OutcomeLabel.UNDEFINED
    
    # Calculate price excursions from signal_price
    max_price = max(future_prices)
    min_price = min(future_prices)
    
    max_excursion_up = max_price - signal_price
    max_excursion_down = signal_price - min_price
    
    if direction.upper() == "UP":
        # Testing resistance - trying to break UP
        # BOUNCE: Falls back DOWN > bounce_threshold, didn't break up > bounce_limit
        # Use epsilon to handle floating point precision
        if max_excursion_down > bounce_threshold and max_excursion_up <= (bounce_limit + epsilon):
            return OutcomeLabel.BOUNCE
        
        # BREAK: Broke UP > break_threshold
        if max_excursion_up > break_threshold:
            return OutcomeLabel.BREAK
            
    elif direction.upper() == "DOWN":
        # Testing support - trying to break DOWN
        # BOUNCE: Bounces UP > bounce_threshold, didn't break down > bounce_limit
        # Use epsilon to handle floating point precision
        if max_excursion_up > bounce_threshold and max_excursion_down <= (bounce_limit + epsilon):
            return OutcomeLabel.BOUNCE
        
        # BREAK: Broke DOWN > break_threshold
        if max_excursion_down > break_threshold:
            return OutcomeLabel.BREAK
    else:
        # Invalid direction
        return OutcomeLabel.UNDEFINED
    
    # No clear outcome - CHOP
    return OutcomeLabel.CHOP


def label_signal_with_future_data(
    signal_price: float,
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
    
    outcome = get_outcome(signal_price, future_prices, direction)
    
    # Get price at 5 minutes forward (or last available price)
    future_price_5min = future_prices[-1] if future_prices else None
    
    return outcome, future_price_5min

