from __future__ import annotations

from typing import List

from .stages.base import Stage


def build_pipeline(product_type: str) -> List[Stage]:
    """Return the ordered list of stages for the given product_type.
    
    Args:
        product_type: One of 'future', 'future_option', 'equity', 'equity_option'
    
    Returns:
        List of Stage instances to execute in order
    """
    
    if product_type == "future":
        from .stages.future import GoldFilterFirst3Hours, SilverConvertUtcToEst
        
        return [
            SilverConvertUtcToEst(),
            GoldFilterFirst3Hours(),
        ]
    
    elif product_type == "future_option":
        from .stages.future_option import GoldFilterFirst3Hours, SilverConvertUtcToEst
        
        return [
            SilverConvertUtcToEst(),
            GoldFilterFirst3Hours(),
        ]
    
    else:
        raise ValueError(
            f"Unknown product_type: {product_type}. "
            f"Must be one of: future, future_option, equity, equity_option"
        )
