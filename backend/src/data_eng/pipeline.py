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
        from .stages.silver.future.convert_utc_to_est import SilverConvertUtcToEst
        from .stages.gold.future.filter_first3h import GoldFilterFirst3Hours
        
        return [
            SilverConvertUtcToEst(),
            GoldFilterFirst3Hours(),
        ]
    
    elif product_type == "future_option":
        from .stages.silver.future_option.convert_utc_to_est import SilverConvertUtcToEst
        from .stages.gold.future_option.filter_first3h import GoldFilterFirst3Hours
        
        return [
            SilverConvertUtcToEst(),
            GoldFilterFirst3Hours(),
        ]
    
    else:
        raise ValueError(
            f"Unknown product_type: {product_type}. "
            f"Must be one of: future, future_option, equity, equity_option"
        )
