from __future__ import annotations

from typing import List

from .stages.base import Stage


def build_pipeline(product_type: str, layer: str = "all") -> List[Stage]:
    """Return the ordered list of stages for the given product_type and layer.
    
    Args:
        product_type: One of 'future', 'future_option', 'equity', 'equity_option'
        layer: One of 'bronze', 'silver', 'gold', 'all'
    
    Returns:
        List of Stage instances to execute in order
    """
    
    if product_type == "future":
        from .stages.bronze.future.process_dbn import BronzeProcessDBN
        from .stages.silver.future.convert_utc_to_est import SilverConvertUtcToEst
        from .stages.silver.future.add_session_levels import SilverAddSessionLevels
        from .stages.silver.future.compute_bar5s_features import SilverComputeBar5sFeatures
        from .stages.silver.future.extract_level_episodes import SilverExtractLevelEpisodes
        from .stages.silver.future.compute_approach_features import SilverComputeApproachFeatures
        from .stages.gold.future.filter_first3h import GoldFilterFirst3Hours
        from .stages.gold.future.filter_band_range import GoldFilterBandRange

        if layer == "bronze":
            return [BronzeProcessDBN()]
        elif layer == "silver":
            return [SilverConvertUtcToEst(), SilverAddSessionLevels(), SilverComputeBar5sFeatures(), SilverExtractLevelEpisodes(), SilverComputeApproachFeatures()]
        elif layer == "gold":
            return [GoldFilterFirst3Hours(), GoldFilterBandRange()]
        elif layer == "all":
            return [
                BronzeProcessDBN(),
                SilverConvertUtcToEst(),
                SilverAddSessionLevels(),
                SilverComputeBar5sFeatures(),
                SilverExtractLevelEpisodes(),
                SilverComputeApproachFeatures(),
                GoldFilterFirst3Hours(),
                GoldFilterBandRange(),
            ]
    
    elif product_type == "future_option":
        from .stages.silver.future_option.convert_utc_to_est import SilverConvertUtcToEst
        from .stages.gold.future_option.filter_first3h import GoldFilterFirst3Hours
        
        if layer == "bronze":
            raise ValueError("Bronze not implemented for future_option")
        elif layer == "silver":
            return [SilverConvertUtcToEst()]
        elif layer == "gold":
            return [GoldFilterFirst3Hours()]
        elif layer == "all":
            return [
                SilverConvertUtcToEst(),
                GoldFilterFirst3Hours(),
            ]
    
    raise ValueError(
        f"Unknown product_type: {product_type}. "
        f"Must be one of: future, future_option, equity, equity_option"
    )
