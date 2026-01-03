"""
Bronze to Silver Global Market Pipeline.

Produces market-wide features at regular intervals (30s) that are NOT level-relative.
These features can be joined with level-specific features during training.

Output: Time series of global market features
- Microstructure: spread, depth, imbalance
- OFI: Total order flow imbalance
- Kinematics: Market velocity, acceleration
- Options: Total GEX, Tide, put/call ratio
- Session context: minutes_since_open, or_active
"""

from typing import List

from src.pipeline.core.pipeline import Pipeline
from src.pipeline.core.stage import BaseStage

# Reuse stages from level pipeline
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.init_market_state import InitMarketStateStage

# Global-specific stages
from src.pipeline.global_stages.generate_time_grid import GenerateTimeGridStage
from src.pipeline.global_stages.compute_market_micro import ComputeMarketMicroStage
from src.pipeline.global_stages.compute_market_ofi import ComputeMarketOFIStage
from src.pipeline.global_stages.compute_market_kinematics import ComputeMarketKinematicsStage
from src.pipeline.global_stages.compute_market_options import ComputeMarketOptionsStage
from src.pipeline.global_stages.compute_market_walls import ComputeMarketWallsStage
from src.pipeline.global_stages.filter_rth_global import FilterRTHGlobalStage


def get_bronze_to_silver_global_stages(interval_seconds: float = 30.0) -> List[BaseStage]:
    """
    Get the ordered list of stages for the global market pipeline.
    
    Args:
        interval_seconds: Interval between time grid events (default 30s)
    
    Returns:
        List of pipeline stages
    """
    return [
        # Stage 0: Load raw data (same as level pipeline)
        LoadBronzeStage(),
        
        # Stage 1: Build OHLCV bars (same as level pipeline)
        BuildAllOHLCVStage(),
        
        # Stage 2: Initialize market state (same as level pipeline)
        InitMarketStateStage(),
        
        # Stage 3: Generate time grid (DIFFERENT - no levels)
        GenerateTimeGridStage(interval_seconds=interval_seconds),
        
        # Stage 4: Compute market microstructure
        ComputeMarketMicroStage(),
        
        # Stage 5: Compute market OFI
        ComputeMarketOFIStage(),
        
        # Stage 6: Compute market kinematics
        ComputeMarketKinematicsStage(),
        
        # Stage 7: Compute market options
        ComputeMarketOptionsStage(),
        
        # Stage 8: Compute market walls (futures + options)
        ComputeMarketWallsStage(),
        
        # Stage 9: Filter to RTH and output
        FilterRTHGlobalStage(),
    ]


class BronzeToSilverGlobalPipeline(Pipeline):
    """
    Pipeline for generating global market features.
    
    Unlike the level-specific pipeline which generates features at level touches,
    this pipeline generates features at regular time intervals (e.g., every 30 seconds).
    
    The output can be joined with level-specific features by timestamp during training.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, interval_seconds: float = 30.0):
        """
        Args:
            interval_seconds: Interval between time grid events (default 30s)
        """
        stages = get_bronze_to_silver_global_stages(interval_seconds)
        super().__init__(
            stages=stages,
            name="bronze_to_silver_global",
            version=self.VERSION,
        )
        self.interval_seconds = interval_seconds
    
    @property
    def description(self) -> str:
        return f"Global market features at {self.interval_seconds}s intervals"


# Register pipeline
def create_pipeline(interval_seconds: float = 30.0) -> Pipeline:
    """Factory function for pipeline registry."""
    return BronzeToSilverGlobalPipeline(interval_seconds=interval_seconds)

