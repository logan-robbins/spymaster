"""Detect level touches stage."""
from typing import Any, Dict, List
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import (
    detect_touches_vectorized,
    detect_dynamic_level_touches,
)


class DetectTouchesStage(BaseStage):
    """Detect all level touches using numpy broadcasting.

    Detects touches for both static and dynamic levels,
    then merges and deduplicates.

    Args:
        touch_tolerance: How close counts as a touch (default: 0.10)
        max_touches: Maximum touches to process (default: 5000)

    Outputs:
        touches_df: DataFrame with touch information
    """

    def __init__(
        self,
        touch_tolerance: float = 0.10,
        max_touches: int = 5000
    ):
        self.touch_tolerance = touch_tolerance
        self.max_touches = max_touches

    @property
    def name(self) -> str:
        return "detect_touches"

    @property
    def required_inputs(self) -> List[str]:
        return ['ohlcv_1min', 'static_level_info', 'dynamic_levels']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        ohlcv_df = ctx.data['ohlcv_1min']
        static_level_info = ctx.data['static_level_info']
        dynamic_levels = ctx.data['dynamic_levels']

        # Detect static level touches
        touches_df = detect_touches_vectorized(
            ohlcv_df, static_level_info,
            touch_tolerance=self.touch_tolerance
        )

        # Detect dynamic level touches
        dynamic_touches = detect_dynamic_level_touches(
            ohlcv_df, dynamic_levels,
            touch_tolerance=self.touch_tolerance
        )

        # Merge and deduplicate
        if not dynamic_touches.empty:
            touches_df = pd.concat([touches_df, dynamic_touches], ignore_index=True)
            touches_df = touches_df.drop_duplicates(
                subset=['ts_ns', 'level_kind_name', 'level_price']
            )

        # Limit touches if needed
        if len(touches_df) > self.max_touches:
            touches_df = touches_df.head(self.max_touches)

        if touches_df.empty:
            raise ValueError("No touches detected")

        return {'touches_df': touches_df}
