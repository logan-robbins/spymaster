"""Compute SMA-based mean reversion features stage (v2.0+)."""
from typing import Any, Dict, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import compute_mean_reversion_features


class ComputeSMAFeaturesStage(BaseStage):
    """Compute SMA-based mean reversion features.

    This stage is used in v2.0+ pipelines to add SMA-200 and SMA-400
    distance features for confluence analysis.

    Outputs:
        signals_df: Updated with mean reversion features
    """

    @property
    def name(self) -> str:
        return "compute_sma_features"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']
        ohlcv_2min = ctx.data.get('ohlcv_2min')

        signals_df = compute_mean_reversion_features(
            signals_df, ohlcv_df, ohlcv_2min=ohlcv_2min
        )

        return {'signals_df': signals_df}
