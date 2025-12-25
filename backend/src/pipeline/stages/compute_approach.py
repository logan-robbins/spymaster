"""Compute approach context features stage (v2.0+)."""
from typing import Any, Dict, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import (
    compute_approach_context_vectorized,
    add_sparse_feature_transforms,
    add_normalized_features,
    compute_attempt_features,
)


class ComputeApproachFeaturesStage(BaseStage):
    """Compute approach context and normalized features.

    This stage is used in v2.0+ pipelines to add:
    - Approach context (velocity, bars, distance)
    - Sparse feature transforms
    - Normalized features
    - Attempt clustering and deterioration trends

    Outputs:
        signals_df: Updated with approach features
    """

    @property
    def name(self) -> str:
        return "compute_approach_features"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']

        # Compute approach context
        signals_df = compute_approach_context_vectorized(signals_df, ohlcv_df)

        # Sparse feature transforms + normalization
        signals_df = add_sparse_feature_transforms(signals_df)
        signals_df = add_normalized_features(signals_df)

        # Attempt clustering + deterioration trends
        signals_df = compute_attempt_features(signals_df)

        return {'signals_df': signals_df}
