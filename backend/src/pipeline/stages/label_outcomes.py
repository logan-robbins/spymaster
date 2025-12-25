"""Label outcomes stage."""
from typing import Any, Dict, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import label_outcomes_vectorized


class LabelOutcomesStage(BaseStage):
    """Label outcomes using competing risks methodology.

    Computes forward-looking outcome labels based on whether
    price breaks through or bounces from the level.

    Outputs:
        signals_df: Updated with outcome labels:
            - outcome: BREAK, BOUNCE, STALL
            - strength_signed: Signed magnitude of move
            - t1_60, t1_120: Confirmation timestamps
            - tradeable_1, tradeable_2: Tradeable flags
    """

    @property
    def name(self) -> str:
        return "label_outcomes"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']

        signals_df = label_outcomes_vectorized(signals_df, ohlcv_df)

        return {'signals_df': signals_df}
