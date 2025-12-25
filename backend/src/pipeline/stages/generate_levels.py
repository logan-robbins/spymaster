"""Generate level universe stage."""
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import (
    LevelInfo,
    generate_level_universe_vectorized,
    compute_dynamic_level_series,
)


class GenerateLevelsStage(BaseStage):
    """Generate level universe using vectorized operations.

    Generates:
    - Static levels: ROUND, STRIKE
    - Dynamic levels: PM_HIGH/LOW, OR_HIGH/LOW, SESSION_HIGH/LOW,
                      SMA_200/400, VWAP, CALL_WALL/PUT_WALL

    Outputs:
        level_info: LevelInfo with all levels
        static_level_info: LevelInfo with only static levels (ROUND, STRIKE)
        dynamic_levels: Dict with per-bar dynamic level values
    """

    @property
    def name(self) -> str:
        return "generate_levels"

    @property
    def required_inputs(self) -> List[str]:
        return ['ohlcv_1min', 'market_state']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        ohlcv_df = ctx.data['ohlcv_1min']
        market_state = ctx.data['market_state']
        ohlcv_2min = ctx.data.get('ohlcv_2min')
        option_trades_df = ctx.data.get('option_trades_df', pd.DataFrame())

        # Generate full level universe
        level_info = generate_level_universe_vectorized(
            ohlcv_df,
            market_state.option_flows,
            ctx.date,
            ohlcv_2min=ohlcv_2min
        )

        # Compute dynamic levels per bar
        dynamic_levels = compute_dynamic_level_series(
            ohlcv_df, ohlcv_2min, option_trades_df, ctx.date
        )

        # Extract static levels (ROUND, STRIKE)
        static_prices = []
        static_kinds = []
        static_names = []
        for price, kind, name in zip(level_info.prices, level_info.kinds, level_info.kind_names):
            if name in ('ROUND', 'STRIKE'):
                static_prices.append(price)
                static_kinds.append(kind)
                static_names.append(name)

        static_level_info = LevelInfo(
            prices=np.array(static_prices, dtype=np.float64),
            kinds=np.array(static_kinds, dtype=np.int8),
            kind_names=static_names
        )

        return {
            'level_info': level_info,
            'static_level_info': static_level_info,
            'dynamic_levels': dynamic_levels,
        }
