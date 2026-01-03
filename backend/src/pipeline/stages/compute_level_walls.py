"""
Compute level-relative wall features (futures + options).

For a given technical level, compute the walls above and below it.
These walls represent liquidity that must be absorbed for price to break through the level.

Futures walls: Resting orders in MBP-10 above/below level
Options walls: Strikes with dealer gamma above/below level
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.compute.walls import (
    compute_futures_walls_at_timestamps,
    compute_options_walls_at_timestamps,
)


class ComputeLevelWallsStage(BaseStage):
    """
    Compute wall features relative to the tested level.
    
    This stage computes walls ABOVE and BELOW the specific level price,
    providing context about the liquidity landscape around the level.
    
    Futures Wall Features (level-relative):
    - futures_wall_above_price: Nearest significant bid/ask above level
    - futures_wall_above_size: Size at that price
    - futures_wall_above_dist: Distance from level to wall above
    - futures_wall_below_price: Nearest significant bid/ask below level
    - futures_wall_below_size: Size at that price
    - futures_wall_below_dist: Distance from level to wall below
    - futures_wall_asymmetry: (above - below) / total depth
    
    Options Wall Features (level-relative):
    - options_wall_above_price: Strike with highest dealer gamma above level
    - options_wall_above_gex: GEX at that strike
    - options_wall_above_dist: Distance from level
    - options_wall_above_type: 'C' or 'P' (call or put wall)
    - options_wall_below_price: Strike with highest dealer gamma below level
    - options_wall_below_gex: GEX at that strike
    - options_wall_below_dist: Distance from level
    - options_wall_below_type: 'C' or 'P'
    """
    
    @property
    def name(self) -> str:
        return "compute_level_walls"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10_snapshots']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        option_trades_df = ctx.data.get('option_trades_df', pd.DataFrame())
        level_price = ctx.data.get('level_price')
        
        if signals_df.empty:
            return {'signals_df': signals_df}
        
        if level_price is None:
            ctx.logger.warning("No level_price in context, skipping level walls computation")
            return {'signals_df': signals_df}
        
        signal_ts = signals_df['ts_ns'].values.astype(np.int64)
        spot_prices = signals_df['spot'].values.astype(np.float64) if 'spot' in signals_df.columns else np.zeros(len(signals_df))
        
        # Compute futures walls relative to level
        futures_walls = compute_futures_walls_at_timestamps(
            signal_ts=signal_ts,
            mbp10_snapshots=mbp10_snapshots,
            level_price=level_price,  # Level-relative mode
        )
        
        for name, values in futures_walls.items():
            signals_df[name] = values
        
        # Compute options walls relative to level
        if not option_trades_df.empty:
            options_walls = compute_options_walls_at_timestamps(
                signal_ts=signal_ts,
                option_trades_df=option_trades_df,
                spot_prices=spot_prices,
                level_price=level_price,  # Level-relative mode
            )
            
            for name, values in options_walls.items():
                signals_df[name] = values
        else:
            # Fill with NaN if no options data
            for name in ['options_wall_above_price', 'options_wall_above_flow', 
                        'options_wall_above_dist',
                        'options_wall_below_price', 'options_wall_below_flow',
                        'options_wall_below_dist']:
                signals_df[name] = np.nan
        
        # Log summary
        above_price = signals_df['futures_wall_above_price'].dropna()
        below_price = signals_df['futures_wall_below_price'].dropna()
        
        ctx.logger.info(
            f"Computed level-relative walls for level {level_price:.2f}: "
            f"futures_wall_above avg dist {signals_df['futures_wall_above_dist'].mean():.2f}pt, "
            f"futures_wall_below avg dist {signals_df['futures_wall_below_dist'].mean():.2f}pt"
        )
        
        return {'signals_df': signals_df}

