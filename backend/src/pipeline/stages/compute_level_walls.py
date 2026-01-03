"""
Stage: Compute Level Walls
Type: Feature Engineering (Liquidity/Gamma)
Input: Signals DataFrame (touches), MBP-10 Snapshots, Options Trades
Output: Signals DataFrame with Wall Features

Transformation:
1. Identifies the nearest significant "Liquidity Walls" above and below the interaction level.
   - Futures Walls: High resting liquidity in the Limit Order Book (LOB).
   - Options Walls: Strikes with large Dealer Gamma.
2. Computes the Distance and Size of these walls relative to the level.
   - Proximity: Exploring if a level is "guarded" by a wall.
   - Asymmetry: Is the wall above stronger than below?

Note: This stage answers "Is there a backstop nearby?"—helping distinguish between a clean break and a trap.
"""

import pandas as pd
import logging
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
    - options_wall_above_flow: Net dealer flow (gamma * size) at that strike
    - options_wall_above_dist: Distance from level
    - options_wall_below_price: Strike with highest dealer gamma below level
    - options_wall_below_flow: Net dealer flow (gamma * size) at that strike
    - options_wall_below_dist: Distance from level
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
        
        # Get level price from Context or DataFrame (assuming single-level batch)
        level_price = ctx.data.get('level_price')
        if level_price is None and not signals_df.empty and 'level_price' in signals_df.columns:
            level_price = signals_df['level_price'].iloc[0]
        
        if signals_df.empty:
            return {'signals_df': signals_df}
        
        if level_price is None:
            # Check for logger in ctx, else use module logger
            log = getattr(ctx, 'logger', logging.getLogger(__name__))
            log.warning("No level_price found in context/signals, skipping walls.")
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
        # ctx.logger might not avail, use module logger
        log = getattr(ctx, 'logger', logging.getLogger(__name__))
        log.info(
            f"Computed level-relative walls for level {level_price:.2f}: "
            f"futures_wall_above avg dist {signals_df['futures_wall_above_dist'].mean():.2f}pt, "
            f"futures_wall_below avg dist {signals_df['futures_wall_below_dist'].mean():.2f}pt"
        )
        
        return {'signals_df': signals_df}

