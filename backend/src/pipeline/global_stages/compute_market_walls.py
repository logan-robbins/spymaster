"""
Stage: Compute Market Walls (Global Pipeline)
Type: Feature Engineering (Market-Wide)
Input: Signals DataFrame (Time Grid), MBP-10 Snapshots, Option Trades
Output: Signals DataFrame with Global Wall Features

Transformation:
1. Identifies "Walls" (Large Resting Liquidity) in the Order Book (Futures):
   - Bid Wall: Price with max resting bid volume.
   - Ask Wall: Price with max resting ask volume.
   - Distances: How far are these walls from the current mid price?
2. Identifies "Option Walls" (High Gamma Strikes):
   - Call Wall: Strike with max positive Call Gamma.
   - Put Wall: Strike with max positive Put Gamma.
   - Distances: How far is spot from these pinning levels?
   
Note: Walls act as magnets or barriers. High gamma strikes often pin price (kill volatility).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.compute.walls import (
    compute_futures_walls_at_timestamps,
    compute_options_walls_at_timestamps,
)


class ComputeMarketWallsStage(BaseStage):
    """
    Compute market-wide wall features from futures and options data.
    
    Futures Wall Features:
    - futures_bid_wall_price: Price with largest resting bid
    - futures_bid_wall_size: Size at that price
    - futures_ask_wall_price: Price with largest resting ask
    - futures_ask_wall_size: Size at that price
    - futures_bid_wall_dist: Distance from mid to bid wall
    - futures_ask_wall_dist: Distance from mid to ask wall
    - futures_total_bid_depth: Total bid depth across 10 levels
    - futures_total_ask_depth: Total ask depth across 10 levels
    
    Options Wall Features:
    - options_call_wall_price: Strike with highest call dealer gamma
    - options_call_wall_gex: GEX at that strike
    - options_put_wall_price: Strike with highest put dealer gamma
    - options_put_wall_gex: GEX at that strike
    - options_call_wall_dist: Distance from spot to call wall
    - options_put_wall_dist: Distance from spot to put wall
    """
    
    @property
    def name(self) -> str:
        return "compute_market_walls"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10_snapshots']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        option_trades_df = ctx.data.get('option_trades_df', pd.DataFrame())
        
        if signals_df.empty:
            return {'signals_df': signals_df}
        
        signal_ts = signals_df['ts_ns'].values.astype(np.int64)
        spot_prices = signals_df['spot'].values.astype(np.float64) if 'spot' in signals_df.columns else np.zeros(len(signals_df))
        
        # Compute futures walls (global mode - no level_price)
        futures_walls = compute_futures_walls_at_timestamps(
            signal_ts=signal_ts,
            mbp10_snapshots=mbp10_snapshots,
            level_price=None,  # Global mode
        )
        
        for name, values in futures_walls.items():
            signals_df[name] = values
        
        # Compute options walls (global mode)
        if not option_trades_df.empty:
            options_walls = compute_options_walls_at_timestamps(
                signal_ts=signal_ts,
                option_trades_df=option_trades_df,
                spot_prices=spot_prices,
                level_price=None,  # Global mode
            )
            
            for name, values in options_walls.items():
                signals_df[name] = values
        else:
            # Fill with NaN if no options data
            for name in ['options_call_wall_price', 'options_call_wall_flow',
                        'options_put_wall_price', 'options_put_wall_flow',
                        'options_call_wall_dist', 'options_put_wall_dist']:
                signals_df[name] = np.nan
        
        logging.getLogger(__name__).info(
            f"Computed market walls: "
            f"futures_bid_wall_price range [{signals_df['futures_bid_wall_price'].min():.2f}, {signals_df['futures_bid_wall_price'].max():.2f}], "
            f"options_call_wall_price range [{signals_df['options_call_wall_price'].min():.2f}, {signals_df['options_call_wall_price'].max():.2f}]"
        )
        
        return {'signals_df': signals_df}

