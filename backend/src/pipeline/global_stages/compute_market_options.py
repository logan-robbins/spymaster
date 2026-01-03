"""
Stage: Compute Market Options (Global Pipeline)
Type: Feature Engineering (Market-Wide)
Input: Signals DataFrame (Time Grid), Option Trades
Output: Signals DataFrame with Global Options Features

Transformation:
1. Aggregates ALL option trades (irrespective of price level).
2. Computes System-Wide Greeks:
   - Total GEX: Net Gamma Exposure ($Bn).
   - GEX Asymmetry: Balance between Bull/Bear positioning.
   - GEX above/below spot: Positioning relative to current price.
3. Computes System-Wide Flow (Tide):
   - Call Tide: Net Call Premium Flow.
   - Put Tide: Net Put Premium Flow.
   - Put/Call Ratio: Volume-based sentiment.
   
Note: These features capture the "Macro Sentiment" and "Structural Volatility" of the entire market.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.compute.gex import compute_gex_features, compute_tide_features


class ComputeMarketOptionsStage(BaseStage):
    """
    Compute market-wide options features.
    
    Unlike level-relative options which compute GEX/Tide above/below a level,
    this computes totals and above/below current spot price.
    
    Features:
    - total_gex: Total gamma exposure
    - total_call_gex, total_put_gex: By option type
    - gex_above_spot, gex_below_spot: Relative to current price
    - gex_asymmetry: (above - below) / total
    - call_tide, put_tide: Premium flow
    - put_call_ratio: Total put volume / call volume
    """
    
    @property
    def name(self) -> str:
        return "compute_market_options"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'option_trades_df']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        option_trades_df = ctx.data.get('option_trades_df', pd.DataFrame())
        
        if signals_df.empty:
            return {'signals_df': signals_df}
        
        # Standardize column name: 'right' -> 'option_type'
        if not option_trades_df.empty and 'right' in option_trades_df.columns and 'option_type' not in option_trades_df.columns:
            option_trades_df = option_trades_df.copy()
            option_trades_df['option_type'] = option_trades_df['right']
        
        signal_ts = signals_df['ts_ns'].values.astype(np.int64)
        spot_prices = signals_df['spot'].values.astype(np.float64) if 'spot' in signals_df.columns else np.zeros(len(signals_df))
        
        # Compute GEX features (global mode)
        gex_features = compute_gex_features(
            signal_ts=signal_ts,
            option_trades_df=option_trades_df,
            spot_prices=spot_prices,
            level_price=None,  # Global mode
        )
        
        for name, values in gex_features.items():
            signals_df[name] = values
        
        # Compute Tide features (global mode)
        tide_features = compute_tide_features(
            signal_ts=signal_ts,
            option_trades_df=option_trades_df,
            level_price=None,  # Global mode
        )
        
        for name, values in tide_features.items():
            signals_df[name] = values
        
        # Compute put/call volume ratio
        if not option_trades_df.empty and 'option_type' in option_trades_df.columns:
            df = option_trades_df.sort_values('ts_event_ns')
            opt_ts = df['ts_event_ns'].values.astype(np.int64)
            opt_types = df['option_type'].values
            sizes = df['size'].values if 'size' in df.columns else np.ones(len(df))
            
            cum_call_vol = np.cumsum(np.where(opt_types == 'C', sizes, 0))
            cum_put_vol = np.cumsum(np.where(opt_types == 'P', sizes, 0))
            
            idx_lookup = np.searchsorted(opt_ts, signal_ts, side='right') - 1
            idx_lookup = np.clip(idx_lookup, 0, len(opt_ts) - 1)
            
            call_vol = np.zeros(len(signals_df))
            put_vol = np.zeros(len(signals_df))
            
            valid = idx_lookup >= 0
            call_vol[valid] = cum_call_vol[idx_lookup[valid]]
            put_vol[valid] = cum_put_vol[idx_lookup[valid]]
            
            signals_df['total_call_volume'] = call_vol
            signals_df['total_put_volume'] = put_vol
            
            # Put/call ratio
            signals_df['put_call_ratio'] = np.where(
                call_vol > 0,
                put_vol / call_vol,
                0.0
            )
        else:
            signals_df['total_call_volume'] = 0.0
            signals_df['total_put_volume'] = 0.0
            signals_df['put_call_ratio'] = 0.0
        
        print(f"  Computed market options for {len(signals_df)} events")
        if 'total_gex' in signals_df.columns:
            print(f"  Total GEX: {signals_df['total_gex'].iloc[-1]:.0f} (final)")
        
        return {'signals_df': signals_df}

