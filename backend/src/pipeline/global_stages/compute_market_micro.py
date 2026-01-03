"""
Compute market-wide microstructure features.

Computes spread, depth, imbalance without level-relative filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.compute.microstructure import compute_market_microstructure


class ComputeMarketMicroStage(BaseStage):
    """
    Compute market-wide microstructure features at each time grid point.
    
    Features:
    - spread: Best ask - best bid
    - spread_pct: Spread as percentage of mid
    - bid_depth: Total bid depth across all 10 levels
    - ask_depth: Total ask depth across all 10 levels
    - depth_imbalance: (bid - ask) / (bid + ask)
    - mid_price: (best ask + best bid) / 2
    """
    
    @property
    def name(self) -> str:
        return "compute_market_micro"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10_snapshots']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        
        if signals_df.empty:
            return {'signals_df': signals_df}
        
        signal_ts = signals_df['ts_ns'].values.astype(np.int64)
        
        # Compute global microstructure (no level filtering)
        micro_features = compute_market_microstructure(
            signal_ts=signal_ts,
            mbp10_snapshots=mbp10_snapshots,
            level_price=None,  # Global mode
        )
        
        # Add features to DataFrame
        for name, values in micro_features.items():
            signals_df[name] = values
        
        # Update spot price from mid_price
        if 'mid_price' in signals_df.columns:
            signals_df['spot'] = signals_df['mid_price']
        
        # Compute derived features
        atr = ctx.data.get('atr', 0.0)
        # Ensure atr is a scalar
        if isinstance(atr, pd.Series):
            atr = atr.iloc[0] if len(atr) > 0 else 0.0
        if atr and atr > 0:
            signals_df['spread_atr'] = signals_df['spread'] / atr
        else:
            signals_df['spread_atr'] = 0.0
        
        print(f"  Computed market microstructure for {len(signals_df)} events")
        print(f"  Spread: {signals_df['spread'].mean():.4f} avg")
        print(f"  Depth imbalance: {signals_df['depth_imbalance'].mean():.4f} avg")
        
        return {'signals_df': signals_df}

