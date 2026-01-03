"""
Stage: Compute Market OFI (Global Pipeline)
Type: Feature Engineering (Market-Wide)
Input: Signals DataFrame (Time Grid), MBP-10 Snapshots
Output: Signals DataFrame with Global OFI Features

Transformation:
1. Replays ALL Order Book Updates (Snapshot-based L1 Deltas).
2. Computes System-Wide Order Flow Imbalance (OFI):
   - Measures net aggressor demand across the entire book (depth-agnostic).
   - No spatial filtering (all price levels contribute).
3. Aggregates over Time Windows (30s, 60s, 120s, 300s).
   - Captures "buying pressure" vs "selling pressure" at market scale.
4. Computes Flow Acceleration:
   - Ratio of short-term (30s) to long-term (120s) flow.
   
Note: High positive Market OFI implies strong broad-based buying pressure, often preceding a trend day.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.compute.ofi import compute_event_ofi, compute_ofi_windows


class ComputeMarketOFIStage(BaseStage):
    """
    Compute market-wide OFI at multiple lookback windows.
    
    Unlike level-relative OFI which filters spatially around a level,
    this computes total order flow imbalance across all price levels.
    
    Features:
    - ofi_30s, ofi_60s, ofi_120s, ofi_300s: Total OFI in lookback window
    - ofi_acceleration: Ratio of short-term to long-term OFI
    """
    
    @property
    def name(self) -> str:
        return "compute_market_ofi"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10_snapshots']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        
        if signals_df.empty or not mbp10_snapshots:
            # Add empty columns
            for w in [30, 60, 120, 300]:
                signals_df[f'ofi_{w}s'] = 0.0
            signals_df['ofi_acceleration'] = 0.0
            return {'signals_df': signals_df}
        
        signal_ts = signals_df['ts_ns'].values.astype(np.int64)
        
        # Compute raw event-based OFI
        ofi_timestamps, ofi_values, action_prices = compute_event_ofi(mbp10_snapshots)
        
        # Compute OFI windows (global mode - no level filtering)
        ofi_features = compute_ofi_windows(
            signal_ts=signal_ts,
            ofi_timestamps=ofi_timestamps,
            ofi_values=ofi_values,
            action_prices=action_prices,
            windows_seconds=[30.0, 60.0, 120.0, 300.0],
            level_price=None,  # Global mode - no spatial filtering
        )
        
        # Add features to DataFrame
        for name, values in ofi_features.items():
            signals_df[name] = values
        
        # Compute OFI acceleration (ratio of short-term to long-term OFI)
        # Only compute when ofi_120s has meaningful magnitude to avoid division instability
        if 'ofi_30s' in signals_df.columns and 'ofi_120s' in signals_df.columns:
            ofi_30 = signals_df['ofi_30s'].values
            ofi_120 = signals_df['ofi_120s'].values
            # Require |ofi_120s| > 5 for meaningful ratio, else set to 0
            signals_df['ofi_acceleration'] = np.divide(
                ofi_30, ofi_120,
                out=np.zeros_like(ofi_30),
                where=(np.abs(ofi_120) > 5)
            )
        
        print(f"  Computed market OFI for {len(signals_df)} events")
        print(f"  OFI_60s: min={signals_df['ofi_60s'].min():.0f}, max={signals_df['ofi_60s'].max():.0f}")
        
        return {'signals_df': signals_df}

