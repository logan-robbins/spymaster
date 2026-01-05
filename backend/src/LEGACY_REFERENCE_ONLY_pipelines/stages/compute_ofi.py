"""
Compute Order Flow Imbalance (OFI) features from ES MBP-10.

Integrated OFI is the "force" proxy at L2.

OFI measures the net pressure from order book updates:
- Bid side: increases in bid size or decreases in ask size → buying pressure
- Ask side: increases in ask size or decreases in bid size → selling pressure

Integrated OFI: Cumulative weighted OFI over top N levels with distance decay.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import MBP10
from src.common.config import CONFIG


def compute_integrated_ofi(
    signals_df: pd.DataFrame,
    mbp10_snapshots: List[MBP10],
    lookback_seconds: float = 60.0
) -> pd.DataFrame:
    """
    Compute integrated OFI from MBP-10 snapshots.
    
    Integrated OFI = Σ_k w_k × OFI_k
    where w_k is distance decay weight for level k (nearer levels weighted higher).
    
    Args:
        signals_df: DataFrame with signals
        mbp10_snapshots: List of MBP-10 snapshots
        lookback_seconds: Window for OFI computation
    
    Returns:
        DataFrame with OFI features added
    """
    if signals_df.empty or not mbp10_snapshots:
        result = signals_df.copy()
        result['integrated_ofi'] = 0.0
        result['ofi_near_level'] = 0.0
        result['ofi_imbalance'] = 0.0
        return result
    
    # Build MBP-10 time series
    mbp_times = []
    mbp_bids = []  # List of bid price/size tuples (up to 10 levels)
    mbp_asks = []
    
    for mbp in mbp10_snapshots:
        mbp_times.append(mbp.ts_event_ns)
        
        bids = []
        asks = []
        for level in mbp.levels[:10]:  # Top 10 levels
            bids.append((level.bid_px, level.bid_sz))
            asks.append((level.ask_px, level.ask_sz))
        
        mbp_bids.append(bids)
        mbp_asks.append(asks)
    
    mbp_times = np.array(mbp_times, dtype=np.int64)
    lookback_ns = int(lookback_seconds * 1e9)
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    level_prices = signals_df['level_price'].values.astype(np.float64)
    
    integrated_ofi = np.zeros(n, dtype=np.float64)
    ofi_near_level = np.zeros(n, dtype=np.float64)
    ofi_imbalance = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        ts = signal_ts[i]
        level = level_prices[i]
        start_ts = ts - lookback_ns
        
        # Find MBP snapshots in window
        mask = (mbp_times >= start_ts) & (mbp_times <= ts)
        indices = np.where(mask)[0]
        
        if len(indices) < 2:
            continue
        
        # Compute OFI between consecutive snapshots
        ofi_values = []
        weights = []
        
        for j in range(len(indices) - 1):
            idx_curr = indices[j + 1]
            idx_prev = indices[j]
            
            bids_curr = mbp_bids[idx_curr]
            asks_curr = mbp_asks[idx_curr]
            bids_prev = mbp_bids[idx_prev]
            asks_prev = mbp_asks[idx_prev]
            
            # Compute OFI for top N levels with distance decay
            ofi_sum = 0.0
            weight_sum = 0.0
            
            for k in range(min(len(bids_curr), len(bids_prev))):
                # Get current and previous bid/ask
                bid_px_curr, bid_sz_curr = bids_curr[k]
                ask_px_curr, ask_sz_curr = asks_curr[k]
                bid_px_prev, bid_sz_prev = bids_prev[k]
                ask_px_prev, ask_sz_prev = asks_prev[k]
                
                # Skip if prices are zero (missing levels)
                if bid_px_curr == 0 or ask_px_curr == 0:
                    continue
                
                # Distance decay weight (closer to level = higher weight)
                mid_px = (bid_px_curr + ask_px_curr) / 2.0
                distance = abs(mid_px - level)
                weight = np.exp(-distance / 10.0)  # Decay scale ~10 points
                
                # OFI calculation
                # Bid side: increase in bid size = buying pressure
                # Ask side: decrease in ask size = buying pressure
                ofi_bid = bid_sz_curr - bid_sz_prev
                ofi_ask = -(ask_sz_curr - ask_sz_prev)
                ofi_level = ofi_bid + ofi_ask
                
                ofi_sum += weight * ofi_level
                weight_sum += weight
            
            if weight_sum > 0:
                ofi_values.append(ofi_sum / weight_sum)
                weights.append(1.0)  # Equal weight across time for now
        
        # Aggregate OFI across snapshots
        if ofi_values:
            integrated_ofi[i] = np.mean(ofi_values)
            
            # OFI near level: Average OFI weighted by proximity
            ofi_near_level[i] = integrated_ofi[i]
            
            # Imbalance: sign and magnitude
            ofi_imbalance[i] = np.sign(integrated_ofi[i]) * np.log1p(abs(integrated_ofi[i]))
    
    result = signals_df.copy()
    result['integrated_ofi'] = integrated_ofi
    result['ofi_near_level'] = ofi_near_level
    result['ofi_imbalance'] = ofi_imbalance
    
    return result


class ComputeOFIStage(BaseStage):
    """
    Compute integrated Order Flow Imbalance from MBP-10.
    
    OFI is the "force" proxy.
    
    Outputs:
        signals_df: Updated with OFI features
    """
    
    @property
    def name(self) -> str:
        return "compute_ofi"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10_snapshots']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        
        signals_df = compute_integrated_ofi(
            signals_df=signals_df,
            mbp10_snapshots=mbp10_snapshots,
            lookback_seconds=CONFIG.W_b  # Use barrier window
        )
        
        return {'signals_df': signals_df}

