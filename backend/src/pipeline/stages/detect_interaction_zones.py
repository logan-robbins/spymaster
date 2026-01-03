"""
Detect interaction zone entries (level-relative event extraction).

Define events as physics interactions with a level
(not single tick touches). This stage replaces the simple touch detection with
zone-based event extraction.

Key improvements:
- Dynamic interaction zone width (scales with ATR)
- Direction assignment from approach side
- Deterministic event IDs (reproducible for retrieval)
- Entry/exit tracking (not just touches)
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_deterministic_event_id(
    date: str,
    level_kind: str,
    level_price: float,
    anchor_ts_ns: int,
    direction: str
) -> str:
    """
    Generate deterministic event ID for retrieval reproducibility.
    
    Deterministic (no random UUIDs) - same inputs produce same ID.
    
    Format: {date}_{level_kind}_{level_price_cents}_{anchor_ts_ns}_{direction}
    Example: 20251203_PM_HIGH_692000_1733248200000000000_UP
    
    Args:
        date: Date string (YYYY-MM-DD)
        level_kind: Level type (PM_HIGH, OR_LOW, SMA_90, etc.)
        level_price: Level price in ES index points
        anchor_ts_ns: Anchor timestamp in nanoseconds
        direction: 'UP' or 'DOWN'
    
    Returns:
        Deterministic event ID string
    """
    # Round price to 2 decimal places (hundredths of a point)
    price_cents = int(level_price * 100)
    
    # Format: 20251203_PM_HIGH_574000_1733248200000000000_UP
    event_id = f"{date}_{level_kind}_{price_cents}_{anchor_ts_ns}_{direction}"
    
    return event_id


from src.common.event_types import FuturesTrade

def detect_entries_from_ticks(
    trades: List[FuturesTrade],
    level_prices: np.ndarray,
    level_name: str,
    date: str,
    atr: pd.Series = None,
    min_separation_sec: float = 300.0
) -> pd.DataFrame:
    """
    Detect level crosses using tick-by-tick analysis.
    
    Detects ANY price cross of the level (not just zone entries):
    - UP cross: prev_price < level AND curr_price >= level
    - DOWN cross: prev_price > level AND curr_price <= level
    
    This catches fast moves that gap through the level zone without
    leaving a tick inside the monitor band.
    
    Single-level pipeline: detects crosses for ONE level type only.
    
    Args:
        trades: List of raw FuturesTrade objects
        level_prices: Array of level prices (typically just one price for static levels)
        level_name: Level type name (e.g., 'PM_HIGH', 'PM_LOW')
        date: Date string
        atr: ATR series for dynamic width (optional)
        min_separation_sec: Debounce window (default 5m)
    """
    if not trades or len(level_prices) == 0:
        return pd.DataFrame()

    # Convert trades to arrays
    ts_ns = np.array([t.ts_event_ns for t in trades], dtype=np.int64)
    prices = np.array([t.price for t in trades], dtype=np.float64)
    
    events = []
    
    # Single-level pipeline: process THE level
    for level_price in level_prices:
        # Detect crosses between consecutive ticks
        # UP cross: prev < level AND curr >= level
        # DOWN cross: prev > level AND curr <= level
        
        if len(prices) < 2:
            continue
        
        prev_prices = prices[:-1]
        curr_prices = prices[1:]
        
        # UP crosses
        up_crosses = (prev_prices < level_price) & (curr_prices >= level_price)
        up_indices = np.where(up_crosses)[0] + 1  # Index in original array
        
        # DOWN crosses  
        down_crosses = (prev_prices > level_price) & (curr_prices <= level_price)
        down_indices = np.where(down_crosses)[0] + 1
        
        # Combine and sort
        all_cross_indices = np.concatenate([up_indices, down_indices])
        all_directions = np.concatenate([
            np.full(len(up_indices), 'UP'),
            np.full(len(down_indices), 'DOWN')
        ])
        
        # Sort by timestamp
        sort_order = np.argsort(ts_ns[all_cross_indices])
        cross_indices = all_cross_indices[sort_order]
        directions = all_directions[sort_order]
        
        # Check if first trade is already at level
        if len(prices) > 0 and abs(prices[0] - level_price) <= CONFIG.MONITOR_BAND:
            cross_indices = np.insert(cross_indices, 0, 0)
            directions = np.insert(directions, 0, 'UP' if prices[0] >= level_price else 'DOWN')
        
        if len(cross_indices) == 0:
            continue
        
        # Debounce
        last_event_ts = -np.inf
        
        for i, direction in zip(cross_indices, directions):
            t = ts_ns[i]
            if (t - last_event_ts) / 1e9 >= min_separation_sec:
                p = prices[i]
                
                event_id = compute_deterministic_event_id(
                    date=date, level_kind=level_name, level_price=level_price,
                    anchor_ts_ns=t, direction=direction
                )
                
                events.append({
                    'event_id': event_id,
                    'ts_ns': t,
                    'timestamp': pd.Timestamp(t, unit='ns', tz='UTC'),
                    'level_price': level_price,
                    'direction': direction,
                    'entry_price': p,
                    'spot': p,
                    'date': date
                })
                
                last_event_ts = t

    if not events:
        return pd.DataFrame()
        
    return pd.DataFrame(events).sort_values('ts_ns').reset_index(drop=True)


class DetectInteractionZonesStage(BaseStage):
    """
    Detect zone-based interaction events using HIGH-FREQUENCY ticks.
    """
    
    @property
    def name(self) -> str:
        return "detect_interaction_zones"
    
    @property
    def required_inputs(self) -> List[str]:
        # Now requires 'trades' instead of 'ohlcv_1min'
        return ['trades', 'level_info']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        trades = ctx.data.get('trades', [])
        level_info = ctx.data['level_info']
        
        # Filter to single level type (case-insensitive)
        target_level = ctx.level.upper() if ctx.level else None
        if not target_level:
            raise ValueError("ctx.level must be specified for level-specific pipeline")
        
        mask = np.array([name.upper() == target_level for name in level_info.kind_names])
        
        if not mask.any():
            raise ValueError(f"Level '{target_level}' not found. Available: {level_info.kind_names}")
        
        filtered_prices = level_info.prices[mask]
        
        events_df = detect_entries_from_ticks(
            trades=trades,
            level_prices=filtered_prices,
            level_name=target_level,
            date=ctx.date,
            min_separation_sec=300.0
        )
        
        num_events = len(events_df)
        if num_events > 0:
            print(f"  Detected {num_events} events for {target_level} from {len(trades)} ticks")
        else:
            print(f"  ⚠️  No events detected for {target_level} (level={filtered_prices[0]:.2f})")
        
        return {'touches_df': events_df}
