"""
Detect interaction zone entries (level-relative event extraction).

Per Final Call v1 spec Section 4: Define events as physics interactions with a level
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
    
    Per Final Call v1 spec Section 4.3:
    - Must be deterministic (no random UUIDs)
    - Must uniquely identify the event
    
    Format: {date}_{level_kind}_{level_price_cents}_{anchor_ts_ns}_{direction}
    
    Args:
        date: Date string (YYYY-MM-DD)
        level_kind: Level type (e.g., 'PM_HIGH', 'SMA_200')
        level_price: Level price in SPX points
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


def detect_interaction_zone_entries(
    ohlcv_df: pd.DataFrame,
    level_prices: np.ndarray,
    level_kinds: np.ndarray,
    level_kind_names: List[str],
    date: str,
    atr: pd.Series = None
) -> pd.DataFrame:
    """
    Detect zone-based interaction events (not simple touches).
    
    Per Final Call v1 spec Section 4:
    - Define interaction zone with dynamic width (k × ATR)
    - Create event when price ENTERS zone from outside
    - Assign direction based on approach side
    - Use deterministic event IDs
    
    Args:
        ohlcv_df: OHLCV bars (SPX index points)
        level_prices: Array of level prices
        level_kinds: Array of level kind codes
        level_kind_names: List of level kind names
        date: Date string (YYYY-MM-DD)
        atr: Optional ATR series for dynamic zone width
    
    Returns:
        DataFrame with interaction events
    """
    if ohlcv_df.empty or len(level_prices) == 0:
        return pd.DataFrame()
    
    # Ensure timestamp column
    df = ohlcv_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    df['ts_ns'] = df['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    
    # Compute dynamic zone width
    # Per Final Call spec: width = max(w_min, k × ATR_RTH)
    w_min = 1.0  # SPX points (≈ $0.25 in SPY equivalent)
    k_atr = 0.5  # ATR multiplier
    
    if atr is not None and len(atr) > 0:
        # Use trailing ATR for each bar
        zone_width = np.maximum(w_min, k_atr * atr.fillna(w_min).values)
    else:
        # Fallback to fixed width
        zone_width = np.full(len(df), w_min)
    
    # Extract arrays for vectorized operations
    timestamps = df['timestamp'].values
    ts_ns = df['ts_ns'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    events = []
    
    # For each level, detect zone entries
    for idx, level_price in enumerate(level_prices):
        level_kind = int(level_kinds[idx])
        level_name = level_kind_names[idx]
        
        # Track whether we're inside the zone
        inside_zone = np.zeros(len(df), dtype=bool)
        
        for i in range(len(df)):
            half_width = zone_width[i]
            lower_bound = level_price - half_width
            upper_bound = level_price + half_width
            
            # Check if bar touched the zone
            inside_zone[i] = (lows[i] <= upper_bound) and (highs[i] >= lower_bound)
        
        # Detect zone entries (transitions from outside → inside)
        # Entry event = first bar where inside_zone=True after outside_zone=True
        for i in range(1, len(inside_zone)):
            if inside_zone[i] and not inside_zone[i-1]:
                # Zone entry detected!
                
                # Determine direction from approach side
                prev_close = closes[i-1]
                if prev_close < level_price:
                    direction = 'UP'  # Approaching from below (testing resistance)
                else:
                    direction = 'DOWN'  # Approaching from above (testing support)
                
                # Generate deterministic event ID
                event_id = compute_deterministic_event_id(
                    date=date,
                    level_kind=level_name,
                    level_price=level_price,
                    anchor_ts_ns=int(ts_ns[i]),
                    direction=direction
                )
                
                events.append({
                    'event_id': event_id,
                    'ts_ns': int(ts_ns[i]),
                    'timestamp': timestamps[i],
                    'level_price': level_price,
                    'level_kind': level_kind,
                    'level_kind_name': level_name,
                    'direction': direction,
                    'entry_price': closes[i],
                    'zone_width': zone_width[i],
                    'date': date
                })
    
    if not events:
        return pd.DataFrame()
    
    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values('ts_ns').reset_index(drop=True)
    
    return events_df


class DetectInteractionZonesStage(BaseStage):
    """
    Detect zone-based interaction events (replaces simple touch detection).
    
    Per Final Call v1 spec Section 4:
    - Dynamic interaction zones (ATR-scaled)
    - Entry events (not continuous touches)
    - Deterministic event IDs
    - Direction from approach side
    
    Outputs:
        touches_df: DataFrame with interaction events (renamed for compatibility)
    """
    
    @property
    def name(self) -> str:
        return "detect_interaction_zones"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['ohlcv_1min', 'level_info', 'atr']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        ohlcv_df = ctx.data['ohlcv_1min']
        level_info = ctx.data['level_info']
        atr = ctx.data.get('atr')
        
        # Detect interaction zone entries
        events_df = detect_interaction_zone_entries(
            ohlcv_df=ohlcv_df,
            level_prices=level_info.prices,
            level_kinds=level_info.kinds,
            level_kind_names=level_info.kind_names,
            date=ctx.date,
            atr=atr
        )
        
        # Return as 'touches_df' for compatibility with downstream stages
        return {'touches_df': events_df}

