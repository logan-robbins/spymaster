from __future__ import annotations

import pandas as pd

from ...base import Stage, StageIO
from .mbp10_bar5s import compute_bar5s_features


class SilverComputeBar5sFeatures(Stage):
    """Silver stage: compute 5-second bar aggregated features from MBP-10 tick data.
    
    - Input:  silver.future.market_by_price_10_session_levels
    - Output: silver.future.market_by_price_10_bar5s
    
    Aggregates tick-level MBP-10 events into 5-second bars with 233 features covering:
    - Meta (event counts)
    - State (spread, imbalances, TWA + EOB)
    - Depth (total & banded quantities/fractions, TWA + EOB)
    - Ladder (price gaps)
    - Shape (per-level sizes/counts)
    - Flow (add/remove/net volume and counts by band)
    - Trade (volume by aggressor side)
    - Wall (z-score based order detection)
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_bar5s_features",
            io=StageIO(
                inputs=["silver.future.market_by_price_10_session_levels"],
                output="silver.future.market_by_price_10_bar5s",
            ),
        )
    
    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame()
        
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "UNKNOWN"
        
        df_bars = compute_bar5s_features(df, symbol)
        
        return df_bars

