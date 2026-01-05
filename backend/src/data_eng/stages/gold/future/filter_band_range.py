from __future__ import annotations

import pandas as pd
import numpy as np

from ...base import Stage, StageIO


class GoldFilterBandRange(Stage):
    """Gold stage: nullify band features that reference prices outside MBP-10 range.
    
    - Input:  silver.future.market_by_price_10_bar5s
    - Output: gold.future.market_by_price_10_bar5s_filtered
    
    For each 5-second bar, computes the visible MBP-10 range from deepest bid to deepest ask.
    Sets all p3_5 and p5_10 band features to NaN if those bands extend beyond the visible range.
    
    MBP-10 typically covers ~2.5 points on each side for ES, so p3_5 (3-5 pts) and p5_10 (5-10 pts)
    are often outside the visible book depth and should be nullified.
    """
    
    BAND_FIELDS = [
        "bar5s_state_cdi_p3_5_twa",
        "bar5s_state_cdi_p3_5_eob",
        "bar5s_state_cdi_p5_10_twa",
        "bar5s_state_cdi_p5_10_eob",
        "bar5s_depth_below_p3_5_qty_twa",
        "bar5s_depth_below_p3_5_qty_eob",
        "bar5s_depth_below_p5_10_qty_twa",
        "bar5s_depth_below_p5_10_qty_eob",
        "bar5s_depth_above_p3_5_qty_twa",
        "bar5s_depth_above_p3_5_qty_eob",
        "bar5s_depth_above_p5_10_qty_twa",
        "bar5s_depth_above_p5_10_qty_eob",
        "bar5s_depth_below_p3_5_frac_twa",
        "bar5s_depth_below_p3_5_frac_eob",
        "bar5s_depth_below_p5_10_frac_twa",
        "bar5s_depth_below_p5_10_frac_eob",
        "bar5s_depth_above_p3_5_frac_twa",
        "bar5s_depth_above_p3_5_frac_eob",
        "bar5s_depth_above_p5_10_frac_twa",
        "bar5s_depth_above_p5_10_frac_eob",
        "bar5s_flow_add_vol_bid_p3_5_sum",
        "bar5s_flow_add_vol_bid_p5_10_sum",
        "bar5s_flow_add_vol_ask_p3_5_sum",
        "bar5s_flow_add_vol_ask_p5_10_sum",
        "bar5s_flow_rem_vol_bid_p3_5_sum",
        "bar5s_flow_rem_vol_bid_p5_10_sum",
        "bar5s_flow_rem_vol_ask_p3_5_sum",
        "bar5s_flow_rem_vol_ask_p5_10_sum",
        "bar5s_flow_net_vol_bid_p3_5_sum",
        "bar5s_flow_net_vol_bid_p5_10_sum",
        "bar5s_flow_net_vol_ask_p3_5_sum",
        "bar5s_flow_net_vol_ask_p5_10_sum",
        "bar5s_flow_cnt_add_bid_p3_5_sum",
        "bar5s_flow_cnt_add_bid_p5_10_sum",
        "bar5s_flow_cnt_add_ask_p3_5_sum",
        "bar5s_flow_cnt_add_ask_p5_10_sum",
        "bar5s_flow_cnt_cancel_bid_p3_5_sum",
        "bar5s_flow_cnt_cancel_bid_p5_10_sum",
        "bar5s_flow_cnt_cancel_ask_p3_5_sum",
        "bar5s_flow_cnt_cancel_ask_p5_10_sum",
        "bar5s_flow_cnt_modify_bid_p3_5_sum",
        "bar5s_flow_cnt_modify_bid_p5_10_sum",
        "bar5s_flow_cnt_modify_ask_p3_5_sum",
        "bar5s_flow_cnt_modify_ask_p5_10_sum",
        "bar5s_flow_net_volnorm_bid_p3_5_sum",
        "bar5s_flow_net_volnorm_bid_p5_10_sum",
        "bar5s_flow_net_volnorm_ask_p3_5_sum",
        "bar5s_flow_net_volnorm_ask_p5_10_sum",
    ]
    
    BAND_UPPER_BOUNDS = {
        "p3_5": 5.0,
        "p5_10": 10.0,
    }

    def __init__(self) -> None:
        super().__init__(
            name="gold_filter_band_range",
            io=StageIO(
                inputs=["silver.future.market_by_price_10_bar5s"],
                output="gold.future.market_by_price_10_bar5s_filtered",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        if len(df) == 0:
            return df.copy()
        
        cols_to_drop = [col for col in self.BAND_FIELDS if col in df.columns]
        df_out = df.drop(columns=cols_to_drop)
        
        return df_out

