from __future__ import annotations

import pandas as pd

from ...base import Stage, StageIO


class SilverFilterBandRange(Stage):
    """Silver stage: nullify deep band features outside MBP-10 visible range.

    - Input:  silver.future.market_by_price_10_bar5s (first 3 hours only)
    - Output: silver.future.market_by_price_10_bar5s_filtered

    Nullifies p3_5 and p5_10 band features since MBP-10 only shows 10 levels total
    (~2.5 points range), making deeper band patterns unreliable.

    After feature reduction, mainly affects normalized flow features in deep bands.
    """

    COLUMNS_TO_DROP = [
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
        "bar5s_shape_bid_px_l00_eob",
        "bar5s_shape_bid_px_l01_eob",
        "bar5s_shape_bid_px_l02_eob",
        "bar5s_shape_bid_px_l03_eob",
        "bar5s_shape_bid_px_l04_eob",
        "bar5s_shape_bid_px_l05_eob",
        "bar5s_shape_bid_px_l06_eob",
        "bar5s_shape_bid_px_l07_eob",
        "bar5s_shape_bid_px_l08_eob",
        "bar5s_shape_bid_px_l09_eob",
        "bar5s_shape_ask_px_l00_eob",
        "bar5s_shape_ask_px_l01_eob",
        "bar5s_shape_ask_px_l02_eob",
        "bar5s_shape_ask_px_l03_eob",
        "bar5s_shape_ask_px_l04_eob",
        "bar5s_shape_ask_px_l05_eob",
        "bar5s_shape_ask_px_l06_eob",
        "bar5s_shape_ask_px_l07_eob",
        "bar5s_shape_ask_px_l08_eob",
        "bar5s_shape_ask_px_l09_eob",
    ]

    def __init__(self) -> None:
        super().__init__(
            name="silver_filter_band_range",
            io=StageIO(
                inputs=["silver.future.market_by_price_10_bar5s"],
                output="silver.future.market_by_price_10_bar5s_filtered",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        if len(df) == 0:
            return df.copy()

        cols_to_drop = [col for col in self.COLUMNS_TO_DROP if col in df.columns]
        df_out = df.drop(columns=cols_to_drop)

        return df_out