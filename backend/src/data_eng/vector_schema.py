from __future__ import annotations

from typing import List

F_DOWN = [
    "f1_ask_com_disp_log",
    "f1_ask_slope_convex_log",
    "f1_ask_slope_inner_log",
    "f1_ask_at_share_delta",
    "f1_ask_near_share_delta",
    "f1_ask_reprice_away_share_rest",
    "f2_ask_pull_add_log_rest",
    "f2_ask_pull_intensity_log_rest",
    "f2_ask_at_pull_share_rest",
    "f2_ask_near_pull_share_rest",
    "f3_bid_com_disp_log",
    "f3_bid_slope_convex_log",
    "f3_bid_slope_inner_log",
    "f3_bid_at_share_delta",
    "f3_bid_near_share_delta",
    "f3_bid_reprice_away_share_rest",
    "f4_bid_pull_add_log_rest",
    "f4_bid_pull_intensity_log_rest",
    "f4_bid_at_pull_share_rest",
    "f4_bid_near_pull_share_rest",
    "f5_vacuum_expansion_log",
    "f6_vacuum_decay_log",
    "f7_vacuum_total_log",
    "f8_ask_bbo_dist_ticks",
    "f9_bid_bbo_dist_ticks",
]

F_UP = [
    "u1_ask_com_disp_log",
    "u2_ask_slope_convex_log",
    "u2_ask_slope_inner_log",
    "u3_ask_at_share_decay",
    "u3_ask_near_share_decay",
    "u4_ask_reprice_away_share_rest",
    "u5_ask_pull_add_log_rest",
    "u6_ask_pull_intensity_log_rest",
    "u7_ask_at_pull_share_rest",
    "u7_ask_near_pull_share_rest",
    "u8_bid_com_approach_log",
    "u9_bid_slope_support_log",
    "u9_bid_slope_inner_log",
    "u10_bid_at_share_rise",
    "u10_bid_near_share_rise",
    "u11_bid_reprice_toward_share_rest",
    "u12_bid_add_pull_log_rest",
    "u13_bid_add_intensity_log",
    "u14_bid_far_pull_share_rest",
    "u15_up_expansion_log",
    "u16_up_flow_log",
    "u17_up_total_log",
    "u18_ask_bbo_dist_ticks",
    "u19_bid_bbo_dist_ticks",
]

X_COLUMNS = [
    col
    for name in F_DOWN
    for col in (name, f"d1_{name}", f"d2_{name}", f"d3_{name}")
] + [
    col
    for name in F_UP
    for col in (name, f"d1_{name}", f"d2_{name}", f"d3_{name}")
]

VECTOR_BLOCKS = [
    "w0",
    "w3_mean",
    "w3_delta",
    "w9_mean",
    "w9_delta",
    "w24_mean",
    "w24_delta",
]

VECTOR_DIM = len(X_COLUMNS) * len(VECTOR_BLOCKS)


def vector_feature_names() -> List[str]:
    names: List[str] = []
    for prefix in VECTOR_BLOCKS:
        for col in X_COLUMNS:
            names.append(f"{prefix}_{col}")
    return names


def vector_feature_rows() -> List[dict]:
    rows: List[dict] = []
    idx = 0
    for block in VECTOR_BLOCKS:
        for base in X_COLUMNS:
            rows.append(
                {
                    "feature_index": idx,
                    "feature_name": f"{block}_{base}",
                    "block": block,
                    "base_feature": base,
                    "vector_dim": VECTOR_DIM,
                }
            )
            idx += 1
    return rows
