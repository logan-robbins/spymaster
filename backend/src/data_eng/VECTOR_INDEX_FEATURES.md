# Vector Index Features

All features are level-relative: prices and depths are measured around P_ref, and near/far buckets are defined in ticks (near <=5, far 15-20). All inputs are ratios/logs and discrete differences only (no raw quantities). Per-window x_k features are computed on 5s windows; d1/d2/d3 are first/second/third differences across consecutive windows. Vector features v_k are rollups of x_k across 3, 9, and 24 windows plus the current window.

Normalization at index build: robust scaling per-dimension using median/MAD (scale 1.4826), clip to +/-8, set zero-MAD dims to 0, then L2-normalize the vector.

## Feature List

| Feature | Meaning |
| --- | --- |
| `w0_f1_ask_com_disp_log` | current 5s value of log ratio of ask COM distance from level (ticks), end vs start |
| `w0_d1_f1_ask_com_disp_log` | current 5s value of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w0_d2_f1_ask_com_disp_log` | current 5s value of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w0_d3_f1_ask_com_disp_log` | current 5s value of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w0_f1_ask_slope_convex_log` | current 5s value of log ratio of ask depth far vs near at window end |
| `w0_d1_f1_ask_slope_convex_log` | current 5s value of first difference of log ratio of ask depth far vs near at window end |
| `w0_d2_f1_ask_slope_convex_log` | current 5s value of second difference of log ratio of ask depth far vs near at window end |
| `w0_d3_f1_ask_slope_convex_log` | current 5s value of third difference of log ratio of ask depth far vs near at window end |
| `w0_f1_ask_near_share_delta` | current 5s value of change in ask near depth share (end - start) |
| `w0_d1_f1_ask_near_share_delta` | current 5s value of first difference of change in ask near depth share (end - start) |
| `w0_d2_f1_ask_near_share_delta` | current 5s value of second difference of change in ask near depth share (end - start) |
| `w0_d3_f1_ask_near_share_delta` | current 5s value of third difference of change in ask near depth share (end - start) |
| `w0_f1_ask_reprice_away_share_rest` | current 5s value of share of ask reprice-away in resting reprices |
| `w0_d1_f1_ask_reprice_away_share_rest` | current 5s value of first difference of share of ask reprice-away in resting reprices |
| `w0_d2_f1_ask_reprice_away_share_rest` | current 5s value of second difference of share of ask reprice-away in resting reprices |
| `w0_d3_f1_ask_reprice_away_share_rest` | current 5s value of third difference of share of ask reprice-away in resting reprices |
| `w0_f2_ask_pull_add_log_rest` | current 5s value of log pull/add ratio for resting ask qty |
| `w0_d1_f2_ask_pull_add_log_rest` | current 5s value of first difference of log pull/add ratio for resting ask qty |
| `w0_d2_f2_ask_pull_add_log_rest` | current 5s value of second difference of log pull/add ratio for resting ask qty |
| `w0_d3_f2_ask_pull_add_log_rest` | current 5s value of third difference of log pull/add ratio for resting ask qty |
| `w0_f2_ask_pull_intensity_log_rest` | current 5s value of log1p ask pull qty / start ask depth total |
| `w0_d1_f2_ask_pull_intensity_log_rest` | current 5s value of first difference of log1p ask pull qty / start ask depth total |
| `w0_d2_f2_ask_pull_intensity_log_rest` | current 5s value of second difference of log1p ask pull qty / start ask depth total |
| `w0_d3_f2_ask_pull_intensity_log_rest` | current 5s value of third difference of log1p ask pull qty / start ask depth total |
| `w0_f2_ask_near_pull_share_rest` | current 5s value of share of ask pulls that were near |
| `w0_d1_f2_ask_near_pull_share_rest` | current 5s value of first difference of share of ask pulls that were near |
| `w0_d2_f2_ask_near_pull_share_rest` | current 5s value of second difference of share of ask pulls that were near |
| `w0_d3_f2_ask_near_pull_share_rest` | current 5s value of third difference of share of ask pulls that were near |
| `w0_f3_bid_com_disp_log` | current 5s value of log ratio of bid COM distance from level (ticks), end vs start |
| `w0_d1_f3_bid_com_disp_log` | current 5s value of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w0_d2_f3_bid_com_disp_log` | current 5s value of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w0_d3_f3_bid_com_disp_log` | current 5s value of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w0_f3_bid_slope_convex_log` | current 5s value of log ratio of bid depth far vs near at window end |
| `w0_d1_f3_bid_slope_convex_log` | current 5s value of first difference of log ratio of bid depth far vs near at window end |
| `w0_d2_f3_bid_slope_convex_log` | current 5s value of second difference of log ratio of bid depth far vs near at window end |
| `w0_d3_f3_bid_slope_convex_log` | current 5s value of third difference of log ratio of bid depth far vs near at window end |
| `w0_f3_bid_near_share_delta` | current 5s value of change in bid near depth share (end - start) |
| `w0_d1_f3_bid_near_share_delta` | current 5s value of first difference of change in bid near depth share (end - start) |
| `w0_d2_f3_bid_near_share_delta` | current 5s value of second difference of change in bid near depth share (end - start) |
| `w0_d3_f3_bid_near_share_delta` | current 5s value of third difference of change in bid near depth share (end - start) |
| `w0_f3_bid_reprice_away_share_rest` | current 5s value of share of bid reprice-away in resting reprices |
| `w0_d1_f3_bid_reprice_away_share_rest` | current 5s value of first difference of share of bid reprice-away in resting reprices |
| `w0_d2_f3_bid_reprice_away_share_rest` | current 5s value of second difference of share of bid reprice-away in resting reprices |
| `w0_d3_f3_bid_reprice_away_share_rest` | current 5s value of third difference of share of bid reprice-away in resting reprices |
| `w0_f4_bid_pull_add_log_rest` | current 5s value of log pull/add ratio for resting bid qty |
| `w0_d1_f4_bid_pull_add_log_rest` | current 5s value of first difference of log pull/add ratio for resting bid qty |
| `w0_d2_f4_bid_pull_add_log_rest` | current 5s value of second difference of log pull/add ratio for resting bid qty |
| `w0_d3_f4_bid_pull_add_log_rest` | current 5s value of third difference of log pull/add ratio for resting bid qty |
| `w0_f4_bid_pull_intensity_log_rest` | current 5s value of log1p bid pull qty / start bid depth total |
| `w0_d1_f4_bid_pull_intensity_log_rest` | current 5s value of first difference of log1p bid pull qty / start bid depth total |
| `w0_d2_f4_bid_pull_intensity_log_rest` | current 5s value of second difference of log1p bid pull qty / start bid depth total |
| `w0_d3_f4_bid_pull_intensity_log_rest` | current 5s value of third difference of log1p bid pull qty / start bid depth total |
| `w0_f4_bid_near_pull_share_rest` | current 5s value of share of bid pulls that were near |
| `w0_d1_f4_bid_near_pull_share_rest` | current 5s value of first difference of share of bid pulls that were near |
| `w0_d2_f4_bid_near_pull_share_rest` | current 5s value of second difference of share of bid pulls that were near |
| `w0_d3_f4_bid_near_pull_share_rest` | current 5s value of third difference of share of bid pulls that were near |
| `w0_f5_vacuum_expansion_log` | current 5s value of f1+f3 (ask+bid COM dispersion logs) |
| `w0_d1_f5_vacuum_expansion_log` | current 5s value of first difference of f1+f3 (ask+bid COM dispersion logs) |
| `w0_d2_f5_vacuum_expansion_log` | current 5s value of second difference of f1+f3 (ask+bid COM dispersion logs) |
| `w0_d3_f5_vacuum_expansion_log` | current 5s value of third difference of f1+f3 (ask+bid COM dispersion logs) |
| `w0_f6_vacuum_decay_log` | current 5s value of f2+f4 (ask+bid pull/add logs) |
| `w0_d1_f6_vacuum_decay_log` | current 5s value of first difference of f2+f4 (ask+bid pull/add logs) |
| `w0_d2_f6_vacuum_decay_log` | current 5s value of second difference of f2+f4 (ask+bid pull/add logs) |
| `w0_d3_f6_vacuum_decay_log` | current 5s value of third difference of f2+f4 (ask+bid pull/add logs) |
| `w0_f7_vacuum_total_log` | current 5s value of f5+f6 (total vacuum log) |
| `w0_d1_f7_vacuum_total_log` | current 5s value of first difference of f5+f6 (total vacuum log) |
| `w0_d2_f7_vacuum_total_log` | current 5s value of second difference of f5+f6 (total vacuum log) |
| `w0_d3_f7_vacuum_total_log` | current 5s value of third difference of f5+f6 (total vacuum log) |
| `w0_u1_ask_com_disp_log` | current 5s value of copy of f1 ask COM dispersion log |
| `w0_d1_u1_ask_com_disp_log` | current 5s value of first difference of copy of f1 ask COM dispersion log |
| `w0_d2_u1_ask_com_disp_log` | current 5s value of second difference of copy of f1 ask COM dispersion log |
| `w0_d3_u1_ask_com_disp_log` | current 5s value of third difference of copy of f1 ask COM dispersion log |
| `w0_u2_ask_slope_convex_log` | current 5s value of copy of f1 ask depth slope log |
| `w0_d1_u2_ask_slope_convex_log` | current 5s value of first difference of copy of f1 ask depth slope log |
| `w0_d2_u2_ask_slope_convex_log` | current 5s value of second difference of copy of f1 ask depth slope log |
| `w0_d3_u2_ask_slope_convex_log` | current 5s value of third difference of copy of f1 ask depth slope log |
| `w0_u3_ask_near_share_decay` | current 5s value of negative of f1 ask near share delta |
| `w0_d1_u3_ask_near_share_decay` | current 5s value of first difference of negative of f1 ask near share delta |
| `w0_d2_u3_ask_near_share_decay` | current 5s value of second difference of negative of f1 ask near share delta |
| `w0_d3_u3_ask_near_share_decay` | current 5s value of third difference of negative of f1 ask near share delta |
| `w0_u4_ask_reprice_away_share_rest` | current 5s value of copy of f1 ask reprice-away share |
| `w0_d1_u4_ask_reprice_away_share_rest` | current 5s value of first difference of copy of f1 ask reprice-away share |
| `w0_d2_u4_ask_reprice_away_share_rest` | current 5s value of second difference of copy of f1 ask reprice-away share |
| `w0_d3_u4_ask_reprice_away_share_rest` | current 5s value of third difference of copy of f1 ask reprice-away share |
| `w0_u5_ask_pull_add_log_rest` | current 5s value of copy of f2 ask pull/add log |
| `w0_d1_u5_ask_pull_add_log_rest` | current 5s value of first difference of copy of f2 ask pull/add log |
| `w0_d2_u5_ask_pull_add_log_rest` | current 5s value of second difference of copy of f2 ask pull/add log |
| `w0_d3_u5_ask_pull_add_log_rest` | current 5s value of third difference of copy of f2 ask pull/add log |
| `w0_u6_ask_pull_intensity_log_rest` | current 5s value of copy of f2 ask pull intensity log |
| `w0_d1_u6_ask_pull_intensity_log_rest` | current 5s value of first difference of copy of f2 ask pull intensity log |
| `w0_d2_u6_ask_pull_intensity_log_rest` | current 5s value of second difference of copy of f2 ask pull intensity log |
| `w0_d3_u6_ask_pull_intensity_log_rest` | current 5s value of third difference of copy of f2 ask pull intensity log |
| `w0_u7_ask_near_pull_share_rest` | current 5s value of copy of f2 ask near pull share |
| `w0_d1_u7_ask_near_pull_share_rest` | current 5s value of first difference of copy of f2 ask near pull share |
| `w0_d2_u7_ask_near_pull_share_rest` | current 5s value of second difference of copy of f2 ask near pull share |
| `w0_d3_u7_ask_near_pull_share_rest` | current 5s value of third difference of copy of f2 ask near pull share |
| `w0_u8_bid_com_approach_log` | current 5s value of negative of f3 bid COM dispersion log |
| `w0_d1_u8_bid_com_approach_log` | current 5s value of first difference of negative of f3 bid COM dispersion log |
| `w0_d2_u8_bid_com_approach_log` | current 5s value of second difference of negative of f3 bid COM dispersion log |
| `w0_d3_u8_bid_com_approach_log` | current 5s value of third difference of negative of f3 bid COM dispersion log |
| `w0_u9_bid_slope_support_log` | current 5s value of negative of f3 bid depth slope log |
| `w0_d1_u9_bid_slope_support_log` | current 5s value of first difference of negative of f3 bid depth slope log |
| `w0_d2_u9_bid_slope_support_log` | current 5s value of second difference of negative of f3 bid depth slope log |
| `w0_d3_u9_bid_slope_support_log` | current 5s value of third difference of negative of f3 bid depth slope log |
| `w0_u10_bid_near_share_rise` | current 5s value of copy of f3 bid near share delta |
| `w0_d1_u10_bid_near_share_rise` | current 5s value of first difference of copy of f3 bid near share delta |
| `w0_d2_u10_bid_near_share_rise` | current 5s value of second difference of copy of f3 bid near share delta |
| `w0_d3_u10_bid_near_share_rise` | current 5s value of third difference of copy of f3 bid near share delta |
| `w0_u11_bid_reprice_toward_share_rest` | current 5s value of 1 - f3 bid reprice-away share |
| `w0_d1_u11_bid_reprice_toward_share_rest` | current 5s value of first difference of 1 - f3 bid reprice-away share |
| `w0_d2_u11_bid_reprice_toward_share_rest` | current 5s value of second difference of 1 - f3 bid reprice-away share |
| `w0_d3_u11_bid_reprice_toward_share_rest` | current 5s value of third difference of 1 - f3 bid reprice-away share |
| `w0_u12_bid_add_pull_log_rest` | current 5s value of negative of f4 bid pull/add log |
| `w0_d1_u12_bid_add_pull_log_rest` | current 5s value of first difference of negative of f4 bid pull/add log |
| `w0_d2_u12_bid_add_pull_log_rest` | current 5s value of second difference of negative of f4 bid pull/add log |
| `w0_d3_u12_bid_add_pull_log_rest` | current 5s value of third difference of negative of f4 bid pull/add log |
| `w0_u13_bid_add_intensity_log` | current 5s value of log1p bid add qty / start bid depth total |
| `w0_d1_u13_bid_add_intensity_log` | current 5s value of first difference of log1p bid add qty / start bid depth total |
| `w0_d2_u13_bid_add_intensity_log` | current 5s value of second difference of log1p bid add qty / start bid depth total |
| `w0_d3_u13_bid_add_intensity_log` | current 5s value of third difference of log1p bid add qty / start bid depth total |
| `w0_u14_bid_far_pull_share_rest` | current 5s value of 1 - f4 bid near pull share |
| `w0_d1_u14_bid_far_pull_share_rest` | current 5s value of first difference of 1 - f4 bid near pull share |
| `w0_d2_u14_bid_far_pull_share_rest` | current 5s value of second difference of 1 - f4 bid near pull share |
| `w0_d3_u14_bid_far_pull_share_rest` | current 5s value of third difference of 1 - f4 bid near pull share |
| `w0_u15_up_expansion_log` | current 5s value of u1+u8 (up expansion log) |
| `w0_d1_u15_up_expansion_log` | current 5s value of first difference of u1+u8 (up expansion log) |
| `w0_d2_u15_up_expansion_log` | current 5s value of second difference of u1+u8 (up expansion log) |
| `w0_d3_u15_up_expansion_log` | current 5s value of third difference of u1+u8 (up expansion log) |
| `w0_u16_up_flow_log` | current 5s value of u5+u12 (up flow log) |
| `w0_d1_u16_up_flow_log` | current 5s value of first difference of u5+u12 (up flow log) |
| `w0_d2_u16_up_flow_log` | current 5s value of second difference of u5+u12 (up flow log) |
| `w0_d3_u16_up_flow_log` | current 5s value of third difference of u5+u12 (up flow log) |
| `w0_u17_up_total_log` | current 5s value of u15+u16 (up total log) |
| `w0_d1_u17_up_total_log` | current 5s value of first difference of u15+u16 (up total log) |
| `w0_d2_u17_up_total_log` | current 5s value of second difference of u15+u16 (up total log) |
| `w0_d3_u17_up_total_log` | current 5s value of third difference of u15+u16 (up total log) |
| `w3_mean_f1_ask_com_disp_log` | 3-window mean (15s) of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_mean_d1_f1_ask_com_disp_log` | 3-window mean (15s) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_mean_d2_f1_ask_com_disp_log` | 3-window mean (15s) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_mean_d3_f1_ask_com_disp_log` | 3-window mean (15s) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_mean_f1_ask_slope_convex_log` | 3-window mean (15s) of log ratio of ask depth far vs near at window end |
| `w3_mean_d1_f1_ask_slope_convex_log` | 3-window mean (15s) of first difference of log ratio of ask depth far vs near at window end |
| `w3_mean_d2_f1_ask_slope_convex_log` | 3-window mean (15s) of second difference of log ratio of ask depth far vs near at window end |
| `w3_mean_d3_f1_ask_slope_convex_log` | 3-window mean (15s) of third difference of log ratio of ask depth far vs near at window end |
| `w3_mean_f1_ask_near_share_delta` | 3-window mean (15s) of change in ask near depth share (end - start) |
| `w3_mean_d1_f1_ask_near_share_delta` | 3-window mean (15s) of first difference of change in ask near depth share (end - start) |
| `w3_mean_d2_f1_ask_near_share_delta` | 3-window mean (15s) of second difference of change in ask near depth share (end - start) |
| `w3_mean_d3_f1_ask_near_share_delta` | 3-window mean (15s) of third difference of change in ask near depth share (end - start) |
| `w3_mean_f1_ask_reprice_away_share_rest` | 3-window mean (15s) of share of ask reprice-away in resting reprices |
| `w3_mean_d1_f1_ask_reprice_away_share_rest` | 3-window mean (15s) of first difference of share of ask reprice-away in resting reprices |
| `w3_mean_d2_f1_ask_reprice_away_share_rest` | 3-window mean (15s) of second difference of share of ask reprice-away in resting reprices |
| `w3_mean_d3_f1_ask_reprice_away_share_rest` | 3-window mean (15s) of third difference of share of ask reprice-away in resting reprices |
| `w3_mean_f2_ask_pull_add_log_rest` | 3-window mean (15s) of log pull/add ratio for resting ask qty |
| `w3_mean_d1_f2_ask_pull_add_log_rest` | 3-window mean (15s) of first difference of log pull/add ratio for resting ask qty |
| `w3_mean_d2_f2_ask_pull_add_log_rest` | 3-window mean (15s) of second difference of log pull/add ratio for resting ask qty |
| `w3_mean_d3_f2_ask_pull_add_log_rest` | 3-window mean (15s) of third difference of log pull/add ratio for resting ask qty |
| `w3_mean_f2_ask_pull_intensity_log_rest` | 3-window mean (15s) of log1p ask pull qty / start ask depth total |
| `w3_mean_d1_f2_ask_pull_intensity_log_rest` | 3-window mean (15s) of first difference of log1p ask pull qty / start ask depth total |
| `w3_mean_d2_f2_ask_pull_intensity_log_rest` | 3-window mean (15s) of second difference of log1p ask pull qty / start ask depth total |
| `w3_mean_d3_f2_ask_pull_intensity_log_rest` | 3-window mean (15s) of third difference of log1p ask pull qty / start ask depth total |
| `w3_mean_f2_ask_near_pull_share_rest` | 3-window mean (15s) of share of ask pulls that were near |
| `w3_mean_d1_f2_ask_near_pull_share_rest` | 3-window mean (15s) of first difference of share of ask pulls that were near |
| `w3_mean_d2_f2_ask_near_pull_share_rest` | 3-window mean (15s) of second difference of share of ask pulls that were near |
| `w3_mean_d3_f2_ask_near_pull_share_rest` | 3-window mean (15s) of third difference of share of ask pulls that were near |
| `w3_mean_f3_bid_com_disp_log` | 3-window mean (15s) of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_mean_d1_f3_bid_com_disp_log` | 3-window mean (15s) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_mean_d2_f3_bid_com_disp_log` | 3-window mean (15s) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_mean_d3_f3_bid_com_disp_log` | 3-window mean (15s) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_mean_f3_bid_slope_convex_log` | 3-window mean (15s) of log ratio of bid depth far vs near at window end |
| `w3_mean_d1_f3_bid_slope_convex_log` | 3-window mean (15s) of first difference of log ratio of bid depth far vs near at window end |
| `w3_mean_d2_f3_bid_slope_convex_log` | 3-window mean (15s) of second difference of log ratio of bid depth far vs near at window end |
| `w3_mean_d3_f3_bid_slope_convex_log` | 3-window mean (15s) of third difference of log ratio of bid depth far vs near at window end |
| `w3_mean_f3_bid_near_share_delta` | 3-window mean (15s) of change in bid near depth share (end - start) |
| `w3_mean_d1_f3_bid_near_share_delta` | 3-window mean (15s) of first difference of change in bid near depth share (end - start) |
| `w3_mean_d2_f3_bid_near_share_delta` | 3-window mean (15s) of second difference of change in bid near depth share (end - start) |
| `w3_mean_d3_f3_bid_near_share_delta` | 3-window mean (15s) of third difference of change in bid near depth share (end - start) |
| `w3_mean_f3_bid_reprice_away_share_rest` | 3-window mean (15s) of share of bid reprice-away in resting reprices |
| `w3_mean_d1_f3_bid_reprice_away_share_rest` | 3-window mean (15s) of first difference of share of bid reprice-away in resting reprices |
| `w3_mean_d2_f3_bid_reprice_away_share_rest` | 3-window mean (15s) of second difference of share of bid reprice-away in resting reprices |
| `w3_mean_d3_f3_bid_reprice_away_share_rest` | 3-window mean (15s) of third difference of share of bid reprice-away in resting reprices |
| `w3_mean_f4_bid_pull_add_log_rest` | 3-window mean (15s) of log pull/add ratio for resting bid qty |
| `w3_mean_d1_f4_bid_pull_add_log_rest` | 3-window mean (15s) of first difference of log pull/add ratio for resting bid qty |
| `w3_mean_d2_f4_bid_pull_add_log_rest` | 3-window mean (15s) of second difference of log pull/add ratio for resting bid qty |
| `w3_mean_d3_f4_bid_pull_add_log_rest` | 3-window mean (15s) of third difference of log pull/add ratio for resting bid qty |
| `w3_mean_f4_bid_pull_intensity_log_rest` | 3-window mean (15s) of log1p bid pull qty / start bid depth total |
| `w3_mean_d1_f4_bid_pull_intensity_log_rest` | 3-window mean (15s) of first difference of log1p bid pull qty / start bid depth total |
| `w3_mean_d2_f4_bid_pull_intensity_log_rest` | 3-window mean (15s) of second difference of log1p bid pull qty / start bid depth total |
| `w3_mean_d3_f4_bid_pull_intensity_log_rest` | 3-window mean (15s) of third difference of log1p bid pull qty / start bid depth total |
| `w3_mean_f4_bid_near_pull_share_rest` | 3-window mean (15s) of share of bid pulls that were near |
| `w3_mean_d1_f4_bid_near_pull_share_rest` | 3-window mean (15s) of first difference of share of bid pulls that were near |
| `w3_mean_d2_f4_bid_near_pull_share_rest` | 3-window mean (15s) of second difference of share of bid pulls that were near |
| `w3_mean_d3_f4_bid_near_pull_share_rest` | 3-window mean (15s) of third difference of share of bid pulls that were near |
| `w3_mean_f5_vacuum_expansion_log` | 3-window mean (15s) of f1+f3 (ask+bid COM dispersion logs) |
| `w3_mean_d1_f5_vacuum_expansion_log` | 3-window mean (15s) of first difference of f1+f3 (ask+bid COM dispersion logs) |
| `w3_mean_d2_f5_vacuum_expansion_log` | 3-window mean (15s) of second difference of f1+f3 (ask+bid COM dispersion logs) |
| `w3_mean_d3_f5_vacuum_expansion_log` | 3-window mean (15s) of third difference of f1+f3 (ask+bid COM dispersion logs) |
| `w3_mean_f6_vacuum_decay_log` | 3-window mean (15s) of f2+f4 (ask+bid pull/add logs) |
| `w3_mean_d1_f6_vacuum_decay_log` | 3-window mean (15s) of first difference of f2+f4 (ask+bid pull/add logs) |
| `w3_mean_d2_f6_vacuum_decay_log` | 3-window mean (15s) of second difference of f2+f4 (ask+bid pull/add logs) |
| `w3_mean_d3_f6_vacuum_decay_log` | 3-window mean (15s) of third difference of f2+f4 (ask+bid pull/add logs) |
| `w3_mean_f7_vacuum_total_log` | 3-window mean (15s) of f5+f6 (total vacuum log) |
| `w3_mean_d1_f7_vacuum_total_log` | 3-window mean (15s) of first difference of f5+f6 (total vacuum log) |
| `w3_mean_d2_f7_vacuum_total_log` | 3-window mean (15s) of second difference of f5+f6 (total vacuum log) |
| `w3_mean_d3_f7_vacuum_total_log` | 3-window mean (15s) of third difference of f5+f6 (total vacuum log) |
| `w3_mean_u1_ask_com_disp_log` | 3-window mean (15s) of copy of f1 ask COM dispersion log |
| `w3_mean_d1_u1_ask_com_disp_log` | 3-window mean (15s) of first difference of copy of f1 ask COM dispersion log |
| `w3_mean_d2_u1_ask_com_disp_log` | 3-window mean (15s) of second difference of copy of f1 ask COM dispersion log |
| `w3_mean_d3_u1_ask_com_disp_log` | 3-window mean (15s) of third difference of copy of f1 ask COM dispersion log |
| `w3_mean_u2_ask_slope_convex_log` | 3-window mean (15s) of copy of f1 ask depth slope log |
| `w3_mean_d1_u2_ask_slope_convex_log` | 3-window mean (15s) of first difference of copy of f1 ask depth slope log |
| `w3_mean_d2_u2_ask_slope_convex_log` | 3-window mean (15s) of second difference of copy of f1 ask depth slope log |
| `w3_mean_d3_u2_ask_slope_convex_log` | 3-window mean (15s) of third difference of copy of f1 ask depth slope log |
| `w3_mean_u3_ask_near_share_decay` | 3-window mean (15s) of negative of f1 ask near share delta |
| `w3_mean_d1_u3_ask_near_share_decay` | 3-window mean (15s) of first difference of negative of f1 ask near share delta |
| `w3_mean_d2_u3_ask_near_share_decay` | 3-window mean (15s) of second difference of negative of f1 ask near share delta |
| `w3_mean_d3_u3_ask_near_share_decay` | 3-window mean (15s) of third difference of negative of f1 ask near share delta |
| `w3_mean_u4_ask_reprice_away_share_rest` | 3-window mean (15s) of copy of f1 ask reprice-away share |
| `w3_mean_d1_u4_ask_reprice_away_share_rest` | 3-window mean (15s) of first difference of copy of f1 ask reprice-away share |
| `w3_mean_d2_u4_ask_reprice_away_share_rest` | 3-window mean (15s) of second difference of copy of f1 ask reprice-away share |
| `w3_mean_d3_u4_ask_reprice_away_share_rest` | 3-window mean (15s) of third difference of copy of f1 ask reprice-away share |
| `w3_mean_u5_ask_pull_add_log_rest` | 3-window mean (15s) of copy of f2 ask pull/add log |
| `w3_mean_d1_u5_ask_pull_add_log_rest` | 3-window mean (15s) of first difference of copy of f2 ask pull/add log |
| `w3_mean_d2_u5_ask_pull_add_log_rest` | 3-window mean (15s) of second difference of copy of f2 ask pull/add log |
| `w3_mean_d3_u5_ask_pull_add_log_rest` | 3-window mean (15s) of third difference of copy of f2 ask pull/add log |
| `w3_mean_u6_ask_pull_intensity_log_rest` | 3-window mean (15s) of copy of f2 ask pull intensity log |
| `w3_mean_d1_u6_ask_pull_intensity_log_rest` | 3-window mean (15s) of first difference of copy of f2 ask pull intensity log |
| `w3_mean_d2_u6_ask_pull_intensity_log_rest` | 3-window mean (15s) of second difference of copy of f2 ask pull intensity log |
| `w3_mean_d3_u6_ask_pull_intensity_log_rest` | 3-window mean (15s) of third difference of copy of f2 ask pull intensity log |
| `w3_mean_u7_ask_near_pull_share_rest` | 3-window mean (15s) of copy of f2 ask near pull share |
| `w3_mean_d1_u7_ask_near_pull_share_rest` | 3-window mean (15s) of first difference of copy of f2 ask near pull share |
| `w3_mean_d2_u7_ask_near_pull_share_rest` | 3-window mean (15s) of second difference of copy of f2 ask near pull share |
| `w3_mean_d3_u7_ask_near_pull_share_rest` | 3-window mean (15s) of third difference of copy of f2 ask near pull share |
| `w3_mean_u8_bid_com_approach_log` | 3-window mean (15s) of negative of f3 bid COM dispersion log |
| `w3_mean_d1_u8_bid_com_approach_log` | 3-window mean (15s) of first difference of negative of f3 bid COM dispersion log |
| `w3_mean_d2_u8_bid_com_approach_log` | 3-window mean (15s) of second difference of negative of f3 bid COM dispersion log |
| `w3_mean_d3_u8_bid_com_approach_log` | 3-window mean (15s) of third difference of negative of f3 bid COM dispersion log |
| `w3_mean_u9_bid_slope_support_log` | 3-window mean (15s) of negative of f3 bid depth slope log |
| `w3_mean_d1_u9_bid_slope_support_log` | 3-window mean (15s) of first difference of negative of f3 bid depth slope log |
| `w3_mean_d2_u9_bid_slope_support_log` | 3-window mean (15s) of second difference of negative of f3 bid depth slope log |
| `w3_mean_d3_u9_bid_slope_support_log` | 3-window mean (15s) of third difference of negative of f3 bid depth slope log |
| `w3_mean_u10_bid_near_share_rise` | 3-window mean (15s) of copy of f3 bid near share delta |
| `w3_mean_d1_u10_bid_near_share_rise` | 3-window mean (15s) of first difference of copy of f3 bid near share delta |
| `w3_mean_d2_u10_bid_near_share_rise` | 3-window mean (15s) of second difference of copy of f3 bid near share delta |
| `w3_mean_d3_u10_bid_near_share_rise` | 3-window mean (15s) of third difference of copy of f3 bid near share delta |
| `w3_mean_u11_bid_reprice_toward_share_rest` | 3-window mean (15s) of 1 - f3 bid reprice-away share |
| `w3_mean_d1_u11_bid_reprice_toward_share_rest` | 3-window mean (15s) of first difference of 1 - f3 bid reprice-away share |
| `w3_mean_d2_u11_bid_reprice_toward_share_rest` | 3-window mean (15s) of second difference of 1 - f3 bid reprice-away share |
| `w3_mean_d3_u11_bid_reprice_toward_share_rest` | 3-window mean (15s) of third difference of 1 - f3 bid reprice-away share |
| `w3_mean_u12_bid_add_pull_log_rest` | 3-window mean (15s) of negative of f4 bid pull/add log |
| `w3_mean_d1_u12_bid_add_pull_log_rest` | 3-window mean (15s) of first difference of negative of f4 bid pull/add log |
| `w3_mean_d2_u12_bid_add_pull_log_rest` | 3-window mean (15s) of second difference of negative of f4 bid pull/add log |
| `w3_mean_d3_u12_bid_add_pull_log_rest` | 3-window mean (15s) of third difference of negative of f4 bid pull/add log |
| `w3_mean_u13_bid_add_intensity_log` | 3-window mean (15s) of log1p bid add qty / start bid depth total |
| `w3_mean_d1_u13_bid_add_intensity_log` | 3-window mean (15s) of first difference of log1p bid add qty / start bid depth total |
| `w3_mean_d2_u13_bid_add_intensity_log` | 3-window mean (15s) of second difference of log1p bid add qty / start bid depth total |
| `w3_mean_d3_u13_bid_add_intensity_log` | 3-window mean (15s) of third difference of log1p bid add qty / start bid depth total |
| `w3_mean_u14_bid_far_pull_share_rest` | 3-window mean (15s) of 1 - f4 bid near pull share |
| `w3_mean_d1_u14_bid_far_pull_share_rest` | 3-window mean (15s) of first difference of 1 - f4 bid near pull share |
| `w3_mean_d2_u14_bid_far_pull_share_rest` | 3-window mean (15s) of second difference of 1 - f4 bid near pull share |
| `w3_mean_d3_u14_bid_far_pull_share_rest` | 3-window mean (15s) of third difference of 1 - f4 bid near pull share |
| `w3_mean_u15_up_expansion_log` | 3-window mean (15s) of u1+u8 (up expansion log) |
| `w3_mean_d1_u15_up_expansion_log` | 3-window mean (15s) of first difference of u1+u8 (up expansion log) |
| `w3_mean_d2_u15_up_expansion_log` | 3-window mean (15s) of second difference of u1+u8 (up expansion log) |
| `w3_mean_d3_u15_up_expansion_log` | 3-window mean (15s) of third difference of u1+u8 (up expansion log) |
| `w3_mean_u16_up_flow_log` | 3-window mean (15s) of u5+u12 (up flow log) |
| `w3_mean_d1_u16_up_flow_log` | 3-window mean (15s) of first difference of u5+u12 (up flow log) |
| `w3_mean_d2_u16_up_flow_log` | 3-window mean (15s) of second difference of u5+u12 (up flow log) |
| `w3_mean_d3_u16_up_flow_log` | 3-window mean (15s) of third difference of u5+u12 (up flow log) |
| `w3_mean_u17_up_total_log` | 3-window mean (15s) of u15+u16 (up total log) |
| `w3_mean_d1_u17_up_total_log` | 3-window mean (15s) of first difference of u15+u16 (up total log) |
| `w3_mean_d2_u17_up_total_log` | 3-window mean (15s) of second difference of u15+u16 (up total log) |
| `w3_mean_d3_u17_up_total_log` | 3-window mean (15s) of third difference of u15+u16 (up total log) |
| `w3_delta_f1_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_delta_d1_f1_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_delta_d2_f1_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_delta_d3_f1_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_delta_f1_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of log ratio of ask depth far vs near at window end |
| `w3_delta_d1_f1_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log ratio of ask depth far vs near at window end |
| `w3_delta_d2_f1_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log ratio of ask depth far vs near at window end |
| `w3_delta_d3_f1_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log ratio of ask depth far vs near at window end |
| `w3_delta_f1_ask_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of change in ask near depth share (end - start) |
| `w3_delta_d1_f1_ask_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of first difference of change in ask near depth share (end - start) |
| `w3_delta_d2_f1_ask_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of second difference of change in ask near depth share (end - start) |
| `w3_delta_d3_f1_ask_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of third difference of change in ask near depth share (end - start) |
| `w3_delta_f1_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of share of ask reprice-away in resting reprices |
| `w3_delta_d1_f1_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of share of ask reprice-away in resting reprices |
| `w3_delta_d2_f1_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of share of ask reprice-away in resting reprices |
| `w3_delta_d3_f1_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of share of ask reprice-away in resting reprices |
| `w3_delta_f2_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of log pull/add ratio for resting ask qty |
| `w3_delta_d1_f2_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log pull/add ratio for resting ask qty |
| `w3_delta_d2_f2_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log pull/add ratio for resting ask qty |
| `w3_delta_d3_f2_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log pull/add ratio for resting ask qty |
| `w3_delta_f2_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of log1p ask pull qty / start ask depth total |
| `w3_delta_d1_f2_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log1p ask pull qty / start ask depth total |
| `w3_delta_d2_f2_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log1p ask pull qty / start ask depth total |
| `w3_delta_d3_f2_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log1p ask pull qty / start ask depth total |
| `w3_delta_f2_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of share of ask pulls that were near |
| `w3_delta_d1_f2_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of share of ask pulls that were near |
| `w3_delta_d2_f2_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of share of ask pulls that were near |
| `w3_delta_d3_f2_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of share of ask pulls that were near |
| `w3_delta_f3_bid_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_delta_d1_f3_bid_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_delta_d2_f3_bid_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_delta_d3_f3_bid_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_delta_f3_bid_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of log ratio of bid depth far vs near at window end |
| `w3_delta_d1_f3_bid_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log ratio of bid depth far vs near at window end |
| `w3_delta_d2_f3_bid_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log ratio of bid depth far vs near at window end |
| `w3_delta_d3_f3_bid_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log ratio of bid depth far vs near at window end |
| `w3_delta_f3_bid_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of change in bid near depth share (end - start) |
| `w3_delta_d1_f3_bid_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of first difference of change in bid near depth share (end - start) |
| `w3_delta_d2_f3_bid_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of second difference of change in bid near depth share (end - start) |
| `w3_delta_d3_f3_bid_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of third difference of change in bid near depth share (end - start) |
| `w3_delta_f3_bid_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of share of bid reprice-away in resting reprices |
| `w3_delta_d1_f3_bid_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of share of bid reprice-away in resting reprices |
| `w3_delta_d2_f3_bid_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of share of bid reprice-away in resting reprices |
| `w3_delta_d3_f3_bid_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of share of bid reprice-away in resting reprices |
| `w3_delta_f4_bid_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of log pull/add ratio for resting bid qty |
| `w3_delta_d1_f4_bid_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log pull/add ratio for resting bid qty |
| `w3_delta_d2_f4_bid_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log pull/add ratio for resting bid qty |
| `w3_delta_d3_f4_bid_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log pull/add ratio for resting bid qty |
| `w3_delta_f4_bid_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of log1p bid pull qty / start bid depth total |
| `w3_delta_d1_f4_bid_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log1p bid pull qty / start bid depth total |
| `w3_delta_d2_f4_bid_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log1p bid pull qty / start bid depth total |
| `w3_delta_d3_f4_bid_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log1p bid pull qty / start bid depth total |
| `w3_delta_f4_bid_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of share of bid pulls that were near |
| `w3_delta_d1_f4_bid_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of share of bid pulls that were near |
| `w3_delta_d2_f4_bid_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of share of bid pulls that were near |
| `w3_delta_d3_f4_bid_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of share of bid pulls that were near |
| `w3_delta_f5_vacuum_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of f1+f3 (ask+bid COM dispersion logs) |
| `w3_delta_d1_f5_vacuum_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of f1+f3 (ask+bid COM dispersion logs) |
| `w3_delta_d2_f5_vacuum_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of f1+f3 (ask+bid COM dispersion logs) |
| `w3_delta_d3_f5_vacuum_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of f1+f3 (ask+bid COM dispersion logs) |
| `w3_delta_f6_vacuum_decay_log` | 3-window delta ((x_t - x_{t-2})/2) of f2+f4 (ask+bid pull/add logs) |
| `w3_delta_d1_f6_vacuum_decay_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of f2+f4 (ask+bid pull/add logs) |
| `w3_delta_d2_f6_vacuum_decay_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of f2+f4 (ask+bid pull/add logs) |
| `w3_delta_d3_f6_vacuum_decay_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of f2+f4 (ask+bid pull/add logs) |
| `w3_delta_f7_vacuum_total_log` | 3-window delta ((x_t - x_{t-2})/2) of f5+f6 (total vacuum log) |
| `w3_delta_d1_f7_vacuum_total_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of f5+f6 (total vacuum log) |
| `w3_delta_d2_f7_vacuum_total_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of f5+f6 (total vacuum log) |
| `w3_delta_d3_f7_vacuum_total_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of f5+f6 (total vacuum log) |
| `w3_delta_u1_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of copy of f1 ask COM dispersion log |
| `w3_delta_d1_u1_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of f1 ask COM dispersion log |
| `w3_delta_d2_u1_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of f1 ask COM dispersion log |
| `w3_delta_d3_u1_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of f1 ask COM dispersion log |
| `w3_delta_u2_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of copy of f1 ask depth slope log |
| `w3_delta_d1_u2_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of f1 ask depth slope log |
| `w3_delta_d2_u2_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of f1 ask depth slope log |
| `w3_delta_d3_u2_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of f1 ask depth slope log |
| `w3_delta_u3_ask_near_share_decay` | 3-window delta ((x_t - x_{t-2})/2) of negative of f1 ask near share delta |
| `w3_delta_d1_u3_ask_near_share_decay` | 3-window delta ((x_t - x_{t-2})/2) of first difference of negative of f1 ask near share delta |
| `w3_delta_d2_u3_ask_near_share_decay` | 3-window delta ((x_t - x_{t-2})/2) of second difference of negative of f1 ask near share delta |
| `w3_delta_d3_u3_ask_near_share_decay` | 3-window delta ((x_t - x_{t-2})/2) of third difference of negative of f1 ask near share delta |
| `w3_delta_u4_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of copy of f1 ask reprice-away share |
| `w3_delta_d1_u4_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of f1 ask reprice-away share |
| `w3_delta_d2_u4_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of f1 ask reprice-away share |
| `w3_delta_d3_u4_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of f1 ask reprice-away share |
| `w3_delta_u5_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of copy of f2 ask pull/add log |
| `w3_delta_d1_u5_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of f2 ask pull/add log |
| `w3_delta_d2_u5_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of f2 ask pull/add log |
| `w3_delta_d3_u5_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of f2 ask pull/add log |
| `w3_delta_u6_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of copy of f2 ask pull intensity log |
| `w3_delta_d1_u6_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of f2 ask pull intensity log |
| `w3_delta_d2_u6_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of f2 ask pull intensity log |
| `w3_delta_d3_u6_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of f2 ask pull intensity log |
| `w3_delta_u7_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of copy of f2 ask near pull share |
| `w3_delta_d1_u7_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of f2 ask near pull share |
| `w3_delta_d2_u7_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of f2 ask near pull share |
| `w3_delta_d3_u7_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of f2 ask near pull share |
| `w3_delta_u8_bid_com_approach_log` | 3-window delta ((x_t - x_{t-2})/2) of negative of f3 bid COM dispersion log |
| `w3_delta_d1_u8_bid_com_approach_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of negative of f3 bid COM dispersion log |
| `w3_delta_d2_u8_bid_com_approach_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of negative of f3 bid COM dispersion log |
| `w3_delta_d3_u8_bid_com_approach_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of negative of f3 bid COM dispersion log |
| `w3_delta_u9_bid_slope_support_log` | 3-window delta ((x_t - x_{t-2})/2) of negative of f3 bid depth slope log |
| `w3_delta_d1_u9_bid_slope_support_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of negative of f3 bid depth slope log |
| `w3_delta_d2_u9_bid_slope_support_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of negative of f3 bid depth slope log |
| `w3_delta_d3_u9_bid_slope_support_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of negative of f3 bid depth slope log |
| `w3_delta_u10_bid_near_share_rise` | 3-window delta ((x_t - x_{t-2})/2) of copy of f3 bid near share delta |
| `w3_delta_d1_u10_bid_near_share_rise` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of f3 bid near share delta |
| `w3_delta_d2_u10_bid_near_share_rise` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of f3 bid near share delta |
| `w3_delta_d3_u10_bid_near_share_rise` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of f3 bid near share delta |
| `w3_delta_u11_bid_reprice_toward_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of 1 - f3 bid reprice-away share |
| `w3_delta_d1_u11_bid_reprice_toward_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of 1 - f3 bid reprice-away share |
| `w3_delta_d2_u11_bid_reprice_toward_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of 1 - f3 bid reprice-away share |
| `w3_delta_d3_u11_bid_reprice_toward_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of 1 - f3 bid reprice-away share |
| `w3_delta_u12_bid_add_pull_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of negative of f4 bid pull/add log |
| `w3_delta_d1_u12_bid_add_pull_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of negative of f4 bid pull/add log |
| `w3_delta_d2_u12_bid_add_pull_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of negative of f4 bid pull/add log |
| `w3_delta_d3_u12_bid_add_pull_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of negative of f4 bid pull/add log |
| `w3_delta_u13_bid_add_intensity_log` | 3-window delta ((x_t - x_{t-2})/2) of log1p bid add qty / start bid depth total |
| `w3_delta_d1_u13_bid_add_intensity_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log1p bid add qty / start bid depth total |
| `w3_delta_d2_u13_bid_add_intensity_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log1p bid add qty / start bid depth total |
| `w3_delta_d3_u13_bid_add_intensity_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log1p bid add qty / start bid depth total |
| `w3_delta_u14_bid_far_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of 1 - f4 bid near pull share |
| `w3_delta_d1_u14_bid_far_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of 1 - f4 bid near pull share |
| `w3_delta_d2_u14_bid_far_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of 1 - f4 bid near pull share |
| `w3_delta_d3_u14_bid_far_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of 1 - f4 bid near pull share |
| `w3_delta_u15_up_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of u1+u8 (up expansion log) |
| `w3_delta_d1_u15_up_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of u1+u8 (up expansion log) |
| `w3_delta_d2_u15_up_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of u1+u8 (up expansion log) |
| `w3_delta_d3_u15_up_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of u1+u8 (up expansion log) |
| `w3_delta_u16_up_flow_log` | 3-window delta ((x_t - x_{t-2})/2) of u5+u12 (up flow log) |
| `w3_delta_d1_u16_up_flow_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of u5+u12 (up flow log) |
| `w3_delta_d2_u16_up_flow_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of u5+u12 (up flow log) |
| `w3_delta_d3_u16_up_flow_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of u5+u12 (up flow log) |
| `w3_delta_u17_up_total_log` | 3-window delta ((x_t - x_{t-2})/2) of u15+u16 (up total log) |
| `w3_delta_d1_u17_up_total_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of u15+u16 (up total log) |
| `w3_delta_d2_u17_up_total_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of u15+u16 (up total log) |
| `w3_delta_d3_u17_up_total_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of u15+u16 (up total log) |
| `w9_mean_f1_ask_com_disp_log` | 9-window mean (45s) of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_mean_d1_f1_ask_com_disp_log` | 9-window mean (45s) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_mean_d2_f1_ask_com_disp_log` | 9-window mean (45s) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_mean_d3_f1_ask_com_disp_log` | 9-window mean (45s) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_mean_f1_ask_slope_convex_log` | 9-window mean (45s) of log ratio of ask depth far vs near at window end |
| `w9_mean_d1_f1_ask_slope_convex_log` | 9-window mean (45s) of first difference of log ratio of ask depth far vs near at window end |
| `w9_mean_d2_f1_ask_slope_convex_log` | 9-window mean (45s) of second difference of log ratio of ask depth far vs near at window end |
| `w9_mean_d3_f1_ask_slope_convex_log` | 9-window mean (45s) of third difference of log ratio of ask depth far vs near at window end |
| `w9_mean_f1_ask_near_share_delta` | 9-window mean (45s) of change in ask near depth share (end - start) |
| `w9_mean_d1_f1_ask_near_share_delta` | 9-window mean (45s) of first difference of change in ask near depth share (end - start) |
| `w9_mean_d2_f1_ask_near_share_delta` | 9-window mean (45s) of second difference of change in ask near depth share (end - start) |
| `w9_mean_d3_f1_ask_near_share_delta` | 9-window mean (45s) of third difference of change in ask near depth share (end - start) |
| `w9_mean_f1_ask_reprice_away_share_rest` | 9-window mean (45s) of share of ask reprice-away in resting reprices |
| `w9_mean_d1_f1_ask_reprice_away_share_rest` | 9-window mean (45s) of first difference of share of ask reprice-away in resting reprices |
| `w9_mean_d2_f1_ask_reprice_away_share_rest` | 9-window mean (45s) of second difference of share of ask reprice-away in resting reprices |
| `w9_mean_d3_f1_ask_reprice_away_share_rest` | 9-window mean (45s) of third difference of share of ask reprice-away in resting reprices |
| `w9_mean_f2_ask_pull_add_log_rest` | 9-window mean (45s) of log pull/add ratio for resting ask qty |
| `w9_mean_d1_f2_ask_pull_add_log_rest` | 9-window mean (45s) of first difference of log pull/add ratio for resting ask qty |
| `w9_mean_d2_f2_ask_pull_add_log_rest` | 9-window mean (45s) of second difference of log pull/add ratio for resting ask qty |
| `w9_mean_d3_f2_ask_pull_add_log_rest` | 9-window mean (45s) of third difference of log pull/add ratio for resting ask qty |
| `w9_mean_f2_ask_pull_intensity_log_rest` | 9-window mean (45s) of log1p ask pull qty / start ask depth total |
| `w9_mean_d1_f2_ask_pull_intensity_log_rest` | 9-window mean (45s) of first difference of log1p ask pull qty / start ask depth total |
| `w9_mean_d2_f2_ask_pull_intensity_log_rest` | 9-window mean (45s) of second difference of log1p ask pull qty / start ask depth total |
| `w9_mean_d3_f2_ask_pull_intensity_log_rest` | 9-window mean (45s) of third difference of log1p ask pull qty / start ask depth total |
| `w9_mean_f2_ask_near_pull_share_rest` | 9-window mean (45s) of share of ask pulls that were near |
| `w9_mean_d1_f2_ask_near_pull_share_rest` | 9-window mean (45s) of first difference of share of ask pulls that were near |
| `w9_mean_d2_f2_ask_near_pull_share_rest` | 9-window mean (45s) of second difference of share of ask pulls that were near |
| `w9_mean_d3_f2_ask_near_pull_share_rest` | 9-window mean (45s) of third difference of share of ask pulls that were near |
| `w9_mean_f3_bid_com_disp_log` | 9-window mean (45s) of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_mean_d1_f3_bid_com_disp_log` | 9-window mean (45s) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_mean_d2_f3_bid_com_disp_log` | 9-window mean (45s) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_mean_d3_f3_bid_com_disp_log` | 9-window mean (45s) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_mean_f3_bid_slope_convex_log` | 9-window mean (45s) of log ratio of bid depth far vs near at window end |
| `w9_mean_d1_f3_bid_slope_convex_log` | 9-window mean (45s) of first difference of log ratio of bid depth far vs near at window end |
| `w9_mean_d2_f3_bid_slope_convex_log` | 9-window mean (45s) of second difference of log ratio of bid depth far vs near at window end |
| `w9_mean_d3_f3_bid_slope_convex_log` | 9-window mean (45s) of third difference of log ratio of bid depth far vs near at window end |
| `w9_mean_f3_bid_near_share_delta` | 9-window mean (45s) of change in bid near depth share (end - start) |
| `w9_mean_d1_f3_bid_near_share_delta` | 9-window mean (45s) of first difference of change in bid near depth share (end - start) |
| `w9_mean_d2_f3_bid_near_share_delta` | 9-window mean (45s) of second difference of change in bid near depth share (end - start) |
| `w9_mean_d3_f3_bid_near_share_delta` | 9-window mean (45s) of third difference of change in bid near depth share (end - start) |
| `w9_mean_f3_bid_reprice_away_share_rest` | 9-window mean (45s) of share of bid reprice-away in resting reprices |
| `w9_mean_d1_f3_bid_reprice_away_share_rest` | 9-window mean (45s) of first difference of share of bid reprice-away in resting reprices |
| `w9_mean_d2_f3_bid_reprice_away_share_rest` | 9-window mean (45s) of second difference of share of bid reprice-away in resting reprices |
| `w9_mean_d3_f3_bid_reprice_away_share_rest` | 9-window mean (45s) of third difference of share of bid reprice-away in resting reprices |
| `w9_mean_f4_bid_pull_add_log_rest` | 9-window mean (45s) of log pull/add ratio for resting bid qty |
| `w9_mean_d1_f4_bid_pull_add_log_rest` | 9-window mean (45s) of first difference of log pull/add ratio for resting bid qty |
| `w9_mean_d2_f4_bid_pull_add_log_rest` | 9-window mean (45s) of second difference of log pull/add ratio for resting bid qty |
| `w9_mean_d3_f4_bid_pull_add_log_rest` | 9-window mean (45s) of third difference of log pull/add ratio for resting bid qty |
| `w9_mean_f4_bid_pull_intensity_log_rest` | 9-window mean (45s) of log1p bid pull qty / start bid depth total |
| `w9_mean_d1_f4_bid_pull_intensity_log_rest` | 9-window mean (45s) of first difference of log1p bid pull qty / start bid depth total |
| `w9_mean_d2_f4_bid_pull_intensity_log_rest` | 9-window mean (45s) of second difference of log1p bid pull qty / start bid depth total |
| `w9_mean_d3_f4_bid_pull_intensity_log_rest` | 9-window mean (45s) of third difference of log1p bid pull qty / start bid depth total |
| `w9_mean_f4_bid_near_pull_share_rest` | 9-window mean (45s) of share of bid pulls that were near |
| `w9_mean_d1_f4_bid_near_pull_share_rest` | 9-window mean (45s) of first difference of share of bid pulls that were near |
| `w9_mean_d2_f4_bid_near_pull_share_rest` | 9-window mean (45s) of second difference of share of bid pulls that were near |
| `w9_mean_d3_f4_bid_near_pull_share_rest` | 9-window mean (45s) of third difference of share of bid pulls that were near |
| `w9_mean_f5_vacuum_expansion_log` | 9-window mean (45s) of f1+f3 (ask+bid COM dispersion logs) |
| `w9_mean_d1_f5_vacuum_expansion_log` | 9-window mean (45s) of first difference of f1+f3 (ask+bid COM dispersion logs) |
| `w9_mean_d2_f5_vacuum_expansion_log` | 9-window mean (45s) of second difference of f1+f3 (ask+bid COM dispersion logs) |
| `w9_mean_d3_f5_vacuum_expansion_log` | 9-window mean (45s) of third difference of f1+f3 (ask+bid COM dispersion logs) |
| `w9_mean_f6_vacuum_decay_log` | 9-window mean (45s) of f2+f4 (ask+bid pull/add logs) |
| `w9_mean_d1_f6_vacuum_decay_log` | 9-window mean (45s) of first difference of f2+f4 (ask+bid pull/add logs) |
| `w9_mean_d2_f6_vacuum_decay_log` | 9-window mean (45s) of second difference of f2+f4 (ask+bid pull/add logs) |
| `w9_mean_d3_f6_vacuum_decay_log` | 9-window mean (45s) of third difference of f2+f4 (ask+bid pull/add logs) |
| `w9_mean_f7_vacuum_total_log` | 9-window mean (45s) of f5+f6 (total vacuum log) |
| `w9_mean_d1_f7_vacuum_total_log` | 9-window mean (45s) of first difference of f5+f6 (total vacuum log) |
| `w9_mean_d2_f7_vacuum_total_log` | 9-window mean (45s) of second difference of f5+f6 (total vacuum log) |
| `w9_mean_d3_f7_vacuum_total_log` | 9-window mean (45s) of third difference of f5+f6 (total vacuum log) |
| `w9_mean_u1_ask_com_disp_log` | 9-window mean (45s) of copy of f1 ask COM dispersion log |
| `w9_mean_d1_u1_ask_com_disp_log` | 9-window mean (45s) of first difference of copy of f1 ask COM dispersion log |
| `w9_mean_d2_u1_ask_com_disp_log` | 9-window mean (45s) of second difference of copy of f1 ask COM dispersion log |
| `w9_mean_d3_u1_ask_com_disp_log` | 9-window mean (45s) of third difference of copy of f1 ask COM dispersion log |
| `w9_mean_u2_ask_slope_convex_log` | 9-window mean (45s) of copy of f1 ask depth slope log |
| `w9_mean_d1_u2_ask_slope_convex_log` | 9-window mean (45s) of first difference of copy of f1 ask depth slope log |
| `w9_mean_d2_u2_ask_slope_convex_log` | 9-window mean (45s) of second difference of copy of f1 ask depth slope log |
| `w9_mean_d3_u2_ask_slope_convex_log` | 9-window mean (45s) of third difference of copy of f1 ask depth slope log |
| `w9_mean_u3_ask_near_share_decay` | 9-window mean (45s) of negative of f1 ask near share delta |
| `w9_mean_d1_u3_ask_near_share_decay` | 9-window mean (45s) of first difference of negative of f1 ask near share delta |
| `w9_mean_d2_u3_ask_near_share_decay` | 9-window mean (45s) of second difference of negative of f1 ask near share delta |
| `w9_mean_d3_u3_ask_near_share_decay` | 9-window mean (45s) of third difference of negative of f1 ask near share delta |
| `w9_mean_u4_ask_reprice_away_share_rest` | 9-window mean (45s) of copy of f1 ask reprice-away share |
| `w9_mean_d1_u4_ask_reprice_away_share_rest` | 9-window mean (45s) of first difference of copy of f1 ask reprice-away share |
| `w9_mean_d2_u4_ask_reprice_away_share_rest` | 9-window mean (45s) of second difference of copy of f1 ask reprice-away share |
| `w9_mean_d3_u4_ask_reprice_away_share_rest` | 9-window mean (45s) of third difference of copy of f1 ask reprice-away share |
| `w9_mean_u5_ask_pull_add_log_rest` | 9-window mean (45s) of copy of f2 ask pull/add log |
| `w9_mean_d1_u5_ask_pull_add_log_rest` | 9-window mean (45s) of first difference of copy of f2 ask pull/add log |
| `w9_mean_d2_u5_ask_pull_add_log_rest` | 9-window mean (45s) of second difference of copy of f2 ask pull/add log |
| `w9_mean_d3_u5_ask_pull_add_log_rest` | 9-window mean (45s) of third difference of copy of f2 ask pull/add log |
| `w9_mean_u6_ask_pull_intensity_log_rest` | 9-window mean (45s) of copy of f2 ask pull intensity log |
| `w9_mean_d1_u6_ask_pull_intensity_log_rest` | 9-window mean (45s) of first difference of copy of f2 ask pull intensity log |
| `w9_mean_d2_u6_ask_pull_intensity_log_rest` | 9-window mean (45s) of second difference of copy of f2 ask pull intensity log |
| `w9_mean_d3_u6_ask_pull_intensity_log_rest` | 9-window mean (45s) of third difference of copy of f2 ask pull intensity log |
| `w9_mean_u7_ask_near_pull_share_rest` | 9-window mean (45s) of copy of f2 ask near pull share |
| `w9_mean_d1_u7_ask_near_pull_share_rest` | 9-window mean (45s) of first difference of copy of f2 ask near pull share |
| `w9_mean_d2_u7_ask_near_pull_share_rest` | 9-window mean (45s) of second difference of copy of f2 ask near pull share |
| `w9_mean_d3_u7_ask_near_pull_share_rest` | 9-window mean (45s) of third difference of copy of f2 ask near pull share |
| `w9_mean_u8_bid_com_approach_log` | 9-window mean (45s) of negative of f3 bid COM dispersion log |
| `w9_mean_d1_u8_bid_com_approach_log` | 9-window mean (45s) of first difference of negative of f3 bid COM dispersion log |
| `w9_mean_d2_u8_bid_com_approach_log` | 9-window mean (45s) of second difference of negative of f3 bid COM dispersion log |
| `w9_mean_d3_u8_bid_com_approach_log` | 9-window mean (45s) of third difference of negative of f3 bid COM dispersion log |
| `w9_mean_u9_bid_slope_support_log` | 9-window mean (45s) of negative of f3 bid depth slope log |
| `w9_mean_d1_u9_bid_slope_support_log` | 9-window mean (45s) of first difference of negative of f3 bid depth slope log |
| `w9_mean_d2_u9_bid_slope_support_log` | 9-window mean (45s) of second difference of negative of f3 bid depth slope log |
| `w9_mean_d3_u9_bid_slope_support_log` | 9-window mean (45s) of third difference of negative of f3 bid depth slope log |
| `w9_mean_u10_bid_near_share_rise` | 9-window mean (45s) of copy of f3 bid near share delta |
| `w9_mean_d1_u10_bid_near_share_rise` | 9-window mean (45s) of first difference of copy of f3 bid near share delta |
| `w9_mean_d2_u10_bid_near_share_rise` | 9-window mean (45s) of second difference of copy of f3 bid near share delta |
| `w9_mean_d3_u10_bid_near_share_rise` | 9-window mean (45s) of third difference of copy of f3 bid near share delta |
| `w9_mean_u11_bid_reprice_toward_share_rest` | 9-window mean (45s) of 1 - f3 bid reprice-away share |
| `w9_mean_d1_u11_bid_reprice_toward_share_rest` | 9-window mean (45s) of first difference of 1 - f3 bid reprice-away share |
| `w9_mean_d2_u11_bid_reprice_toward_share_rest` | 9-window mean (45s) of second difference of 1 - f3 bid reprice-away share |
| `w9_mean_d3_u11_bid_reprice_toward_share_rest` | 9-window mean (45s) of third difference of 1 - f3 bid reprice-away share |
| `w9_mean_u12_bid_add_pull_log_rest` | 9-window mean (45s) of negative of f4 bid pull/add log |
| `w9_mean_d1_u12_bid_add_pull_log_rest` | 9-window mean (45s) of first difference of negative of f4 bid pull/add log |
| `w9_mean_d2_u12_bid_add_pull_log_rest` | 9-window mean (45s) of second difference of negative of f4 bid pull/add log |
| `w9_mean_d3_u12_bid_add_pull_log_rest` | 9-window mean (45s) of third difference of negative of f4 bid pull/add log |
| `w9_mean_u13_bid_add_intensity_log` | 9-window mean (45s) of log1p bid add qty / start bid depth total |
| `w9_mean_d1_u13_bid_add_intensity_log` | 9-window mean (45s) of first difference of log1p bid add qty / start bid depth total |
| `w9_mean_d2_u13_bid_add_intensity_log` | 9-window mean (45s) of second difference of log1p bid add qty / start bid depth total |
| `w9_mean_d3_u13_bid_add_intensity_log` | 9-window mean (45s) of third difference of log1p bid add qty / start bid depth total |
| `w9_mean_u14_bid_far_pull_share_rest` | 9-window mean (45s) of 1 - f4 bid near pull share |
| `w9_mean_d1_u14_bid_far_pull_share_rest` | 9-window mean (45s) of first difference of 1 - f4 bid near pull share |
| `w9_mean_d2_u14_bid_far_pull_share_rest` | 9-window mean (45s) of second difference of 1 - f4 bid near pull share |
| `w9_mean_d3_u14_bid_far_pull_share_rest` | 9-window mean (45s) of third difference of 1 - f4 bid near pull share |
| `w9_mean_u15_up_expansion_log` | 9-window mean (45s) of u1+u8 (up expansion log) |
| `w9_mean_d1_u15_up_expansion_log` | 9-window mean (45s) of first difference of u1+u8 (up expansion log) |
| `w9_mean_d2_u15_up_expansion_log` | 9-window mean (45s) of second difference of u1+u8 (up expansion log) |
| `w9_mean_d3_u15_up_expansion_log` | 9-window mean (45s) of third difference of u1+u8 (up expansion log) |
| `w9_mean_u16_up_flow_log` | 9-window mean (45s) of u5+u12 (up flow log) |
| `w9_mean_d1_u16_up_flow_log` | 9-window mean (45s) of first difference of u5+u12 (up flow log) |
| `w9_mean_d2_u16_up_flow_log` | 9-window mean (45s) of second difference of u5+u12 (up flow log) |
| `w9_mean_d3_u16_up_flow_log` | 9-window mean (45s) of third difference of u5+u12 (up flow log) |
| `w9_mean_u17_up_total_log` | 9-window mean (45s) of u15+u16 (up total log) |
| `w9_mean_d1_u17_up_total_log` | 9-window mean (45s) of first difference of u15+u16 (up total log) |
| `w9_mean_d2_u17_up_total_log` | 9-window mean (45s) of second difference of u15+u16 (up total log) |
| `w9_mean_d3_u17_up_total_log` | 9-window mean (45s) of third difference of u15+u16 (up total log) |
| `w9_delta_f1_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_delta_d1_f1_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_delta_d2_f1_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_delta_d3_f1_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_delta_f1_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of log ratio of ask depth far vs near at window end |
| `w9_delta_d1_f1_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log ratio of ask depth far vs near at window end |
| `w9_delta_d2_f1_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log ratio of ask depth far vs near at window end |
| `w9_delta_d3_f1_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log ratio of ask depth far vs near at window end |
| `w9_delta_f1_ask_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of change in ask near depth share (end - start) |
| `w9_delta_d1_f1_ask_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of first difference of change in ask near depth share (end - start) |
| `w9_delta_d2_f1_ask_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of second difference of change in ask near depth share (end - start) |
| `w9_delta_d3_f1_ask_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of third difference of change in ask near depth share (end - start) |
| `w9_delta_f1_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of share of ask reprice-away in resting reprices |
| `w9_delta_d1_f1_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of share of ask reprice-away in resting reprices |
| `w9_delta_d2_f1_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of share of ask reprice-away in resting reprices |
| `w9_delta_d3_f1_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of share of ask reprice-away in resting reprices |
| `w9_delta_f2_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of log pull/add ratio for resting ask qty |
| `w9_delta_d1_f2_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log pull/add ratio for resting ask qty |
| `w9_delta_d2_f2_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log pull/add ratio for resting ask qty |
| `w9_delta_d3_f2_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log pull/add ratio for resting ask qty |
| `w9_delta_f2_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of log1p ask pull qty / start ask depth total |
| `w9_delta_d1_f2_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log1p ask pull qty / start ask depth total |
| `w9_delta_d2_f2_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log1p ask pull qty / start ask depth total |
| `w9_delta_d3_f2_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log1p ask pull qty / start ask depth total |
| `w9_delta_f2_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of share of ask pulls that were near |
| `w9_delta_d1_f2_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of share of ask pulls that were near |
| `w9_delta_d2_f2_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of share of ask pulls that were near |
| `w9_delta_d3_f2_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of share of ask pulls that were near |
| `w9_delta_f3_bid_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_delta_d1_f3_bid_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_delta_d2_f3_bid_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_delta_d3_f3_bid_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_delta_f3_bid_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of log ratio of bid depth far vs near at window end |
| `w9_delta_d1_f3_bid_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log ratio of bid depth far vs near at window end |
| `w9_delta_d2_f3_bid_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log ratio of bid depth far vs near at window end |
| `w9_delta_d3_f3_bid_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log ratio of bid depth far vs near at window end |
| `w9_delta_f3_bid_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of change in bid near depth share (end - start) |
| `w9_delta_d1_f3_bid_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of first difference of change in bid near depth share (end - start) |
| `w9_delta_d2_f3_bid_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of second difference of change in bid near depth share (end - start) |
| `w9_delta_d3_f3_bid_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of third difference of change in bid near depth share (end - start) |
| `w9_delta_f3_bid_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of share of bid reprice-away in resting reprices |
| `w9_delta_d1_f3_bid_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of share of bid reprice-away in resting reprices |
| `w9_delta_d2_f3_bid_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of share of bid reprice-away in resting reprices |
| `w9_delta_d3_f3_bid_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of share of bid reprice-away in resting reprices |
| `w9_delta_f4_bid_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of log pull/add ratio for resting bid qty |
| `w9_delta_d1_f4_bid_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log pull/add ratio for resting bid qty |
| `w9_delta_d2_f4_bid_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log pull/add ratio for resting bid qty |
| `w9_delta_d3_f4_bid_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log pull/add ratio for resting bid qty |
| `w9_delta_f4_bid_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of log1p bid pull qty / start bid depth total |
| `w9_delta_d1_f4_bid_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log1p bid pull qty / start bid depth total |
| `w9_delta_d2_f4_bid_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log1p bid pull qty / start bid depth total |
| `w9_delta_d3_f4_bid_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log1p bid pull qty / start bid depth total |
| `w9_delta_f4_bid_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of share of bid pulls that were near |
| `w9_delta_d1_f4_bid_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of share of bid pulls that were near |
| `w9_delta_d2_f4_bid_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of share of bid pulls that were near |
| `w9_delta_d3_f4_bid_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of share of bid pulls that were near |
| `w9_delta_f5_vacuum_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of f1+f3 (ask+bid COM dispersion logs) |
| `w9_delta_d1_f5_vacuum_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of f1+f3 (ask+bid COM dispersion logs) |
| `w9_delta_d2_f5_vacuum_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of f1+f3 (ask+bid COM dispersion logs) |
| `w9_delta_d3_f5_vacuum_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of f1+f3 (ask+bid COM dispersion logs) |
| `w9_delta_f6_vacuum_decay_log` | 9-window delta ((x_t - x_{t-8})/8) of f2+f4 (ask+bid pull/add logs) |
| `w9_delta_d1_f6_vacuum_decay_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of f2+f4 (ask+bid pull/add logs) |
| `w9_delta_d2_f6_vacuum_decay_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of f2+f4 (ask+bid pull/add logs) |
| `w9_delta_d3_f6_vacuum_decay_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of f2+f4 (ask+bid pull/add logs) |
| `w9_delta_f7_vacuum_total_log` | 9-window delta ((x_t - x_{t-8})/8) of f5+f6 (total vacuum log) |
| `w9_delta_d1_f7_vacuum_total_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of f5+f6 (total vacuum log) |
| `w9_delta_d2_f7_vacuum_total_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of f5+f6 (total vacuum log) |
| `w9_delta_d3_f7_vacuum_total_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of f5+f6 (total vacuum log) |
| `w9_delta_u1_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of copy of f1 ask COM dispersion log |
| `w9_delta_d1_u1_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of f1 ask COM dispersion log |
| `w9_delta_d2_u1_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of f1 ask COM dispersion log |
| `w9_delta_d3_u1_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of f1 ask COM dispersion log |
| `w9_delta_u2_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of copy of f1 ask depth slope log |
| `w9_delta_d1_u2_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of f1 ask depth slope log |
| `w9_delta_d2_u2_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of f1 ask depth slope log |
| `w9_delta_d3_u2_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of f1 ask depth slope log |
| `w9_delta_u3_ask_near_share_decay` | 9-window delta ((x_t - x_{t-8})/8) of negative of f1 ask near share delta |
| `w9_delta_d1_u3_ask_near_share_decay` | 9-window delta ((x_t - x_{t-8})/8) of first difference of negative of f1 ask near share delta |
| `w9_delta_d2_u3_ask_near_share_decay` | 9-window delta ((x_t - x_{t-8})/8) of second difference of negative of f1 ask near share delta |
| `w9_delta_d3_u3_ask_near_share_decay` | 9-window delta ((x_t - x_{t-8})/8) of third difference of negative of f1 ask near share delta |
| `w9_delta_u4_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of copy of f1 ask reprice-away share |
| `w9_delta_d1_u4_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of f1 ask reprice-away share |
| `w9_delta_d2_u4_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of f1 ask reprice-away share |
| `w9_delta_d3_u4_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of f1 ask reprice-away share |
| `w9_delta_u5_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of copy of f2 ask pull/add log |
| `w9_delta_d1_u5_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of f2 ask pull/add log |
| `w9_delta_d2_u5_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of f2 ask pull/add log |
| `w9_delta_d3_u5_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of f2 ask pull/add log |
| `w9_delta_u6_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of copy of f2 ask pull intensity log |
| `w9_delta_d1_u6_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of f2 ask pull intensity log |
| `w9_delta_d2_u6_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of f2 ask pull intensity log |
| `w9_delta_d3_u6_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of f2 ask pull intensity log |
| `w9_delta_u7_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of copy of f2 ask near pull share |
| `w9_delta_d1_u7_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of f2 ask near pull share |
| `w9_delta_d2_u7_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of f2 ask near pull share |
| `w9_delta_d3_u7_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of f2 ask near pull share |
| `w9_delta_u8_bid_com_approach_log` | 9-window delta ((x_t - x_{t-8})/8) of negative of f3 bid COM dispersion log |
| `w9_delta_d1_u8_bid_com_approach_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of negative of f3 bid COM dispersion log |
| `w9_delta_d2_u8_bid_com_approach_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of negative of f3 bid COM dispersion log |
| `w9_delta_d3_u8_bid_com_approach_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of negative of f3 bid COM dispersion log |
| `w9_delta_u9_bid_slope_support_log` | 9-window delta ((x_t - x_{t-8})/8) of negative of f3 bid depth slope log |
| `w9_delta_d1_u9_bid_slope_support_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of negative of f3 bid depth slope log |
| `w9_delta_d2_u9_bid_slope_support_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of negative of f3 bid depth slope log |
| `w9_delta_d3_u9_bid_slope_support_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of negative of f3 bid depth slope log |
| `w9_delta_u10_bid_near_share_rise` | 9-window delta ((x_t - x_{t-8})/8) of copy of f3 bid near share delta |
| `w9_delta_d1_u10_bid_near_share_rise` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of f3 bid near share delta |
| `w9_delta_d2_u10_bid_near_share_rise` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of f3 bid near share delta |
| `w9_delta_d3_u10_bid_near_share_rise` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of f3 bid near share delta |
| `w9_delta_u11_bid_reprice_toward_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of 1 - f3 bid reprice-away share |
| `w9_delta_d1_u11_bid_reprice_toward_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of 1 - f3 bid reprice-away share |
| `w9_delta_d2_u11_bid_reprice_toward_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of 1 - f3 bid reprice-away share |
| `w9_delta_d3_u11_bid_reprice_toward_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of 1 - f3 bid reprice-away share |
| `w9_delta_u12_bid_add_pull_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of negative of f4 bid pull/add log |
| `w9_delta_d1_u12_bid_add_pull_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of negative of f4 bid pull/add log |
| `w9_delta_d2_u12_bid_add_pull_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of negative of f4 bid pull/add log |
| `w9_delta_d3_u12_bid_add_pull_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of negative of f4 bid pull/add log |
| `w9_delta_u13_bid_add_intensity_log` | 9-window delta ((x_t - x_{t-8})/8) of log1p bid add qty / start bid depth total |
| `w9_delta_d1_u13_bid_add_intensity_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log1p bid add qty / start bid depth total |
| `w9_delta_d2_u13_bid_add_intensity_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log1p bid add qty / start bid depth total |
| `w9_delta_d3_u13_bid_add_intensity_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log1p bid add qty / start bid depth total |
| `w9_delta_u14_bid_far_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of 1 - f4 bid near pull share |
| `w9_delta_d1_u14_bid_far_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of 1 - f4 bid near pull share |
| `w9_delta_d2_u14_bid_far_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of 1 - f4 bid near pull share |
| `w9_delta_d3_u14_bid_far_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of 1 - f4 bid near pull share |
| `w9_delta_u15_up_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of u1+u8 (up expansion log) |
| `w9_delta_d1_u15_up_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of u1+u8 (up expansion log) |
| `w9_delta_d2_u15_up_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of u1+u8 (up expansion log) |
| `w9_delta_d3_u15_up_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of u1+u8 (up expansion log) |
| `w9_delta_u16_up_flow_log` | 9-window delta ((x_t - x_{t-8})/8) of u5+u12 (up flow log) |
| `w9_delta_d1_u16_up_flow_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of u5+u12 (up flow log) |
| `w9_delta_d2_u16_up_flow_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of u5+u12 (up flow log) |
| `w9_delta_d3_u16_up_flow_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of u5+u12 (up flow log) |
| `w9_delta_u17_up_total_log` | 9-window delta ((x_t - x_{t-8})/8) of u15+u16 (up total log) |
| `w9_delta_d1_u17_up_total_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of u15+u16 (up total log) |
| `w9_delta_d2_u17_up_total_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of u15+u16 (up total log) |
| `w9_delta_d3_u17_up_total_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of u15+u16 (up total log) |
| `w24_mean_f1_ask_com_disp_log` | 24-window mean (120s) of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_mean_d1_f1_ask_com_disp_log` | 24-window mean (120s) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_mean_d2_f1_ask_com_disp_log` | 24-window mean (120s) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_mean_d3_f1_ask_com_disp_log` | 24-window mean (120s) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_mean_f1_ask_slope_convex_log` | 24-window mean (120s) of log ratio of ask depth far vs near at window end |
| `w24_mean_d1_f1_ask_slope_convex_log` | 24-window mean (120s) of first difference of log ratio of ask depth far vs near at window end |
| `w24_mean_d2_f1_ask_slope_convex_log` | 24-window mean (120s) of second difference of log ratio of ask depth far vs near at window end |
| `w24_mean_d3_f1_ask_slope_convex_log` | 24-window mean (120s) of third difference of log ratio of ask depth far vs near at window end |
| `w24_mean_f1_ask_near_share_delta` | 24-window mean (120s) of change in ask near depth share (end - start) |
| `w24_mean_d1_f1_ask_near_share_delta` | 24-window mean (120s) of first difference of change in ask near depth share (end - start) |
| `w24_mean_d2_f1_ask_near_share_delta` | 24-window mean (120s) of second difference of change in ask near depth share (end - start) |
| `w24_mean_d3_f1_ask_near_share_delta` | 24-window mean (120s) of third difference of change in ask near depth share (end - start) |
| `w24_mean_f1_ask_reprice_away_share_rest` | 24-window mean (120s) of share of ask reprice-away in resting reprices |
| `w24_mean_d1_f1_ask_reprice_away_share_rest` | 24-window mean (120s) of first difference of share of ask reprice-away in resting reprices |
| `w24_mean_d2_f1_ask_reprice_away_share_rest` | 24-window mean (120s) of second difference of share of ask reprice-away in resting reprices |
| `w24_mean_d3_f1_ask_reprice_away_share_rest` | 24-window mean (120s) of third difference of share of ask reprice-away in resting reprices |
| `w24_mean_f2_ask_pull_add_log_rest` | 24-window mean (120s) of log pull/add ratio for resting ask qty |
| `w24_mean_d1_f2_ask_pull_add_log_rest` | 24-window mean (120s) of first difference of log pull/add ratio for resting ask qty |
| `w24_mean_d2_f2_ask_pull_add_log_rest` | 24-window mean (120s) of second difference of log pull/add ratio for resting ask qty |
| `w24_mean_d3_f2_ask_pull_add_log_rest` | 24-window mean (120s) of third difference of log pull/add ratio for resting ask qty |
| `w24_mean_f2_ask_pull_intensity_log_rest` | 24-window mean (120s) of log1p ask pull qty / start ask depth total |
| `w24_mean_d1_f2_ask_pull_intensity_log_rest` | 24-window mean (120s) of first difference of log1p ask pull qty / start ask depth total |
| `w24_mean_d2_f2_ask_pull_intensity_log_rest` | 24-window mean (120s) of second difference of log1p ask pull qty / start ask depth total |
| `w24_mean_d3_f2_ask_pull_intensity_log_rest` | 24-window mean (120s) of third difference of log1p ask pull qty / start ask depth total |
| `w24_mean_f2_ask_near_pull_share_rest` | 24-window mean (120s) of share of ask pulls that were near |
| `w24_mean_d1_f2_ask_near_pull_share_rest` | 24-window mean (120s) of first difference of share of ask pulls that were near |
| `w24_mean_d2_f2_ask_near_pull_share_rest` | 24-window mean (120s) of second difference of share of ask pulls that were near |
| `w24_mean_d3_f2_ask_near_pull_share_rest` | 24-window mean (120s) of third difference of share of ask pulls that were near |
| `w24_mean_f3_bid_com_disp_log` | 24-window mean (120s) of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_mean_d1_f3_bid_com_disp_log` | 24-window mean (120s) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_mean_d2_f3_bid_com_disp_log` | 24-window mean (120s) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_mean_d3_f3_bid_com_disp_log` | 24-window mean (120s) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_mean_f3_bid_slope_convex_log` | 24-window mean (120s) of log ratio of bid depth far vs near at window end |
| `w24_mean_d1_f3_bid_slope_convex_log` | 24-window mean (120s) of first difference of log ratio of bid depth far vs near at window end |
| `w24_mean_d2_f3_bid_slope_convex_log` | 24-window mean (120s) of second difference of log ratio of bid depth far vs near at window end |
| `w24_mean_d3_f3_bid_slope_convex_log` | 24-window mean (120s) of third difference of log ratio of bid depth far vs near at window end |
| `w24_mean_f3_bid_near_share_delta` | 24-window mean (120s) of change in bid near depth share (end - start) |
| `w24_mean_d1_f3_bid_near_share_delta` | 24-window mean (120s) of first difference of change in bid near depth share (end - start) |
| `w24_mean_d2_f3_bid_near_share_delta` | 24-window mean (120s) of second difference of change in bid near depth share (end - start) |
| `w24_mean_d3_f3_bid_near_share_delta` | 24-window mean (120s) of third difference of change in bid near depth share (end - start) |
| `w24_mean_f3_bid_reprice_away_share_rest` | 24-window mean (120s) of share of bid reprice-away in resting reprices |
| `w24_mean_d1_f3_bid_reprice_away_share_rest` | 24-window mean (120s) of first difference of share of bid reprice-away in resting reprices |
| `w24_mean_d2_f3_bid_reprice_away_share_rest` | 24-window mean (120s) of second difference of share of bid reprice-away in resting reprices |
| `w24_mean_d3_f3_bid_reprice_away_share_rest` | 24-window mean (120s) of third difference of share of bid reprice-away in resting reprices |
| `w24_mean_f4_bid_pull_add_log_rest` | 24-window mean (120s) of log pull/add ratio for resting bid qty |
| `w24_mean_d1_f4_bid_pull_add_log_rest` | 24-window mean (120s) of first difference of log pull/add ratio for resting bid qty |
| `w24_mean_d2_f4_bid_pull_add_log_rest` | 24-window mean (120s) of second difference of log pull/add ratio for resting bid qty |
| `w24_mean_d3_f4_bid_pull_add_log_rest` | 24-window mean (120s) of third difference of log pull/add ratio for resting bid qty |
| `w24_mean_f4_bid_pull_intensity_log_rest` | 24-window mean (120s) of log1p bid pull qty / start bid depth total |
| `w24_mean_d1_f4_bid_pull_intensity_log_rest` | 24-window mean (120s) of first difference of log1p bid pull qty / start bid depth total |
| `w24_mean_d2_f4_bid_pull_intensity_log_rest` | 24-window mean (120s) of second difference of log1p bid pull qty / start bid depth total |
| `w24_mean_d3_f4_bid_pull_intensity_log_rest` | 24-window mean (120s) of third difference of log1p bid pull qty / start bid depth total |
| `w24_mean_f4_bid_near_pull_share_rest` | 24-window mean (120s) of share of bid pulls that were near |
| `w24_mean_d1_f4_bid_near_pull_share_rest` | 24-window mean (120s) of first difference of share of bid pulls that were near |
| `w24_mean_d2_f4_bid_near_pull_share_rest` | 24-window mean (120s) of second difference of share of bid pulls that were near |
| `w24_mean_d3_f4_bid_near_pull_share_rest` | 24-window mean (120s) of third difference of share of bid pulls that were near |
| `w24_mean_f5_vacuum_expansion_log` | 24-window mean (120s) of f1+f3 (ask+bid COM dispersion logs) |
| `w24_mean_d1_f5_vacuum_expansion_log` | 24-window mean (120s) of first difference of f1+f3 (ask+bid COM dispersion logs) |
| `w24_mean_d2_f5_vacuum_expansion_log` | 24-window mean (120s) of second difference of f1+f3 (ask+bid COM dispersion logs) |
| `w24_mean_d3_f5_vacuum_expansion_log` | 24-window mean (120s) of third difference of f1+f3 (ask+bid COM dispersion logs) |
| `w24_mean_f6_vacuum_decay_log` | 24-window mean (120s) of f2+f4 (ask+bid pull/add logs) |
| `w24_mean_d1_f6_vacuum_decay_log` | 24-window mean (120s) of first difference of f2+f4 (ask+bid pull/add logs) |
| `w24_mean_d2_f6_vacuum_decay_log` | 24-window mean (120s) of second difference of f2+f4 (ask+bid pull/add logs) |
| `w24_mean_d3_f6_vacuum_decay_log` | 24-window mean (120s) of third difference of f2+f4 (ask+bid pull/add logs) |
| `w24_mean_f7_vacuum_total_log` | 24-window mean (120s) of f5+f6 (total vacuum log) |
| `w24_mean_d1_f7_vacuum_total_log` | 24-window mean (120s) of first difference of f5+f6 (total vacuum log) |
| `w24_mean_d2_f7_vacuum_total_log` | 24-window mean (120s) of second difference of f5+f6 (total vacuum log) |
| `w24_mean_d3_f7_vacuum_total_log` | 24-window mean (120s) of third difference of f5+f6 (total vacuum log) |
| `w24_mean_u1_ask_com_disp_log` | 24-window mean (120s) of copy of f1 ask COM dispersion log |
| `w24_mean_d1_u1_ask_com_disp_log` | 24-window mean (120s) of first difference of copy of f1 ask COM dispersion log |
| `w24_mean_d2_u1_ask_com_disp_log` | 24-window mean (120s) of second difference of copy of f1 ask COM dispersion log |
| `w24_mean_d3_u1_ask_com_disp_log` | 24-window mean (120s) of third difference of copy of f1 ask COM dispersion log |
| `w24_mean_u2_ask_slope_convex_log` | 24-window mean (120s) of copy of f1 ask depth slope log |
| `w24_mean_d1_u2_ask_slope_convex_log` | 24-window mean (120s) of first difference of copy of f1 ask depth slope log |
| `w24_mean_d2_u2_ask_slope_convex_log` | 24-window mean (120s) of second difference of copy of f1 ask depth slope log |
| `w24_mean_d3_u2_ask_slope_convex_log` | 24-window mean (120s) of third difference of copy of f1 ask depth slope log |
| `w24_mean_u3_ask_near_share_decay` | 24-window mean (120s) of negative of f1 ask near share delta |
| `w24_mean_d1_u3_ask_near_share_decay` | 24-window mean (120s) of first difference of negative of f1 ask near share delta |
| `w24_mean_d2_u3_ask_near_share_decay` | 24-window mean (120s) of second difference of negative of f1 ask near share delta |
| `w24_mean_d3_u3_ask_near_share_decay` | 24-window mean (120s) of third difference of negative of f1 ask near share delta |
| `w24_mean_u4_ask_reprice_away_share_rest` | 24-window mean (120s) of copy of f1 ask reprice-away share |
| `w24_mean_d1_u4_ask_reprice_away_share_rest` | 24-window mean (120s) of first difference of copy of f1 ask reprice-away share |
| `w24_mean_d2_u4_ask_reprice_away_share_rest` | 24-window mean (120s) of second difference of copy of f1 ask reprice-away share |
| `w24_mean_d3_u4_ask_reprice_away_share_rest` | 24-window mean (120s) of third difference of copy of f1 ask reprice-away share |
| `w24_mean_u5_ask_pull_add_log_rest` | 24-window mean (120s) of copy of f2 ask pull/add log |
| `w24_mean_d1_u5_ask_pull_add_log_rest` | 24-window mean (120s) of first difference of copy of f2 ask pull/add log |
| `w24_mean_d2_u5_ask_pull_add_log_rest` | 24-window mean (120s) of second difference of copy of f2 ask pull/add log |
| `w24_mean_d3_u5_ask_pull_add_log_rest` | 24-window mean (120s) of third difference of copy of f2 ask pull/add log |
| `w24_mean_u6_ask_pull_intensity_log_rest` | 24-window mean (120s) of copy of f2 ask pull intensity log |
| `w24_mean_d1_u6_ask_pull_intensity_log_rest` | 24-window mean (120s) of first difference of copy of f2 ask pull intensity log |
| `w24_mean_d2_u6_ask_pull_intensity_log_rest` | 24-window mean (120s) of second difference of copy of f2 ask pull intensity log |
| `w24_mean_d3_u6_ask_pull_intensity_log_rest` | 24-window mean (120s) of third difference of copy of f2 ask pull intensity log |
| `w24_mean_u7_ask_near_pull_share_rest` | 24-window mean (120s) of copy of f2 ask near pull share |
| `w24_mean_d1_u7_ask_near_pull_share_rest` | 24-window mean (120s) of first difference of copy of f2 ask near pull share |
| `w24_mean_d2_u7_ask_near_pull_share_rest` | 24-window mean (120s) of second difference of copy of f2 ask near pull share |
| `w24_mean_d3_u7_ask_near_pull_share_rest` | 24-window mean (120s) of third difference of copy of f2 ask near pull share |
| `w24_mean_u8_bid_com_approach_log` | 24-window mean (120s) of negative of f3 bid COM dispersion log |
| `w24_mean_d1_u8_bid_com_approach_log` | 24-window mean (120s) of first difference of negative of f3 bid COM dispersion log |
| `w24_mean_d2_u8_bid_com_approach_log` | 24-window mean (120s) of second difference of negative of f3 bid COM dispersion log |
| `w24_mean_d3_u8_bid_com_approach_log` | 24-window mean (120s) of third difference of negative of f3 bid COM dispersion log |
| `w24_mean_u9_bid_slope_support_log` | 24-window mean (120s) of negative of f3 bid depth slope log |
| `w24_mean_d1_u9_bid_slope_support_log` | 24-window mean (120s) of first difference of negative of f3 bid depth slope log |
| `w24_mean_d2_u9_bid_slope_support_log` | 24-window mean (120s) of second difference of negative of f3 bid depth slope log |
| `w24_mean_d3_u9_bid_slope_support_log` | 24-window mean (120s) of third difference of negative of f3 bid depth slope log |
| `w24_mean_u10_bid_near_share_rise` | 24-window mean (120s) of copy of f3 bid near share delta |
| `w24_mean_d1_u10_bid_near_share_rise` | 24-window mean (120s) of first difference of copy of f3 bid near share delta |
| `w24_mean_d2_u10_bid_near_share_rise` | 24-window mean (120s) of second difference of copy of f3 bid near share delta |
| `w24_mean_d3_u10_bid_near_share_rise` | 24-window mean (120s) of third difference of copy of f3 bid near share delta |
| `w24_mean_u11_bid_reprice_toward_share_rest` | 24-window mean (120s) of 1 - f3 bid reprice-away share |
| `w24_mean_d1_u11_bid_reprice_toward_share_rest` | 24-window mean (120s) of first difference of 1 - f3 bid reprice-away share |
| `w24_mean_d2_u11_bid_reprice_toward_share_rest` | 24-window mean (120s) of second difference of 1 - f3 bid reprice-away share |
| `w24_mean_d3_u11_bid_reprice_toward_share_rest` | 24-window mean (120s) of third difference of 1 - f3 bid reprice-away share |
| `w24_mean_u12_bid_add_pull_log_rest` | 24-window mean (120s) of negative of f4 bid pull/add log |
| `w24_mean_d1_u12_bid_add_pull_log_rest` | 24-window mean (120s) of first difference of negative of f4 bid pull/add log |
| `w24_mean_d2_u12_bid_add_pull_log_rest` | 24-window mean (120s) of second difference of negative of f4 bid pull/add log |
| `w24_mean_d3_u12_bid_add_pull_log_rest` | 24-window mean (120s) of third difference of negative of f4 bid pull/add log |
| `w24_mean_u13_bid_add_intensity_log` | 24-window mean (120s) of log1p bid add qty / start bid depth total |
| `w24_mean_d1_u13_bid_add_intensity_log` | 24-window mean (120s) of first difference of log1p bid add qty / start bid depth total |
| `w24_mean_d2_u13_bid_add_intensity_log` | 24-window mean (120s) of second difference of log1p bid add qty / start bid depth total |
| `w24_mean_d3_u13_bid_add_intensity_log` | 24-window mean (120s) of third difference of log1p bid add qty / start bid depth total |
| `w24_mean_u14_bid_far_pull_share_rest` | 24-window mean (120s) of 1 - f4 bid near pull share |
| `w24_mean_d1_u14_bid_far_pull_share_rest` | 24-window mean (120s) of first difference of 1 - f4 bid near pull share |
| `w24_mean_d2_u14_bid_far_pull_share_rest` | 24-window mean (120s) of second difference of 1 - f4 bid near pull share |
| `w24_mean_d3_u14_bid_far_pull_share_rest` | 24-window mean (120s) of third difference of 1 - f4 bid near pull share |
| `w24_mean_u15_up_expansion_log` | 24-window mean (120s) of u1+u8 (up expansion log) |
| `w24_mean_d1_u15_up_expansion_log` | 24-window mean (120s) of first difference of u1+u8 (up expansion log) |
| `w24_mean_d2_u15_up_expansion_log` | 24-window mean (120s) of second difference of u1+u8 (up expansion log) |
| `w24_mean_d3_u15_up_expansion_log` | 24-window mean (120s) of third difference of u1+u8 (up expansion log) |
| `w24_mean_u16_up_flow_log` | 24-window mean (120s) of u5+u12 (up flow log) |
| `w24_mean_d1_u16_up_flow_log` | 24-window mean (120s) of first difference of u5+u12 (up flow log) |
| `w24_mean_d2_u16_up_flow_log` | 24-window mean (120s) of second difference of u5+u12 (up flow log) |
| `w24_mean_d3_u16_up_flow_log` | 24-window mean (120s) of third difference of u5+u12 (up flow log) |
| `w24_mean_u17_up_total_log` | 24-window mean (120s) of u15+u16 (up total log) |
| `w24_mean_d1_u17_up_total_log` | 24-window mean (120s) of first difference of u15+u16 (up total log) |
| `w24_mean_d2_u17_up_total_log` | 24-window mean (120s) of second difference of u15+u16 (up total log) |
| `w24_mean_d3_u17_up_total_log` | 24-window mean (120s) of third difference of u15+u16 (up total log) |
| `w24_delta_f1_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_delta_d1_f1_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_delta_d2_f1_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_delta_d3_f1_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_delta_f1_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of log ratio of ask depth far vs near at window end |
| `w24_delta_d1_f1_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log ratio of ask depth far vs near at window end |
| `w24_delta_d2_f1_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log ratio of ask depth far vs near at window end |
| `w24_delta_d3_f1_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log ratio of ask depth far vs near at window end |
| `w24_delta_f1_ask_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of change in ask near depth share (end - start) |
| `w24_delta_d1_f1_ask_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of first difference of change in ask near depth share (end - start) |
| `w24_delta_d2_f1_ask_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of second difference of change in ask near depth share (end - start) |
| `w24_delta_d3_f1_ask_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of third difference of change in ask near depth share (end - start) |
| `w24_delta_f1_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of share of ask reprice-away in resting reprices |
| `w24_delta_d1_f1_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of share of ask reprice-away in resting reprices |
| `w24_delta_d2_f1_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of share of ask reprice-away in resting reprices |
| `w24_delta_d3_f1_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of share of ask reprice-away in resting reprices |
| `w24_delta_f2_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of log pull/add ratio for resting ask qty |
| `w24_delta_d1_f2_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log pull/add ratio for resting ask qty |
| `w24_delta_d2_f2_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log pull/add ratio for resting ask qty |
| `w24_delta_d3_f2_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log pull/add ratio for resting ask qty |
| `w24_delta_f2_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of log1p ask pull qty / start ask depth total |
| `w24_delta_d1_f2_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log1p ask pull qty / start ask depth total |
| `w24_delta_d2_f2_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log1p ask pull qty / start ask depth total |
| `w24_delta_d3_f2_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log1p ask pull qty / start ask depth total |
| `w24_delta_f2_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of share of ask pulls that were near |
| `w24_delta_d1_f2_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of share of ask pulls that were near |
| `w24_delta_d2_f2_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of share of ask pulls that were near |
| `w24_delta_d3_f2_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of share of ask pulls that were near |
| `w24_delta_f3_bid_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_delta_d1_f3_bid_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_delta_d2_f3_bid_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_delta_d3_f3_bid_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_delta_f3_bid_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of log ratio of bid depth far vs near at window end |
| `w24_delta_d1_f3_bid_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log ratio of bid depth far vs near at window end |
| `w24_delta_d2_f3_bid_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log ratio of bid depth far vs near at window end |
| `w24_delta_d3_f3_bid_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log ratio of bid depth far vs near at window end |
| `w24_delta_f3_bid_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of change in bid near depth share (end - start) |
| `w24_delta_d1_f3_bid_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of first difference of change in bid near depth share (end - start) |
| `w24_delta_d2_f3_bid_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of second difference of change in bid near depth share (end - start) |
| `w24_delta_d3_f3_bid_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of third difference of change in bid near depth share (end - start) |
| `w24_delta_f3_bid_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of share of bid reprice-away in resting reprices |
| `w24_delta_d1_f3_bid_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of share of bid reprice-away in resting reprices |
| `w24_delta_d2_f3_bid_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of share of bid reprice-away in resting reprices |
| `w24_delta_d3_f3_bid_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of share of bid reprice-away in resting reprices |
| `w24_delta_f4_bid_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of log pull/add ratio for resting bid qty |
| `w24_delta_d1_f4_bid_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log pull/add ratio for resting bid qty |
| `w24_delta_d2_f4_bid_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log pull/add ratio for resting bid qty |
| `w24_delta_d3_f4_bid_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log pull/add ratio for resting bid qty |
| `w24_delta_f4_bid_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of log1p bid pull qty / start bid depth total |
| `w24_delta_d1_f4_bid_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log1p bid pull qty / start bid depth total |
| `w24_delta_d2_f4_bid_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log1p bid pull qty / start bid depth total |
| `w24_delta_d3_f4_bid_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log1p bid pull qty / start bid depth total |
| `w24_delta_f4_bid_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of share of bid pulls that were near |
| `w24_delta_d1_f4_bid_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of share of bid pulls that were near |
| `w24_delta_d2_f4_bid_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of share of bid pulls that were near |
| `w24_delta_d3_f4_bid_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of share of bid pulls that were near |
| `w24_delta_f5_vacuum_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of f1+f3 (ask+bid COM dispersion logs) |
| `w24_delta_d1_f5_vacuum_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of f1+f3 (ask+bid COM dispersion logs) |
| `w24_delta_d2_f5_vacuum_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of f1+f3 (ask+bid COM dispersion logs) |
| `w24_delta_d3_f5_vacuum_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of f1+f3 (ask+bid COM dispersion logs) |
| `w24_delta_f6_vacuum_decay_log` | 24-window delta ((x_t - x_{t-23})/23) of f2+f4 (ask+bid pull/add logs) |
| `w24_delta_d1_f6_vacuum_decay_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of f2+f4 (ask+bid pull/add logs) |
| `w24_delta_d2_f6_vacuum_decay_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of f2+f4 (ask+bid pull/add logs) |
| `w24_delta_d3_f6_vacuum_decay_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of f2+f4 (ask+bid pull/add logs) |
| `w24_delta_f7_vacuum_total_log` | 24-window delta ((x_t - x_{t-23})/23) of f5+f6 (total vacuum log) |
| `w24_delta_d1_f7_vacuum_total_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of f5+f6 (total vacuum log) |
| `w24_delta_d2_f7_vacuum_total_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of f5+f6 (total vacuum log) |
| `w24_delta_d3_f7_vacuum_total_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of f5+f6 (total vacuum log) |
| `w24_delta_u1_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of copy of f1 ask COM dispersion log |
| `w24_delta_d1_u1_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of f1 ask COM dispersion log |
| `w24_delta_d2_u1_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of f1 ask COM dispersion log |
| `w24_delta_d3_u1_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of f1 ask COM dispersion log |
| `w24_delta_u2_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of copy of f1 ask depth slope log |
| `w24_delta_d1_u2_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of f1 ask depth slope log |
| `w24_delta_d2_u2_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of f1 ask depth slope log |
| `w24_delta_d3_u2_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of f1 ask depth slope log |
| `w24_delta_u3_ask_near_share_decay` | 24-window delta ((x_t - x_{t-23})/23) of negative of f1 ask near share delta |
| `w24_delta_d1_u3_ask_near_share_decay` | 24-window delta ((x_t - x_{t-23})/23) of first difference of negative of f1 ask near share delta |
| `w24_delta_d2_u3_ask_near_share_decay` | 24-window delta ((x_t - x_{t-23})/23) of second difference of negative of f1 ask near share delta |
| `w24_delta_d3_u3_ask_near_share_decay` | 24-window delta ((x_t - x_{t-23})/23) of third difference of negative of f1 ask near share delta |
| `w24_delta_u4_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of copy of f1 ask reprice-away share |
| `w24_delta_d1_u4_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of f1 ask reprice-away share |
| `w24_delta_d2_u4_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of f1 ask reprice-away share |
| `w24_delta_d3_u4_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of f1 ask reprice-away share |
| `w24_delta_u5_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of copy of f2 ask pull/add log |
| `w24_delta_d1_u5_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of f2 ask pull/add log |
| `w24_delta_d2_u5_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of f2 ask pull/add log |
| `w24_delta_d3_u5_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of f2 ask pull/add log |
| `w24_delta_u6_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of copy of f2 ask pull intensity log |
| `w24_delta_d1_u6_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of f2 ask pull intensity log |
| `w24_delta_d2_u6_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of f2 ask pull intensity log |
| `w24_delta_d3_u6_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of f2 ask pull intensity log |
| `w24_delta_u7_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of copy of f2 ask near pull share |
| `w24_delta_d1_u7_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of f2 ask near pull share |
| `w24_delta_d2_u7_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of f2 ask near pull share |
| `w24_delta_d3_u7_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of f2 ask near pull share |
| `w24_delta_u8_bid_com_approach_log` | 24-window delta ((x_t - x_{t-23})/23) of negative of f3 bid COM dispersion log |
| `w24_delta_d1_u8_bid_com_approach_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of negative of f3 bid COM dispersion log |
| `w24_delta_d2_u8_bid_com_approach_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of negative of f3 bid COM dispersion log |
| `w24_delta_d3_u8_bid_com_approach_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of negative of f3 bid COM dispersion log |
| `w24_delta_u9_bid_slope_support_log` | 24-window delta ((x_t - x_{t-23})/23) of negative of f3 bid depth slope log |
| `w24_delta_d1_u9_bid_slope_support_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of negative of f3 bid depth slope log |
| `w24_delta_d2_u9_bid_slope_support_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of negative of f3 bid depth slope log |
| `w24_delta_d3_u9_bid_slope_support_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of negative of f3 bid depth slope log |
| `w24_delta_u10_bid_near_share_rise` | 24-window delta ((x_t - x_{t-23})/23) of copy of f3 bid near share delta |
| `w24_delta_d1_u10_bid_near_share_rise` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of f3 bid near share delta |
| `w24_delta_d2_u10_bid_near_share_rise` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of f3 bid near share delta |
| `w24_delta_d3_u10_bid_near_share_rise` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of f3 bid near share delta |
| `w24_delta_u11_bid_reprice_toward_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of 1 - f3 bid reprice-away share |
| `w24_delta_d1_u11_bid_reprice_toward_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of 1 - f3 bid reprice-away share |
| `w24_delta_d2_u11_bid_reprice_toward_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of 1 - f3 bid reprice-away share |
| `w24_delta_d3_u11_bid_reprice_toward_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of 1 - f3 bid reprice-away share |
| `w24_delta_u12_bid_add_pull_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of negative of f4 bid pull/add log |
| `w24_delta_d1_u12_bid_add_pull_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of negative of f4 bid pull/add log |
| `w24_delta_d2_u12_bid_add_pull_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of negative of f4 bid pull/add log |
| `w24_delta_d3_u12_bid_add_pull_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of negative of f4 bid pull/add log |
| `w24_delta_u13_bid_add_intensity_log` | 24-window delta ((x_t - x_{t-23})/23) of log1p bid add qty / start bid depth total |
| `w24_delta_d1_u13_bid_add_intensity_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log1p bid add qty / start bid depth total |
| `w24_delta_d2_u13_bid_add_intensity_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log1p bid add qty / start bid depth total |
| `w24_delta_d3_u13_bid_add_intensity_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log1p bid add qty / start bid depth total |
| `w24_delta_u14_bid_far_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of 1 - f4 bid near pull share |
| `w24_delta_d1_u14_bid_far_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of 1 - f4 bid near pull share |
| `w24_delta_d2_u14_bid_far_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of 1 - f4 bid near pull share |
| `w24_delta_d3_u14_bid_far_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of 1 - f4 bid near pull share |
| `w24_delta_u15_up_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of u1+u8 (up expansion log) |
| `w24_delta_d1_u15_up_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of u1+u8 (up expansion log) |
| `w24_delta_d2_u15_up_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of u1+u8 (up expansion log) |
| `w24_delta_d3_u15_up_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of u1+u8 (up expansion log) |
| `w24_delta_u16_up_flow_log` | 24-window delta ((x_t - x_{t-23})/23) of u5+u12 (up flow log) |
| `w24_delta_d1_u16_up_flow_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of u5+u12 (up flow log) |
| `w24_delta_d2_u16_up_flow_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of u5+u12 (up flow log) |
| `w24_delta_d3_u16_up_flow_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of u5+u12 (up flow log) |
| `w24_delta_u17_up_total_log` | 24-window delta ((x_t - x_{t-23})/23) of u15+u16 (up total log) |
| `w24_delta_d1_u17_up_total_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of u15+u16 (up total log) |
| `w24_delta_d2_u17_up_total_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of u15+u16 (up total log) |
| `w24_delta_d3_u17_up_total_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of u15+u16 (up total log) |