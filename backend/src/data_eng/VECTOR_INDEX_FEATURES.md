# Vector Index Features

All features are level-relative: prices and depths are measured around P_ref, and near/far buckets are defined in ticks (near <=5, far 15-20). All inputs are ratios/logs and discrete differences only (no raw quantities). Prefix mapping: dn_ = approaching from above, up_ = approaching from below.

Normalization at index build: robust scaling per-dimension using median/MAD (scale 1.4826), clip to +/-8, set zero-MAD dims to 0, then L2-normalize the vector.

## Base Features (x_k)

| Feature | Meaning |
| --- | --- |
| `dn_ask_com_disp_log` | log ratio of ask COM distance from level (ticks), end vs start |
| `dn_ask_slope_convex_log` | log ratio of ask depth far vs near at window end |
| `dn_ask_near_share_delta` | change in ask near depth share (end - start) |
| `dn_ask_reprice_away_share_rest` | share of ask reprice-away in resting reprices |
| `dn_ask_pull_add_log_rest` | log pull/add ratio for resting ask qty |
| `dn_ask_pull_intensity_log_rest` | log1p ask pull qty / start ask depth total |
| `dn_ask_near_pull_share_rest` | share of ask pulls that were near |
| `dn_bid_com_disp_log` | log ratio of bid COM distance from level (ticks), end vs start |
| `dn_bid_slope_convex_log` | log ratio of bid depth far vs near at window end |
| `dn_bid_near_share_delta` | change in bid near depth share (end - start) |
| `dn_bid_reprice_away_share_rest` | share of bid reprice-away in resting reprices |
| `dn_bid_pull_add_log_rest` | log pull/add ratio for resting bid qty |
| `dn_bid_pull_intensity_log_rest` | log1p bid pull qty / start bid depth total |
| `dn_bid_near_pull_share_rest` | share of bid pulls that were near |
| `dn_vacuum_expansion_log` | dn_ask_com_disp_log + dn_bid_com_disp_log |
| `dn_vacuum_decay_log` | dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `dn_vacuum_total_log` | dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `up_ask_com_disp_log` | copy of dn_ask_com_disp_log |
| `up_ask_slope_convex_log` | copy of dn_ask_slope_convex_log |
| `up_ask_near_share_decay` | negative of dn_ask_near_share_delta |
| `up_ask_reprice_away_share_rest` | copy of dn_ask_reprice_away_share_rest |
| `up_ask_pull_add_log_rest` | copy of dn_ask_pull_add_log_rest |
| `up_ask_pull_intensity_log_rest` | copy of dn_ask_pull_intensity_log_rest |
| `up_ask_near_pull_share_rest` | copy of dn_ask_near_pull_share_rest |
| `up_bid_com_approach_log` | negative of dn_bid_com_disp_log |
| `up_bid_slope_support_log` | negative of dn_bid_slope_convex_log |
| `up_bid_near_share_rise` | copy of dn_bid_near_share_delta |
| `up_bid_reprice_toward_share_rest` | 1 - dn_bid_reprice_away_share_rest |
| `up_bid_add_pull_log_rest` | negative of dn_bid_pull_add_log_rest |
| `up_bid_add_intensity_log` | log1p bid add qty / start bid depth total |
| `up_bid_far_pull_share_rest` | 1 - dn_bid_near_pull_share_rest |
| `up_expansion_log` | up_ask_com_disp_log + up_bid_com_approach_log |
| `up_flow_log` | up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `up_total_log` | up_expansion_log + up_flow_log |

## Rollups and Derivatives

- d1_, d2_, d3_ are first/second/third differences across consecutive 5s windows of the base feature.
- w0_ is the current 5s value; w3_mean/w9_mean/w24_mean are rolling means over 3/9/24 windows.
- w3_delta/w9_delta/w24_delta are scaled deltas using (x_t - x_{t-k})/k for k in {2, 8, 23}.

## Vector Feature List

| Feature | Meaning |
| --- | --- |
| `w0_dn_ask_com_disp_log` | current 5s value of log ratio of ask COM distance from level (ticks), end vs start |
| `w0_d1_dn_ask_com_disp_log` | current 5s value of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w0_d2_dn_ask_com_disp_log` | current 5s value of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w0_d3_dn_ask_com_disp_log` | current 5s value of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w0_dn_ask_slope_convex_log` | current 5s value of log ratio of ask depth far vs near at window end |
| `w0_d1_dn_ask_slope_convex_log` | current 5s value of first difference of log ratio of ask depth far vs near at window end |
| `w0_d2_dn_ask_slope_convex_log` | current 5s value of second difference of log ratio of ask depth far vs near at window end |
| `w0_d3_dn_ask_slope_convex_log` | current 5s value of third difference of log ratio of ask depth far vs near at window end |
| `w0_dn_ask_near_share_delta` | current 5s value of change in ask near depth share (end - start) |
| `w0_d1_dn_ask_near_share_delta` | current 5s value of first difference of change in ask near depth share (end - start) |
| `w0_d2_dn_ask_near_share_delta` | current 5s value of second difference of change in ask near depth share (end - start) |
| `w0_d3_dn_ask_near_share_delta` | current 5s value of third difference of change in ask near depth share (end - start) |
| `w0_dn_ask_reprice_away_share_rest` | current 5s value of share of ask reprice-away in resting reprices |
| `w0_d1_dn_ask_reprice_away_share_rest` | current 5s value of first difference of share of ask reprice-away in resting reprices |
| `w0_d2_dn_ask_reprice_away_share_rest` | current 5s value of second difference of share of ask reprice-away in resting reprices |
| `w0_d3_dn_ask_reprice_away_share_rest` | current 5s value of third difference of share of ask reprice-away in resting reprices |
| `w0_dn_ask_pull_add_log_rest` | current 5s value of log pull/add ratio for resting ask qty |
| `w0_d1_dn_ask_pull_add_log_rest` | current 5s value of first difference of log pull/add ratio for resting ask qty |
| `w0_d2_dn_ask_pull_add_log_rest` | current 5s value of second difference of log pull/add ratio for resting ask qty |
| `w0_d3_dn_ask_pull_add_log_rest` | current 5s value of third difference of log pull/add ratio for resting ask qty |
| `w0_dn_ask_pull_intensity_log_rest` | current 5s value of log1p ask pull qty / start ask depth total |
| `w0_d1_dn_ask_pull_intensity_log_rest` | current 5s value of first difference of log1p ask pull qty / start ask depth total |
| `w0_d2_dn_ask_pull_intensity_log_rest` | current 5s value of second difference of log1p ask pull qty / start ask depth total |
| `w0_d3_dn_ask_pull_intensity_log_rest` | current 5s value of third difference of log1p ask pull qty / start ask depth total |
| `w0_dn_ask_near_pull_share_rest` | current 5s value of share of ask pulls that were near |
| `w0_d1_dn_ask_near_pull_share_rest` | current 5s value of first difference of share of ask pulls that were near |
| `w0_d2_dn_ask_near_pull_share_rest` | current 5s value of second difference of share of ask pulls that were near |
| `w0_d3_dn_ask_near_pull_share_rest` | current 5s value of third difference of share of ask pulls that were near |
| `w0_dn_bid_com_disp_log` | current 5s value of log ratio of bid COM distance from level (ticks), end vs start |
| `w0_d1_dn_bid_com_disp_log` | current 5s value of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w0_d2_dn_bid_com_disp_log` | current 5s value of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w0_d3_dn_bid_com_disp_log` | current 5s value of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w0_dn_bid_slope_convex_log` | current 5s value of log ratio of bid depth far vs near at window end |
| `w0_d1_dn_bid_slope_convex_log` | current 5s value of first difference of log ratio of bid depth far vs near at window end |
| `w0_d2_dn_bid_slope_convex_log` | current 5s value of second difference of log ratio of bid depth far vs near at window end |
| `w0_d3_dn_bid_slope_convex_log` | current 5s value of third difference of log ratio of bid depth far vs near at window end |
| `w0_dn_bid_near_share_delta` | current 5s value of change in bid near depth share (end - start) |
| `w0_d1_dn_bid_near_share_delta` | current 5s value of first difference of change in bid near depth share (end - start) |
| `w0_d2_dn_bid_near_share_delta` | current 5s value of second difference of change in bid near depth share (end - start) |
| `w0_d3_dn_bid_near_share_delta` | current 5s value of third difference of change in bid near depth share (end - start) |
| `w0_dn_bid_reprice_away_share_rest` | current 5s value of share of bid reprice-away in resting reprices |
| `w0_d1_dn_bid_reprice_away_share_rest` | current 5s value of first difference of share of bid reprice-away in resting reprices |
| `w0_d2_dn_bid_reprice_away_share_rest` | current 5s value of second difference of share of bid reprice-away in resting reprices |
| `w0_d3_dn_bid_reprice_away_share_rest` | current 5s value of third difference of share of bid reprice-away in resting reprices |
| `w0_dn_bid_pull_add_log_rest` | current 5s value of log pull/add ratio for resting bid qty |
| `w0_d1_dn_bid_pull_add_log_rest` | current 5s value of first difference of log pull/add ratio for resting bid qty |
| `w0_d2_dn_bid_pull_add_log_rest` | current 5s value of second difference of log pull/add ratio for resting bid qty |
| `w0_d3_dn_bid_pull_add_log_rest` | current 5s value of third difference of log pull/add ratio for resting bid qty |
| `w0_dn_bid_pull_intensity_log_rest` | current 5s value of log1p bid pull qty / start bid depth total |
| `w0_d1_dn_bid_pull_intensity_log_rest` | current 5s value of first difference of log1p bid pull qty / start bid depth total |
| `w0_d2_dn_bid_pull_intensity_log_rest` | current 5s value of second difference of log1p bid pull qty / start bid depth total |
| `w0_d3_dn_bid_pull_intensity_log_rest` | current 5s value of third difference of log1p bid pull qty / start bid depth total |
| `w0_dn_bid_near_pull_share_rest` | current 5s value of share of bid pulls that were near |
| `w0_d1_dn_bid_near_pull_share_rest` | current 5s value of first difference of share of bid pulls that were near |
| `w0_d2_dn_bid_near_pull_share_rest` | current 5s value of second difference of share of bid pulls that were near |
| `w0_d3_dn_bid_near_pull_share_rest` | current 5s value of third difference of share of bid pulls that were near |
| `w0_dn_vacuum_expansion_log` | current 5s value of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w0_d1_dn_vacuum_expansion_log` | current 5s value of first difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w0_d2_dn_vacuum_expansion_log` | current 5s value of second difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w0_d3_dn_vacuum_expansion_log` | current 5s value of third difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w0_dn_vacuum_decay_log` | current 5s value of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w0_d1_dn_vacuum_decay_log` | current 5s value of first difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w0_d2_dn_vacuum_decay_log` | current 5s value of second difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w0_d3_dn_vacuum_decay_log` | current 5s value of third difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w0_dn_vacuum_total_log` | current 5s value of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w0_d1_dn_vacuum_total_log` | current 5s value of first difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w0_d2_dn_vacuum_total_log` | current 5s value of second difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w0_d3_dn_vacuum_total_log` | current 5s value of third difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w0_up_ask_com_disp_log` | current 5s value of copy of dn_ask_com_disp_log |
| `w0_d1_up_ask_com_disp_log` | current 5s value of first difference of copy of dn_ask_com_disp_log |
| `w0_d2_up_ask_com_disp_log` | current 5s value of second difference of copy of dn_ask_com_disp_log |
| `w0_d3_up_ask_com_disp_log` | current 5s value of third difference of copy of dn_ask_com_disp_log |
| `w0_up_ask_slope_convex_log` | current 5s value of copy of dn_ask_slope_convex_log |
| `w0_d1_up_ask_slope_convex_log` | current 5s value of first difference of copy of dn_ask_slope_convex_log |
| `w0_d2_up_ask_slope_convex_log` | current 5s value of second difference of copy of dn_ask_slope_convex_log |
| `w0_d3_up_ask_slope_convex_log` | current 5s value of third difference of copy of dn_ask_slope_convex_log |
| `w0_up_ask_near_share_decay` | current 5s value of negative of dn_ask_near_share_delta |
| `w0_d1_up_ask_near_share_decay` | current 5s value of first difference of negative of dn_ask_near_share_delta |
| `w0_d2_up_ask_near_share_decay` | current 5s value of second difference of negative of dn_ask_near_share_delta |
| `w0_d3_up_ask_near_share_decay` | current 5s value of third difference of negative of dn_ask_near_share_delta |
| `w0_up_ask_reprice_away_share_rest` | current 5s value of copy of dn_ask_reprice_away_share_rest |
| `w0_d1_up_ask_reprice_away_share_rest` | current 5s value of first difference of copy of dn_ask_reprice_away_share_rest |
| `w0_d2_up_ask_reprice_away_share_rest` | current 5s value of second difference of copy of dn_ask_reprice_away_share_rest |
| `w0_d3_up_ask_reprice_away_share_rest` | current 5s value of third difference of copy of dn_ask_reprice_away_share_rest |
| `w0_up_ask_pull_add_log_rest` | current 5s value of copy of dn_ask_pull_add_log_rest |
| `w0_d1_up_ask_pull_add_log_rest` | current 5s value of first difference of copy of dn_ask_pull_add_log_rest |
| `w0_d2_up_ask_pull_add_log_rest` | current 5s value of second difference of copy of dn_ask_pull_add_log_rest |
| `w0_d3_up_ask_pull_add_log_rest` | current 5s value of third difference of copy of dn_ask_pull_add_log_rest |
| `w0_up_ask_pull_intensity_log_rest` | current 5s value of copy of dn_ask_pull_intensity_log_rest |
| `w0_d1_up_ask_pull_intensity_log_rest` | current 5s value of first difference of copy of dn_ask_pull_intensity_log_rest |
| `w0_d2_up_ask_pull_intensity_log_rest` | current 5s value of second difference of copy of dn_ask_pull_intensity_log_rest |
| `w0_d3_up_ask_pull_intensity_log_rest` | current 5s value of third difference of copy of dn_ask_pull_intensity_log_rest |
| `w0_up_ask_near_pull_share_rest` | current 5s value of copy of dn_ask_near_pull_share_rest |
| `w0_d1_up_ask_near_pull_share_rest` | current 5s value of first difference of copy of dn_ask_near_pull_share_rest |
| `w0_d2_up_ask_near_pull_share_rest` | current 5s value of second difference of copy of dn_ask_near_pull_share_rest |
| `w0_d3_up_ask_near_pull_share_rest` | current 5s value of third difference of copy of dn_ask_near_pull_share_rest |
| `w0_up_bid_com_approach_log` | current 5s value of negative of dn_bid_com_disp_log |
| `w0_d1_up_bid_com_approach_log` | current 5s value of first difference of negative of dn_bid_com_disp_log |
| `w0_d2_up_bid_com_approach_log` | current 5s value of second difference of negative of dn_bid_com_disp_log |
| `w0_d3_up_bid_com_approach_log` | current 5s value of third difference of negative of dn_bid_com_disp_log |
| `w0_up_bid_slope_support_log` | current 5s value of negative of dn_bid_slope_convex_log |
| `w0_d1_up_bid_slope_support_log` | current 5s value of first difference of negative of dn_bid_slope_convex_log |
| `w0_d2_up_bid_slope_support_log` | current 5s value of second difference of negative of dn_bid_slope_convex_log |
| `w0_d3_up_bid_slope_support_log` | current 5s value of third difference of negative of dn_bid_slope_convex_log |
| `w0_up_bid_near_share_rise` | current 5s value of copy of dn_bid_near_share_delta |
| `w0_d1_up_bid_near_share_rise` | current 5s value of first difference of copy of dn_bid_near_share_delta |
| `w0_d2_up_bid_near_share_rise` | current 5s value of second difference of copy of dn_bid_near_share_delta |
| `w0_d3_up_bid_near_share_rise` | current 5s value of third difference of copy of dn_bid_near_share_delta |
| `w0_up_bid_reprice_toward_share_rest` | current 5s value of 1 - dn_bid_reprice_away_share_rest |
| `w0_d1_up_bid_reprice_toward_share_rest` | current 5s value of first difference of 1 - dn_bid_reprice_away_share_rest |
| `w0_d2_up_bid_reprice_toward_share_rest` | current 5s value of second difference of 1 - dn_bid_reprice_away_share_rest |
| `w0_d3_up_bid_reprice_toward_share_rest` | current 5s value of third difference of 1 - dn_bid_reprice_away_share_rest |
| `w0_up_bid_add_pull_log_rest` | current 5s value of negative of dn_bid_pull_add_log_rest |
| `w0_d1_up_bid_add_pull_log_rest` | current 5s value of first difference of negative of dn_bid_pull_add_log_rest |
| `w0_d2_up_bid_add_pull_log_rest` | current 5s value of second difference of negative of dn_bid_pull_add_log_rest |
| `w0_d3_up_bid_add_pull_log_rest` | current 5s value of third difference of negative of dn_bid_pull_add_log_rest |
| `w0_up_bid_add_intensity_log` | current 5s value of log1p bid add qty / start bid depth total |
| `w0_d1_up_bid_add_intensity_log` | current 5s value of first difference of log1p bid add qty / start bid depth total |
| `w0_d2_up_bid_add_intensity_log` | current 5s value of second difference of log1p bid add qty / start bid depth total |
| `w0_d3_up_bid_add_intensity_log` | current 5s value of third difference of log1p bid add qty / start bid depth total |
| `w0_up_bid_far_pull_share_rest` | current 5s value of 1 - dn_bid_near_pull_share_rest |
| `w0_d1_up_bid_far_pull_share_rest` | current 5s value of first difference of 1 - dn_bid_near_pull_share_rest |
| `w0_d2_up_bid_far_pull_share_rest` | current 5s value of second difference of 1 - dn_bid_near_pull_share_rest |
| `w0_d3_up_bid_far_pull_share_rest` | current 5s value of third difference of 1 - dn_bid_near_pull_share_rest |
| `w0_up_expansion_log` | current 5s value of up_ask_com_disp_log + up_bid_com_approach_log |
| `w0_d1_up_expansion_log` | current 5s value of first difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w0_d2_up_expansion_log` | current 5s value of second difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w0_d3_up_expansion_log` | current 5s value of third difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w0_up_flow_log` | current 5s value of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w0_d1_up_flow_log` | current 5s value of first difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w0_d2_up_flow_log` | current 5s value of second difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w0_d3_up_flow_log` | current 5s value of third difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w0_up_total_log` | current 5s value of up_expansion_log + up_flow_log |
| `w0_d1_up_total_log` | current 5s value of first difference of up_expansion_log + up_flow_log |
| `w0_d2_up_total_log` | current 5s value of second difference of up_expansion_log + up_flow_log |
| `w0_d3_up_total_log` | current 5s value of third difference of up_expansion_log + up_flow_log |
| `w3_mean_dn_ask_com_disp_log` | 3-window mean (15s) of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_mean_d1_dn_ask_com_disp_log` | 3-window mean (15s) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_mean_d2_dn_ask_com_disp_log` | 3-window mean (15s) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_mean_d3_dn_ask_com_disp_log` | 3-window mean (15s) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_mean_dn_ask_slope_convex_log` | 3-window mean (15s) of log ratio of ask depth far vs near at window end |
| `w3_mean_d1_dn_ask_slope_convex_log` | 3-window mean (15s) of first difference of log ratio of ask depth far vs near at window end |
| `w3_mean_d2_dn_ask_slope_convex_log` | 3-window mean (15s) of second difference of log ratio of ask depth far vs near at window end |
| `w3_mean_d3_dn_ask_slope_convex_log` | 3-window mean (15s) of third difference of log ratio of ask depth far vs near at window end |
| `w3_mean_dn_ask_near_share_delta` | 3-window mean (15s) of change in ask near depth share (end - start) |
| `w3_mean_d1_dn_ask_near_share_delta` | 3-window mean (15s) of first difference of change in ask near depth share (end - start) |
| `w3_mean_d2_dn_ask_near_share_delta` | 3-window mean (15s) of second difference of change in ask near depth share (end - start) |
| `w3_mean_d3_dn_ask_near_share_delta` | 3-window mean (15s) of third difference of change in ask near depth share (end - start) |
| `w3_mean_dn_ask_reprice_away_share_rest` | 3-window mean (15s) of share of ask reprice-away in resting reprices |
| `w3_mean_d1_dn_ask_reprice_away_share_rest` | 3-window mean (15s) of first difference of share of ask reprice-away in resting reprices |
| `w3_mean_d2_dn_ask_reprice_away_share_rest` | 3-window mean (15s) of second difference of share of ask reprice-away in resting reprices |
| `w3_mean_d3_dn_ask_reprice_away_share_rest` | 3-window mean (15s) of third difference of share of ask reprice-away in resting reprices |
| `w3_mean_dn_ask_pull_add_log_rest` | 3-window mean (15s) of log pull/add ratio for resting ask qty |
| `w3_mean_d1_dn_ask_pull_add_log_rest` | 3-window mean (15s) of first difference of log pull/add ratio for resting ask qty |
| `w3_mean_d2_dn_ask_pull_add_log_rest` | 3-window mean (15s) of second difference of log pull/add ratio for resting ask qty |
| `w3_mean_d3_dn_ask_pull_add_log_rest` | 3-window mean (15s) of third difference of log pull/add ratio for resting ask qty |
| `w3_mean_dn_ask_pull_intensity_log_rest` | 3-window mean (15s) of log1p ask pull qty / start ask depth total |
| `w3_mean_d1_dn_ask_pull_intensity_log_rest` | 3-window mean (15s) of first difference of log1p ask pull qty / start ask depth total |
| `w3_mean_d2_dn_ask_pull_intensity_log_rest` | 3-window mean (15s) of second difference of log1p ask pull qty / start ask depth total |
| `w3_mean_d3_dn_ask_pull_intensity_log_rest` | 3-window mean (15s) of third difference of log1p ask pull qty / start ask depth total |
| `w3_mean_dn_ask_near_pull_share_rest` | 3-window mean (15s) of share of ask pulls that were near |
| `w3_mean_d1_dn_ask_near_pull_share_rest` | 3-window mean (15s) of first difference of share of ask pulls that were near |
| `w3_mean_d2_dn_ask_near_pull_share_rest` | 3-window mean (15s) of second difference of share of ask pulls that were near |
| `w3_mean_d3_dn_ask_near_pull_share_rest` | 3-window mean (15s) of third difference of share of ask pulls that were near |
| `w3_mean_dn_bid_com_disp_log` | 3-window mean (15s) of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_mean_d1_dn_bid_com_disp_log` | 3-window mean (15s) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_mean_d2_dn_bid_com_disp_log` | 3-window mean (15s) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_mean_d3_dn_bid_com_disp_log` | 3-window mean (15s) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_mean_dn_bid_slope_convex_log` | 3-window mean (15s) of log ratio of bid depth far vs near at window end |
| `w3_mean_d1_dn_bid_slope_convex_log` | 3-window mean (15s) of first difference of log ratio of bid depth far vs near at window end |
| `w3_mean_d2_dn_bid_slope_convex_log` | 3-window mean (15s) of second difference of log ratio of bid depth far vs near at window end |
| `w3_mean_d3_dn_bid_slope_convex_log` | 3-window mean (15s) of third difference of log ratio of bid depth far vs near at window end |
| `w3_mean_dn_bid_near_share_delta` | 3-window mean (15s) of change in bid near depth share (end - start) |
| `w3_mean_d1_dn_bid_near_share_delta` | 3-window mean (15s) of first difference of change in bid near depth share (end - start) |
| `w3_mean_d2_dn_bid_near_share_delta` | 3-window mean (15s) of second difference of change in bid near depth share (end - start) |
| `w3_mean_d3_dn_bid_near_share_delta` | 3-window mean (15s) of third difference of change in bid near depth share (end - start) |
| `w3_mean_dn_bid_reprice_away_share_rest` | 3-window mean (15s) of share of bid reprice-away in resting reprices |
| `w3_mean_d1_dn_bid_reprice_away_share_rest` | 3-window mean (15s) of first difference of share of bid reprice-away in resting reprices |
| `w3_mean_d2_dn_bid_reprice_away_share_rest` | 3-window mean (15s) of second difference of share of bid reprice-away in resting reprices |
| `w3_mean_d3_dn_bid_reprice_away_share_rest` | 3-window mean (15s) of third difference of share of bid reprice-away in resting reprices |
| `w3_mean_dn_bid_pull_add_log_rest` | 3-window mean (15s) of log pull/add ratio for resting bid qty |
| `w3_mean_d1_dn_bid_pull_add_log_rest` | 3-window mean (15s) of first difference of log pull/add ratio for resting bid qty |
| `w3_mean_d2_dn_bid_pull_add_log_rest` | 3-window mean (15s) of second difference of log pull/add ratio for resting bid qty |
| `w3_mean_d3_dn_bid_pull_add_log_rest` | 3-window mean (15s) of third difference of log pull/add ratio for resting bid qty |
| `w3_mean_dn_bid_pull_intensity_log_rest` | 3-window mean (15s) of log1p bid pull qty / start bid depth total |
| `w3_mean_d1_dn_bid_pull_intensity_log_rest` | 3-window mean (15s) of first difference of log1p bid pull qty / start bid depth total |
| `w3_mean_d2_dn_bid_pull_intensity_log_rest` | 3-window mean (15s) of second difference of log1p bid pull qty / start bid depth total |
| `w3_mean_d3_dn_bid_pull_intensity_log_rest` | 3-window mean (15s) of third difference of log1p bid pull qty / start bid depth total |
| `w3_mean_dn_bid_near_pull_share_rest` | 3-window mean (15s) of share of bid pulls that were near |
| `w3_mean_d1_dn_bid_near_pull_share_rest` | 3-window mean (15s) of first difference of share of bid pulls that were near |
| `w3_mean_d2_dn_bid_near_pull_share_rest` | 3-window mean (15s) of second difference of share of bid pulls that were near |
| `w3_mean_d3_dn_bid_near_pull_share_rest` | 3-window mean (15s) of third difference of share of bid pulls that were near |
| `w3_mean_dn_vacuum_expansion_log` | 3-window mean (15s) of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w3_mean_d1_dn_vacuum_expansion_log` | 3-window mean (15s) of first difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w3_mean_d2_dn_vacuum_expansion_log` | 3-window mean (15s) of second difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w3_mean_d3_dn_vacuum_expansion_log` | 3-window mean (15s) of third difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w3_mean_dn_vacuum_decay_log` | 3-window mean (15s) of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w3_mean_d1_dn_vacuum_decay_log` | 3-window mean (15s) of first difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w3_mean_d2_dn_vacuum_decay_log` | 3-window mean (15s) of second difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w3_mean_d3_dn_vacuum_decay_log` | 3-window mean (15s) of third difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w3_mean_dn_vacuum_total_log` | 3-window mean (15s) of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w3_mean_d1_dn_vacuum_total_log` | 3-window mean (15s) of first difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w3_mean_d2_dn_vacuum_total_log` | 3-window mean (15s) of second difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w3_mean_d3_dn_vacuum_total_log` | 3-window mean (15s) of third difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w3_mean_up_ask_com_disp_log` | 3-window mean (15s) of copy of dn_ask_com_disp_log |
| `w3_mean_d1_up_ask_com_disp_log` | 3-window mean (15s) of first difference of copy of dn_ask_com_disp_log |
| `w3_mean_d2_up_ask_com_disp_log` | 3-window mean (15s) of second difference of copy of dn_ask_com_disp_log |
| `w3_mean_d3_up_ask_com_disp_log` | 3-window mean (15s) of third difference of copy of dn_ask_com_disp_log |
| `w3_mean_up_ask_slope_convex_log` | 3-window mean (15s) of copy of dn_ask_slope_convex_log |
| `w3_mean_d1_up_ask_slope_convex_log` | 3-window mean (15s) of first difference of copy of dn_ask_slope_convex_log |
| `w3_mean_d2_up_ask_slope_convex_log` | 3-window mean (15s) of second difference of copy of dn_ask_slope_convex_log |
| `w3_mean_d3_up_ask_slope_convex_log` | 3-window mean (15s) of third difference of copy of dn_ask_slope_convex_log |
| `w3_mean_up_ask_near_share_decay` | 3-window mean (15s) of negative of dn_ask_near_share_delta |
| `w3_mean_d1_up_ask_near_share_decay` | 3-window mean (15s) of first difference of negative of dn_ask_near_share_delta |
| `w3_mean_d2_up_ask_near_share_decay` | 3-window mean (15s) of second difference of negative of dn_ask_near_share_delta |
| `w3_mean_d3_up_ask_near_share_decay` | 3-window mean (15s) of third difference of negative of dn_ask_near_share_delta |
| `w3_mean_up_ask_reprice_away_share_rest` | 3-window mean (15s) of copy of dn_ask_reprice_away_share_rest |
| `w3_mean_d1_up_ask_reprice_away_share_rest` | 3-window mean (15s) of first difference of copy of dn_ask_reprice_away_share_rest |
| `w3_mean_d2_up_ask_reprice_away_share_rest` | 3-window mean (15s) of second difference of copy of dn_ask_reprice_away_share_rest |
| `w3_mean_d3_up_ask_reprice_away_share_rest` | 3-window mean (15s) of third difference of copy of dn_ask_reprice_away_share_rest |
| `w3_mean_up_ask_pull_add_log_rest` | 3-window mean (15s) of copy of dn_ask_pull_add_log_rest |
| `w3_mean_d1_up_ask_pull_add_log_rest` | 3-window mean (15s) of first difference of copy of dn_ask_pull_add_log_rest |
| `w3_mean_d2_up_ask_pull_add_log_rest` | 3-window mean (15s) of second difference of copy of dn_ask_pull_add_log_rest |
| `w3_mean_d3_up_ask_pull_add_log_rest` | 3-window mean (15s) of third difference of copy of dn_ask_pull_add_log_rest |
| `w3_mean_up_ask_pull_intensity_log_rest` | 3-window mean (15s) of copy of dn_ask_pull_intensity_log_rest |
| `w3_mean_d1_up_ask_pull_intensity_log_rest` | 3-window mean (15s) of first difference of copy of dn_ask_pull_intensity_log_rest |
| `w3_mean_d2_up_ask_pull_intensity_log_rest` | 3-window mean (15s) of second difference of copy of dn_ask_pull_intensity_log_rest |
| `w3_mean_d3_up_ask_pull_intensity_log_rest` | 3-window mean (15s) of third difference of copy of dn_ask_pull_intensity_log_rest |
| `w3_mean_up_ask_near_pull_share_rest` | 3-window mean (15s) of copy of dn_ask_near_pull_share_rest |
| `w3_mean_d1_up_ask_near_pull_share_rest` | 3-window mean (15s) of first difference of copy of dn_ask_near_pull_share_rest |
| `w3_mean_d2_up_ask_near_pull_share_rest` | 3-window mean (15s) of second difference of copy of dn_ask_near_pull_share_rest |
| `w3_mean_d3_up_ask_near_pull_share_rest` | 3-window mean (15s) of third difference of copy of dn_ask_near_pull_share_rest |
| `w3_mean_up_bid_com_approach_log` | 3-window mean (15s) of negative of dn_bid_com_disp_log |
| `w3_mean_d1_up_bid_com_approach_log` | 3-window mean (15s) of first difference of negative of dn_bid_com_disp_log |
| `w3_mean_d2_up_bid_com_approach_log` | 3-window mean (15s) of second difference of negative of dn_bid_com_disp_log |
| `w3_mean_d3_up_bid_com_approach_log` | 3-window mean (15s) of third difference of negative of dn_bid_com_disp_log |
| `w3_mean_up_bid_slope_support_log` | 3-window mean (15s) of negative of dn_bid_slope_convex_log |
| `w3_mean_d1_up_bid_slope_support_log` | 3-window mean (15s) of first difference of negative of dn_bid_slope_convex_log |
| `w3_mean_d2_up_bid_slope_support_log` | 3-window mean (15s) of second difference of negative of dn_bid_slope_convex_log |
| `w3_mean_d3_up_bid_slope_support_log` | 3-window mean (15s) of third difference of negative of dn_bid_slope_convex_log |
| `w3_mean_up_bid_near_share_rise` | 3-window mean (15s) of copy of dn_bid_near_share_delta |
| `w3_mean_d1_up_bid_near_share_rise` | 3-window mean (15s) of first difference of copy of dn_bid_near_share_delta |
| `w3_mean_d2_up_bid_near_share_rise` | 3-window mean (15s) of second difference of copy of dn_bid_near_share_delta |
| `w3_mean_d3_up_bid_near_share_rise` | 3-window mean (15s) of third difference of copy of dn_bid_near_share_delta |
| `w3_mean_up_bid_reprice_toward_share_rest` | 3-window mean (15s) of 1 - dn_bid_reprice_away_share_rest |
| `w3_mean_d1_up_bid_reprice_toward_share_rest` | 3-window mean (15s) of first difference of 1 - dn_bid_reprice_away_share_rest |
| `w3_mean_d2_up_bid_reprice_toward_share_rest` | 3-window mean (15s) of second difference of 1 - dn_bid_reprice_away_share_rest |
| `w3_mean_d3_up_bid_reprice_toward_share_rest` | 3-window mean (15s) of third difference of 1 - dn_bid_reprice_away_share_rest |
| `w3_mean_up_bid_add_pull_log_rest` | 3-window mean (15s) of negative of dn_bid_pull_add_log_rest |
| `w3_mean_d1_up_bid_add_pull_log_rest` | 3-window mean (15s) of first difference of negative of dn_bid_pull_add_log_rest |
| `w3_mean_d2_up_bid_add_pull_log_rest` | 3-window mean (15s) of second difference of negative of dn_bid_pull_add_log_rest |
| `w3_mean_d3_up_bid_add_pull_log_rest` | 3-window mean (15s) of third difference of negative of dn_bid_pull_add_log_rest |
| `w3_mean_up_bid_add_intensity_log` | 3-window mean (15s) of log1p bid add qty / start bid depth total |
| `w3_mean_d1_up_bid_add_intensity_log` | 3-window mean (15s) of first difference of log1p bid add qty / start bid depth total |
| `w3_mean_d2_up_bid_add_intensity_log` | 3-window mean (15s) of second difference of log1p bid add qty / start bid depth total |
| `w3_mean_d3_up_bid_add_intensity_log` | 3-window mean (15s) of third difference of log1p bid add qty / start bid depth total |
| `w3_mean_up_bid_far_pull_share_rest` | 3-window mean (15s) of 1 - dn_bid_near_pull_share_rest |
| `w3_mean_d1_up_bid_far_pull_share_rest` | 3-window mean (15s) of first difference of 1 - dn_bid_near_pull_share_rest |
| `w3_mean_d2_up_bid_far_pull_share_rest` | 3-window mean (15s) of second difference of 1 - dn_bid_near_pull_share_rest |
| `w3_mean_d3_up_bid_far_pull_share_rest` | 3-window mean (15s) of third difference of 1 - dn_bid_near_pull_share_rest |
| `w3_mean_up_expansion_log` | 3-window mean (15s) of up_ask_com_disp_log + up_bid_com_approach_log |
| `w3_mean_d1_up_expansion_log` | 3-window mean (15s) of first difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w3_mean_d2_up_expansion_log` | 3-window mean (15s) of second difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w3_mean_d3_up_expansion_log` | 3-window mean (15s) of third difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w3_mean_up_flow_log` | 3-window mean (15s) of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w3_mean_d1_up_flow_log` | 3-window mean (15s) of first difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w3_mean_d2_up_flow_log` | 3-window mean (15s) of second difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w3_mean_d3_up_flow_log` | 3-window mean (15s) of third difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w3_mean_up_total_log` | 3-window mean (15s) of up_expansion_log + up_flow_log |
| `w3_mean_d1_up_total_log` | 3-window mean (15s) of first difference of up_expansion_log + up_flow_log |
| `w3_mean_d2_up_total_log` | 3-window mean (15s) of second difference of up_expansion_log + up_flow_log |
| `w3_mean_d3_up_total_log` | 3-window mean (15s) of third difference of up_expansion_log + up_flow_log |
| `w3_delta_dn_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_delta_d1_dn_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_delta_d2_dn_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_delta_d3_dn_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w3_delta_dn_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of log ratio of ask depth far vs near at window end |
| `w3_delta_d1_dn_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log ratio of ask depth far vs near at window end |
| `w3_delta_d2_dn_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log ratio of ask depth far vs near at window end |
| `w3_delta_d3_dn_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log ratio of ask depth far vs near at window end |
| `w3_delta_dn_ask_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of change in ask near depth share (end - start) |
| `w3_delta_d1_dn_ask_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of first difference of change in ask near depth share (end - start) |
| `w3_delta_d2_dn_ask_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of second difference of change in ask near depth share (end - start) |
| `w3_delta_d3_dn_ask_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of third difference of change in ask near depth share (end - start) |
| `w3_delta_dn_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of share of ask reprice-away in resting reprices |
| `w3_delta_d1_dn_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of share of ask reprice-away in resting reprices |
| `w3_delta_d2_dn_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of share of ask reprice-away in resting reprices |
| `w3_delta_d3_dn_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of share of ask reprice-away in resting reprices |
| `w3_delta_dn_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of log pull/add ratio for resting ask qty |
| `w3_delta_d1_dn_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log pull/add ratio for resting ask qty |
| `w3_delta_d2_dn_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log pull/add ratio for resting ask qty |
| `w3_delta_d3_dn_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log pull/add ratio for resting ask qty |
| `w3_delta_dn_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of log1p ask pull qty / start ask depth total |
| `w3_delta_d1_dn_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log1p ask pull qty / start ask depth total |
| `w3_delta_d2_dn_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log1p ask pull qty / start ask depth total |
| `w3_delta_d3_dn_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log1p ask pull qty / start ask depth total |
| `w3_delta_dn_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of share of ask pulls that were near |
| `w3_delta_d1_dn_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of share of ask pulls that were near |
| `w3_delta_d2_dn_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of share of ask pulls that were near |
| `w3_delta_d3_dn_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of share of ask pulls that were near |
| `w3_delta_dn_bid_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_delta_d1_dn_bid_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_delta_d2_dn_bid_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_delta_d3_dn_bid_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w3_delta_dn_bid_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of log ratio of bid depth far vs near at window end |
| `w3_delta_d1_dn_bid_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log ratio of bid depth far vs near at window end |
| `w3_delta_d2_dn_bid_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log ratio of bid depth far vs near at window end |
| `w3_delta_d3_dn_bid_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log ratio of bid depth far vs near at window end |
| `w3_delta_dn_bid_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of change in bid near depth share (end - start) |
| `w3_delta_d1_dn_bid_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of first difference of change in bid near depth share (end - start) |
| `w3_delta_d2_dn_bid_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of second difference of change in bid near depth share (end - start) |
| `w3_delta_d3_dn_bid_near_share_delta` | 3-window delta ((x_t - x_{t-2})/2) of third difference of change in bid near depth share (end - start) |
| `w3_delta_dn_bid_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of share of bid reprice-away in resting reprices |
| `w3_delta_d1_dn_bid_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of share of bid reprice-away in resting reprices |
| `w3_delta_d2_dn_bid_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of share of bid reprice-away in resting reprices |
| `w3_delta_d3_dn_bid_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of share of bid reprice-away in resting reprices |
| `w3_delta_dn_bid_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of log pull/add ratio for resting bid qty |
| `w3_delta_d1_dn_bid_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log pull/add ratio for resting bid qty |
| `w3_delta_d2_dn_bid_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log pull/add ratio for resting bid qty |
| `w3_delta_d3_dn_bid_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log pull/add ratio for resting bid qty |
| `w3_delta_dn_bid_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of log1p bid pull qty / start bid depth total |
| `w3_delta_d1_dn_bid_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log1p bid pull qty / start bid depth total |
| `w3_delta_d2_dn_bid_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log1p bid pull qty / start bid depth total |
| `w3_delta_d3_dn_bid_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log1p bid pull qty / start bid depth total |
| `w3_delta_dn_bid_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of share of bid pulls that were near |
| `w3_delta_d1_dn_bid_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of share of bid pulls that were near |
| `w3_delta_d2_dn_bid_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of share of bid pulls that were near |
| `w3_delta_d3_dn_bid_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of share of bid pulls that were near |
| `w3_delta_dn_vacuum_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w3_delta_d1_dn_vacuum_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w3_delta_d2_dn_vacuum_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w3_delta_d3_dn_vacuum_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w3_delta_dn_vacuum_decay_log` | 3-window delta ((x_t - x_{t-2})/2) of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w3_delta_d1_dn_vacuum_decay_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w3_delta_d2_dn_vacuum_decay_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w3_delta_d3_dn_vacuum_decay_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w3_delta_dn_vacuum_total_log` | 3-window delta ((x_t - x_{t-2})/2) of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w3_delta_d1_dn_vacuum_total_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w3_delta_d2_dn_vacuum_total_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w3_delta_d3_dn_vacuum_total_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w3_delta_up_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of copy of dn_ask_com_disp_log |
| `w3_delta_d1_up_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of dn_ask_com_disp_log |
| `w3_delta_d2_up_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of dn_ask_com_disp_log |
| `w3_delta_d3_up_ask_com_disp_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of dn_ask_com_disp_log |
| `w3_delta_up_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of copy of dn_ask_slope_convex_log |
| `w3_delta_d1_up_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of dn_ask_slope_convex_log |
| `w3_delta_d2_up_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of dn_ask_slope_convex_log |
| `w3_delta_d3_up_ask_slope_convex_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of dn_ask_slope_convex_log |
| `w3_delta_up_ask_near_share_decay` | 3-window delta ((x_t - x_{t-2})/2) of negative of dn_ask_near_share_delta |
| `w3_delta_d1_up_ask_near_share_decay` | 3-window delta ((x_t - x_{t-2})/2) of first difference of negative of dn_ask_near_share_delta |
| `w3_delta_d2_up_ask_near_share_decay` | 3-window delta ((x_t - x_{t-2})/2) of second difference of negative of dn_ask_near_share_delta |
| `w3_delta_d3_up_ask_near_share_decay` | 3-window delta ((x_t - x_{t-2})/2) of third difference of negative of dn_ask_near_share_delta |
| `w3_delta_up_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of copy of dn_ask_reprice_away_share_rest |
| `w3_delta_d1_up_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of dn_ask_reprice_away_share_rest |
| `w3_delta_d2_up_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of dn_ask_reprice_away_share_rest |
| `w3_delta_d3_up_ask_reprice_away_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of dn_ask_reprice_away_share_rest |
| `w3_delta_up_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of copy of dn_ask_pull_add_log_rest |
| `w3_delta_d1_up_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of dn_ask_pull_add_log_rest |
| `w3_delta_d2_up_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of dn_ask_pull_add_log_rest |
| `w3_delta_d3_up_ask_pull_add_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of dn_ask_pull_add_log_rest |
| `w3_delta_up_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of copy of dn_ask_pull_intensity_log_rest |
| `w3_delta_d1_up_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of dn_ask_pull_intensity_log_rest |
| `w3_delta_d2_up_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of dn_ask_pull_intensity_log_rest |
| `w3_delta_d3_up_ask_pull_intensity_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of dn_ask_pull_intensity_log_rest |
| `w3_delta_up_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of copy of dn_ask_near_pull_share_rest |
| `w3_delta_d1_up_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of dn_ask_near_pull_share_rest |
| `w3_delta_d2_up_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of dn_ask_near_pull_share_rest |
| `w3_delta_d3_up_ask_near_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of dn_ask_near_pull_share_rest |
| `w3_delta_up_bid_com_approach_log` | 3-window delta ((x_t - x_{t-2})/2) of negative of dn_bid_com_disp_log |
| `w3_delta_d1_up_bid_com_approach_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of negative of dn_bid_com_disp_log |
| `w3_delta_d2_up_bid_com_approach_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of negative of dn_bid_com_disp_log |
| `w3_delta_d3_up_bid_com_approach_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of negative of dn_bid_com_disp_log |
| `w3_delta_up_bid_slope_support_log` | 3-window delta ((x_t - x_{t-2})/2) of negative of dn_bid_slope_convex_log |
| `w3_delta_d1_up_bid_slope_support_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of negative of dn_bid_slope_convex_log |
| `w3_delta_d2_up_bid_slope_support_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of negative of dn_bid_slope_convex_log |
| `w3_delta_d3_up_bid_slope_support_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of negative of dn_bid_slope_convex_log |
| `w3_delta_up_bid_near_share_rise` | 3-window delta ((x_t - x_{t-2})/2) of copy of dn_bid_near_share_delta |
| `w3_delta_d1_up_bid_near_share_rise` | 3-window delta ((x_t - x_{t-2})/2) of first difference of copy of dn_bid_near_share_delta |
| `w3_delta_d2_up_bid_near_share_rise` | 3-window delta ((x_t - x_{t-2})/2) of second difference of copy of dn_bid_near_share_delta |
| `w3_delta_d3_up_bid_near_share_rise` | 3-window delta ((x_t - x_{t-2})/2) of third difference of copy of dn_bid_near_share_delta |
| `w3_delta_up_bid_reprice_toward_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of 1 - dn_bid_reprice_away_share_rest |
| `w3_delta_d1_up_bid_reprice_toward_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of 1 - dn_bid_reprice_away_share_rest |
| `w3_delta_d2_up_bid_reprice_toward_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of 1 - dn_bid_reprice_away_share_rest |
| `w3_delta_d3_up_bid_reprice_toward_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of 1 - dn_bid_reprice_away_share_rest |
| `w3_delta_up_bid_add_pull_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of negative of dn_bid_pull_add_log_rest |
| `w3_delta_d1_up_bid_add_pull_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of negative of dn_bid_pull_add_log_rest |
| `w3_delta_d2_up_bid_add_pull_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of negative of dn_bid_pull_add_log_rest |
| `w3_delta_d3_up_bid_add_pull_log_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of negative of dn_bid_pull_add_log_rest |
| `w3_delta_up_bid_add_intensity_log` | 3-window delta ((x_t - x_{t-2})/2) of log1p bid add qty / start bid depth total |
| `w3_delta_d1_up_bid_add_intensity_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of log1p bid add qty / start bid depth total |
| `w3_delta_d2_up_bid_add_intensity_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of log1p bid add qty / start bid depth total |
| `w3_delta_d3_up_bid_add_intensity_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of log1p bid add qty / start bid depth total |
| `w3_delta_up_bid_far_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of 1 - dn_bid_near_pull_share_rest |
| `w3_delta_d1_up_bid_far_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of first difference of 1 - dn_bid_near_pull_share_rest |
| `w3_delta_d2_up_bid_far_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of second difference of 1 - dn_bid_near_pull_share_rest |
| `w3_delta_d3_up_bid_far_pull_share_rest` | 3-window delta ((x_t - x_{t-2})/2) of third difference of 1 - dn_bid_near_pull_share_rest |
| `w3_delta_up_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of up_ask_com_disp_log + up_bid_com_approach_log |
| `w3_delta_d1_up_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w3_delta_d2_up_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w3_delta_d3_up_expansion_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w3_delta_up_flow_log` | 3-window delta ((x_t - x_{t-2})/2) of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w3_delta_d1_up_flow_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w3_delta_d2_up_flow_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w3_delta_d3_up_flow_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w3_delta_up_total_log` | 3-window delta ((x_t - x_{t-2})/2) of up_expansion_log + up_flow_log |
| `w3_delta_d1_up_total_log` | 3-window delta ((x_t - x_{t-2})/2) of first difference of up_expansion_log + up_flow_log |
| `w3_delta_d2_up_total_log` | 3-window delta ((x_t - x_{t-2})/2) of second difference of up_expansion_log + up_flow_log |
| `w3_delta_d3_up_total_log` | 3-window delta ((x_t - x_{t-2})/2) of third difference of up_expansion_log + up_flow_log |
| `w9_mean_dn_ask_com_disp_log` | 9-window mean (45s) of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_mean_d1_dn_ask_com_disp_log` | 9-window mean (45s) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_mean_d2_dn_ask_com_disp_log` | 9-window mean (45s) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_mean_d3_dn_ask_com_disp_log` | 9-window mean (45s) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_mean_dn_ask_slope_convex_log` | 9-window mean (45s) of log ratio of ask depth far vs near at window end |
| `w9_mean_d1_dn_ask_slope_convex_log` | 9-window mean (45s) of first difference of log ratio of ask depth far vs near at window end |
| `w9_mean_d2_dn_ask_slope_convex_log` | 9-window mean (45s) of second difference of log ratio of ask depth far vs near at window end |
| `w9_mean_d3_dn_ask_slope_convex_log` | 9-window mean (45s) of third difference of log ratio of ask depth far vs near at window end |
| `w9_mean_dn_ask_near_share_delta` | 9-window mean (45s) of change in ask near depth share (end - start) |
| `w9_mean_d1_dn_ask_near_share_delta` | 9-window mean (45s) of first difference of change in ask near depth share (end - start) |
| `w9_mean_d2_dn_ask_near_share_delta` | 9-window mean (45s) of second difference of change in ask near depth share (end - start) |
| `w9_mean_d3_dn_ask_near_share_delta` | 9-window mean (45s) of third difference of change in ask near depth share (end - start) |
| `w9_mean_dn_ask_reprice_away_share_rest` | 9-window mean (45s) of share of ask reprice-away in resting reprices |
| `w9_mean_d1_dn_ask_reprice_away_share_rest` | 9-window mean (45s) of first difference of share of ask reprice-away in resting reprices |
| `w9_mean_d2_dn_ask_reprice_away_share_rest` | 9-window mean (45s) of second difference of share of ask reprice-away in resting reprices |
| `w9_mean_d3_dn_ask_reprice_away_share_rest` | 9-window mean (45s) of third difference of share of ask reprice-away in resting reprices |
| `w9_mean_dn_ask_pull_add_log_rest` | 9-window mean (45s) of log pull/add ratio for resting ask qty |
| `w9_mean_d1_dn_ask_pull_add_log_rest` | 9-window mean (45s) of first difference of log pull/add ratio for resting ask qty |
| `w9_mean_d2_dn_ask_pull_add_log_rest` | 9-window mean (45s) of second difference of log pull/add ratio for resting ask qty |
| `w9_mean_d3_dn_ask_pull_add_log_rest` | 9-window mean (45s) of third difference of log pull/add ratio for resting ask qty |
| `w9_mean_dn_ask_pull_intensity_log_rest` | 9-window mean (45s) of log1p ask pull qty / start ask depth total |
| `w9_mean_d1_dn_ask_pull_intensity_log_rest` | 9-window mean (45s) of first difference of log1p ask pull qty / start ask depth total |
| `w9_mean_d2_dn_ask_pull_intensity_log_rest` | 9-window mean (45s) of second difference of log1p ask pull qty / start ask depth total |
| `w9_mean_d3_dn_ask_pull_intensity_log_rest` | 9-window mean (45s) of third difference of log1p ask pull qty / start ask depth total |
| `w9_mean_dn_ask_near_pull_share_rest` | 9-window mean (45s) of share of ask pulls that were near |
| `w9_mean_d1_dn_ask_near_pull_share_rest` | 9-window mean (45s) of first difference of share of ask pulls that were near |
| `w9_mean_d2_dn_ask_near_pull_share_rest` | 9-window mean (45s) of second difference of share of ask pulls that were near |
| `w9_mean_d3_dn_ask_near_pull_share_rest` | 9-window mean (45s) of third difference of share of ask pulls that were near |
| `w9_mean_dn_bid_com_disp_log` | 9-window mean (45s) of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_mean_d1_dn_bid_com_disp_log` | 9-window mean (45s) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_mean_d2_dn_bid_com_disp_log` | 9-window mean (45s) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_mean_d3_dn_bid_com_disp_log` | 9-window mean (45s) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_mean_dn_bid_slope_convex_log` | 9-window mean (45s) of log ratio of bid depth far vs near at window end |
| `w9_mean_d1_dn_bid_slope_convex_log` | 9-window mean (45s) of first difference of log ratio of bid depth far vs near at window end |
| `w9_mean_d2_dn_bid_slope_convex_log` | 9-window mean (45s) of second difference of log ratio of bid depth far vs near at window end |
| `w9_mean_d3_dn_bid_slope_convex_log` | 9-window mean (45s) of third difference of log ratio of bid depth far vs near at window end |
| `w9_mean_dn_bid_near_share_delta` | 9-window mean (45s) of change in bid near depth share (end - start) |
| `w9_mean_d1_dn_bid_near_share_delta` | 9-window mean (45s) of first difference of change in bid near depth share (end - start) |
| `w9_mean_d2_dn_bid_near_share_delta` | 9-window mean (45s) of second difference of change in bid near depth share (end - start) |
| `w9_mean_d3_dn_bid_near_share_delta` | 9-window mean (45s) of third difference of change in bid near depth share (end - start) |
| `w9_mean_dn_bid_reprice_away_share_rest` | 9-window mean (45s) of share of bid reprice-away in resting reprices |
| `w9_mean_d1_dn_bid_reprice_away_share_rest` | 9-window mean (45s) of first difference of share of bid reprice-away in resting reprices |
| `w9_mean_d2_dn_bid_reprice_away_share_rest` | 9-window mean (45s) of second difference of share of bid reprice-away in resting reprices |
| `w9_mean_d3_dn_bid_reprice_away_share_rest` | 9-window mean (45s) of third difference of share of bid reprice-away in resting reprices |
| `w9_mean_dn_bid_pull_add_log_rest` | 9-window mean (45s) of log pull/add ratio for resting bid qty |
| `w9_mean_d1_dn_bid_pull_add_log_rest` | 9-window mean (45s) of first difference of log pull/add ratio for resting bid qty |
| `w9_mean_d2_dn_bid_pull_add_log_rest` | 9-window mean (45s) of second difference of log pull/add ratio for resting bid qty |
| `w9_mean_d3_dn_bid_pull_add_log_rest` | 9-window mean (45s) of third difference of log pull/add ratio for resting bid qty |
| `w9_mean_dn_bid_pull_intensity_log_rest` | 9-window mean (45s) of log1p bid pull qty / start bid depth total |
| `w9_mean_d1_dn_bid_pull_intensity_log_rest` | 9-window mean (45s) of first difference of log1p bid pull qty / start bid depth total |
| `w9_mean_d2_dn_bid_pull_intensity_log_rest` | 9-window mean (45s) of second difference of log1p bid pull qty / start bid depth total |
| `w9_mean_d3_dn_bid_pull_intensity_log_rest` | 9-window mean (45s) of third difference of log1p bid pull qty / start bid depth total |
| `w9_mean_dn_bid_near_pull_share_rest` | 9-window mean (45s) of share of bid pulls that were near |
| `w9_mean_d1_dn_bid_near_pull_share_rest` | 9-window mean (45s) of first difference of share of bid pulls that were near |
| `w9_mean_d2_dn_bid_near_pull_share_rest` | 9-window mean (45s) of second difference of share of bid pulls that were near |
| `w9_mean_d3_dn_bid_near_pull_share_rest` | 9-window mean (45s) of third difference of share of bid pulls that were near |
| `w9_mean_dn_vacuum_expansion_log` | 9-window mean (45s) of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w9_mean_d1_dn_vacuum_expansion_log` | 9-window mean (45s) of first difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w9_mean_d2_dn_vacuum_expansion_log` | 9-window mean (45s) of second difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w9_mean_d3_dn_vacuum_expansion_log` | 9-window mean (45s) of third difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w9_mean_dn_vacuum_decay_log` | 9-window mean (45s) of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w9_mean_d1_dn_vacuum_decay_log` | 9-window mean (45s) of first difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w9_mean_d2_dn_vacuum_decay_log` | 9-window mean (45s) of second difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w9_mean_d3_dn_vacuum_decay_log` | 9-window mean (45s) of third difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w9_mean_dn_vacuum_total_log` | 9-window mean (45s) of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w9_mean_d1_dn_vacuum_total_log` | 9-window mean (45s) of first difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w9_mean_d2_dn_vacuum_total_log` | 9-window mean (45s) of second difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w9_mean_d3_dn_vacuum_total_log` | 9-window mean (45s) of third difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w9_mean_up_ask_com_disp_log` | 9-window mean (45s) of copy of dn_ask_com_disp_log |
| `w9_mean_d1_up_ask_com_disp_log` | 9-window mean (45s) of first difference of copy of dn_ask_com_disp_log |
| `w9_mean_d2_up_ask_com_disp_log` | 9-window mean (45s) of second difference of copy of dn_ask_com_disp_log |
| `w9_mean_d3_up_ask_com_disp_log` | 9-window mean (45s) of third difference of copy of dn_ask_com_disp_log |
| `w9_mean_up_ask_slope_convex_log` | 9-window mean (45s) of copy of dn_ask_slope_convex_log |
| `w9_mean_d1_up_ask_slope_convex_log` | 9-window mean (45s) of first difference of copy of dn_ask_slope_convex_log |
| `w9_mean_d2_up_ask_slope_convex_log` | 9-window mean (45s) of second difference of copy of dn_ask_slope_convex_log |
| `w9_mean_d3_up_ask_slope_convex_log` | 9-window mean (45s) of third difference of copy of dn_ask_slope_convex_log |
| `w9_mean_up_ask_near_share_decay` | 9-window mean (45s) of negative of dn_ask_near_share_delta |
| `w9_mean_d1_up_ask_near_share_decay` | 9-window mean (45s) of first difference of negative of dn_ask_near_share_delta |
| `w9_mean_d2_up_ask_near_share_decay` | 9-window mean (45s) of second difference of negative of dn_ask_near_share_delta |
| `w9_mean_d3_up_ask_near_share_decay` | 9-window mean (45s) of third difference of negative of dn_ask_near_share_delta |
| `w9_mean_up_ask_reprice_away_share_rest` | 9-window mean (45s) of copy of dn_ask_reprice_away_share_rest |
| `w9_mean_d1_up_ask_reprice_away_share_rest` | 9-window mean (45s) of first difference of copy of dn_ask_reprice_away_share_rest |
| `w9_mean_d2_up_ask_reprice_away_share_rest` | 9-window mean (45s) of second difference of copy of dn_ask_reprice_away_share_rest |
| `w9_mean_d3_up_ask_reprice_away_share_rest` | 9-window mean (45s) of third difference of copy of dn_ask_reprice_away_share_rest |
| `w9_mean_up_ask_pull_add_log_rest` | 9-window mean (45s) of copy of dn_ask_pull_add_log_rest |
| `w9_mean_d1_up_ask_pull_add_log_rest` | 9-window mean (45s) of first difference of copy of dn_ask_pull_add_log_rest |
| `w9_mean_d2_up_ask_pull_add_log_rest` | 9-window mean (45s) of second difference of copy of dn_ask_pull_add_log_rest |
| `w9_mean_d3_up_ask_pull_add_log_rest` | 9-window mean (45s) of third difference of copy of dn_ask_pull_add_log_rest |
| `w9_mean_up_ask_pull_intensity_log_rest` | 9-window mean (45s) of copy of dn_ask_pull_intensity_log_rest |
| `w9_mean_d1_up_ask_pull_intensity_log_rest` | 9-window mean (45s) of first difference of copy of dn_ask_pull_intensity_log_rest |
| `w9_mean_d2_up_ask_pull_intensity_log_rest` | 9-window mean (45s) of second difference of copy of dn_ask_pull_intensity_log_rest |
| `w9_mean_d3_up_ask_pull_intensity_log_rest` | 9-window mean (45s) of third difference of copy of dn_ask_pull_intensity_log_rest |
| `w9_mean_up_ask_near_pull_share_rest` | 9-window mean (45s) of copy of dn_ask_near_pull_share_rest |
| `w9_mean_d1_up_ask_near_pull_share_rest` | 9-window mean (45s) of first difference of copy of dn_ask_near_pull_share_rest |
| `w9_mean_d2_up_ask_near_pull_share_rest` | 9-window mean (45s) of second difference of copy of dn_ask_near_pull_share_rest |
| `w9_mean_d3_up_ask_near_pull_share_rest` | 9-window mean (45s) of third difference of copy of dn_ask_near_pull_share_rest |
| `w9_mean_up_bid_com_approach_log` | 9-window mean (45s) of negative of dn_bid_com_disp_log |
| `w9_mean_d1_up_bid_com_approach_log` | 9-window mean (45s) of first difference of negative of dn_bid_com_disp_log |
| `w9_mean_d2_up_bid_com_approach_log` | 9-window mean (45s) of second difference of negative of dn_bid_com_disp_log |
| `w9_mean_d3_up_bid_com_approach_log` | 9-window mean (45s) of third difference of negative of dn_bid_com_disp_log |
| `w9_mean_up_bid_slope_support_log` | 9-window mean (45s) of negative of dn_bid_slope_convex_log |
| `w9_mean_d1_up_bid_slope_support_log` | 9-window mean (45s) of first difference of negative of dn_bid_slope_convex_log |
| `w9_mean_d2_up_bid_slope_support_log` | 9-window mean (45s) of second difference of negative of dn_bid_slope_convex_log |
| `w9_mean_d3_up_bid_slope_support_log` | 9-window mean (45s) of third difference of negative of dn_bid_slope_convex_log |
| `w9_mean_up_bid_near_share_rise` | 9-window mean (45s) of copy of dn_bid_near_share_delta |
| `w9_mean_d1_up_bid_near_share_rise` | 9-window mean (45s) of first difference of copy of dn_bid_near_share_delta |
| `w9_mean_d2_up_bid_near_share_rise` | 9-window mean (45s) of second difference of copy of dn_bid_near_share_delta |
| `w9_mean_d3_up_bid_near_share_rise` | 9-window mean (45s) of third difference of copy of dn_bid_near_share_delta |
| `w9_mean_up_bid_reprice_toward_share_rest` | 9-window mean (45s) of 1 - dn_bid_reprice_away_share_rest |
| `w9_mean_d1_up_bid_reprice_toward_share_rest` | 9-window mean (45s) of first difference of 1 - dn_bid_reprice_away_share_rest |
| `w9_mean_d2_up_bid_reprice_toward_share_rest` | 9-window mean (45s) of second difference of 1 - dn_bid_reprice_away_share_rest |
| `w9_mean_d3_up_bid_reprice_toward_share_rest` | 9-window mean (45s) of third difference of 1 - dn_bid_reprice_away_share_rest |
| `w9_mean_up_bid_add_pull_log_rest` | 9-window mean (45s) of negative of dn_bid_pull_add_log_rest |
| `w9_mean_d1_up_bid_add_pull_log_rest` | 9-window mean (45s) of first difference of negative of dn_bid_pull_add_log_rest |
| `w9_mean_d2_up_bid_add_pull_log_rest` | 9-window mean (45s) of second difference of negative of dn_bid_pull_add_log_rest |
| `w9_mean_d3_up_bid_add_pull_log_rest` | 9-window mean (45s) of third difference of negative of dn_bid_pull_add_log_rest |
| `w9_mean_up_bid_add_intensity_log` | 9-window mean (45s) of log1p bid add qty / start bid depth total |
| `w9_mean_d1_up_bid_add_intensity_log` | 9-window mean (45s) of first difference of log1p bid add qty / start bid depth total |
| `w9_mean_d2_up_bid_add_intensity_log` | 9-window mean (45s) of second difference of log1p bid add qty / start bid depth total |
| `w9_mean_d3_up_bid_add_intensity_log` | 9-window mean (45s) of third difference of log1p bid add qty / start bid depth total |
| `w9_mean_up_bid_far_pull_share_rest` | 9-window mean (45s) of 1 - dn_bid_near_pull_share_rest |
| `w9_mean_d1_up_bid_far_pull_share_rest` | 9-window mean (45s) of first difference of 1 - dn_bid_near_pull_share_rest |
| `w9_mean_d2_up_bid_far_pull_share_rest` | 9-window mean (45s) of second difference of 1 - dn_bid_near_pull_share_rest |
| `w9_mean_d3_up_bid_far_pull_share_rest` | 9-window mean (45s) of third difference of 1 - dn_bid_near_pull_share_rest |
| `w9_mean_up_expansion_log` | 9-window mean (45s) of up_ask_com_disp_log + up_bid_com_approach_log |
| `w9_mean_d1_up_expansion_log` | 9-window mean (45s) of first difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w9_mean_d2_up_expansion_log` | 9-window mean (45s) of second difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w9_mean_d3_up_expansion_log` | 9-window mean (45s) of third difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w9_mean_up_flow_log` | 9-window mean (45s) of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w9_mean_d1_up_flow_log` | 9-window mean (45s) of first difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w9_mean_d2_up_flow_log` | 9-window mean (45s) of second difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w9_mean_d3_up_flow_log` | 9-window mean (45s) of third difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w9_mean_up_total_log` | 9-window mean (45s) of up_expansion_log + up_flow_log |
| `w9_mean_d1_up_total_log` | 9-window mean (45s) of first difference of up_expansion_log + up_flow_log |
| `w9_mean_d2_up_total_log` | 9-window mean (45s) of second difference of up_expansion_log + up_flow_log |
| `w9_mean_d3_up_total_log` | 9-window mean (45s) of third difference of up_expansion_log + up_flow_log |
| `w9_delta_dn_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_delta_d1_dn_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_delta_d2_dn_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_delta_d3_dn_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w9_delta_dn_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of log ratio of ask depth far vs near at window end |
| `w9_delta_d1_dn_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log ratio of ask depth far vs near at window end |
| `w9_delta_d2_dn_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log ratio of ask depth far vs near at window end |
| `w9_delta_d3_dn_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log ratio of ask depth far vs near at window end |
| `w9_delta_dn_ask_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of change in ask near depth share (end - start) |
| `w9_delta_d1_dn_ask_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of first difference of change in ask near depth share (end - start) |
| `w9_delta_d2_dn_ask_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of second difference of change in ask near depth share (end - start) |
| `w9_delta_d3_dn_ask_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of third difference of change in ask near depth share (end - start) |
| `w9_delta_dn_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of share of ask reprice-away in resting reprices |
| `w9_delta_d1_dn_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of share of ask reprice-away in resting reprices |
| `w9_delta_d2_dn_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of share of ask reprice-away in resting reprices |
| `w9_delta_d3_dn_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of share of ask reprice-away in resting reprices |
| `w9_delta_dn_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of log pull/add ratio for resting ask qty |
| `w9_delta_d1_dn_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log pull/add ratio for resting ask qty |
| `w9_delta_d2_dn_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log pull/add ratio for resting ask qty |
| `w9_delta_d3_dn_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log pull/add ratio for resting ask qty |
| `w9_delta_dn_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of log1p ask pull qty / start ask depth total |
| `w9_delta_d1_dn_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log1p ask pull qty / start ask depth total |
| `w9_delta_d2_dn_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log1p ask pull qty / start ask depth total |
| `w9_delta_d3_dn_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log1p ask pull qty / start ask depth total |
| `w9_delta_dn_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of share of ask pulls that were near |
| `w9_delta_d1_dn_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of share of ask pulls that were near |
| `w9_delta_d2_dn_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of share of ask pulls that were near |
| `w9_delta_d3_dn_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of share of ask pulls that were near |
| `w9_delta_dn_bid_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_delta_d1_dn_bid_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_delta_d2_dn_bid_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_delta_d3_dn_bid_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w9_delta_dn_bid_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of log ratio of bid depth far vs near at window end |
| `w9_delta_d1_dn_bid_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log ratio of bid depth far vs near at window end |
| `w9_delta_d2_dn_bid_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log ratio of bid depth far vs near at window end |
| `w9_delta_d3_dn_bid_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log ratio of bid depth far vs near at window end |
| `w9_delta_dn_bid_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of change in bid near depth share (end - start) |
| `w9_delta_d1_dn_bid_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of first difference of change in bid near depth share (end - start) |
| `w9_delta_d2_dn_bid_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of second difference of change in bid near depth share (end - start) |
| `w9_delta_d3_dn_bid_near_share_delta` | 9-window delta ((x_t - x_{t-8})/8) of third difference of change in bid near depth share (end - start) |
| `w9_delta_dn_bid_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of share of bid reprice-away in resting reprices |
| `w9_delta_d1_dn_bid_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of share of bid reprice-away in resting reprices |
| `w9_delta_d2_dn_bid_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of share of bid reprice-away in resting reprices |
| `w9_delta_d3_dn_bid_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of share of bid reprice-away in resting reprices |
| `w9_delta_dn_bid_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of log pull/add ratio for resting bid qty |
| `w9_delta_d1_dn_bid_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log pull/add ratio for resting bid qty |
| `w9_delta_d2_dn_bid_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log pull/add ratio for resting bid qty |
| `w9_delta_d3_dn_bid_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log pull/add ratio for resting bid qty |
| `w9_delta_dn_bid_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of log1p bid pull qty / start bid depth total |
| `w9_delta_d1_dn_bid_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log1p bid pull qty / start bid depth total |
| `w9_delta_d2_dn_bid_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log1p bid pull qty / start bid depth total |
| `w9_delta_d3_dn_bid_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log1p bid pull qty / start bid depth total |
| `w9_delta_dn_bid_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of share of bid pulls that were near |
| `w9_delta_d1_dn_bid_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of share of bid pulls that were near |
| `w9_delta_d2_dn_bid_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of share of bid pulls that were near |
| `w9_delta_d3_dn_bid_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of share of bid pulls that were near |
| `w9_delta_dn_vacuum_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w9_delta_d1_dn_vacuum_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w9_delta_d2_dn_vacuum_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w9_delta_d3_dn_vacuum_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w9_delta_dn_vacuum_decay_log` | 9-window delta ((x_t - x_{t-8})/8) of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w9_delta_d1_dn_vacuum_decay_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w9_delta_d2_dn_vacuum_decay_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w9_delta_d3_dn_vacuum_decay_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w9_delta_dn_vacuum_total_log` | 9-window delta ((x_t - x_{t-8})/8) of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w9_delta_d1_dn_vacuum_total_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w9_delta_d2_dn_vacuum_total_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w9_delta_d3_dn_vacuum_total_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w9_delta_up_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of copy of dn_ask_com_disp_log |
| `w9_delta_d1_up_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of dn_ask_com_disp_log |
| `w9_delta_d2_up_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of dn_ask_com_disp_log |
| `w9_delta_d3_up_ask_com_disp_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of dn_ask_com_disp_log |
| `w9_delta_up_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of copy of dn_ask_slope_convex_log |
| `w9_delta_d1_up_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of dn_ask_slope_convex_log |
| `w9_delta_d2_up_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of dn_ask_slope_convex_log |
| `w9_delta_d3_up_ask_slope_convex_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of dn_ask_slope_convex_log |
| `w9_delta_up_ask_near_share_decay` | 9-window delta ((x_t - x_{t-8})/8) of negative of dn_ask_near_share_delta |
| `w9_delta_d1_up_ask_near_share_decay` | 9-window delta ((x_t - x_{t-8})/8) of first difference of negative of dn_ask_near_share_delta |
| `w9_delta_d2_up_ask_near_share_decay` | 9-window delta ((x_t - x_{t-8})/8) of second difference of negative of dn_ask_near_share_delta |
| `w9_delta_d3_up_ask_near_share_decay` | 9-window delta ((x_t - x_{t-8})/8) of third difference of negative of dn_ask_near_share_delta |
| `w9_delta_up_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of copy of dn_ask_reprice_away_share_rest |
| `w9_delta_d1_up_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of dn_ask_reprice_away_share_rest |
| `w9_delta_d2_up_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of dn_ask_reprice_away_share_rest |
| `w9_delta_d3_up_ask_reprice_away_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of dn_ask_reprice_away_share_rest |
| `w9_delta_up_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of copy of dn_ask_pull_add_log_rest |
| `w9_delta_d1_up_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of dn_ask_pull_add_log_rest |
| `w9_delta_d2_up_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of dn_ask_pull_add_log_rest |
| `w9_delta_d3_up_ask_pull_add_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of dn_ask_pull_add_log_rest |
| `w9_delta_up_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of copy of dn_ask_pull_intensity_log_rest |
| `w9_delta_d1_up_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of dn_ask_pull_intensity_log_rest |
| `w9_delta_d2_up_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of dn_ask_pull_intensity_log_rest |
| `w9_delta_d3_up_ask_pull_intensity_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of dn_ask_pull_intensity_log_rest |
| `w9_delta_up_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of copy of dn_ask_near_pull_share_rest |
| `w9_delta_d1_up_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of dn_ask_near_pull_share_rest |
| `w9_delta_d2_up_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of dn_ask_near_pull_share_rest |
| `w9_delta_d3_up_ask_near_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of dn_ask_near_pull_share_rest |
| `w9_delta_up_bid_com_approach_log` | 9-window delta ((x_t - x_{t-8})/8) of negative of dn_bid_com_disp_log |
| `w9_delta_d1_up_bid_com_approach_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of negative of dn_bid_com_disp_log |
| `w9_delta_d2_up_bid_com_approach_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of negative of dn_bid_com_disp_log |
| `w9_delta_d3_up_bid_com_approach_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of negative of dn_bid_com_disp_log |
| `w9_delta_up_bid_slope_support_log` | 9-window delta ((x_t - x_{t-8})/8) of negative of dn_bid_slope_convex_log |
| `w9_delta_d1_up_bid_slope_support_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of negative of dn_bid_slope_convex_log |
| `w9_delta_d2_up_bid_slope_support_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of negative of dn_bid_slope_convex_log |
| `w9_delta_d3_up_bid_slope_support_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of negative of dn_bid_slope_convex_log |
| `w9_delta_up_bid_near_share_rise` | 9-window delta ((x_t - x_{t-8})/8) of copy of dn_bid_near_share_delta |
| `w9_delta_d1_up_bid_near_share_rise` | 9-window delta ((x_t - x_{t-8})/8) of first difference of copy of dn_bid_near_share_delta |
| `w9_delta_d2_up_bid_near_share_rise` | 9-window delta ((x_t - x_{t-8})/8) of second difference of copy of dn_bid_near_share_delta |
| `w9_delta_d3_up_bid_near_share_rise` | 9-window delta ((x_t - x_{t-8})/8) of third difference of copy of dn_bid_near_share_delta |
| `w9_delta_up_bid_reprice_toward_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of 1 - dn_bid_reprice_away_share_rest |
| `w9_delta_d1_up_bid_reprice_toward_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of 1 - dn_bid_reprice_away_share_rest |
| `w9_delta_d2_up_bid_reprice_toward_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of 1 - dn_bid_reprice_away_share_rest |
| `w9_delta_d3_up_bid_reprice_toward_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of 1 - dn_bid_reprice_away_share_rest |
| `w9_delta_up_bid_add_pull_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of negative of dn_bid_pull_add_log_rest |
| `w9_delta_d1_up_bid_add_pull_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of negative of dn_bid_pull_add_log_rest |
| `w9_delta_d2_up_bid_add_pull_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of negative of dn_bid_pull_add_log_rest |
| `w9_delta_d3_up_bid_add_pull_log_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of negative of dn_bid_pull_add_log_rest |
| `w9_delta_up_bid_add_intensity_log` | 9-window delta ((x_t - x_{t-8})/8) of log1p bid add qty / start bid depth total |
| `w9_delta_d1_up_bid_add_intensity_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of log1p bid add qty / start bid depth total |
| `w9_delta_d2_up_bid_add_intensity_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of log1p bid add qty / start bid depth total |
| `w9_delta_d3_up_bid_add_intensity_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of log1p bid add qty / start bid depth total |
| `w9_delta_up_bid_far_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of 1 - dn_bid_near_pull_share_rest |
| `w9_delta_d1_up_bid_far_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of first difference of 1 - dn_bid_near_pull_share_rest |
| `w9_delta_d2_up_bid_far_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of second difference of 1 - dn_bid_near_pull_share_rest |
| `w9_delta_d3_up_bid_far_pull_share_rest` | 9-window delta ((x_t - x_{t-8})/8) of third difference of 1 - dn_bid_near_pull_share_rest |
| `w9_delta_up_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of up_ask_com_disp_log + up_bid_com_approach_log |
| `w9_delta_d1_up_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w9_delta_d2_up_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w9_delta_d3_up_expansion_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w9_delta_up_flow_log` | 9-window delta ((x_t - x_{t-8})/8) of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w9_delta_d1_up_flow_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w9_delta_d2_up_flow_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w9_delta_d3_up_flow_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w9_delta_up_total_log` | 9-window delta ((x_t - x_{t-8})/8) of up_expansion_log + up_flow_log |
| `w9_delta_d1_up_total_log` | 9-window delta ((x_t - x_{t-8})/8) of first difference of up_expansion_log + up_flow_log |
| `w9_delta_d2_up_total_log` | 9-window delta ((x_t - x_{t-8})/8) of second difference of up_expansion_log + up_flow_log |
| `w9_delta_d3_up_total_log` | 9-window delta ((x_t - x_{t-8})/8) of third difference of up_expansion_log + up_flow_log |
| `w24_mean_dn_ask_com_disp_log` | 24-window mean (120s) of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_mean_d1_dn_ask_com_disp_log` | 24-window mean (120s) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_mean_d2_dn_ask_com_disp_log` | 24-window mean (120s) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_mean_d3_dn_ask_com_disp_log` | 24-window mean (120s) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_mean_dn_ask_slope_convex_log` | 24-window mean (120s) of log ratio of ask depth far vs near at window end |
| `w24_mean_d1_dn_ask_slope_convex_log` | 24-window mean (120s) of first difference of log ratio of ask depth far vs near at window end |
| `w24_mean_d2_dn_ask_slope_convex_log` | 24-window mean (120s) of second difference of log ratio of ask depth far vs near at window end |
| `w24_mean_d3_dn_ask_slope_convex_log` | 24-window mean (120s) of third difference of log ratio of ask depth far vs near at window end |
| `w24_mean_dn_ask_near_share_delta` | 24-window mean (120s) of change in ask near depth share (end - start) |
| `w24_mean_d1_dn_ask_near_share_delta` | 24-window mean (120s) of first difference of change in ask near depth share (end - start) |
| `w24_mean_d2_dn_ask_near_share_delta` | 24-window mean (120s) of second difference of change in ask near depth share (end - start) |
| `w24_mean_d3_dn_ask_near_share_delta` | 24-window mean (120s) of third difference of change in ask near depth share (end - start) |
| `w24_mean_dn_ask_reprice_away_share_rest` | 24-window mean (120s) of share of ask reprice-away in resting reprices |
| `w24_mean_d1_dn_ask_reprice_away_share_rest` | 24-window mean (120s) of first difference of share of ask reprice-away in resting reprices |
| `w24_mean_d2_dn_ask_reprice_away_share_rest` | 24-window mean (120s) of second difference of share of ask reprice-away in resting reprices |
| `w24_mean_d3_dn_ask_reprice_away_share_rest` | 24-window mean (120s) of third difference of share of ask reprice-away in resting reprices |
| `w24_mean_dn_ask_pull_add_log_rest` | 24-window mean (120s) of log pull/add ratio for resting ask qty |
| `w24_mean_d1_dn_ask_pull_add_log_rest` | 24-window mean (120s) of first difference of log pull/add ratio for resting ask qty |
| `w24_mean_d2_dn_ask_pull_add_log_rest` | 24-window mean (120s) of second difference of log pull/add ratio for resting ask qty |
| `w24_mean_d3_dn_ask_pull_add_log_rest` | 24-window mean (120s) of third difference of log pull/add ratio for resting ask qty |
| `w24_mean_dn_ask_pull_intensity_log_rest` | 24-window mean (120s) of log1p ask pull qty / start ask depth total |
| `w24_mean_d1_dn_ask_pull_intensity_log_rest` | 24-window mean (120s) of first difference of log1p ask pull qty / start ask depth total |
| `w24_mean_d2_dn_ask_pull_intensity_log_rest` | 24-window mean (120s) of second difference of log1p ask pull qty / start ask depth total |
| `w24_mean_d3_dn_ask_pull_intensity_log_rest` | 24-window mean (120s) of third difference of log1p ask pull qty / start ask depth total |
| `w24_mean_dn_ask_near_pull_share_rest` | 24-window mean (120s) of share of ask pulls that were near |
| `w24_mean_d1_dn_ask_near_pull_share_rest` | 24-window mean (120s) of first difference of share of ask pulls that were near |
| `w24_mean_d2_dn_ask_near_pull_share_rest` | 24-window mean (120s) of second difference of share of ask pulls that were near |
| `w24_mean_d3_dn_ask_near_pull_share_rest` | 24-window mean (120s) of third difference of share of ask pulls that were near |
| `w24_mean_dn_bid_com_disp_log` | 24-window mean (120s) of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_mean_d1_dn_bid_com_disp_log` | 24-window mean (120s) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_mean_d2_dn_bid_com_disp_log` | 24-window mean (120s) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_mean_d3_dn_bid_com_disp_log` | 24-window mean (120s) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_mean_dn_bid_slope_convex_log` | 24-window mean (120s) of log ratio of bid depth far vs near at window end |
| `w24_mean_d1_dn_bid_slope_convex_log` | 24-window mean (120s) of first difference of log ratio of bid depth far vs near at window end |
| `w24_mean_d2_dn_bid_slope_convex_log` | 24-window mean (120s) of second difference of log ratio of bid depth far vs near at window end |
| `w24_mean_d3_dn_bid_slope_convex_log` | 24-window mean (120s) of third difference of log ratio of bid depth far vs near at window end |
| `w24_mean_dn_bid_near_share_delta` | 24-window mean (120s) of change in bid near depth share (end - start) |
| `w24_mean_d1_dn_bid_near_share_delta` | 24-window mean (120s) of first difference of change in bid near depth share (end - start) |
| `w24_mean_d2_dn_bid_near_share_delta` | 24-window mean (120s) of second difference of change in bid near depth share (end - start) |
| `w24_mean_d3_dn_bid_near_share_delta` | 24-window mean (120s) of third difference of change in bid near depth share (end - start) |
| `w24_mean_dn_bid_reprice_away_share_rest` | 24-window mean (120s) of share of bid reprice-away in resting reprices |
| `w24_mean_d1_dn_bid_reprice_away_share_rest` | 24-window mean (120s) of first difference of share of bid reprice-away in resting reprices |
| `w24_mean_d2_dn_bid_reprice_away_share_rest` | 24-window mean (120s) of second difference of share of bid reprice-away in resting reprices |
| `w24_mean_d3_dn_bid_reprice_away_share_rest` | 24-window mean (120s) of third difference of share of bid reprice-away in resting reprices |
| `w24_mean_dn_bid_pull_add_log_rest` | 24-window mean (120s) of log pull/add ratio for resting bid qty |
| `w24_mean_d1_dn_bid_pull_add_log_rest` | 24-window mean (120s) of first difference of log pull/add ratio for resting bid qty |
| `w24_mean_d2_dn_bid_pull_add_log_rest` | 24-window mean (120s) of second difference of log pull/add ratio for resting bid qty |
| `w24_mean_d3_dn_bid_pull_add_log_rest` | 24-window mean (120s) of third difference of log pull/add ratio for resting bid qty |
| `w24_mean_dn_bid_pull_intensity_log_rest` | 24-window mean (120s) of log1p bid pull qty / start bid depth total |
| `w24_mean_d1_dn_bid_pull_intensity_log_rest` | 24-window mean (120s) of first difference of log1p bid pull qty / start bid depth total |
| `w24_mean_d2_dn_bid_pull_intensity_log_rest` | 24-window mean (120s) of second difference of log1p bid pull qty / start bid depth total |
| `w24_mean_d3_dn_bid_pull_intensity_log_rest` | 24-window mean (120s) of third difference of log1p bid pull qty / start bid depth total |
| `w24_mean_dn_bid_near_pull_share_rest` | 24-window mean (120s) of share of bid pulls that were near |
| `w24_mean_d1_dn_bid_near_pull_share_rest` | 24-window mean (120s) of first difference of share of bid pulls that were near |
| `w24_mean_d2_dn_bid_near_pull_share_rest` | 24-window mean (120s) of second difference of share of bid pulls that were near |
| `w24_mean_d3_dn_bid_near_pull_share_rest` | 24-window mean (120s) of third difference of share of bid pulls that were near |
| `w24_mean_dn_vacuum_expansion_log` | 24-window mean (120s) of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w24_mean_d1_dn_vacuum_expansion_log` | 24-window mean (120s) of first difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w24_mean_d2_dn_vacuum_expansion_log` | 24-window mean (120s) of second difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w24_mean_d3_dn_vacuum_expansion_log` | 24-window mean (120s) of third difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w24_mean_dn_vacuum_decay_log` | 24-window mean (120s) of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w24_mean_d1_dn_vacuum_decay_log` | 24-window mean (120s) of first difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w24_mean_d2_dn_vacuum_decay_log` | 24-window mean (120s) of second difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w24_mean_d3_dn_vacuum_decay_log` | 24-window mean (120s) of third difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w24_mean_dn_vacuum_total_log` | 24-window mean (120s) of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w24_mean_d1_dn_vacuum_total_log` | 24-window mean (120s) of first difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w24_mean_d2_dn_vacuum_total_log` | 24-window mean (120s) of second difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w24_mean_d3_dn_vacuum_total_log` | 24-window mean (120s) of third difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w24_mean_up_ask_com_disp_log` | 24-window mean (120s) of copy of dn_ask_com_disp_log |
| `w24_mean_d1_up_ask_com_disp_log` | 24-window mean (120s) of first difference of copy of dn_ask_com_disp_log |
| `w24_mean_d2_up_ask_com_disp_log` | 24-window mean (120s) of second difference of copy of dn_ask_com_disp_log |
| `w24_mean_d3_up_ask_com_disp_log` | 24-window mean (120s) of third difference of copy of dn_ask_com_disp_log |
| `w24_mean_up_ask_slope_convex_log` | 24-window mean (120s) of copy of dn_ask_slope_convex_log |
| `w24_mean_d1_up_ask_slope_convex_log` | 24-window mean (120s) of first difference of copy of dn_ask_slope_convex_log |
| `w24_mean_d2_up_ask_slope_convex_log` | 24-window mean (120s) of second difference of copy of dn_ask_slope_convex_log |
| `w24_mean_d3_up_ask_slope_convex_log` | 24-window mean (120s) of third difference of copy of dn_ask_slope_convex_log |
| `w24_mean_up_ask_near_share_decay` | 24-window mean (120s) of negative of dn_ask_near_share_delta |
| `w24_mean_d1_up_ask_near_share_decay` | 24-window mean (120s) of first difference of negative of dn_ask_near_share_delta |
| `w24_mean_d2_up_ask_near_share_decay` | 24-window mean (120s) of second difference of negative of dn_ask_near_share_delta |
| `w24_mean_d3_up_ask_near_share_decay` | 24-window mean (120s) of third difference of negative of dn_ask_near_share_delta |
| `w24_mean_up_ask_reprice_away_share_rest` | 24-window mean (120s) of copy of dn_ask_reprice_away_share_rest |
| `w24_mean_d1_up_ask_reprice_away_share_rest` | 24-window mean (120s) of first difference of copy of dn_ask_reprice_away_share_rest |
| `w24_mean_d2_up_ask_reprice_away_share_rest` | 24-window mean (120s) of second difference of copy of dn_ask_reprice_away_share_rest |
| `w24_mean_d3_up_ask_reprice_away_share_rest` | 24-window mean (120s) of third difference of copy of dn_ask_reprice_away_share_rest |
| `w24_mean_up_ask_pull_add_log_rest` | 24-window mean (120s) of copy of dn_ask_pull_add_log_rest |
| `w24_mean_d1_up_ask_pull_add_log_rest` | 24-window mean (120s) of first difference of copy of dn_ask_pull_add_log_rest |
| `w24_mean_d2_up_ask_pull_add_log_rest` | 24-window mean (120s) of second difference of copy of dn_ask_pull_add_log_rest |
| `w24_mean_d3_up_ask_pull_add_log_rest` | 24-window mean (120s) of third difference of copy of dn_ask_pull_add_log_rest |
| `w24_mean_up_ask_pull_intensity_log_rest` | 24-window mean (120s) of copy of dn_ask_pull_intensity_log_rest |
| `w24_mean_d1_up_ask_pull_intensity_log_rest` | 24-window mean (120s) of first difference of copy of dn_ask_pull_intensity_log_rest |
| `w24_mean_d2_up_ask_pull_intensity_log_rest` | 24-window mean (120s) of second difference of copy of dn_ask_pull_intensity_log_rest |
| `w24_mean_d3_up_ask_pull_intensity_log_rest` | 24-window mean (120s) of third difference of copy of dn_ask_pull_intensity_log_rest |
| `w24_mean_up_ask_near_pull_share_rest` | 24-window mean (120s) of copy of dn_ask_near_pull_share_rest |
| `w24_mean_d1_up_ask_near_pull_share_rest` | 24-window mean (120s) of first difference of copy of dn_ask_near_pull_share_rest |
| `w24_mean_d2_up_ask_near_pull_share_rest` | 24-window mean (120s) of second difference of copy of dn_ask_near_pull_share_rest |
| `w24_mean_d3_up_ask_near_pull_share_rest` | 24-window mean (120s) of third difference of copy of dn_ask_near_pull_share_rest |
| `w24_mean_up_bid_com_approach_log` | 24-window mean (120s) of negative of dn_bid_com_disp_log |
| `w24_mean_d1_up_bid_com_approach_log` | 24-window mean (120s) of first difference of negative of dn_bid_com_disp_log |
| `w24_mean_d2_up_bid_com_approach_log` | 24-window mean (120s) of second difference of negative of dn_bid_com_disp_log |
| `w24_mean_d3_up_bid_com_approach_log` | 24-window mean (120s) of third difference of negative of dn_bid_com_disp_log |
| `w24_mean_up_bid_slope_support_log` | 24-window mean (120s) of negative of dn_bid_slope_convex_log |
| `w24_mean_d1_up_bid_slope_support_log` | 24-window mean (120s) of first difference of negative of dn_bid_slope_convex_log |
| `w24_mean_d2_up_bid_slope_support_log` | 24-window mean (120s) of second difference of negative of dn_bid_slope_convex_log |
| `w24_mean_d3_up_bid_slope_support_log` | 24-window mean (120s) of third difference of negative of dn_bid_slope_convex_log |
| `w24_mean_up_bid_near_share_rise` | 24-window mean (120s) of copy of dn_bid_near_share_delta |
| `w24_mean_d1_up_bid_near_share_rise` | 24-window mean (120s) of first difference of copy of dn_bid_near_share_delta |
| `w24_mean_d2_up_bid_near_share_rise` | 24-window mean (120s) of second difference of copy of dn_bid_near_share_delta |
| `w24_mean_d3_up_bid_near_share_rise` | 24-window mean (120s) of third difference of copy of dn_bid_near_share_delta |
| `w24_mean_up_bid_reprice_toward_share_rest` | 24-window mean (120s) of 1 - dn_bid_reprice_away_share_rest |
| `w24_mean_d1_up_bid_reprice_toward_share_rest` | 24-window mean (120s) of first difference of 1 - dn_bid_reprice_away_share_rest |
| `w24_mean_d2_up_bid_reprice_toward_share_rest` | 24-window mean (120s) of second difference of 1 - dn_bid_reprice_away_share_rest |
| `w24_mean_d3_up_bid_reprice_toward_share_rest` | 24-window mean (120s) of third difference of 1 - dn_bid_reprice_away_share_rest |
| `w24_mean_up_bid_add_pull_log_rest` | 24-window mean (120s) of negative of dn_bid_pull_add_log_rest |
| `w24_mean_d1_up_bid_add_pull_log_rest` | 24-window mean (120s) of first difference of negative of dn_bid_pull_add_log_rest |
| `w24_mean_d2_up_bid_add_pull_log_rest` | 24-window mean (120s) of second difference of negative of dn_bid_pull_add_log_rest |
| `w24_mean_d3_up_bid_add_pull_log_rest` | 24-window mean (120s) of third difference of negative of dn_bid_pull_add_log_rest |
| `w24_mean_up_bid_add_intensity_log` | 24-window mean (120s) of log1p bid add qty / start bid depth total |
| `w24_mean_d1_up_bid_add_intensity_log` | 24-window mean (120s) of first difference of log1p bid add qty / start bid depth total |
| `w24_mean_d2_up_bid_add_intensity_log` | 24-window mean (120s) of second difference of log1p bid add qty / start bid depth total |
| `w24_mean_d3_up_bid_add_intensity_log` | 24-window mean (120s) of third difference of log1p bid add qty / start bid depth total |
| `w24_mean_up_bid_far_pull_share_rest` | 24-window mean (120s) of 1 - dn_bid_near_pull_share_rest |
| `w24_mean_d1_up_bid_far_pull_share_rest` | 24-window mean (120s) of first difference of 1 - dn_bid_near_pull_share_rest |
| `w24_mean_d2_up_bid_far_pull_share_rest` | 24-window mean (120s) of second difference of 1 - dn_bid_near_pull_share_rest |
| `w24_mean_d3_up_bid_far_pull_share_rest` | 24-window mean (120s) of third difference of 1 - dn_bid_near_pull_share_rest |
| `w24_mean_up_expansion_log` | 24-window mean (120s) of up_ask_com_disp_log + up_bid_com_approach_log |
| `w24_mean_d1_up_expansion_log` | 24-window mean (120s) of first difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w24_mean_d2_up_expansion_log` | 24-window mean (120s) of second difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w24_mean_d3_up_expansion_log` | 24-window mean (120s) of third difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w24_mean_up_flow_log` | 24-window mean (120s) of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w24_mean_d1_up_flow_log` | 24-window mean (120s) of first difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w24_mean_d2_up_flow_log` | 24-window mean (120s) of second difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w24_mean_d3_up_flow_log` | 24-window mean (120s) of third difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w24_mean_up_total_log` | 24-window mean (120s) of up_expansion_log + up_flow_log |
| `w24_mean_d1_up_total_log` | 24-window mean (120s) of first difference of up_expansion_log + up_flow_log |
| `w24_mean_d2_up_total_log` | 24-window mean (120s) of second difference of up_expansion_log + up_flow_log |
| `w24_mean_d3_up_total_log` | 24-window mean (120s) of third difference of up_expansion_log + up_flow_log |
| `w24_delta_dn_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_delta_d1_dn_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_delta_d2_dn_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_delta_d3_dn_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log ratio of ask COM distance from level (ticks), end vs start |
| `w24_delta_dn_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of log ratio of ask depth far vs near at window end |
| `w24_delta_d1_dn_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log ratio of ask depth far vs near at window end |
| `w24_delta_d2_dn_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log ratio of ask depth far vs near at window end |
| `w24_delta_d3_dn_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log ratio of ask depth far vs near at window end |
| `w24_delta_dn_ask_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of change in ask near depth share (end - start) |
| `w24_delta_d1_dn_ask_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of first difference of change in ask near depth share (end - start) |
| `w24_delta_d2_dn_ask_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of second difference of change in ask near depth share (end - start) |
| `w24_delta_d3_dn_ask_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of third difference of change in ask near depth share (end - start) |
| `w24_delta_dn_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of share of ask reprice-away in resting reprices |
| `w24_delta_d1_dn_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of share of ask reprice-away in resting reprices |
| `w24_delta_d2_dn_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of share of ask reprice-away in resting reprices |
| `w24_delta_d3_dn_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of share of ask reprice-away in resting reprices |
| `w24_delta_dn_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of log pull/add ratio for resting ask qty |
| `w24_delta_d1_dn_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log pull/add ratio for resting ask qty |
| `w24_delta_d2_dn_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log pull/add ratio for resting ask qty |
| `w24_delta_d3_dn_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log pull/add ratio for resting ask qty |
| `w24_delta_dn_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of log1p ask pull qty / start ask depth total |
| `w24_delta_d1_dn_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log1p ask pull qty / start ask depth total |
| `w24_delta_d2_dn_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log1p ask pull qty / start ask depth total |
| `w24_delta_d3_dn_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log1p ask pull qty / start ask depth total |
| `w24_delta_dn_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of share of ask pulls that were near |
| `w24_delta_d1_dn_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of share of ask pulls that were near |
| `w24_delta_d2_dn_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of share of ask pulls that were near |
| `w24_delta_d3_dn_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of share of ask pulls that were near |
| `w24_delta_dn_bid_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_delta_d1_dn_bid_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_delta_d2_dn_bid_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_delta_d3_dn_bid_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log ratio of bid COM distance from level (ticks), end vs start |
| `w24_delta_dn_bid_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of log ratio of bid depth far vs near at window end |
| `w24_delta_d1_dn_bid_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log ratio of bid depth far vs near at window end |
| `w24_delta_d2_dn_bid_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log ratio of bid depth far vs near at window end |
| `w24_delta_d3_dn_bid_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log ratio of bid depth far vs near at window end |
| `w24_delta_dn_bid_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of change in bid near depth share (end - start) |
| `w24_delta_d1_dn_bid_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of first difference of change in bid near depth share (end - start) |
| `w24_delta_d2_dn_bid_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of second difference of change in bid near depth share (end - start) |
| `w24_delta_d3_dn_bid_near_share_delta` | 24-window delta ((x_t - x_{t-23})/23) of third difference of change in bid near depth share (end - start) |
| `w24_delta_dn_bid_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of share of bid reprice-away in resting reprices |
| `w24_delta_d1_dn_bid_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of share of bid reprice-away in resting reprices |
| `w24_delta_d2_dn_bid_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of share of bid reprice-away in resting reprices |
| `w24_delta_d3_dn_bid_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of share of bid reprice-away in resting reprices |
| `w24_delta_dn_bid_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of log pull/add ratio for resting bid qty |
| `w24_delta_d1_dn_bid_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log pull/add ratio for resting bid qty |
| `w24_delta_d2_dn_bid_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log pull/add ratio for resting bid qty |
| `w24_delta_d3_dn_bid_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log pull/add ratio for resting bid qty |
| `w24_delta_dn_bid_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of log1p bid pull qty / start bid depth total |
| `w24_delta_d1_dn_bid_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log1p bid pull qty / start bid depth total |
| `w24_delta_d2_dn_bid_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log1p bid pull qty / start bid depth total |
| `w24_delta_d3_dn_bid_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log1p bid pull qty / start bid depth total |
| `w24_delta_dn_bid_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of share of bid pulls that were near |
| `w24_delta_d1_dn_bid_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of share of bid pulls that were near |
| `w24_delta_d2_dn_bid_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of share of bid pulls that were near |
| `w24_delta_d3_dn_bid_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of share of bid pulls that were near |
| `w24_delta_dn_vacuum_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w24_delta_d1_dn_vacuum_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w24_delta_d2_dn_vacuum_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w24_delta_d3_dn_vacuum_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of dn_ask_com_disp_log + dn_bid_com_disp_log |
| `w24_delta_dn_vacuum_decay_log` | 24-window delta ((x_t - x_{t-23})/23) of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w24_delta_d1_dn_vacuum_decay_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w24_delta_d2_dn_vacuum_decay_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w24_delta_d3_dn_vacuum_decay_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of dn_ask_pull_add_log_rest + dn_bid_pull_add_log_rest |
| `w24_delta_dn_vacuum_total_log` | 24-window delta ((x_t - x_{t-23})/23) of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w24_delta_d1_dn_vacuum_total_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w24_delta_d2_dn_vacuum_total_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w24_delta_d3_dn_vacuum_total_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of dn_vacuum_expansion_log + dn_vacuum_decay_log |
| `w24_delta_up_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of copy of dn_ask_com_disp_log |
| `w24_delta_d1_up_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of dn_ask_com_disp_log |
| `w24_delta_d2_up_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of dn_ask_com_disp_log |
| `w24_delta_d3_up_ask_com_disp_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of dn_ask_com_disp_log |
| `w24_delta_up_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of copy of dn_ask_slope_convex_log |
| `w24_delta_d1_up_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of dn_ask_slope_convex_log |
| `w24_delta_d2_up_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of dn_ask_slope_convex_log |
| `w24_delta_d3_up_ask_slope_convex_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of dn_ask_slope_convex_log |
| `w24_delta_up_ask_near_share_decay` | 24-window delta ((x_t - x_{t-23})/23) of negative of dn_ask_near_share_delta |
| `w24_delta_d1_up_ask_near_share_decay` | 24-window delta ((x_t - x_{t-23})/23) of first difference of negative of dn_ask_near_share_delta |
| `w24_delta_d2_up_ask_near_share_decay` | 24-window delta ((x_t - x_{t-23})/23) of second difference of negative of dn_ask_near_share_delta |
| `w24_delta_d3_up_ask_near_share_decay` | 24-window delta ((x_t - x_{t-23})/23) of third difference of negative of dn_ask_near_share_delta |
| `w24_delta_up_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of copy of dn_ask_reprice_away_share_rest |
| `w24_delta_d1_up_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of dn_ask_reprice_away_share_rest |
| `w24_delta_d2_up_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of dn_ask_reprice_away_share_rest |
| `w24_delta_d3_up_ask_reprice_away_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of dn_ask_reprice_away_share_rest |
| `w24_delta_up_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of copy of dn_ask_pull_add_log_rest |
| `w24_delta_d1_up_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of dn_ask_pull_add_log_rest |
| `w24_delta_d2_up_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of dn_ask_pull_add_log_rest |
| `w24_delta_d3_up_ask_pull_add_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of dn_ask_pull_add_log_rest |
| `w24_delta_up_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of copy of dn_ask_pull_intensity_log_rest |
| `w24_delta_d1_up_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of dn_ask_pull_intensity_log_rest |
| `w24_delta_d2_up_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of dn_ask_pull_intensity_log_rest |
| `w24_delta_d3_up_ask_pull_intensity_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of dn_ask_pull_intensity_log_rest |
| `w24_delta_up_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of copy of dn_ask_near_pull_share_rest |
| `w24_delta_d1_up_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of dn_ask_near_pull_share_rest |
| `w24_delta_d2_up_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of dn_ask_near_pull_share_rest |
| `w24_delta_d3_up_ask_near_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of dn_ask_near_pull_share_rest |
| `w24_delta_up_bid_com_approach_log` | 24-window delta ((x_t - x_{t-23})/23) of negative of dn_bid_com_disp_log |
| `w24_delta_d1_up_bid_com_approach_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of negative of dn_bid_com_disp_log |
| `w24_delta_d2_up_bid_com_approach_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of negative of dn_bid_com_disp_log |
| `w24_delta_d3_up_bid_com_approach_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of negative of dn_bid_com_disp_log |
| `w24_delta_up_bid_slope_support_log` | 24-window delta ((x_t - x_{t-23})/23) of negative of dn_bid_slope_convex_log |
| `w24_delta_d1_up_bid_slope_support_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of negative of dn_bid_slope_convex_log |
| `w24_delta_d2_up_bid_slope_support_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of negative of dn_bid_slope_convex_log |
| `w24_delta_d3_up_bid_slope_support_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of negative of dn_bid_slope_convex_log |
| `w24_delta_up_bid_near_share_rise` | 24-window delta ((x_t - x_{t-23})/23) of copy of dn_bid_near_share_delta |
| `w24_delta_d1_up_bid_near_share_rise` | 24-window delta ((x_t - x_{t-23})/23) of first difference of copy of dn_bid_near_share_delta |
| `w24_delta_d2_up_bid_near_share_rise` | 24-window delta ((x_t - x_{t-23})/23) of second difference of copy of dn_bid_near_share_delta |
| `w24_delta_d3_up_bid_near_share_rise` | 24-window delta ((x_t - x_{t-23})/23) of third difference of copy of dn_bid_near_share_delta |
| `w24_delta_up_bid_reprice_toward_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of 1 - dn_bid_reprice_away_share_rest |
| `w24_delta_d1_up_bid_reprice_toward_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of 1 - dn_bid_reprice_away_share_rest |
| `w24_delta_d2_up_bid_reprice_toward_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of 1 - dn_bid_reprice_away_share_rest |
| `w24_delta_d3_up_bid_reprice_toward_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of 1 - dn_bid_reprice_away_share_rest |
| `w24_delta_up_bid_add_pull_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of negative of dn_bid_pull_add_log_rest |
| `w24_delta_d1_up_bid_add_pull_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of negative of dn_bid_pull_add_log_rest |
| `w24_delta_d2_up_bid_add_pull_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of negative of dn_bid_pull_add_log_rest |
| `w24_delta_d3_up_bid_add_pull_log_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of negative of dn_bid_pull_add_log_rest |
| `w24_delta_up_bid_add_intensity_log` | 24-window delta ((x_t - x_{t-23})/23) of log1p bid add qty / start bid depth total |
| `w24_delta_d1_up_bid_add_intensity_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of log1p bid add qty / start bid depth total |
| `w24_delta_d2_up_bid_add_intensity_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of log1p bid add qty / start bid depth total |
| `w24_delta_d3_up_bid_add_intensity_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of log1p bid add qty / start bid depth total |
| `w24_delta_up_bid_far_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of 1 - dn_bid_near_pull_share_rest |
| `w24_delta_d1_up_bid_far_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of first difference of 1 - dn_bid_near_pull_share_rest |
| `w24_delta_d2_up_bid_far_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of second difference of 1 - dn_bid_near_pull_share_rest |
| `w24_delta_d3_up_bid_far_pull_share_rest` | 24-window delta ((x_t - x_{t-23})/23) of third difference of 1 - dn_bid_near_pull_share_rest |
| `w24_delta_up_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of up_ask_com_disp_log + up_bid_com_approach_log |
| `w24_delta_d1_up_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w24_delta_d2_up_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w24_delta_d3_up_expansion_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of up_ask_com_disp_log + up_bid_com_approach_log |
| `w24_delta_up_flow_log` | 24-window delta ((x_t - x_{t-23})/23) of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w24_delta_d1_up_flow_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w24_delta_d2_up_flow_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w24_delta_d3_up_flow_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of up_ask_pull_add_log_rest + up_bid_add_pull_log_rest |
| `w24_delta_up_total_log` | 24-window delta ((x_t - x_{t-23})/23) of up_expansion_log + up_flow_log |
| `w24_delta_d1_up_total_log` | 24-window delta ((x_t - x_{t-23})/23) of first difference of up_expansion_log + up_flow_log |
| `w24_delta_d2_up_total_log` | 24-window delta ((x_t - x_{t-23})/23) of second difference of up_expansion_log + up_flow_log |
| `w24_delta_d3_up_total_log` | 24-window delta ((x_t - x_{t-23})/23) of third difference of up_expansion_log + up_flow_log |