# Silver Layer Schema

## ES Pipeline Features (v3.0.0 - December 2025)
**Path**: `silver/features/es_pipeline/version={canonical_version}/date=YYYY-MM-DD/*.parquet`

**⚠️ IMPORTANT SCHEMA UPDATE (v3.0.0)**:
- **Outcome Labels Changed**: Now using **REJECT** instead of **BOUNCE**
- **Semantics**: First-crossing with 1.0 ATR threshold
  - `BREAK`: Price crossed 1 ATR in direction of approach
  - `REJECT`: Price reversed 1 ATR in opposite direction  
  - `CHOP`: Neither threshold crossed within horizon
- **New Fields**: `excursion_favorable` and `excursion_adverse` (ATR-normalized)
- **Multi-Horizon**: outcome_2min, outcome_4min, outcome_8min (independent labels)

| Field | Type | Description |
|-------|------|-------------|
| event_id | string | Deterministic interaction event ID |
| ts_ns | int64 | Event timestamp (nanoseconds UTC) |
| timestamp | timestamp[ns] | Event timestamp (datetime UTC) |
| level_price | double | Tested level price |
| level_kind | int8 | Level kind code |
| level_kind_name | string | Level kind name |
| direction | string | Approach direction (UP/DOWN) |
| entry_price | double | Price at zone entry |
| zone_width | double | Interaction zone width (points) |
| date | string | Session date (YYYY-MM-DD) |
| barrier_state | string | Barrier state classification |
| barrier_delta_liq | double | Net liquidity change at barrier |
| barrier_replenishment_ratio | double | Barrier replenishment ratio |
| wall_ratio | double | Depth ratio in barrier zone |
| tape_imbalance | double | Tape buy-sell imbalance |
| tape_buy_vol | int64 | Tape buy volume |
| tape_sell_vol | int64 | Tape sell volume |
| tape_velocity | double | Tape trade velocity |
| sweep_detected | bool | Sweep detected flag |
| fuel_effect | string | Gamma fuel effect (AMPLIFY/DAMPEN/NEUTRAL) |
| gamma_exposure | double | Net dealer gamma exposure |
| velocity_1min | double | Level-frame velocity over 1min window |
| acceleration_1min | double | Level-frame acceleration over 1min window |
| jerk_1min | double | Level-frame jerk over 1min window |
| velocity_3min | double | Level-frame velocity over 3min window |
| acceleration_3min | double | Level-frame acceleration over 3min window |
| jerk_3min | double | Level-frame jerk over 3min window |
| momentum_trend_3min | double | Momentum trend proxy over 3min window |
| velocity_5min | double | Level-frame velocity over 5min window |
| acceleration_5min | double | Level-frame acceleration over 5min window |
| jerk_5min | double | Level-frame jerk over 5min window |
| momentum_trend_5min | double | Momentum trend proxy over 5min window |
| velocity_10min | double | Level-frame velocity over 10min window |
| acceleration_10min | double | Level-frame acceleration over 10min window |
| jerk_10min | double | Level-frame jerk over 10min window |
| momentum_trend_10min | double | Momentum trend proxy over 10min window |
| velocity_20min | double | Level-frame velocity over 20min window |
| acceleration_20min | double | Level-frame acceleration over 20min window |
| jerk_20min | double | Level-frame jerk over 20min window |
| momentum_trend_20min | double | Momentum trend proxy over 20min window |
| ofi_30s | double | Integrated OFI over 30s window |
| ofi_near_level_30s | double | Mean OFI over 30s window |
| ofi_60s | double | Integrated OFI over 60s window |
| ofi_near_level_60s | double | Mean OFI over 60s window |
| ofi_120s | double | Integrated OFI over 120s window |
| ofi_near_level_120s | double | Mean OFI over 120s window |
| ofi_300s | double | Integrated OFI over 300s window |
| ofi_near_level_300s | double | Mean OFI over 300s window |
| ofi_acceleration | double | OFI acceleration ratio (30s vs 120s) |
| barrier_delta_1min | double | Barrier depth change over 1min window |
| barrier_pct_change_1min | double | Barrier depth percent change over 1min window |
| barrier_delta_3min | double | Barrier depth change over 3min window |
| barrier_pct_change_3min | double | Barrier depth percent change over 3min window |
| barrier_delta_5min | double | Barrier depth change over 5min window |
| barrier_pct_change_5min | double | Barrier depth percent change over 5min window |
| barrier_depth_current | double | Current barrier depth |
| dist_to_pm_high | double | Signed distance (spot - PM_HIGH) |
| dist_to_pm_high_atr | double | Signed distance (spot - PM_HIGH) normalized by ATR |
| dist_to_pm_low | double | Signed distance (spot - PM_LOW) |
| dist_to_pm_low_atr | double | Signed distance (spot - PM_LOW) normalized by ATR |
| dist_to_or_high | double | Signed distance (spot - OR_HIGH) |
| dist_to_or_high_atr | double | Signed distance (spot - OR_HIGH) normalized by ATR |
| dist_to_or_low | double | Signed distance (spot - OR_LOW) |
| dist_to_or_low_atr | double | Signed distance (spot - OR_LOW) normalized by ATR |
| dist_to_sma_200 | double | Signed distance (spot - SMA_200) |
| dist_to_sma_200_atr | double | Signed distance (spot - SMA_200) normalized by ATR |
| dist_to_sma_400 | double | Signed distance (spot - SMA_400) |
| dist_to_sma_400_atr | double | Signed distance (spot - SMA_400) normalized by ATR |
| dist_to_tested_level | double | Distance to tested level (points) |
| level_stacking_2pt | int8 | Count of levels within 2 points |
| level_stacking_5pt | int8 | Count of levels within 5 points |
| level_stacking_10pt | int8 | Count of levels within 10 points |
| gex_above_1strike | double | Dealer GEX above level within +1 strike band |
| gex_below_1strike | double | Dealer GEX below level within -1 strike band |
| call_gex_above_1strike | double | Call dealer GEX above level within +1 strike band |
| put_gex_below_1strike | double | Put dealer GEX below level within -1 strike band |
| gex_above_2strike | double | Dealer GEX above level within +2 strike band |
| gex_below_2strike | double | Dealer GEX below level within -2 strike band |
| call_gex_above_2strike | double | Call dealer GEX above level within +2 strike band |
| put_gex_below_2strike | double | Put dealer GEX below level within -2 strike band |
| gex_above_3strike | double | Dealer GEX above level within +3 strike band |
| gex_below_3strike | double | Dealer GEX below level within -3 strike band |
| call_gex_above_3strike | double | Call dealer GEX above level within +3 strike band |
| put_gex_below_3strike | double | Put dealer GEX below level within -3 strike band |
| gex_asymmetry | double | GEX above minus below (max band) |
| gex_ratio | double | GEX ratio above/below (max band) |
| net_gex_2strike | double | Net GEX within ±max band |
| predicted_accel | double | Predicted acceleration from F=ma proxy |
| accel_residual | double | Actual minus predicted acceleration |
| force_mass_ratio | double | Force-to-mass ratio |
| atr | double | ATR value at event |
| approach_velocity | double | Approach velocity toward level |
| approach_bars | int32 | Consecutive bars moving toward level |
| approach_distance | double | Approach distance toward level |
| prior_touches | int32 | Prior touches at level (RTH) |
| minutes_since_open | double | Minutes since RTH open |
| bars_since_open | int32 | Bars since RTH open (1-min bars) |
| wall_ratio_nonzero | int8 | Indicator wall_ratio != 0 |
| wall_ratio_log | double | Signed log transform of wall_ratio |
| barrier_delta_liq_nonzero | int8 | Indicator barrier_delta_liq != 0 |
| barrier_delta_liq_log | double | Signed log transform of barrier_delta_liq |
| spot | double | Spot price at event |
| distance_signed | double | Signed level distance (level - spot) |
| distance_signed_atr | double | distance_signed normalized by ATR |
| distance_signed_pct | double | distance_signed normalized by spot |
| dist_to_pm_high_pct | double | dist_to_pm_high normalized by spot |
| dist_to_pm_low_pct | double | dist_to_pm_low normalized by spot |
| dist_to_sma_200_pct | double | dist_to_sma_200 normalized by spot |
| dist_to_sma_400_pct | double | dist_to_sma_400 normalized by spot |
| approach_distance_atr | double | approach_distance normalized by ATR |
| approach_distance_pct | double | approach_distance normalized by spot |
| level_price_pct | double | Level price normalized by spot |
| attempt_index | int64 | Attempt index within touch cluster |
| attempt_cluster_id | int64 | Attempt cluster ID |
| barrier_replenishment_trend | double | Trend in barrier_replenishment_ratio within cluster |
| barrier_delta_liq_trend | double | Trend in barrier_delta_liq within cluster |
| tape_velocity_trend | double | Trend in tape_velocity within cluster |
| tape_imbalance_trend | double | Trend in tape_imbalance within cluster |
| outcome_2min | string | Outcome label (2min window) |
| excursion_max_2min | double | Max favorable excursion (2min window) |
| excursion_min_2min | double | Max adverse excursion (2min window) |
| strength_signed_2min | double | Signed strength (2min window) |
| strength_abs_2min | double | Absolute strength (2min window) |
| time_to_threshold_1_2min | double | Time to threshold_1 (sec) (2min window) Nullable. |
| time_to_threshold_2_2min | double | Time to threshold_2 (sec) (2min window) Nullable. |
| time_to_break_1_2min | double | Time to break threshold_1 (sec) (2min window) Nullable. |
| time_to_break_2_2min | double | Time to break threshold_2 (sec) (2min window) Nullable. |
| time_to_bounce_1_2min | double | Time to bounce threshold_1 (sec) (2min window) Nullable. |
| time_to_bounce_2_2min | double | Time to bounce threshold_2 (sec) (2min window) Nullable. |
| tradeable_1_2min | int8 | Tradeable flag threshold_1 (2min window) |
| tradeable_2_2min | int8 | Tradeable flag threshold_2 (2min window) |
| confirm_ts_ns_2min | int64 | Touch confirmation timestamp (ns) (2min window) Nullable. |
| anchor_spot_2min | double | Spot at confirmation (2min window) |
| future_price_2min | double | Last price in horizon (2min window) |
| outcome_4min | string | Outcome label (4min window) |
| excursion_max_4min | double | Max favorable excursion (4min window) |
| excursion_min_4min | double | Max adverse excursion (4min window) |
| strength_signed_4min | double | Signed strength (4min window) |
| strength_abs_4min | double | Absolute strength (4min window) |
| time_to_threshold_1_4min | double | Time to threshold_1 (sec) (4min window) Nullable. |
| time_to_threshold_2_4min | double | Time to threshold_2 (sec) (4min window) Nullable. |
| time_to_break_1_4min | double | Time to break threshold_1 (sec) (4min window) Nullable. |
| time_to_break_2_4min | double | Time to break threshold_2 (sec) (4min window) Nullable. |
| time_to_bounce_1_4min | double | Time to bounce threshold_1 (sec) (4min window) Nullable. |
| time_to_bounce_2_4min | double | Time to bounce threshold_2 (sec) (4min window) Nullable. |
| tradeable_1_4min | int8 | Tradeable flag threshold_1 (4min window) |
| tradeable_2_4min | int8 | Tradeable flag threshold_2 (4min window) |
| confirm_ts_ns_4min | int64 | Touch confirmation timestamp (ns) (4min window) Nullable. |
| anchor_spot_4min | double | Spot at confirmation (4min window) |
| future_price_4min | double | Last price in horizon (4min window) |
| outcome_8min | string | Outcome label (8min window) |
| excursion_max_8min | double | Max favorable excursion (8min window) |
| excursion_min_8min | double | Max adverse excursion (8min window) |
| strength_signed_8min | double | Signed strength (8min window) |
| strength_abs_8min | double | Absolute strength (8min window) |
| time_to_threshold_1_8min | double | Time to threshold_1 (sec) (8min window) Nullable. |
| time_to_threshold_2_8min | double | Time to threshold_2 (sec) (8min window) Nullable. |
| time_to_break_1_8min | double | Time to break threshold_1 (sec) (8min window) Nullable. |
| time_to_break_2_8min | double | Time to break threshold_2 (sec) (8min window) Nullable. |
| time_to_bounce_1_8min | double | Time to bounce threshold_1 (sec) (8min window) Nullable. |
| time_to_bounce_2_8min | double | Time to bounce threshold_2 (sec) (8min window) Nullable. |
| tradeable_1_8min | int8 | Tradeable flag threshold_1 (8min window) |
| tradeable_2_8min | int8 | Tradeable flag threshold_2 (8min window) |
| confirm_ts_ns_8min | int64 | Touch confirmation timestamp (ns) (8min window) Nullable. |
| anchor_spot_8min | double | Spot at confirmation (8min window) |
| future_price_8min | double | Last price in horizon (8min window) |
| outcome | string | Outcome label (primary 4min window) |
| excursion_max | double | Max favorable excursion (primary 4min window) |
| excursion_min | double | Max adverse excursion (primary 4min window) |
| strength_signed | double | Signed strength (primary 4min window) |
| strength_abs | double | Absolute strength (primary 4min window) |
| time_to_threshold_1 | double | Time to threshold_1 (primary 4min window, sec) Nullable. |
| time_to_threshold_2 | double | Time to threshold_2 (primary 4min window, sec) Nullable. |
| time_to_break_1 | double | Time to break threshold_1 (primary 4min window, sec) Nullable. |
| time_to_break_2 | double | Time to break threshold_2 (primary 4min window, sec) Nullable. |
| time_to_bounce_1 | double | Time to bounce threshold_1 (primary 4min window, sec) Nullable. |
| time_to_bounce_2 | double | Time to bounce threshold_2 (primary 4min window, sec) Nullable. |
| tradeable_1 | int8 | Tradeable flag threshold_1 (primary 4min window) |
| tradeable_2 | int8 | Tradeable flag threshold_2 (primary 4min window) |
| confirm_ts_ns | int64 | Touch confirmation timestamp (primary 4min window, ns) Nullable. |
| anchor_spot | double | Spot at confirmation (primary 4min window) |
| future_price | double | Last price in horizon (primary 4min window) |
