from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from data_eng.stages.silver.future.mbp10_bar5s.level_relative import (
    compute_level_relative_depth_features,
    compute_level_relative_wall_features,
    BANDS,
)


def load_bar5s_with_levels(symbol: str, dt: str) -> pd.DataFrame:
    lake_path = Path(__file__).parents[4] / "lake"
    bar5s_path = lake_path / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_bar5s/dt={dt}"
    levels_path = lake_path / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_session_levels/dt={dt}"

    bar5s_files = list(bar5s_path.glob("*.parquet"))
    levels_files = list(levels_path.glob("*.parquet"))

    df_bar5s = pd.concat([pd.read_parquet(f) for f in bar5s_files])
    df_levels = pd.concat([pd.read_parquet(f) for f in levels_files])

    pm_high = df_levels["pm_high"].iloc[0]
    return df_bar5s, pm_high


def extract_book_state(row: pd.Series) -> tuple:
    bid_px = np.array([row[f"bar5s_shape_bid_px_l{i:02d}_eob"] for i in range(10)])
    ask_px = np.array([row[f"bar5s_shape_ask_px_l{i:02d}_eob"] for i in range(10)])
    bid_sz = np.array([row[f"bar5s_shape_bid_sz_l{i:02d}_eob"] for i in range(10)])
    ask_sz = np.array([row[f"bar5s_shape_ask_sz_l{i:02d}_eob"] for i in range(10)])
    return bid_px, ask_px, bid_sz, ask_sz


def main():
    symbol = "ESH6"
    dt = "2025-12-18"

    print(f"Loading data for {symbol} on {dt}...")
    df_bar5s, pm_high = load_bar5s_with_levels(symbol, dt)

    print(f"PM_HIGH = {pm_high}")

    df_bar5s["dist_to_level"] = (df_bar5s["bar5s_microprice_eob"] - pm_high).abs()
    df_at_level = df_bar5s[df_bar5s["dist_to_level"] <= 1.0].sort_values("dist_to_level")

    print(f"Bars within 1 pt of PM_HIGH: {len(df_at_level)}")

    if len(df_at_level) == 0:
        print("No bars at level, exiting")
        return

    sample_bar = df_at_level.iloc[0]
    bid_px, ask_px, bid_sz, ask_sz = extract_book_state(sample_bar)

    microprice = sample_bar["bar5s_microprice_eob"]
    print(f"\nSample bar microprice: {microprice:.3f}")
    print(f"Distance to PM_HIGH: {microprice - pm_high:.3f} pts")

    print("\n=== BOOK STATE ===")
    print(f"Bid L0: {bid_px[0]:.2f} @ {bid_sz[0]:.0f}")
    print(f"Ask L0: {ask_px[0]:.2f} @ {ask_sz[0]:.0f}")
    print(f"PM_HIGH: {pm_high}")

    print("\n=== MID-RELATIVE DEPTH (reference: microprice) ===")
    for band in BANDS:
        below_key = f"bar5s_depth_below_{band}_qty_eob"
        above_key = f"bar5s_depth_above_{band}_qty_eob"
        if below_key in sample_bar:
            print(f"  {band}: below={sample_bar[below_key]:.0f}, above={sample_bar[above_key]:.0f}")

    print("\n=== LEVEL-RELATIVE DEPTH (reference: PM_HIGH) ===")
    depth_features = compute_level_relative_depth_features(
        bid_px, ask_px, bid_sz, ask_sz, pm_high
    )
    for band in BANDS:
        below_key = f"bar5s_lvldepth_below_{band}_qty_eob"
        above_key = f"bar5s_lvldepth_above_{band}_qty_eob"
        print(f"  {band}: below={depth_features[below_key]:.0f}, above={depth_features[above_key]:.0f}")

    print("\n=== COMPARISON ===")
    total_mid_below = sum(sample_bar.get(f"bar5s_depth_below_{b}_qty_eob", 0) for b in BANDS)
    total_mid_above = sum(sample_bar.get(f"bar5s_depth_above_{b}_qty_eob", 0) for b in BANDS)
    total_lvl_below = depth_features["bar5s_lvldepth_below_total_qty_eob"]
    total_lvl_above = depth_features["bar5s_lvldepth_above_total_qty_eob"]

    print(f"MID-relative: below={total_mid_below:.0f}, above={total_mid_above:.0f}")
    print(f"LVL-relative: below={total_lvl_below:.0f}, above={total_lvl_above:.0f}")

    print("\n=== WALL DISTANCE COMPARISON ===")
    wall_features = compute_level_relative_wall_features(
        bid_px, ask_px, bid_sz, ask_sz, pm_high
    )

    mid_wall_bid = sample_bar.get("bar5s_wall_bid_nearest_strong_dist_pts_eob", np.nan)
    mid_wall_ask = sample_bar.get("bar5s_wall_ask_nearest_strong_dist_pts_eob", np.nan)
    lvl_wall_bid = wall_features["bar5s_lvlwall_bid_nearest_strong_dist_pts_eob"]
    lvl_wall_ask = wall_features["bar5s_lvlwall_ask_nearest_strong_dist_pts_eob"]

    print(f"MID-relative: bid_wall={mid_wall_bid:.2f}pts, ask_wall={mid_wall_ask:.2f}pts")
    print(f"LVL-relative: bid_wall={lvl_wall_bid:.2f}pts, ask_wall={lvl_wall_ask:.2f}pts")

    print("\n✓ Level-relative feature verification complete")


if __name__ == "__main__":
    main()
