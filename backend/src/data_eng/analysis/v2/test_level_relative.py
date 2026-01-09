from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from data_eng.stages.silver.future.mbp10_bar5s.level_relative import (
    compute_level_relative_depth_features,
    compute_level_relative_wall_features,
    compute_all_level_relative_features,
    BANDS,
)


def load_bar5s_with_levels(symbol: str, dt: str) -> pd.DataFrame:
    lake_path = Path(__file__).parents[4] / "lake"
    bar5s_path = lake_path / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_bar5s/dt={dt}"
    levels_path = lake_path / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_session_levels/dt={dt}"

    bar5s_files = list(bar5s_path.glob("*.parquet"))
    levels_files = list(levels_path.glob("*.parquet"))

    if not bar5s_files:
        raise FileNotFoundError(f"No bar5s data for {symbol} on {dt}")
    if not levels_files:
        raise FileNotFoundError(f"No levels data for {symbol} on {dt}")

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
    print(f"Loaded {len(df_bar5s)} bars")

    microprice_col = "bar5s_microprice_eob"
    if microprice_col in df_bar5s.columns:
        close_to_level = df_bar5s[
            (df_bar5s[microprice_col] >= pm_high - 5) &
            (df_bar5s[microprice_col] <= pm_high + 5)
        ]
        print(f"\nBars within 5 pts of PM_HIGH: {len(close_to_level)}")
    else:
        close_to_level = df_bar5s.head(10)

    if len(close_to_level) == 0:
        print("No bars close to PM_HIGH, using first 5 bars instead")
        close_to_level = df_bar5s.head(5)

    sample_bar = close_to_level.iloc[0]
    bid_px, ask_px, bid_sz, ask_sz = extract_book_state(sample_bar)

    microprice = sample_bar.get("bar5s_microprice_eob", (bid_px[0] + ask_px[0]) / 2)
    print(f"\nSample bar microprice: {microprice:.2f}")
    print(f"Distance to PM_HIGH: {microprice - pm_high:.2f} pts")

    print("\n=== MID-RELATIVE DEPTH (existing) ===")
    for band in BANDS[:3]:
        below_key = f"bar5s_depth_below_{band}_qty_eob"
        above_key = f"bar5s_depth_above_{band}_qty_eob"
        if below_key in sample_bar:
            print(f"  {band}: below={sample_bar[below_key]:.0f}, above={sample_bar[above_key]:.0f}")

    print("\n=== LEVEL-RELATIVE DEPTH (new) ===")
    depth_features = compute_level_relative_depth_features(
        bid_px, ask_px, bid_sz, ask_sz, pm_high
    )
    for band in BANDS[:3]:
        below_key = f"bar5s_lvldepth_below_{band}_qty_eob"
        above_key = f"bar5s_lvldepth_above_{band}_qty_eob"
        print(f"  {band}: below={depth_features[below_key]:.0f}, above={depth_features[above_key]:.0f}")

    print("\n=== MID-RELATIVE WALL (existing) ===")
    wall_bid_key = "bar5s_wall_bid_nearest_strong_dist_pts_eob"
    wall_ask_key = "bar5s_wall_ask_nearest_strong_dist_pts_eob"
    if wall_bid_key in sample_bar:
        print(f"  Bid wall dist: {sample_bar[wall_bid_key]:.2f} pts")
        print(f"  Ask wall dist: {sample_bar[wall_ask_key]:.2f} pts")

    print("\n=== LEVEL-RELATIVE WALL (new) ===")
    wall_features = compute_level_relative_wall_features(
        bid_px, ask_px, bid_sz, ask_sz, pm_high
    )
    print(f"  Bid wall dist from level: {wall_features['bar5s_lvlwall_bid_nearest_strong_dist_pts_eob']:.2f} pts")
    print(f"  Ask wall dist from level: {wall_features['bar5s_lvlwall_ask_nearest_strong_dist_pts_eob']:.2f} pts")

    print("\n=== BOOK STATE DEBUG ===")
    print(f"Bid prices: {bid_px[:5]}")
    print(f"Ask prices: {ask_px[:5]}")
    print(f"PM_HIGH: {pm_high}")
    print(f"Microprice: {microprice:.2f}")

    all_features = compute_all_level_relative_features(
        bid_px, ask_px, bid_sz, ask_sz, pm_high
    )
    print(f"\n=== COMPUTED {len(all_features)} LEVEL-RELATIVE FEATURES ===")

    print("\nDepth imbalance (below - above) / total:")
    print(f"  Level-relative: {all_features.get('bar5s_lvldepth_imbal_eob', 'N/A'):.4f}")

    print("\n✓ Verification complete")


if __name__ == "__main__":
    main()
