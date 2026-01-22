
import databento as db
from databento.common.enums import PriceType
import pandas as pd
from pathlib import Path

def inspect_stats():
    backend_dir = Path("/Users/loganrobbins/research/qmachina/spymaster/backend")
    raw_stats_path = backend_dir / "lake/raw/source=databento/product_type=future_option_mbo/symbol=ES/table=statistics/glbx-mdp3-20260106.statistics.dbn"
    raw_def_path = backend_dir / "lake/raw/source=databento/dataset=definition/glbx-mdp3-20260106.definition.dbn"
    dt = "2026-01-06"

    # 1. Load 0DTE IDs
    print(f"Loading definitions for {dt}...")
    store_def = db.DBNStore.from_file(str(raw_def_path))
    df_def = store_def.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    
    df_def = df_def.sort_values("ts_event").groupby("instrument_id", as_index=False).last()
    df_def = df_def.loc[df_def["instrument_class"].isin(["C", "P"])].copy()
    
    exp_dates = (
        pd.to_datetime(df_def["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("Etc/GMT+5")
        .dt.date.astype(str)
    )
    
    target_ids = set(df_def.loc[exp_dates == dt, "instrument_id"].tolist())
    print(f"Target 0DTE IDs: {len(target_ids)}")
    
    if not target_ids:
        return

    # 2. Load Statistics
    if not raw_stats_path.exists():
        print("FAIL: Statistics file does not exist.")
        return

    print(f"Loading Statistics: {raw_stats_path}")
    store_stats = db.DBNStore.from_file(str(raw_stats_path))
    df_stats = store_stats.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=False)
    
    print(f"Loaded {len(df_stats)} statistics records.")
    
    # Filter for Open Interest (stat_type=1 usually? or just check IDs)
    # Checking IDs is sufficient to know if the CONTRACTS are present.
    stats_ids = set(df_stats["instrument_id"].unique())
    
    overlap = stats_ids.intersection(target_ids)
    print(f"Unique IDs in Stats: {len(stats_ids)}")
    print(f"Overlap with 0DTE: {len(overlap)}")
    
    if len(overlap) > 0:
        print("SUCCESS: Statistics contain 0DTE records.")
    else:
        print("FAIL: No 0DTE records found in Statistics file.")

if __name__ == "__main__":
    inspect_stats()
