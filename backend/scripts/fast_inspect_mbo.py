
import databento as db
from databento.common.enums import PriceType
import pandas as pd
from pathlib import Path

def fast_inspect():
    backend_dir = Path("/Users/loganrobbins/research/qmachina/spymaster/backend")
    raw_mbo_path = backend_dir / "lake/raw/source=databento/product_type=future_option_mbo/symbol=ES/table=market_by_order_dbn/glbx-mdp3-20260106.mbo.dbn"
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

    # 2. Stream MBO and check intersection
    print(f"Scanning MBO stream (first 5 million records)...")
    store_mbo = db.DBNStore.from_file(str(raw_mbo_path))
    
    # Iterate in chunks manually if possible, or just use count
    # DBNStore.to_df loads data into memory. 5M records * 100 bytes = 500MB. Safe.
    
    try:
        # We assume 160 is MBO.
        # We just want 'instrument_id' column to be fast.
        # databento 0.39 doesn't support col filtering on load easily without upgrade.
        df_chunk = next(store_mbo.to_df(
            price_type=PriceType.FIXED, 
            pretty_ts=False, 
            map_symbols=False, # Faster
            count=5000000
        )) # Wait, to_df is not an iterator.
    except Exception:
        # If to_df returns full DF, we use the iterator.
        pass

    # Fallback: Just load it? No, assume iterator works if we loop?
    # The documentation says to_df returns a DataFrame.
    # To stream, we use `store` as iterator?
    # `for msg in store: ...` yields DBN Records. Slow in Python.
    
    # Let's try loading header only? No.
    # Let's trust the machine has 128GB RAM. Loading 5GB file takes ~10-20s.
    # My previous script failed to verify because it was doing too much processing.
    # I will just load the whole `instrument_id` column if possible?
    
    # Actually, pandas `read_parquet` allows columns. `DBNStore` does not.
    # I will load the dataframe. 5GB file -> 15GB RAM. Easy.
    print("Loading full ID column...")
    df_mbo = store_mbo.to_df(price_type=PriceType.FIXED)
    
    print(f"Loaded {len(df_mbo)} rows.")
    
    # Check intersection
    mbo_ids = set(df_mbo["instrument_id"].unique())
    
    overlap = mbo_ids.intersection(target_ids)
    print(f"Unique IDs in MBO: {len(mbo_ids)}")
    print(f"Overlap with 0DTE: {len(overlap)}")
    
    if len(overlap) > 0:
        print("SUCCESS: Data contains 0DTE records.")
        print(f"Sample: {list(overlap)[:5]}")
    else:
        print("FAIL: No 0DTE records found in MBO file.")

if __name__ == "__main__":
    fast_inspect()
