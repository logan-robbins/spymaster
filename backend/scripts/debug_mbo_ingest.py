
import databento as db
from databento.common.enums import PriceType
import pandas as pd
from pathlib import Path
import sys

def debug_ingest():
    backend_dir = Path("/Users/loganrobbins/research/qmachina/spymaster/backend")
    raw_mbo_path = backend_dir / "lake/raw/source=databento/product_type=future_option_mbo/symbol=ES/table=market_by_order_dbn/glbx-mdp3-20260106.mbo.dbn"
    raw_def_path = backend_dir / "lake/raw/source=databento/dataset=definition/glbx-mdp3-20260106.definition.dbn"
    
    dt = "2026-01-06"
    
    print(f"Loading definitions: {raw_def_path}")
    store_def = db.DBNStore.from_file(str(raw_def_path))
    df_def = store_def.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    
    # Logic from ingest.py _load_definitions
    df_def = df_def.sort_values("ts_event").groupby("instrument_id", as_index=False).last()
    df_def = df_def.loc[df_def["instrument_class"].isin(["C", "P"])].copy()
    exp_dates = (
        pd.to_datetime(df_def["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("Etc/GMT+5")
        .dt.date.astype(str)
    )
    df_def_target = df_def.loc[exp_dates == dt].copy()
    meta_ids = set(df_def_target["instrument_id"].tolist())
    print(f"Definition IDs matching {dt}: {len(meta_ids)}")
    if len(meta_ids) == 0:
        print("FAIL: No definitions found.")
        return

    print(f"\nLoading MBO: {raw_mbo_path}")
    # Inspect first N records
    store_mbo = db.DBNStore.from_file(str(raw_mbo_path))
    # It might be huge, iterate? No ingest.py reads whole file.
    # We will read 100k or iterate.
    
    # Try reading first 100k
    df_raw = next(store_mbo.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True, count=100000))
    print(df_raw.head()) # Not iterable if not generator? to_df is usually eager unless count specified?
    # Databento 0.39 to_df returns DataFrame.
    # Ah, DBNStore.to_df loads everything unless iterator used.
    # Let's assume standard behavior.
    
    # Wait, to_df with count might not work on file store?
    # Let's just use the store iterator explicitly if possible?
    # Or just load the file, it's 5GB.
    # I'll rely on the fact that I can stream it or just load small chunk.
    # `store.to_df` loads ALL.
    # `store` is iterable yielding records? No.
    # db.DBNStore is iterable?
    pass

    # Quick check using `head` logic manually if library allows.
    # Actually, iterate messages?
    total_mbo = 0
    analyzed = 0
    
    for df_raw in store_mbo.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True, count=100000): # Returns generator if count? No.
        # Databento API: to_df() returns DataFrame.
        # usually.
        # If I want chunking, I need to use `store.to_df` on a `dbn` iterator?
        pass
        
    # Re-write relying on `to_df` being efficient or I take the hit?
    # 5GB into RAM (128GB avail) is fine.
    
    df_raw = store_mbo.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    print(f"Loaded {len(df_raw)} MBO records")
    
    df = df_raw.loc[df_raw["rtype"] == 160].copy()
    print(f"RTYPE 160: {len(df)}")
    
    df = df.loc[df["symbol"].notna()].copy()
    print(f"Symbol notna: {len(df)}")
    
    is_spread = df["symbol"].str.startswith("UD:", na=False).to_numpy()
    df = df.loc[~is_spread].copy()
    print(f"Not Spread: {len(df)}")
    
    is_calendar = df["symbol"].str.contains("-", regex=False, na=False).to_numpy()
    df = df.loc[~is_calendar].copy()
    print(f"Not Calendar: {len(df)}")
    
    df_final = df.loc[df["instrument_id"].isin(meta_ids)].copy()
    print(f"Matched ID: {len(df_final)}")
    
    if len(df_final) == 0:
        print("FAIL: No match on IDs.")
        # Print sample IDs from MBO
        print(f"Sample MBO IDs: {df['instrument_id'].head(10).tolist()}")
        print(f"Sample Meta IDs: {list(meta_ids)[:10]}")

if __name__ == "__main__":
    debug_ingest()
