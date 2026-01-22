
import pandas as pd
import databento as db
from pathlib import Path
from databento.common.enums import PriceType

def debug_definitions(lake_root: Path, date_compact: str):
    base = lake_root / "raw" / "source=databento" / "dataset=definition"
    files = sorted(base.glob(f"*{date_compact}*.dbn*"))
    print(f"Found {len(files)} definition files.")
    
    dfs = []
    for path in files:
        print(f"Loading {path}")
        store = db.DBNStore.from_file(str(path))
        df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
        dfs.append(df)
        
    if not dfs:
        print("No dfs.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total definition rows: {len(df_all)}")
    
    # Filter C/P
    df_all = df_all.loc[df_all["instrument_class"].isin({"C", "P"})].copy()
    print(f"Option definitions: {len(df_all)}")

    # Check Expirations
    # Raw expiration is nanoseconds? Or other?
    # DBN usually ns epoch.
    
    # Sample
    print("Sample expirations (raw):")
    print(df_all["expiration"].head())
    
    # Convert
    ts_exp = pd.to_datetime(df_all["expiration"].astype("int64"), utc=True)
    print("\nSample expirations (UTC):")
    print(ts_exp.head())
    
    # Convert to ETC/GMT+5
    # NOTE: Etc/GMT+5 is GMT MINUS 5 hours (confusingly).
    ts_local = ts_exp.dt.tz_convert("Etc/GMT+5")
    print("\nSample expirations (Etc/GMT+5):")
    print(ts_local.head())
    
    # Unique dates
    unique_dates = ts_local.dt.date.unique()
    print("\nUnique Expiration Dates (Local):")
    for d in sorted(unique_dates):
        print(d)
        
    # Load Statistics to see what is actually trading/active
    stats_path = lake_root / "raw/source=databento/product_type=future_option_mbo/symbol=ES/table=statistics"
    stats_files = sorted(stats_path.glob(f"*{date_compact}*.dbn*"))
    print(f"\nFound {len(stats_files)} statistics files.")

    stats_ids = set()
    for path in stats_files:
        print(f"Loading {path}")
        df_stats = store.to_df(pretty_ts=False, map_symbols=True) # Map symbols!
        # Stat type might be missing if map_symbols changes view?
        # Ensure we keep what we have
        cols = ["instrument_id", "symbol"]
        if "stat_type" in df_stats.columns:
            cols.append("stat_type")
        if "expiration" in df_stats.columns:
            cols.append("expiration")
            
        df_stats = df_stats[cols]
        stats_ids.update(df_stats["instrument_id"].unique())
        
        # Keep map of id->symbol and id->expiration
        stats_id_map = dict(zip(df_stats["instrument_id"], df_stats["symbol"]))
        if "expiration" in df_stats.columns:
             stats_exp_map = dict(zip(df_stats["instrument_id"], df_stats["expiration"]))
        else:
             stats_exp_map = {}
        
    print(f"Total unique instruments in statistics: {len(stats_ids)}")
    
    session_date_str = "2026-01-06"
    print(f"\nTarget Session Date: {session_date_str}")
    
    def_ids = set(df_all["instrument_id"].unique())
    print(f"Total unique instruments in definitions: {len(def_ids)}")
    
    overlap = stats_ids.intersection(def_ids)
    print(f"Overlap count: {len(overlap)}")
    
    missing_def = stats_ids - def_ids
    print(f"Instruments in stats but missing definition: {len(missing_def)}")

    if missing_def:
        print("Sample missing instruments (ID -> Symbol):")
        sample_missing = list(missing_def)[:20]
        # Sort for stability
        sample_missing.sort()
        for mid in sample_missing:
            sym = stats_id_map.get(mid, "Unknown")
            print(f"  {mid}: {sym}")
    
    # Check if any missing IDs look like they should have been 0DTE? 
    # Can't know without def. 
    # But we can check if any *matched* IDs were 0DTE.
    
    # Filter definitions to what we found in stats
    active_defs = df_all.loc[df_all["instrument_id"].isin(stats_ids)].copy()
    
    ts_exp_active = pd.to_datetime(active_defs["expiration"].astype("int64"), utc=True)
    ts_local_active = ts_exp_active.dt.tz_convert("Etc/GMT+5")
    
    matches_active = active_defs.loc[ts_local_active.dt.date.astype(str) == session_date_str]
    
    print(f"0DTE Matches among active instruments: {len(matches_active)}")
    if not matches_active.empty:
        print(matches_active[["instrument_id", "symbol", "expiration", "strike_price"]].head())
    else:
        print("No active instruments match 0DTE session date.")
        
    # Print distinct active dates
    print("\nDistinct active dates:")
    print(sorted(ts_local_active.dt.date.unique()))


if __name__ == "__main__":
    lake_root = Path("lake")
    debug_definitions(lake_root, "20260106")
