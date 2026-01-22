import databento as db
import pandas as pd
from pathlib import Path

def extract_parents():
    def_path = Path("lake/raw/source=databento/dataset=definition/glbx-mdp3-20260106.definition.dbn")
    print(f"Loading {def_path}...")
    store = db.DBNStore.from_file(str(def_path))
    df = store.to_df(pretty_ts=False, map_symbols=True)
    
    # Filter for C/P only
    df = df[df["instrument_class"].isin(["C", "P"])]
    
    # Filter for Underlying = ES?
    # 'underlying' column is what we want.
    # But wait, 0DTEs might list 'underlying' as 'ESZ6'?
    # We want to find the ASSETS that map to ES options.
    
    # Let's inspect 'asset', 'underlying', 'symbol'
    # We know from previous run that assets like 'E1B', 'EW' exist.
    # We want any asset where the underlying *looks* like ES.
    
    # Standard: Underlying is ES
    # Weekly: Underlying is ES
    # Let's check unique underlyings too.
    
    print("Columns:", df.columns.tolist())
    
    # Simple filter: Asset starts with 'E' and underlying starts with 'ES'
    # Or just check all assets for the 11,850 matches we found earlier?
    
    # Re-create the 0DTE filter
    exp_dates = pd.to_datetime(df["expiration"].astype("int64"), utc=True).dt.tz_convert("Etc/GMT+5").dt.date.astype(str)
    target = df[exp_dates == "2026-01-06"]
    
    print(f"\n0DTE Target Count: {len(target)}")
    if not target.empty:
        assets = target["asset"].unique()
        print("\nAssets for 0DTE (2026-01-06):")
        print(sorted(assets))
        
    # Also check broad ES assets
    print("\nAll Assets associated with 'ES' underlying:")
    if "underlying" in df.columns:
        es_related = df[df["underlying"].astype(str).str.startswith("ES")]
        print(sorted(es_related["asset"].unique()))

if __name__ == "__main__":
    extract_parents()
