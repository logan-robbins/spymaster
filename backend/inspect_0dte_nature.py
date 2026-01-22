import databento as db
import pandas as pd
from pathlib import Path

def inspect_nature():
    lake_root = Path("lake")
    date_compact = "20260106"
    
    # Load definitions
    def_path = lake_root / f"raw/source=databento/dataset=definition/glbx-mdp3-{date_compact}.definition.dbn"
    print(f"Loading definitions from {def_path}")
    store = db.DBNStore.from_file(str(def_path))
    df = store.to_df(pretty_ts=False, map_symbols=True)
    
    # Filter for 2026-01-06 expiration
    exp_dates = pd.to_datetime(df["expiration"].astype("int64"), utc=True).dt.tz_convert("Etc/GMT+5").dt.date.astype(str)
    target = df[exp_dates == "2026-01-06"].copy()
    
    print(f"Total Matches: {len(target)}")
    
    # Check breakdown
    if "security_type" in target.columns:
        print("\nBreakdown by Security Type:")
        print(target["security_type"].value_counts())
        
    if "instrument_class" in target.columns:
        print("\nBreakdown by Instrument Class:")
        print(target["instrument_class"].value_counts())
        
    if "user_defined_instrument" in target.columns:
        print("\nUser Defined Instrument Flag:")
        print(target["user_defined_instrument"].value_counts())
        
    print("\nSample Symbols (Outrights vs Spreads?):")
    print(target["symbol"].head(10))
    
    # Check if 'OO' (Option Outright) vs others? 
    # CME typically uses 'OOF' for Options on Futures.
    
if __name__ == "__main__":
    inspect_nature()
