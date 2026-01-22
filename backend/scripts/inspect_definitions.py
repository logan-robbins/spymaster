
import databento as db
from databento.common.enums import PriceType
import pandas as pd
from pathlib import Path

def inspect():
    dbn_path = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake/raw/source=databento/dataset=definition/glbx-mdp3-20260106.definition.dbn")
    if not dbn_path.exists():
        print(f"File not found: {dbn_path}")
        return

    print(f"Reading {dbn_path}...")
    store = db.DBNStore.from_file(str(dbn_path))
    df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    
    print(f"Total definitions: {len(df)}")
    
    # Filter for Options
    df = df.loc[df["instrument_class"].isin(["C", "P"])].copy()
    print(f"Option definitions: {len(df)}")
    
    if df.empty:
        return

    # Check Expirations
    df["exp_dt_utc"] = pd.to_datetime(df["expiration"].astype("int64"), utc=True)
    df["exp_dt_est"] = df["exp_dt_utc"].dt.tz_convert("Etc/GMT+5") # Trying the code's timezone
    df["exp_date_est"] = df["exp_dt_est"].dt.date.astype(str)
    
    counts = df["exp_date_est"].value_counts().sort_index()
    print("\nExpiration Counts (EST Date):")
    print(counts)
    
    target = "2026-01-06"
    if target in counts:
        print(f"\nFOUND {counts[target]} contracts expiring on {target}.")
    else:
        print(f"\nNO contracts found expiring on {target}.")

if __name__ == "__main__":
    inspect()
