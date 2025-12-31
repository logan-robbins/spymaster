
import pandas as pd
from pathlib import Path

# Find a sample metadata file - Corrected path with version
base_dir = Path('/Users/loganrobbins/research/qmachina/spymaster/backend/data/gold/episodes/es_level_episodes/version=3.1.0/metadata')
# Just grab the first date available
try:
    date_dir = next(base_dir.glob('date=*'))
    parquet_file = date_dir / 'metadata.parquet'

    print(f"Checking schema for: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    print("\n--- Columns ---")
    for col in sorted(df.columns):
        print(f"- {col}")

    print("\n--- Sample Excursion Data ---")
    cols_to_check = ['excursion_favorable', 'excursion_adverse', 'strength_abs']
    existing_cols = [c for c in cols_to_check if c in df.columns]
    if existing_cols:
        print(df[existing_cols].head())
    else:
        print("CRITICAL: Excursion columns missing!")
except StopIteration:
    print(f"No date partitions found in {base_dir}")
except Exception as e:
    print(f"Error: {e}")
