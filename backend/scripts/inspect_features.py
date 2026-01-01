import pandas as pd
from pathlib import Path

# Path to a recently processed file
file_path = "data/silver/features/es_pipeline/version=3.1.0/date=2025-12-17/signals.parquet"
df = pd.read_parquet(file_path)

print(f"Loaded {len(df)} signals")

# Kinematics
cols_kin = ['velocity_1min', 'velocity_5min', 'acceleration_1min', 'jerk_1min']
print("\n--- Kinematics (Sample) ---")
print(df[cols_kin].head())

# OFI
cols_ofi = ['ofi_30s', 'ofi_60s', 'ofi_acceleration']
print("\n--- Non-Zero Counts ---")
print((df[cols_kin] != 0).sum())
