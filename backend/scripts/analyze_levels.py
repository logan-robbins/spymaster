
import pandas as pd
import numpy as np

# Load the predictions parquet file
file_path = '/Users/loganrobbins/research/qmachina/spymaster/backend/data/ml/ablation_results/predictions_20251231_114845.parquet'
df = pd.read_parquet(file_path)

# Filter out CHOP if needed, or include it. Let's focus on BREAK vs REJECT base rates.
# Counts per level
print("--- Base Rates per Level (December 2025 Test Set) ---")
level_stats = df.groupby('level_kind').apply(lambda x: pd.Series({
    'Total': len(x),
    'Rejects': (x['actual_outcome'] == 'REJECT').sum(),
    'Breaks': (x['actual_outcome'] == 'BREAK').sum(),
    'Reject_Rate': (x['actual_outcome'] == 'REJECT').mean() * 100,
    'Break_Rate': (x['actual_outcome'] == 'BREAK').mean() * 100
})).sort_values('Reject_Rate', ascending=False)

print(level_stats.to_markdown())

print("\n--- System Precision per Level (Value Add?) ---")
# Does the system add value on top of the base rate?
# If predicting BREAK, what is precision?
break_preds = df[df['predicted_outcome'] == 'BREAK']
if len(break_preds) > 0:
    break_stats = break_preds.groupby('level_kind').apply(lambda x: pd.Series({
        'Predicted_Breaks': len(x),
        'Correct_Breaks': (x['actual_outcome'] == 'BREAK').sum(),
        'Precision': (x['actual_outcome'] == 'BREAK').mean() * 100
    })).sort_values('Precision', ascending=False)
    print(break_stats.to_markdown())
else:
    print("No BREAK predictions made.")
