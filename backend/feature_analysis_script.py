"""Systematic feature analysis - 10 features at a time."""
from pathlib import Path
import pandas as pd
import numpy as np

def load_data():
    """Load silver approach data for analysis."""
    lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
    sample_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"
    df = pd.read_parquet(sample_path)
    return df

def analyze_feature_batch(df, start_idx, batch_size=10):
    """Analyze a batch of features."""
    # Skip metadata/context fields that are expected to have special patterns
    skip_fields = {
        'symbol', 'episode_id', 'touch_id', 'level_type', 'outcome',  # metadata
        'level_price', 'trigger_bar_ts', 'bar_ts',  # context with expected patterns
        'bar_index_in_episode', 'bar_index_in_touch', 'bars_to_trigger',  # episode structure
        'bar5s_meta_clear_cnt_sum'  # Expected to be zero (no clear events in MBP-10)
    }
    all_features = [col for col in df.columns if col not in skip_fields]
    end_idx = min(start_idx + batch_size, len(all_features))

    batch_features = all_features[start_idx:end_idx]
    print(f"\n{'='*80}")
    print(f"ANALYZING FEATURES {start_idx+1}-{end_idx} of {len(all_features)}")
    print(f"{'='*80}")

    issues_found = []

    for i, field in enumerate(batch_features, 1):
        col = df[field]
        n_total = len(col)
        n_null = col.isna().sum()
        pct_null = 100 * n_null / n_total

        issue = None

        # Check for problematic patterns
        if pct_null > 95:
            issue = f">{pct_null:.0f}% nulls"
        elif col.dtype in ['int64', 'float64', 'int32', 'float32']:
            n_zero = (col == 0).sum()
            pct_zero = 100 * n_zero / n_total
            std = col.std()

            if std == 0:
                issue = "Zero variance"
            elif pct_zero > 98:
                issue = f">{pct_zero:.0f}% zeros"
            elif np.isinf(col).any():
                issue = "Contains inf values"
            elif col.min() == col.max() and col.nunique() == 1:
                issue = f"Constant value: {col.iloc[0]}"

        print(f"{i:2d}. {field:50} | Null%: {pct_null:5.1f} | Type: {str(col.dtype):8} | Range: {col.min():>8.2f} - {col.max():>8.2f}", end="")

        if issue:
            print(f" | ⚠️  {issue}")
            issues_found.append({"field": field, "issue": issue})
        else:
            print(" | ✅ OK")

    print(f"\nBatch {start_idx//batch_size + 1} complete: {len(issues_found)} issues found")
    return issues_found, end_idx

def investigate_feature(df, feature_name):
    """Deep investigation of a specific feature."""
    print(f"\n{'='*80}")
    print(f"INVESTIGATING: {feature_name}")
    print(f"{'='*80}")

    col = df[feature_name]

    # Basic stats
    print("Basic Statistics:")
    print(f"  Type: {col.dtype}")
    print(f"  Count: {len(col)}")
    print(f"  Nulls: {col.isna().sum()} ({100*col.isna().sum()/len(col):.1f}%)")

    if col.dtype in ['int64', 'float64', 'int32', 'float32']:
        print(f"  Mean: {col.mean():.6f}")
        print(f"  Std: {col.std():.6f}")
        print(f"  Min: {col.min():.6f}")
        print(f"  Max: {col.max():.6f}")
        print(f"  Zeros: {(col == 0).sum()} ({100*(col == 0).sum()/len(col):.1f}%)")

        # Check for suspicious patterns
        if col.std() == 0:
            print("  ⚠️  ZERO VARIANCE!")
        elif np.isinf(col).any():
            print("  ⚠️  CONTAINS INFINITY!")
        elif col.min() == col.max():
            print(f"  ⚠️  CONSTANT VALUE: {col.iloc[0]}")

    # Sample values
    print(f"\nSample values (first 10 non-null):")
    non_null = col.dropna()
    if len(non_null) > 0:
        print(f"  {non_null.head(10).tolist()}")

    # Distribution
    if col.dtype in ['int64', 'float64', 'int32', 'float32']:
        print("\nValue distribution:")
        try:
            bins = pd.cut(col.dropna(), bins=10)
            print(f"  {bins.value_counts().sort_index()}")
        except:
            print("  Could not create bins")

def main():
    df = load_data()
    print(f"Loaded {len(df)} rows with {len(df.columns)} features")

    # Get all features to analyze
    all_features = [col for col in df.columns if not col.startswith('is_') and col not in ['symbol', 'episode_id', 'touch_id', 'level_type', 'outcome']]
    print(f"Will analyze {len(all_features)} features")

    batch_size = 10
    start_idx = 0
    all_issues = []

    while start_idx < len(all_features):
        issues, next_start = analyze_feature_batch(df, start_idx, batch_size)
        all_issues.extend(issues)

        if issues:
            print(f"\n⚠️  ISSUES FOUND IN BATCH! Stopping for investigation.")
            for issue in issues:
                investigate_feature(df, issue['field'])
            break

        start_idx = next_start

        # Continue automatically for systematic analysis
        if start_idx >= len(all_features):
            break

    print("\nFINAL SUMMARY:")
    print(f"Features analyzed: {min(start_idx, len(all_features))}")
    print(f"Issues found: {len(all_issues)}")

    if all_issues:
        print("\nIssues requiring investigation:")
        for issue in all_issues:
            print(f"  - {issue['field']}: {issue['issue']}")

if __name__ == "__main__":
    main()
