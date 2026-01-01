"""
Feature Interaction Discovery

Discovers and quantifies feature interactions (combos) that predict BREAK vs REJECT.
Tests hypothesized interactions from PENTAVIEW_RESEARCH.md:
- sigma_s × proximity: "Close + clean setup"
- sigma_d × gamma_exposure: "Dealer positioning reinforcement"
- ofi_acceleration × barrier_delta: "Flow meeting weak/strong resistance"
- sigma_s × level_stacking: "Setup quality × confluence"

Usage:
    cd backend
    uv run python -m scripts.discover_feature_interactions \
        --version v4.0.0 \
        --start 2025-11-03 \
        --end 2025-12-19 \
        --horizon 4min \
        --output-json data/ml/feature_interactions.json
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(version: str, start_date: str, end_date: str):
    """Load episodes and streams."""
    episodes_dir = Path(f'data/gold/episodes/es_level_episodes/version={version}/metadata')
    all_meta = []
    for date_dir in sorted(episodes_dir.glob('date=*')):
        date_str = date_dir.name.split('=')[1]
        if start_date <= date_str <= end_date:
            meta_path = date_dir / 'metadata.parquet'
            if meta_path.exists():
                all_meta.append(pd.read_parquet(meta_path))
    episodes = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    
    stream_dir = Path(f'data/gold/streams/pentaview/version={version}')
    all_streams = []
    for date_dir in sorted(stream_dir.glob('date=*')):
        stream_path = date_dir / 'stream_bars.parquet'
        if stream_path.exists():
            all_streams.append(pd.read_parquet(stream_path))
    streams = pd.concat(all_streams, ignore_index=True) if all_streams else pd.DataFrame()
    
    return episodes, streams


def test_hypothesized_interactions(
    df: pd.DataFrame,
    outcome_col: str = 'outcome_4min'
) -> Dict[str, Dict]:
    """
    Test specific hypothesized interactions from research.
    
    For each interaction:
    1. Compute interaction term
    2. Test marginal contribution beyond main effects
    3. Quantify interaction strength
    """
    # Filter to BREAK/REJECT
    mask = df[outcome_col].isin(['BREAK', 'REJECT'])
    df_filtered = df[mask].copy()
    y = (df_filtered[outcome_col] == 'BREAK').astype(int).values
    
    interactions_to_test = [
        ('sigma_s', 'proximity', 'Clean Setup × Close Approach'),
        ('sigma_d', 'gamma_exposure', 'Dealer Stream × Gamma Exposure'),
        ('ofi_acceleration', 'barrier_delta_liq', 'Flow Acceleration × Barrier Change'),
        ('sigma_s', 'level_stacking_5pt', 'Setup Quality × Confluence'),
        ('sigma_b_slope', 'barrier_replenishment_ratio', 'Barrier Slope × Replenishment'),
        ('velocity_2min', 'distance_signed_atr', 'Momentum × Distance'),
    ]
    
    results = {}
    
    for feat1, feat2, description in interactions_to_test:
        # Check if both features exist
        if feat1 not in df_filtered.columns or feat2 not in df_filtered.columns:
            print(f"  Skipping {description}: features not found")
            continue
        
        # Prepare data
        X_main = df_filtered[[feat1, feat2]].fillna(0).values
        X_interaction = X_main.copy()
        
        # Add interaction term
        interaction_term = X_main[:, 0] * X_main[:, 1]
        X_with_interaction = np.column_stack([X_main, interaction_term])
        
        # Scale
        scaler = StandardScaler()
        X_main_scaled = scaler.fit_transform(X_main)
        X_interaction_scaled = scaler.fit_transform(X_with_interaction)
        
        # Train models
        model_main = LogisticRegression(random_state=42, max_iter=1000)
        model_interaction = LogisticRegression(random_state=42, max_iter=1000)
        
        # CV scores
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scores_main = cross_val_score(model_main, X_main_scaled, y, cv=cv, scoring='roc_auc')
        scores_interaction = cross_val_score(model_interaction, X_interaction_scaled, y, cv=cv, scoring='roc_auc')
        
        # Improvement
        auc_main = np.mean(scores_main)
        auc_interaction = np.mean(scores_interaction)
        improvement = auc_interaction - auc_main
        
        # Fit full model to get coefficient
        model_interaction.fit(X_interaction_scaled, y)
        interaction_coef = float(model_interaction.coef_[0, 2])  # Third coefficient is interaction
        
        # Test significance via permutation
        n_perm = 100
        perm_improvements = []
        for _ in range(n_perm):
            y_perm = np.random.permutation(y)
            s_main = cross_val_score(model_main, X_main_scaled, y_perm, cv=cv, scoring='roc_auc')
            s_int = cross_val_score(model_interaction, X_interaction_scaled, y_perm, cv=cv, scoring='roc_auc')
            perm_improvements.append(np.mean(s_int) - np.mean(s_main))
        
        p_value = (np.sum(np.array(perm_improvements) >= improvement) + 1) / (n_perm + 1)
        
        results[f'{feat1}_x_{feat2}'] = {
            'description': description,
            'feat1': feat1,
            'feat2': feat2,
            'auc_main_effects': float(auc_main),
            'auc_with_interaction': float(auc_interaction),
            'improvement': float(improvement),
            'interaction_coefficient': float(interaction_coef),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'n_samples': len(y)
        }
        
        sig_str = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
        print(f"  {description}:")
        print(f"    AUC improvement: {improvement:+.4f} (p={p_value:.3f}){sig_str}")
        print(f"    Interaction coef: {interaction_coef:+.3f}")
    
    return results


def discover_pairwise_interactions(
    df: pd.DataFrame,
    features: List[str],
    outcome_col: str = 'outcome_4min',
    top_k: int = 10
) -> List[Dict]:
    """
    Discover top pairwise interactions by testing all combinations.
    
    Warning: This is O(n^2) in features, so limit feature set.
    """
    # Filter
    mask = df[outcome_col].isin(['BREAK', 'REJECT'])
    df_filtered = df[mask].copy()
    y = (df_filtered[outcome_col] == 'BREAK').astype(int).values
    
    # Limit to available features
    available_features = [f for f in features if f in df_filtered.columns]
    print(f"  Testing {len(available_features)} features")
    print(f"  Total pairwise combinations: {len(list(combinations(available_features, 2)))}")
    
    interaction_scores = []
    
    # Test all pairs
    for feat1, feat2 in combinations(available_features, 2):
        X_main = df_filtered[[feat1, feat2]].fillna(0).values
        interaction_term = X_main[:, 0] * X_main[:, 1]
        X_with_interaction = np.column_stack([X_main, interaction_term])
        
        # Quick single-split test (faster than full CV)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_interaction, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Coefficient of interaction term
        interaction_coef = float(model.coef_[0, 2])
        
        # Score
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_test)) == 2:
            y_pred = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            
            interaction_scores.append({
                'feat1': feat1,
                'feat2': feat2,
                'interaction_coef': interaction_coef,
                'auc': auc,
                'abs_coef': abs(interaction_coef)
            })
    
    # Sort by absolute coefficient (proxy for interaction strength)
    interaction_scores.sort(key=lambda x: x['abs_coef'], reverse=True)
    
    return interaction_scores[:top_k]


def analyze_interaction_patterns(
    df: pd.DataFrame,
    feat1: str,
    feat2: str,
    outcome_col: str = 'outcome_4min'
) -> Dict:
    """
    Analyze how outcome varies across bins of feat1 × feat2.
    
    Creates 2x2 grid (low/high for each feature) and reports BREAK rates.
    """
    mask = df[outcome_col].isin(['BREAK', 'REJECT'])
    df_filtered = df[mask].copy()
    
    # Median split
    feat1_median = df_filtered[feat1].median()
    feat2_median = df_filtered[feat2].median()
    
    df_filtered['feat1_high'] = df_filtered[feat1] > feat1_median
    df_filtered['feat2_high'] = df_filtered[feat2] > feat2_median
    
    # 2x2 grid
    pattern = {}
    for f1_high in [False, True]:
        for f2_high in [False, True]:
            cell_mask = (df_filtered['feat1_high'] == f1_high) & (df_filtered['feat2_high'] == f2_high)
            cell_df = df_filtered[cell_mask]
            
            if len(cell_df) > 0:
                break_rate = (cell_df[outcome_col] == 'BREAK').mean()
                pattern[f'{feat1}_{"high" if f1_high else "low"}_x_{feat2}_{"high" if f2_high else "low"}'] = {
                    'n': len(cell_df),
                    'break_rate': float(break_rate)
                }
    
    return pattern


def main():
    parser = argparse.ArgumentParser(description='Discover feature interactions')
    parser.add_argument('--version', default='v4.0.0')
    parser.add_argument('--start', default='2025-11-03')
    parser.add_argument('--end', default='2025-12-19')
    parser.add_argument('--horizon', default='4min')
    parser.add_argument('--output-json', default='data/ml/feature_interactions.json')
    parser.add_argument('--discover-all', action='store_true', help='Test all pairwise combinations (slow)')
    
    args = parser.parse_args()
    
    outcome_col = f'outcome_{args.horizon}'
    
    print("=== Feature Interaction Discovery ===\n")
    
    # Load data
    print("Loading data...")
    episodes, streams = load_data(args.version, args.start, args.end)
    
    # Merge
    episodes['ts_str'] = episodes['timestamp'].astype(str)
    streams['ts_str'] = streams['timestamp'].astype(str)
    merged = episodes.merge(streams, on=['ts_str', 'level_kind'], how='left')
    
    # Filter to predictable segment
    predictable_mask = (
        (merged['direction'] == 'UP') &
        (merged['level_kind'].isin(['PM_LOW', 'OR_LOW'])) &
        (~merged['time_bucket'].isin(['T0_15']))
    )
    predictable = merged[predictable_mask].copy()
    
    print(f"Predictable segment: {len(predictable)} episodes")
    
    # 1. Test hypothesized interactions
    print("\n=== 1. Hypothesized Interactions ===")
    hypothesized = test_hypothesized_interactions(predictable, outcome_col)
    
    # 2. Discover top interactions (if requested)
    discovered = None
    if args.discover_all:
        print("\n=== 2. Discovering All Pairwise Interactions ===")
        print("Warning: This tests O(n^2) combinations. May take a few minutes...")
        
        all_features = [
            'sigma_s', 'sigma_r', 'sigma_b_slope', 'sigma_p_slope', 'sigma_d',
            'proximity', 'distance_signed_atr', 'level_stacking_5pt',
            'gamma_exposure', 'ofi_acceleration', 'barrier_delta_liq',
            'barrier_replenishment_ratio', 'velocity_2min', 'attempt_number'
        ]
        
        discovered = discover_pairwise_interactions(predictable, all_features, outcome_col, top_k=15)
        
        print("\nTop 10 interactions by coefficient magnitude:")
        for i, interaction in enumerate(discovered[:10], 1):
            print(f"{i}. {interaction['feat1']} × {interaction['feat2']}")
            print(f"   Coef: {interaction['interaction_coef']:+.3f}, AUC: {interaction['auc']:.3f}")
    
    # 3. Analyze patterns for significant interactions
    print("\n=== 3. Interaction Patterns ===")
    patterns = {}
    
    for interaction_key, interaction_data in hypothesized.items():
        if interaction_data['significant']:
            feat1 = interaction_data['feat1']
            feat2 = interaction_data['feat2']
            
            print(f"\n{interaction_data['description']}:")
            pattern = analyze_interaction_patterns(predictable, feat1, feat2, outcome_col)
            
            for cell_key, cell_data in pattern.items():
                print(f"  {cell_key}: {cell_data['break_rate']:.1%} (n={cell_data['n']})")
            
            patterns[interaction_key] = pattern
    
    # Build output
    output = {
        'meta': {
            'version': args.version,
            'start_date': args.start,
            'end_date': args.end,
            'horizon': args.horizon,
            'n_episodes': len(predictable)
        },
        'hypothesized_interactions': hypothesized,
        'discovered_interactions': [
            {
                'feat1': d['feat1'],
                'feat2': d['feat2'],
                'interaction_coef': d['interaction_coef'],
                'auc': d['auc']
            }
            for d in (discovered or [])
        ] if discovered else None,
        'interaction_patterns': patterns,
        'summary': {
            'n_significant_hypothesized': sum(1 for v in hypothesized.values() if v['significant']),
            'most_important_interaction': None
        }
    }
    
    # Find most important
    if hypothesized:
        most_important = max(hypothesized.items(), key=lambda x: x[1]['improvement'])
        output['summary']['most_important_interaction'] = {
            'name': most_important[1]['description'],
            'improvement': most_important[1]['improvement'],
            'p_value': most_important[1]['p_value']
        }
    
    # Save
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Analysis saved to: {output_path}")
    print(f"Significant hypothesized interactions: {output['summary']['n_significant_hypothesized']}")
    
    if output['summary']['most_important_interaction']:
        mi = output['summary']['most_important_interaction']
        print(f"Most important: {mi['name']} (+{mi['improvement']:.4f} AUC, p={mi['p_value']:.3f})")
    
    print(f"\n=== Trader Takeaway ===")
    print("Interactions show when combinations of factors create outsized edge:")
    for interaction_key, interaction_data in hypothesized.items():
        if interaction_data['significant'] and interaction_data['improvement'] > 0.02:
            print(f"  - {interaction_data['description']}: +{interaction_data['improvement']:.3f} AUC")


if __name__ == '__main__':
    main()

