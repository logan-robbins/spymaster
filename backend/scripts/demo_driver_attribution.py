"""
Demo Driver Attribution

Shows how to use the DriverAttributor to explain BREAK/REJECT predictions.
Generates example attributions for high-confidence BREAK and REJECT cases.

Usage:
    cd backend
    uv run python -m scripts.demo_driver_attribution --version v4.0.0
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.driver_attribution import DriverAttributor, compute_historical_stats
from src.ml.episode_vector import construct_episodes_from_events, get_feature_names


def load_trained_model(model_path: Path):
    """Load the trained break predictor model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train a model first using the commands in PENTAVIEW_RESEARCH.md"
        )
    
    artifact = joblib.load(model_path)
    return artifact['model'], artifact.get('scaler'), artifact.get('features')


def load_episode_data(version: str, start_date: str, end_date: str):
    """
    Load episode vectors and metadata for analysis.
    
    Returns:
        episodes_df: DataFrame with metadata
        vectors: (N, 144) array of episode vectors
        stream_features: (N, K) array of stream features (if available)
    """
    base_dir = Path('data/gold/episodes/es_level_episodes')
    episodes_dir = base_dir / f'version={version}' / 'metadata'
    
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes not found: {episodes_dir}")
    
    # Load metadata
    all_meta = []
    for date_dir in sorted(episodes_dir.glob('date=*')):
        date_str = date_dir.name.split('=')[1]
        if start_date <= date_str <= end_date:
            meta_path = date_dir / 'metadata.parquet'
            if meta_path.exists():
                all_meta.append(pd.read_parquet(meta_path))
    
    episodes_df = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    
    print(f"Loaded {len(episodes_df)} episodes from {start_date} to {end_date}")
    
    # Reconstruct episode vectors (to get full 144D)
    # We need signals + state tables
    state_dir = Path('data/silver/state/es_level_state') / f'version={version}'
    signals_dir = Path('data/bronze/futures/es/levels') / f'version={version}'
    
    # For demo, just load first 100 episodes
    # In production, you'd load all or use stored .npy
    print("Reconstructing episode vectors (limited to 100 for demo)...")
    
    # This is a simplified version - in production you'd need full state+signals
    # For now, let's just use random vectors as a placeholder
    # TODO: Actually implement full reconstruction
    n_episodes = min(len(episodes_df), 100)
    vectors = np.random.randn(n_episodes, 144)  # Placeholder
    
    print(f"Warning: Using placeholder vectors for demo. Full reconstruction requires state+signals tables.")
    
    return episodes_df.iloc[:n_episodes], vectors


def main():
    parser = argparse.ArgumentParser(description='Demo driver attribution system')
    parser.add_argument('--version', default='v4.0.0', help='Data version')
    parser.add_argument('--model-path', default='data/ml/break_predictor_v1.joblib', help='Model path')
    parser.add_argument('--start', default='2025-11-03', help='Start date')
    parser.add_argument('--end', default='2025-12-19', help='End date')
    parser.add_argument('--output-json', default='data/ml/driver_attribution_demo.json', help='Output JSON')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model_path = Path(args.model_path)
    
    try:
        model, scaler, stream_features = load_trained_model(model_path)
        print(f"Model loaded. Features: {stream_features}")
    except FileNotFoundError as e:
        print(str(e))
        print("\nNote: This demo requires a trained model.")
        print("See PENTAVIEW_RESEARCH.md section 'Train New Model' for training commands.")
        return
    
    # Load episode data
    print(f"\nLoading episodes (version={args.version})...")
    episodes_df, episode_vectors = load_episode_data(
        args.version, args.start, args.end
    )
    
    if len(episodes_df) == 0:
        print("No episodes found in date range")
        return
    
    # Filter to predictable segment (UP + LOW levels)
    predictable_mask = (
        (episodes_df['direction'] == 'UP') &
        (episodes_df['level_kind'].isin(['PM_LOW', 'OR_LOW']))
    )
    
    predictable_df = episodes_df[predictable_mask].reset_index(drop=True)
    predictable_vectors = episode_vectors[predictable_mask]
    
    print(f"Predictable segment: {len(predictable_df)} episodes")
    print(f"BREAK rate: {(predictable_df['outcome_4min'] == 'BREAK').mean():.1%}")
    
    # Compute historical stats for percentiles
    print("\nComputing historical statistics for percentile calculations...")
    historical_stats = compute_historical_stats(predictable_vectors)
    
    # Create attributor
    # Note: The trained model uses only 5 stream features, not full 144D
    # For full 144D attribution, we'd need a 144D model
    print("\nNote: Current model uses 5 stream features, not full 144D vector.")
    print("For demo purposes, showing concept with stream-only model.")
    
    # We'll create a dummy full-vector model for demonstration
    # In production, you'd train a real RandomForest on 144D
    from sklearn.ensemble import RandomForestClassifier
    print("\nTraining demo 144D model for attribution (5 trees, fast)...")
    
    y = (predictable_df['outcome_4min'] == 'BREAK').astype(int).values
    
    # Quick fit on subset
    demo_model = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42)
    demo_model.fit(predictable_vectors, y)
    
    print(f"Demo model fitted: {demo_model.score(predictable_vectors, y):.2%} accuracy")
    
    # Create attributor
    attributor = DriverAttributor(
        model=demo_model,
        scaler=None,  # No scaling for demo
        feature_names=get_feature_names(),
        base_rate=0.455,
        historical_stats=historical_stats
    )
    
    # Find interesting examples
    print("\nGenerating predictions and finding interesting cases...")
    p_breaks = demo_model.predict_proba(predictable_vectors)[:, 1]
    
    # Find high-confidence BREAK that actually broke
    break_mask = (p_breaks >= 0.6) & (y == 1)
    if break_mask.sum() > 0:
        break_idx = np.where(break_mask)[0][0]
        print(f"\nExample 1: High-confidence BREAK (predicted: {p_breaks[break_idx]:.2%}, actual: BREAK)")
        
        attribution_break = attributor.explain(predictable_vectors[break_idx], p_break=p_breaks[break_idx])
        
        print(f"  Edge: {attribution_break.edge:+.1%}")
        print(f"  Confidence: {attribution_break.confidence}")
        print(f"  BREAK Drivers:")
        for driver in attribution_break.break_drivers[:3]:
            print(f"    - {driver.name}: {driver.contribution:+.3f} ({driver.trader_description})")
        print(f"  REJECT Drivers:")
        for driver in attribution_break.reject_drivers[:3]:
            print(f"    - {driver.name}: {driver.contribution:+.3f} ({driver.trader_description})")
    
    # Find high-confidence REJECT that actually rejected
    reject_mask = (p_breaks <= 0.4) & (y == 0)
    if reject_mask.sum() > 0:
        reject_idx = np.where(reject_mask)[0][0]
        print(f"\nExample 2: High-confidence REJECT (predicted: {p_breaks[reject_idx]:.2%}, actual: REJECT)")
        
        attribution_reject = attributor.explain(predictable_vectors[reject_idx], p_break=p_breaks[reject_idx])
        
        print(f"  Edge: {attribution_reject.edge:+.1%}")
        print(f"  Confidence: {attribution_reject.confidence}")
        print(f"  REJECT Drivers:")
        for driver in attribution_reject.reject_drivers[:3]:
            print(f"    - {driver.name}: {driver.contribution:+.3f} ({driver.trader_description})")
        print(f"  BREAK Drivers:")
        for driver in attribution_reject.break_drivers[:3]:
            print(f"    - {driver.name}: {driver.contribution:+.3f} ({driver.trader_description})")
    
    # Save example attributions to JSON
    output = {
        'meta': {
            'version': args.version,
            'n_episodes': len(predictable_df),
            'break_rate': float((predictable_df['outcome_4min'] == 'BREAK').mean()),
            'model_path': str(model_path),
            'demo_mode': True,
            'note': 'This is a demo with fast-fitted model. Production would use trained 144D model.'
        },
        'examples': []
    }
    
    if break_mask.sum() > 0:
        output['examples'].append({
            'case': 'high_confidence_break',
            'episode_id': predictable_df.iloc[break_idx]['episode_id'],
            'timestamp': str(predictable_df.iloc[break_idx]['timestamp']),
            'level_kind': predictable_df.iloc[break_idx]['level_kind'],
            'attribution': attribution_break.to_dict()
        })
    
    if reject_mask.sum() > 0:
        output['examples'].append({
            'case': 'high_confidence_reject',
            'episode_id': predictable_df.iloc[reject_idx]['episode_id'],
            'timestamp': str(predictable_df.iloc[reject_idx]['timestamp']),
            'level_kind': predictable_df.iloc[reject_idx]['level_kind'],
            'attribution': attribution_reject.to_dict()
        })
    
    # Write output
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDriver attribution examples saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  - Attribution system demonstrates marginal contribution decomposition")
    print(f"  - Section-level contributions show which physics domains matter")
    print(f"  - Feature-level drivers map to trader-understandable language")
    print(f"  - Percentiles contextualize current values vs historical")
    
    print(f"\nNext steps:")
    print(f"  1. Train full 144D model (not just 5 stream features)")
    print(f"  2. Integrate into real-time signal generation")
    print(f"  3. Display drivers in UI dashboard")


if __name__ == '__main__':
    main()
