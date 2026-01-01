"""
Build training dataset for stream projection models.

Per STREAMS.md Section 7.3:
- Loads historical stream bars from gold/streams/pentaview/
- Extracts lookback history (L=20) and future targets (H=10) for each bar
- Saves training samples to gold/training/projection_samples/

Usage:
    uv run python -m scripts.build_projection_dataset \
        --start 2025-11-01 --end 2025-12-31 \
        --lookback 20 --horizon 10
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_time_bucket(minutes_since_open: float) -> str:
    """Map minutes since open to time bucket."""
    if minutes_since_open < 15:
        return 'T0_15'
    elif minutes_since_open < 30:
        return 'T15_30'
    elif minutes_since_open < 60:
        return 'T30_60'
    elif minutes_since_open < 120:
        return 'T60_120'
    else:
        return 'T120_180'


def build_training_sample(
    stream_bars_df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
    horizon: int = 10,
    stream_names: List[str] = None
) -> Optional[Dict]:
    """
    Build one training sample for projection model.
    
    Per STREAMS.md Section 6 (Training Data Construction):
    - Extract L=20 bars of history before idx
    - Extract H=10 bars of future after idx
    - Include cross-stream context and static features
    
    Args:
        stream_bars_df: DataFrame with stream bars for one level
        idx: Current bar index (anchor point)
        lookback: History window (L)
        horizon: Forecast window (H)
        stream_names: List of stream names to extract
    
    Returns:
        Training sample dictionary or None if insufficient data
    """
    if stream_names is None:
        stream_names = ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']
    
    # Check bounds
    if idx < lookback or idx + horizon >= len(stream_bars_df):
        return None
    
    bar = stream_bars_df.iloc[idx]
    
    # Extract lookback history
    hist_start = idx - lookback
    hist_slice = stream_bars_df.iloc[hist_start:idx]
    
    # Extract future targets
    future_slice = stream_bars_df.iloc[idx+1:idx+1+horizon]
    
    # Build sample for each stream
    samples = {}
    
    for stream_name in stream_names:
        if stream_name not in stream_bars_df.columns:
            continue
        
        # Lookback history
        stream_hist = hist_slice[stream_name].values
        slope_hist = hist_slice.get(f'{stream_name}_slope', pd.Series([0.0]*len(hist_slice))).values
        
        # Future targets
        future_target = future_slice[stream_name].values
        current_value = bar[stream_name]
        
        # Cross-stream context (last 5 bars of other streams)
        cross_streams = {}
        for other_stream in ['sigma_m', 'sigma_f', 'sigma_b', 'sigma_d', 'sigma_s']:
            if other_stream != stream_name and other_stream in stream_bars_df.columns:
                cross_streams[other_stream] = hist_slice[other_stream].values[-5:]
        
        # Static features
        minutes_since_open = bar.get('minutes_since_open', 0.0)
        static_features = {
            'level_kind_encoded': _encode_level_kind(bar.get('level_kind', 'OR_HIGH')),
            'direction_encoded': 1.0 if bar.get('direction', 'UP') == 'UP' else -1.0,
            'time_bucket_encoded': _encode_time_bucket(get_time_bucket(minutes_since_open)),
            'attempt_index': float(bar.get('attempt_index', 0)),
            'level_stacking_5pt': float(bar.get('level_stacking_5pt', 0)),
            'atr': float(bar.get('atr', 1.0))
        }
        
        # Sample weight (from setup quality)
        sigma_s = bar.get('sigma_s', 0.0)
        setup_weight = (sigma_s + 1.0) / 2.0  # Map [-1,1] to [0,1]
        
        samples[stream_name] = {
            'sample_id': f"{bar['timestamp']}_{bar['level_kind']}_{stream_name}",
            'timestamp': bar['timestamp'],
            'level_kind': bar['level_kind'],
            'direction': bar.get('direction', 'UP'),
            'time_bucket': get_time_bucket(minutes_since_open),
            'stream_hist': stream_hist,
            'slope_hist': slope_hist,
            'current_value': current_value,
            'future_target': future_target,
            'cross_streams': cross_streams,
            'static_features': static_features,
            'setup_weight': setup_weight,
            'atr': bar.get('atr', 1.0),
            'spot': bar.get('spot', 0.0)
        }
    
    return samples


def _encode_level_kind(level_kind: str) -> float:
    """Encode level_kind to numeric."""
    mapping = {
        'PM_HIGH': 0.0,
        'PM_LOW': 1.0,
        'OR_HIGH': 2.0,
        'OR_LOW': 3.0,
        'SMA_90': 4.0,
        'EMA_20': 5.0
    }
    return mapping.get(level_kind, 0.0)


def _encode_time_bucket(time_bucket: str) -> float:
    """Encode time bucket to numeric."""
    mapping = {
        'T0_15': 0.0,
        'T15_30': 1.0,
        'T30_60': 2.0,
        'T60_120': 3.0,
        'T120_180': 4.0
    }
    return mapping.get(time_bucket, 0.0)


def load_stream_bars(
    data_root: Path,
    canonical_version: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load stream bars from gold layer.
    
    Args:
        data_root: Backend data root
        canonical_version: Version string (e.g., '3.1.0')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Concatenated DataFrame with all stream bars
    """
    stream_dir = data_root / 'gold' / 'streams' / 'pentaview' / f'version={canonical_version}'
    
    if not stream_dir.exists():
        raise FileNotFoundError(f"Stream directory not found: {stream_dir}")
    
    logger.info(f"Loading stream bars from {stream_dir}...")
    
    # Find all date partitions
    date_dirs = sorted(stream_dir.glob('date=*'))
    
    # Filter by date range
    start_pd = pd.Timestamp(start_date)
    end_pd = pd.Timestamp(end_date)
    
    all_dfs = []
    for date_dir in date_dirs:
        # Extract date from directory name
        date_str = date_dir.name.split('=')[1].split('_')[0]  # Handle timestamp suffix
        date_pd = pd.Timestamp(date_str)
        
        if date_pd < start_pd or date_pd > end_pd:
            continue
        
        parquet_file = date_dir / 'stream_bars.parquet'
        if not parquet_file.exists():
            logger.warning(f"  Missing stream_bars.parquet for {date_str}")
            continue
        
        df = pd.read_parquet(parquet_file)
        df['date'] = date_str
        all_dfs.append(df)
        logger.info(f"  Loaded {len(df):,} bars from {date_str}")
    
    if not all_dfs:
        raise ValueError(f"No stream bars found between {start_date} and {end_date}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total: {len(combined):,} stream bars across {len(all_dfs)} dates")
    
    return combined


def build_dataset(
    stream_bars_df: pd.DataFrame,
    lookback: int = 20,
    horizon: int = 10,
    stream_names: List[str] = None,
    min_setup_quality: float = -0.25
) -> Dict[str, List[Dict]]:
    """
    Build training dataset from stream bars.
    
    Args:
        stream_bars_df: Stream bars DataFrame
        lookback: History window (L)
        horizon: Forecast window (H)
        stream_names: Streams to build samples for
        min_setup_quality: Minimum sigma_s to include sample
    
    Returns:
        Dictionary mapping stream_name -> list of training samples
    """
    if stream_names is None:
        stream_names = ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']
    
    logger.info(f"Building training samples (L={lookback}, H={horizon})...")
    
    # Group by level_kind to ensure temporal continuity
    grouped = stream_bars_df.groupby(['date', 'level_kind'])
    
    dataset = {name: [] for name in stream_names}
    total_candidates = 0
    filtered_low_quality = 0
    
    for (date, level_kind), group_df in grouped:
        group_df = group_df.sort_values('timestamp').reset_index(drop=True)
        
        for idx in range(lookback, len(group_df) - horizon):
            total_candidates += 1
            
            # Filter by setup quality
            bar = group_df.iloc[idx]
            sigma_s = bar.get('sigma_s', 0.0)
            if sigma_s < min_setup_quality:
                filtered_low_quality += 1
                continue
            
            samples = build_training_sample(
                stream_bars_df=group_df,
                idx=idx,
                lookback=lookback,
                horizon=horizon,
                stream_names=stream_names
            )
            
            if samples:
                for stream_name, sample in samples.items():
                    dataset[stream_name].append(sample)
    
    logger.info(f"  Candidates: {total_candidates:,}")
    logger.info(f"  Filtered (low quality): {filtered_low_quality:,}")
    
    for stream_name in stream_names:
        n = len(dataset[stream_name])
        logger.info(f"  {stream_name}: {n:,} training samples")
    
    return dataset


def save_dataset(
    dataset: Dict[str, List[Dict]],
    output_dir: Path,
    version: str = 'v1'
) -> None:
    """
    Save training dataset to disk.
    
    Args:
        dataset: Dictionary mapping stream_name -> samples
        output_dir: Output directory
        version: Dataset version string
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving training dataset to {output_dir}...")
    
    for stream_name, samples in dataset.items():
        if not samples:
            continue
        
        output_file = output_dir / f'projection_samples_{stream_name}_{version}.npz'
        
        # Convert to numpy arrays for efficient storage
        arrays = {}
        
        # Extract common fields
        n_samples = len(samples)
        lookback = len(samples[0]['stream_hist'])
        horizon = len(samples[0]['future_target'])
        n_cross = len(samples[0]['cross_streams'])
        n_static = len(samples[0]['static_features'])
        
        # Allocate arrays
        stream_hist = np.zeros((n_samples, lookback), dtype=np.float32)
        slope_hist = np.zeros((n_samples, lookback), dtype=np.float32)
        current_value = np.zeros(n_samples, dtype=np.float32)
        future_target = np.zeros((n_samples, horizon), dtype=np.float32)
        setup_weight = np.zeros(n_samples, dtype=np.float32)
        
        # Cross-streams (fixed at 5 bars per stream)
        cross_stream_names = sorted(samples[0]['cross_streams'].keys())
        cross_streams = np.zeros((n_samples, n_cross, 5), dtype=np.float32)
        
        # Static features
        static_feature_names = sorted(samples[0]['static_features'].keys())
        static_features = np.zeros((n_samples, n_static), dtype=np.float32)
        
        # Metadata
        sample_ids = []
        timestamps = []
        level_kinds = []
        
        # Fill arrays
        for i, sample in enumerate(samples):
            stream_hist[i] = sample['stream_hist']
            slope_hist[i] = sample['slope_hist']
            current_value[i] = sample['current_value']
            future_target[i] = sample['future_target']
            setup_weight[i] = sample['setup_weight']
            
            for j, cs_name in enumerate(cross_stream_names):
                cross_streams[i, j] = sample['cross_streams'][cs_name]
            
            for j, sf_name in enumerate(static_feature_names):
                static_features[i, j] = sample['static_features'][sf_name]
            
            sample_ids.append(sample['sample_id'])
            timestamps.append(str(sample['timestamp']))
            level_kinds.append(sample['level_kind'])
        
        # Save to compressed npz
        np.savez_compressed(
            output_file,
            stream_hist=stream_hist,
            slope_hist=slope_hist,
            current_value=current_value,
            future_target=future_target,
            setup_weight=setup_weight,
            cross_streams=cross_streams,
            static_features=static_features,
            cross_stream_names=np.array(cross_stream_names),
            static_feature_names=np.array(static_feature_names),
            sample_ids=np.array(sample_ids),
            timestamps=np.array(timestamps),
            level_kinds=np.array(level_kinds)
        )
        
        logger.info(f"  Saved {stream_name}: {n_samples:,} samples to {output_file.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description='Build training dataset for stream projection models')
    parser.add_argument('--data-root', type=str, default='data', help='Backend data root')
    parser.add_argument('--canonical-version', type=str, default='3.1.0', help='Canonical version')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookback', type=int, default=20, help='History window (L)')
    parser.add_argument('--horizon', type=int, default=10, help='Forecast window (H)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--streams', type=str, default='sigma_p,sigma_m,sigma_f,sigma_b,sigma_r',
                       help='Comma-separated stream names')
    parser.add_argument('--min-setup-quality', type=float, default=-0.25,
                       help='Minimum sigma_s to include sample')
    parser.add_argument('--version', type=str, default='v1', help='Dataset version')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root).resolve()
    stream_names = [s.strip() for s in args.streams.split(',')]
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root / 'gold' / 'training' / 'projection_samples'
    
    # Load stream bars
    stream_bars_df = load_stream_bars(
        data_root=data_root,
        canonical_version=args.canonical_version,
        start_date=args.start,
        end_date=args.end
    )
    
    # Build training dataset
    dataset = build_dataset(
        stream_bars_df=stream_bars_df,
        lookback=args.lookback,
        horizon=args.horizon,
        stream_names=stream_names,
        min_setup_quality=args.min_setup_quality
    )
    
    # Save to disk
    save_dataset(
        dataset=dataset,
        output_dir=output_dir,
        version=args.version
    )
    
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

