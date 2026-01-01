"""
Build Level Interaction Dataset for Pentaview Analysis

Unbiased data collection: Capture ALL state table features when price is near levels,
and label what actually happens next. Let ablation studies discover the patterns.

Labels (neutral):
- MOVE_UP: Stream increases >threshold in next 5min
- MOVE_DOWN: Stream decreases >threshold in next 5min  
- FLAT: No significant move

Features (complete state table):
- All 86 state table features
- All 5 Pentaview streams
- Level context, proximity, type
- OFI, GEX, barriers, tape, velocity
- No assumptions about relationships

Usage:
    uv run python scripts/build_pentaview_level_dataset.py --start 2025-11-17 --end 2025-12-17
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PentaviewLevelDatasetBuilder:
    """Build unbiased level-interaction dataset."""
    
    def __init__(
        self,
        data_root: Path = Path("data"),
        canonical_version: str = "v4.0.0",
        move_threshold: float = 0.10,
        lookback_bars: int = 8,
        max_distance_atr: float = 1.5  # Only include bars near levels
    ):
        self.data_root = data_root
        self.canonical_version = canonical_version
        self.move_threshold = move_threshold
        self.lookback_bars = lookback_bars
        self.max_distance_atr = max_distance_atr
        
        self.stream_names = ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']
    
    def load_state_with_streams(self, date: str) -> Optional[pd.DataFrame]:
        """Load state table with Pentaview streams merged in."""
        # Load state table (full 86 features)
        state_path = (
            self.data_root / "silver" / "state" / "es_level_state" /
            f"version={self.canonical_version}" / f"date={date}" / "state.parquet"
        )
        
        if not state_path.exists():
            return None
        
        state_df = pd.read_parquet(state_path)
        
        # Load stream bars
        date_partition = f"date={date}_00:00:00"
        stream_path = (
            self.data_root / "gold" / "streams" / "pentaview" /
            f"version={self.canonical_version}" / date_partition / "stream_bars.parquet"
        )
        
        if not stream_path.exists():
            logger.warning(f"Streams not found for {date}, using state table only")
            return state_df
        
        stream_df = pd.read_parquet(stream_path)
        
        # Merge streams into state
        merge_keys = ['timestamp', 'level_kind']
        stream_cols = ['timestamp', 'level_kind'] + [s for s in self.stream_names if s in stream_df.columns]
        
        df = state_df.merge(
            stream_df[stream_cols],
            on=merge_keys,
            how='left',
            suffixes=('', '_stream')
        )
        
        return df
    
    def build_sample(
        self,
        df: pd.DataFrame,
        idx: int,
        target_stream: str
    ) -> Optional[Dict[str, Any]]:
        """Build one training sample with ALL state table features."""
        
        # Check bounds
        if idx < self.lookback_bars or idx + 10 >= len(df):
            return None
        
        current_row = df.iloc[idx]
        future = df.iloc[idx + 1:idx + 11]
        
        # Get target stream value
        if target_stream not in current_row or pd.isna(current_row[target_stream]):
            return None
        
        current_val = float(current_row[target_stream])
        
        # Compute future movement
        future_vals = future[target_stream].values
        if len(future_vals) < 10 or np.any(pd.isna(future_vals)):
            return None
        
        max_future = np.max(future_vals)
        min_future = np.min(future_vals)
        
        up_move = max_future - current_val
        down_move = current_val - min_future
        
        # Determine label (UNBIASED - just what happened)
        if up_move >= self.move_threshold:
            label = 'MOVE_UP'
            magnitude = up_move
            bars_to_extreme = int(np.argmax(future_vals) + 1)
        elif down_move >= self.move_threshold:
            label = 'MOVE_DOWN'
            magnitude = down_move
            bars_to_extreme = int(np.argmin(future_vals) + 1)
        else:
            label = 'FLAT'
            magnitude = max(up_move, down_move)
            bars_to_extreme = 0
        
        # Extract ALL features from current row (exclude metadata/future-looking)
        exclude_cols = {
            'timestamp', 'ts_ns', 'date',  # Metadata
            'level_active',  # Boolean, not useful
            # Don't exclude: We want level_kind, level_price, etc as features
        }
        
        features = {}
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            val = current_row[col]
            
            # Handle categoricals
            if col == 'level_kind':
                for lk in ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20']:
                    features[f'level_is_{lk}'] = 1.0 if val == lk else 0.0
            elif col == 'barrier_state':
                for bs in ['STRONG_SUPPORT', 'WEAK_SUPPORT', 'NEUTRAL', 'WEAK_RESISTANCE', 'STRONG_RESISTANCE']:
                    features[f'barrier_state_{bs}'] = 1.0 if val == bs else 0.0
            elif col == 'fuel_effect':
                for fe in ['AMPLIFY', 'NEUTRAL', 'DAMPEN']:
                    features[f'fuel_{fe}'] = 1.0 if val == fe else 0.0
            elif isinstance(val, (int, float, np.integer, np.floating)):
                features[col] = float(val) if not pd.isna(val) else 0.0
            elif isinstance(val, bool):
                features[col] = 1.0 if val else 0.0
        
        return {
            'features': features,
            'label': label,
            'magnitude': magnitude,
            'bars_to_extreme': bars_to_extreme,
            'target_stream': target_stream,
            'level_kind': current_row['level_kind'],
            'distance_atr': float(current_row.get('distance_signed_atr', 0)),
            'timestamp': current_row['timestamp'],
            'current_stream_value': current_val
        }
    
    def build_dataset(self, dates: List[str]) -> pd.DataFrame:
        """Build comprehensive dataset across dates."""
        all_samples = []
        
        logger.info(f"Building Pentaview Level Interaction Dataset")
        logger.info(f"{'='*70}")
        logger.info(f"Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        logger.info(f"Move threshold: {self.move_threshold}")
        logger.info(f"Max distance: {self.max_distance_atr} ATR")
        logger.info(f"Lookback: {self.lookback_bars} bars")
        logger.info("")
        
        for date in dates:
            df = self.load_state_with_streams(date)
            if df is None:
                logger.warning(f"  {date}: No data")
                continue
            
            # FILTER: Only bars near levels (where Pentaview matters)
            if 'distance_signed_atr' in df.columns:
                df_near = df[df['distance_signed_atr'].abs() < self.max_distance_atr].copy()
            else:
                df_near = df.copy()
            
            if len(df_near) < self.lookback_bars + 10:
                logger.warning(f"  {date}: Insufficient data near levels")
                continue
            
            date_samples = 0
            
            # Build samples for each stream
            for stream in self.stream_names:
                if stream not in df_near.columns:
                    continue
                
                for idx in range(self.lookback_bars, len(df_near) - 10):
                    sample = self.build_sample(df_near, idx, stream)
                    if sample:
                        all_samples.append(sample)
                        date_samples += 1
            
            logger.info(f"  {date}: {date_samples:,} samples")
        
        if not all_samples:
            logger.error("No samples generated!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        feature_names = sorted(all_samples[0]['features'].keys())
        
        data = {
            'label': [s['label'] for s in all_samples],
            'magnitude': [s['magnitude'] for s in all_samples],
            'bars_to_extreme': [s['bars_to_extreme'] for s in all_samples],
            'target_stream': [s['target_stream'] for s in all_samples],
            'level_kind': [s['level_kind'] for s in all_samples],
            'distance_atr': [s['distance_atr'] for s in all_samples],
            'timestamp': [s['timestamp'] for s in all_samples],
            'current_stream_value': [s['current_stream_value'] for s in all_samples]
        }
        
        # Add all feature columns
        for fname in feature_names:
            data[fname] = [s['features'].get(fname, np.nan) for s in all_samples]
        
        dataset = pd.DataFrame(data)
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("DATASET SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total samples: {len(dataset):,}")
        logger.info(f"Features: {len(feature_names)}")
        
        logger.info(f"\nLabel distribution:")
        for label in ['MOVE_UP', 'MOVE_DOWN', 'FLAT']:
            count = (dataset['label'] == label).sum()
            pct = count / len(dataset) * 100
            logger.info(f"  {label:12s}: {count:6,} ({pct:5.1f}%)")
        
        logger.info(f"\nLevel type distribution:")
        for level in dataset['level_kind'].value_counts().head(10).items():
            logger.info(f"  {level[0]}: {level[1]:,}")
        
        logger.info(f"\nStream distribution:")
        for stream in self.stream_names:
            count = (dataset['target_stream'] == stream).sum()
            logger.info(f"  {stream}: {count:,}")
        
        logger.info(f"\nProximity distribution:")
        at_level = (dataset['distance_atr'].abs() < 0.5).sum()
        near_level = ((dataset['distance_atr'].abs() >= 0.5) & (dataset['distance_atr'].abs() < 1.5)).sum()
        logger.info(f"  AT_LEVEL (< 0.5 ATR): {at_level:,} ({at_level/len(dataset)*100:.1f}%)")
        logger.info(f"  NEAR_LEVEL (0.5-1.5 ATR): {near_level:,} ({near_level/len(dataset)*100:.1f}%)")
        
        return dataset


def main():
    parser = argparse.ArgumentParser(description='Build Pentaview level interaction dataset')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-root', type=str, default='data', help='Data root')
    parser.add_argument('--canonical-version', type=str, default='v4.0.0', help='Data version')
    parser.add_argument('--threshold', type=float, default=0.10, help='Move threshold')
    parser.add_argument('--lookback', type=int, default=8, help='History bars')
    parser.add_argument('--max-distance', type=float, default=1.5, help='Max distance ATR')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    
    args = parser.parse_args()
    
    # Generate date range
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    dates = pd.date_range(start, end, freq='D').strftime('%Y-%m-%d').tolist()
    
    # Build dataset
    builder = PentaviewLevelDatasetBuilder(
        data_root=Path(args.data_root),
        canonical_version=args.canonical_version,
        move_threshold=args.threshold,
        lookback_bars=args.lookback,
        max_distance_atr=args.max_distance
    )
    
    dataset = builder.build_dataset(dates)
    
    if dataset.empty:
        logger.error("Failed to build dataset")
        return 1
    
    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.data_root) / 'gold' / 'training' / 'pentaview_level_interactions'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'dataset_{args.start}_{args.end}.parquet'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    
    logger.info(f"\n✅ Saved to: {output_path}")
    logger.info(f"   Samples: {len(dataset):,}")
    logger.info(f"   Features: {len([c for c in dataset.columns if c.startswith('level_') or c.startswith('sigma_') or c.startswith('ofi_') or 'gex' in c])}")
    
    return 0


if __name__ == "__main__":
    exit(main())

