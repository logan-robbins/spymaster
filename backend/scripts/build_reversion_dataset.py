"""
Build Mean-Reversion Training Dataset for Pentaview Streams

Strategy shift: Predict REVERSALS not TRAJECTORIES

Labels:
- REVERT_DOWN: Stream will drop >0.10 in next 5min (fade overbought)
- REVERT_UP: Stream will rise >0.10 in next 5min (buy oversold)
- NO_MOVE: Stream stays within ±0.10 (sit out)

Features:
- Current stream state (all 5 streams)
- Recent history (last 8 bars = 4 minutes)
- Extremity (how far from neutral)
- Alignment (are other streams confirming or diverging)

Usage:
    uv run python scripts/build_reversion_dataset.py --start 2025-11-17 --end 2025-12-17
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ReversionDatasetBuilder:
    """Build training data for mean-reversion classification."""
    
    def __init__(
        self,
        data_root: Path = Path("data"),
        canonical_version: str = "v4.0.0",
        reversion_threshold: float = 0.10,
        lookback_bars: int = 8
    ):
        self.data_root = data_root
        self.canonical_version = canonical_version
        self.reversion_threshold = reversion_threshold
        self.lookback_bars = lookback_bars
        
        self.stream_names = ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']
    
    def load_state_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load state table with full features including streams."""
        # Load state table which has rich features
        state_path = (
            self.data_root / "silver" / "state" / "es_level_state" /
            f"version={self.canonical_version}" / f"date={date}" / "state.parquet"
        )
        
        if not state_path.exists():
            logger.warning(f"State data not found: {state_path}")
            return None
        
        df = pd.read_parquet(state_path)
        
        # Load corresponding stream bars to merge in computed streams
        date_partition = f"date={date}_00:00:00"
        stream_path = (
            self.data_root / "gold" / "streams" / "pentaview" /
            f"version={self.canonical_version}" / date_partition / "stream_bars.parquet"
        )
        
        if stream_path.exists():
            stream_df = pd.read_parquet(stream_path)
            # Merge streams into state table
            merge_keys = ['timestamp', 'level_kind']
            df = df.merge(
                stream_df[['timestamp', 'level_kind', 'sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']],
                on=merge_keys,
                how='left'
            )
        
        return df
    
    def build_sample(
        self,
        df: pd.DataFrame,
        idx: int,
        target_stream: str
    ) -> Optional[Dict[str, Any]]:
        """
        Build one training sample.
        
        Returns sample dict or None if invalid
        """
        # Check bounds
        if idx < self.lookback_bars or idx + 10 >= len(df):
            return None
        
        current_row = df.iloc[idx]
        history = df.iloc[idx - self.lookback_bars:idx]
        future = df.iloc[idx + 1:idx + 11]
        
        # Get current stream value
        current_val = current_row[target_stream]
        
        if pd.isna(current_val):
            return None
        
        # Compute future max/min
        future_vals = future[target_stream].values
        max_future = np.max(future_vals)
        min_future = np.min(future_vals)
        
        up_move = max_future - current_val
        down_move = current_val - min_future
        
        # Determine label
        if up_move >= self.reversion_threshold:
            label = 'REVERT_UP'
            magnitude = up_move
        elif down_move >= self.reversion_threshold:
            label = 'REVERT_DOWN'
            magnitude = down_move
        else:
            label = 'NO_MOVE'
            magnitude = max(up_move, down_move)
        
        # Build features
        features = {}
        
        # Current state (all 5 streams)
        for stream in self.stream_names:
            if stream in current_row:
                features[f'{stream}_current'] = float(current_row[stream])
        
        # History statistics (last 8 bars)
        for stream in self.stream_names:
            if stream in history.columns:
                hist_vals = history[stream].values
                features[f'{stream}_mean_8bar'] = float(np.mean(hist_vals))
                features[f'{stream}_std_8bar'] = float(np.std(hist_vals))
                features[f'{stream}_trend_8bar'] = float(hist_vals[-1] - hist_vals[0])  # Net change
        
        # Extremity features
        features[f'{target_stream}_abs'] = abs(current_val)
        features[f'{target_stream}_squared'] = current_val ** 2
        
        # Alignment features
        other_streams = [s for s in self.stream_names if s != target_stream and s in current_row]
        if len(other_streams) >= 4:
            other_vals = [current_row[s] for s in other_streams]
            features['alignment_score'] = np.mean(other_vals) * np.sign(current_val)  # Do others agree?
            features['n_aligned'] = sum(1 for v in other_vals if np.sign(v) == np.sign(current_val))
            features['divergence_score'] = np.std(other_vals)
        
        # Timing features
        if 'minutes_since_open' in current_row:
            features['minutes_since_open'] = float(current_row['minutes_since_open'])
        
        # Market context
        if 'atr' in current_row:
            features['atr'] = float(current_row['atr'])
        if 'spot' in current_row:
            features['spot'] = float(current_row['spot'])
        
        # Multi-timeframe velocity (market tide)
        for window in ['1min', '3min', '5min', '10min', '20min']:
            vel_col = f'velocity_{window}'
            if vel_col in current_row:
                features[vel_col] = float(current_row[vel_col])
        
        # Order flow imbalance (buying/selling pressure)
        for window in ['30s', '60s', '120s', '300s']:
            ofi_col = f'ofi_{window}'
            if ofi_col in current_row:
                features[ofi_col] = float(current_row[ofi_col])
        
        # Gamma exposure (put/call premium impact)
        if 'call_gex_above_2strike' in current_row:
            features['call_gex'] = float(current_row['call_gex_above_2strike'])
        if 'put_gex_below_2strike' in current_row:
            features['put_gex'] = float(current_row['put_gex_below_2strike'])
        
        # Level stacking (confluence of structural levels)
        for n_pt in ['2pt', '5pt', '10pt']:
            stack_col = f'level_stacking_{n_pt}'
            if stack_col in current_row:
                features[stack_col] = float(current_row[stack_col])
        
        # CRITICAL: Level proximity and type
        if 'level_price' in current_row and 'spot' in current_row:
            distance = current_row['spot'] - current_row['level_price']
            features['distance'] = float(distance)
            
            if 'atr' in current_row:
                features['distance_atr'] = float(distance / current_row['atr'])
                features['at_level'] = 1.0 if abs(distance / current_row['atr']) < 0.5 else 0.0
                features['near_level'] = 1.0 if 0.5 <= abs(distance / current_row['atr']) < 1.5 else 0.0
        
        # Level type encoding
        if 'level_kind' in current_row:
            level_kind = current_row['level_kind']
            features['is_pm_high'] = 1.0 if level_kind == 'PM_HIGH' else 0.0
            features['is_pm_low'] = 1.0 if level_kind == 'PM_LOW' else 0.0
            features['is_or_high'] = 1.0 if level_kind == 'OR_HIGH' else 0.0
            features['is_or_low'] = 1.0 if level_kind == 'OR_LOW' else 0.0
            features['is_sma'] = 1.0 if 'SMA' in level_kind else 0.0
        
        # Direction (approaching from above or below)
        if 'direction' in current_row:
            features['approaching_from_below'] = 1.0 if current_row['direction'] == 'UP' else 0.0
        
        return {
            'features': features,
            'label': label,
            'magnitude': magnitude,
            'target_stream': target_stream,
            'timestamp': current_row['timestamp'],
            'current_value': current_val
        }
    
    def build_dataset(self, dates: List[str]) -> pd.DataFrame:
        """Build dataset across multiple dates."""
        all_samples = []
        
        logger.info(f"Building mean-reversion dataset...")
        logger.info(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        logger.info(f"  Reversion threshold: {self.reversion_threshold}")
        logger.info(f"  Lookback: {self.lookback_bars} bars")
        
        for date in dates:
            df = self.load_state_data(date)
            if df is None:
                logger.warning(f"  Skipping {date} (no data)")
                continue
            
            # FILTER: Only include samples NEAR levels (< 1.5 ATR away)
            # This is where streams have predictive power
            if 'distance_signed_atr' in df.columns:
                df = df[df['distance_signed_atr'].abs() < 1.5].copy()
            
            if len(df) < self.lookback_bars + 10:
                logger.warning(f"  Skipping {date} (insufficient data after filtering)")
                continue
            
            date_samples = 0
            
            # Build samples for each stream
            for stream in self.stream_names:
                if stream not in df.columns:
                    continue
                
                for idx in range(self.lookback_bars, len(df) - 10):
                    sample = self.build_sample(df, idx, stream)
                    if sample:
                        all_samples.append(sample)
                        date_samples += 1
            
            logger.info(f"  {date}: {date_samples:,} samples")
        
        # Convert to DataFrame
        if not all_samples:
            return pd.DataFrame()
        
        # Extract features and labels
        feature_names = list(all_samples[0]['features'].keys())
        
        data = {
            'label': [s['label'] for s in all_samples],
            'magnitude': [s['magnitude'] for s in all_samples],
            'target_stream': [s['target_stream'] for s in all_samples],
            'timestamp': [s['timestamp'] for s in all_samples],
            'current_value': [s['current_value'] for s in all_samples]
        }
        
        # Add feature columns
        for fname in feature_names:
            data[fname] = [s['features'].get(fname, np.nan) for s in all_samples]
        
        dataset = pd.DataFrame(data)
        
        # Summary stats
        logger.info(f"\n{'='*70}")
        logger.info("DATASET SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total samples: {len(dataset):,}")
        logger.info(f"\nLabel distribution:")
        for label in ['REVERT_UP', 'REVERT_DOWN', 'NO_MOVE']:
            count = (dataset['label'] == label).sum()
            pct = count / len(dataset) * 100
            logger.info(f"  {label:12s}: {count:6,} ({pct:5.1f}%)")
        
        logger.info(f"\nStream distribution:")
        for stream in self.stream_names:
            count = (dataset['target_stream'] == stream).sum()
            logger.info(f"  {stream}: {count:,}")
        
        return dataset


def main():
    parser = argparse.ArgumentParser(description='Build mean-reversion training dataset')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-root', type=str, default='data', help='Data root')
    parser.add_argument('--canonical-version', type=str, default='v4.0.0', help='Data version')
    parser.add_argument('--threshold', type=float, default=0.10, help='Reversion threshold')
    parser.add_argument('--lookback', type=int, default=8, help='History bars')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    
    args = parser.parse_args()
    
    # Generate date range
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    dates = pd.date_range(start, end, freq='D').strftime('%Y-%m-%d').tolist()
    
    # Build dataset
    builder = ReversionDatasetBuilder(
        data_root=Path(args.data_root),
        canonical_version=args.canonical_version,
        reversion_threshold=args.threshold,
        lookback_bars=args.lookback
    )
    
    dataset = builder.build_dataset(dates)
    
    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.data_root) / 'gold' / 'training' / 'reversion_samples'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'reversion_dataset_{args.start}_{args.end}.parquet'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    
    logger.info(f"\n✅ Saved to: {output_path}")
    logger.info(f"   Size: {len(dataset):,} samples")


if __name__ == "__main__":
    main()

