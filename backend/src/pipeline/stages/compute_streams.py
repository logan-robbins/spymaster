"""Compute Pentaview streams from state table - STREAMS.md."""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from scipy.fft import dct

from src.pipeline.core.stage import BaseStage, StageContext
from src.ml.stream_builder import compute_all_streams, compute_derivatives
from src.ml.stream_normalization import load_stream_normalization_stats
from src.common.data_paths import date_partition

logger = logging.getLogger(__name__)


class ComputeStreamsStage(BaseStage):
    """
    Compute Pentaview streams from Stage 16 state table.
    
    Per STREAMS.md:
    1. Aggregate 30s state table → 2-minute bars
    2. Compute 5 canonical streams (Σ_M, Σ_F, Σ_B, Σ_D, Σ_S)
    3. Compute merged streams (Σ_P, Σ_R)
    4. Compute derivatives (slope, curvature, jerk)
    5. Compute alignment/divergence metrics
    
    Outputs:
        gold/streams/pentaview/version={canonical_version}/date=YYYY-MM-DD/*.parquet
    """
    
    def __init__(
        self,
        stream_stats_path: str = None,
        bar_freq: str = '2min',
        lookback_bars: int = 40,
        output_dir: str = 'gold/streams/pentaview'
    ):
        """
        Initialize stage.
        
        Args:
            stream_stats_path: Path to stream normalization stats JSON
            bar_freq: Bar frequency for aggregation (default '2min')
            lookback_bars: Number of bars for DCT trajectory (default 40 = 20 min @ 30s)
            output_dir: Output directory relative to DATA_ROOT
        """
        self.stream_stats_path = stream_stats_path
        self.bar_freq = bar_freq
        self.lookback_bars = lookback_bars
        self.output_dir = output_dir
    
    @property
    def name(self) -> str:
        return "compute_streams"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['state_df']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        state_df = ctx.data['state_df']
        date = pd.Timestamp(ctx.date)
        
        logger.info(f"  Computing Pentaview streams from {len(state_df):,} state samples...")
        
        # Load normalization statistics
        if self.stream_stats_path:
            stats_path = Path(self.stream_stats_path)
        else:
            # Default: gold/streams/normalization/current.json
            data_root = ctx.config.get("DATA_ROOT")
            stats_path = (Path(data_root) if data_root else Path("data")) / "gold" / "streams" / "normalization" / "current.json"
        
        if not stats_path.exists():
            logger.warning(f"  Stream normalization stats not found: {stats_path}")
            logger.warning("  Skipping stream computation. Run normalization first.")
            return {
                'streams_df': pd.DataFrame(),
                'n_bars': 0,
                'n_levels': 0
            }
        
        try:
            norm_stats = load_stream_normalization_stats(stats_path)
        except Exception as e:
            logger.error(f"  Failed to load stream stats: {e}")
            return {
                'streams_df': pd.DataFrame(),
                'n_bars': 0,
                'n_levels': 0
            }
        
        # Process each level separately
        all_stream_bars = []
        
        for level_kind in state_df['level_kind'].unique():
            level_df = state_df[state_df['level_kind'] == level_kind].copy()
            
            # Aggregate to 2-minute bars
            bar_df = self._aggregate_to_bars(level_df)
            
            if bar_df.empty:
                continue
            
            # Compute streams for each bar
            stream_bars = self._compute_streams_for_level(
                bar_df=bar_df,
                norm_stats=norm_stats,
                level_kind=level_kind
            )
            
            all_stream_bars.append(stream_bars)
        
        # Concatenate all levels
        if all_stream_bars:
            streams_df = pd.concat(all_stream_bars, ignore_index=True)
        else:
            streams_df = pd.DataFrame()
        
        logger.info(f"  Generated {len(streams_df):,} stream bars across {len(all_stream_bars)} levels")
        
        # Save to gold layer
        canonical_version = ctx.config.get('PIPELINE_CANONICAL_VERSION') or ctx.config.get('CANONICAL_VERSION', '3.1.0')
        output_path = self._get_output_path(ctx, canonical_version, date)
        
        if not streams_df.empty:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            streams_df.to_parquet(output_path, index=False, engine='pyarrow')
            logger.info(f"  Saved stream bars to {output_path}")
        
        # Also set 'signals' key for pipeline compatibility (pipeline.run() extracts this)
        return {
            'streams_df': streams_df,
            'signals': streams_df,  # For pipeline.run() compatibility
            'n_bars': len(streams_df),
            'n_levels': len(all_stream_bars)
        }
    
    def _aggregate_to_bars(self, level_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 30s state samples to 2-minute bars.
        
        Uses last value for most features (forward-fill semantics).
        """
        if level_df.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(level_df['timestamp']):
            level_df['timestamp'] = pd.to_datetime(level_df['timestamp'])
        
        # Set timestamp as index for resampling
        level_df = level_df.set_index('timestamp').sort_index()
        
        # Resample to 2-minute bars (use last value for most features)
        bar_df = level_df.resample(self.bar_freq).last()
        
        # Drop rows with all NaN (gaps in data)
        bar_df = bar_df.dropna(how='all')
        
        # Reset index
        bar_df = bar_df.reset_index()
        
        return bar_df
    
    def _compute_streams_for_level(
        self,
        bar_df: pd.DataFrame,
        norm_stats: Dict[str, Any],
        level_kind: str
    ) -> pd.DataFrame:
        """
        Compute streams for all bars of a single level.
        
        Args:
            bar_df: 2-minute bars for this level
            norm_stats: Normalization statistics
            level_kind: Level identifier
        
        Returns:
            DataFrame with stream values per bar
        """
        stream_rows = []
        
        for idx, bar_row in bar_df.iterrows():
            # Determine stratum for stratified normalization
            stratum = self._get_stratum(bar_row)
            
            # Compute DCT coefficients if we have enough history
            dct_coeffs = None
            if idx >= self.lookback_bars:
                lookback_start = max(0, idx - self.lookback_bars)
                history_df = bar_df.iloc[lookback_start:idx+1]
                dct_coeffs = self._compute_dct_coefficients(history_df)
            
            # Compute all streams
            streams = compute_all_streams(
                bar_row=bar_row,
                stats=norm_stats,
                dct_coeffs=dct_coeffs,
                stratum=stratum
            )
            
            # Package result
            stream_row = {
                'timestamp': bar_row['timestamp'],
                'level_kind': level_kind,
                'direction': bar_row.get('direction', 'UNKNOWN'),
                'spot': bar_row.get('spot', 0.0),
                'atr': bar_row.get('atr', 0.0),
                'level_price': bar_row.get('level_price', 0.0),
                **streams  # Add all stream values
            }
            
            stream_rows.append(stream_row)
        
        stream_df = pd.DataFrame(stream_rows)
        
        # Compute derivatives for key streams
        if len(stream_df) >= 3:
            for stream_name in ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b']:
                if stream_name in stream_df.columns:
                    derivs = compute_derivatives(stream_df.copy(), stream_name)
                    stream_df[f'{stream_name}_smooth'] = derivs['smooth']
                    stream_df[f'{stream_name}_slope'] = derivs['slope']
                    stream_df[f'{stream_name}_curvature'] = derivs['curvature']
                    stream_df[f'{stream_name}_jerk'] = derivs['jerk']
        
        return stream_df
    
    def _compute_dct_coefficients(self, history_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute DCT-II coefficients for key trajectory series.
        
        Per STREAMS.md Section 1.6:
        - d_atr: distance trajectory
        - ofi_60s: order flow trajectory
        - barrier_delta_liq_log: liquidity trajectory
        - tape_imbalance: aggression trajectory
        
        Returns first 8 coefficients per series.
        """
        dct_coeffs = {}
        
        # Series to transform
        series_names = ['distance_signed_atr', 'ofi_60s', 'barrier_delta_liq_log', 'tape_imbalance']
        dct_keys = ['d_atr', 'ofi_60s', 'barrier_delta_liq_log', 'tape_imbalance']
        
        for series_name, dct_key in zip(series_names, dct_keys):
            if series_name in history_df.columns:
                values = history_df[series_name].fillna(0.0).values
                
                # Pad or truncate to lookback_bars length
                if len(values) < self.lookback_bars:
                    values = np.pad(values, (self.lookback_bars - len(values), 0), mode='edge')
                elif len(values) > self.lookback_bars:
                    values = values[-self.lookback_bars:]
                
                # Compute DCT-II (orthonormal)
                coeffs = dct(values, type=2, norm='ortho')
                
                # Keep first 8 coefficients
                dct_coeffs[dct_key] = coeffs[:8]
        
        return dct_coeffs
    
    def _get_stratum(self, bar_row: pd.Series) -> str:
        """
        Determine normalization stratum from bar context.
        
        Per STREAMS.md Section 2.2: stratify by time_bucket.
        """
        minutes_since_open = bar_row.get('minutes_since_open', 0.0)
        
        # Map to time bucket per IMPLEMENTATION_READY.md
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
    
    def _get_output_path(self, ctx: StageContext, canonical_version: str, date: pd.Timestamp) -> Path:
        """Get output path for stream bars."""
        data_root = ctx.config.get("DATA_ROOT")
        base_path = Path(data_root) if data_root else Path("data")
        
        output_dir = base_path / self.output_dir / f"version={canonical_version}" / date_partition(date)
        return output_dir / "stream_bars.parquet"

