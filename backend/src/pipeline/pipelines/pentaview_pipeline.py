"""Pentaview Pipeline - Stream computation from state table.

Per STREAMS.md:
- Reads Stage 16 state table output (30s cadence)
- Aggregates to 2-minute bars
- Computes 5 canonical streams (Momentum, Flow, Barrier, Dealer, Setup)
- Computes merged streams (Pressure, Structure)
- Computes derivatives (slope, curvature, jerk)
- Outputs to gold/streams/pentaview/version={canonical_version}/

This is a separate pipeline from es_pipeline that can run in batch or real-time mode.
It depends on es_pipeline Stage 16 output existing.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from src.pipeline.core.pipeline import Pipeline
from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.stages.compute_streams import ComputeStreamsStage
from src.common.lake_paths import canonical_state_dir, date_partition

logger = logging.getLogger(__name__)


class LoadStateTableStage(BaseStage):
    """
    Load Stage 16 state table output from es_pipeline.
    
    Reads from: silver/state/es_level_state/version={canonical_version}/date=YYYY-MM-DD/*.parquet
    """
    
    @property
    def name(self) -> str:
        return "load_state_table"
    
    @property
    def required_inputs(self) -> List[str]:
        return []
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        # Get date as string (YYYY-MM-DD format)
        if isinstance(ctx.date, str):
            date_str = ctx.date
        else:
            date_str = pd.Timestamp(ctx.date).strftime('%Y-%m-%d')
        
        canonical_version = ctx.config.get('PIPELINE_CANONICAL_VERSION') or ctx.config.get('CANONICAL_VERSION', '3.1.0')
        
        # Get state table path (try versioned first, fall back to unversioned)
        data_root = ctx.config.get("DATA_ROOT")
        base_path = Path(data_root) if data_root else Path("data")
        
        state_dir_versioned = base_path / "silver" / "state" / "es_level_state" / f"version={canonical_version}" / date_partition(date_str)
        state_dir_unversioned = base_path / "silver" / "state" / "es_level_state" / date_partition(date_str)
        
        if state_dir_versioned.exists():
            state_dir = state_dir_versioned
        elif state_dir_unversioned.exists():
            state_dir = state_dir_unversioned
        else:
            logger.error(f"  State table directory not found: {state_dir_versioned} or {state_dir_unversioned}")
            logger.error("  Run es_pipeline Stage 16 first to generate state table")
            return {
                'state_df': pd.DataFrame(),
                'n_samples': 0
            }
        
        # Load all parquet files in date partition
        parquet_files = list(state_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.error(f"  No parquet files found in {state_dir}")
            return {
                'state_df': pd.DataFrame(),
                'n_samples': 0
            }
        
        logger.info(f"  Loading state table from {len(parquet_files)} files...")
        
        dfs = []
        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            dfs.append(df)
        
        state_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp
        if 'timestamp' in state_df.columns:
            state_df = state_df.sort_values('timestamp')
        
        # Add missing columns if needed
        if 'direction' not in state_df.columns:
            # Infer direction from distance_signed_atr
            state_df['direction'] = state_df['distance_signed_atr'].apply(
                lambda d: 'UP' if d < 0 else 'DOWN'
            )
        
        if 'fuel_effect_encoded' not in state_df.columns and 'fuel_effect' in state_df.columns:
            # Map fuel_effect string to encoded value
            fuel_map = {'AMPLIFY': 1, 'NEUTRAL': 0, 'DAMPEN': -1}
            state_df['fuel_effect_encoded'] = state_df['fuel_effect'].map(fuel_map).fillna(0).astype(int)
        
        if 'barrier_state_encoded' not in state_df.columns and 'barrier_state' in state_df.columns:
            # Map barrier_state string to encoded value
            barrier_map = {'STRONG_SUPPORT': 2, 'WEAK_SUPPORT': 1, 'NEUTRAL': 0, 'WEAK_RESISTANCE': -1, 'STRONG_RESISTANCE': -2, 'WEAK': 0}
            state_df['barrier_state_encoded'] = state_df['barrier_state'].map(barrier_map).fillna(0).astype(int)
        
        # Add barrier_delta_liq_log if missing (log transform of barrier_delta_liq)
        if 'barrier_delta_liq_log' not in state_df.columns and 'barrier_delta_liq' in state_df.columns:
            state_df['barrier_delta_liq_log'] = state_df['barrier_delta_liq'].apply(
                lambda x: np.sign(x) * np.log1p(np.abs(x)) if pd.notna(x) else 0.0
            )
        
        # Add wall_ratio_log if missing
        if 'wall_ratio_log' not in state_df.columns and 'wall_ratio' in state_df.columns:
            state_df['wall_ratio_log'] = state_df['wall_ratio'].apply(
                lambda x: np.log(max(x, 1e-6)) if pd.notna(x) and x > 0 else 0.0
            )
        
        logger.info(f"  Loaded {len(state_df):,} state samples")
        logger.info(f"    Timestamp range: {state_df['timestamp'].min()} to {state_df['timestamp'].max()}")
        logger.info(f"    Level kinds: {state_df['level_kind'].unique().tolist()}")
        
        return {
            'state_df': state_df,
            'n_samples': len(state_df)
        }


def build_pentaview_pipeline() -> Pipeline:
    """
    Build Pentaview stream computation pipeline.
    
    Stage sequence:
    0. LoadStateTable (from es_pipeline Stage 16 output)
    1. ComputeStreams (aggregate to 2-min bars, compute 5 streams + derivatives)
    
    Returns:
        Pipeline instance
    """
    return Pipeline(
        name="pentaview_pipeline",
        version="1.0.0",
        stages=[
            LoadStateTableStage(),
            ComputeStreamsStage(
                stream_stats_path=None,  # Use default: gold/streams/normalization/current.json
                bar_freq='30s',
                lookback_bars=40,  # 20 minutes @ 30s cadence for DCT
                output_dir='gold/streams/pentaview'
            )
        ]
    )

