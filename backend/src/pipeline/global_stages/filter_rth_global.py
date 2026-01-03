"""
Stage: Filter RTH Global (Global Pipeline)
Type: Data Filtering & Schema Enforcement
Input: Signals DataFrame (Global Features)
Output: Canonical Silver Signals (Global, Filtered)

Transformation:
1. Filters events to the Training Window (08:30 - 12:30 ET).
   - Ensures time grid aligns with level-based events.
2. Drops intermediate timestamp columns.
3. Writes the Global Signals to the Silver Data Lake (partitioned by Date/Market).
   - This dataset provides the "System/Market State" needed by the Gold Layer.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Set
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG
from src.common.data_paths import canonical_signals_dir, date_partition
from src.pipeline.core.feature_definitions import (
    IDENTITY_FEATURES,
    MARKET_STATE_FEATURES,
    GLOBAL_OPTIONS_FEATURES,
    GLOBAL_OFI_FEATURES,
    GLOBAL_KINEMATICS_FEATURES,
    GLOBAL_MICRO_FEATURES,
    GLOBAL_WALL_FEATURES,
)

logger = logging.getLogger(__name__)


class FilterRTHGlobalStage(BaseStage):
    """
    Filter global market features to training window.
    
    Output schema includes:
    - Identity: event_id, ts_ns, timestamp, date
    - Session context: minutes_since_open, bars_since_open, or_active
    - Market state: spot, atr, volatility
    - Microstructure: spread, bid_depth, ask_depth, depth_imbalance
    - OFI: ofi_30s, ofi_60s, ofi_120s, ofi_300s
    - Kinematics: velocity_*, acceleration_*, jerk_*
    - Options: total_gex, call_tide, put_tide, put_call_ratio
    """
    
    @property
    def name(self) -> str:
        return "filter_rth_global"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        
        if signals_df.empty:
            return {
                'signals': signals_df,
                'signals_df': signals_df,
                'signals_output_path': None,
            }
        
        # Training window: 08:30-12:30 ET
        session_start = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(
            hours=CONFIG.TRAINING_START_HOUR, minutes=CONFIG.TRAINING_START_MINUTE
        )
        session_end = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(
            hours=CONFIG.TRAINING_END_HOUR, minutes=CONFIG.TRAINING_END_MINUTE
        )
        session_start_ns = session_start.tz_convert("UTC").value
        session_end_ns = session_end.tz_convert("UTC").value
        
        rth_mask = (
            (signals_df["ts_ns"] >= session_start_ns) &
            (signals_df["ts_ns"] <= session_end_ns)
        )
        signals_df = signals_df.loc[rth_mask].copy()
        
        # Drop intermediate columns
        cols_to_drop = ['timestamp_dt', 'time_et']
        signals_df = signals_df.drop(
            columns=[c for c in cols_to_drop if c in signals_df.columns]
        )
        
        logger.info(f"  Filtered to {len(signals_df)} events in training window")
        
        # Validate output schema
        required_cols = set(IDENTITY_FEATURES) | {'spot'} # Validates Identity + Spot
        # We expect at least some features from each category
        
        missing = [c for c in required_cols if c not in signals_df.columns]
        if missing:
            logger.warning(f"  Missing minimal required columns: {missing}")
            
        # Optional: Check feature groups coverage
        def check_coverage(name, expected, actual):
            present = [c for c in expected if c in actual]
            if not present:
                logger.warning(f"  Missing ALL {name} features! Expected e.g. {expected[:3]}")
            else:
                logger.debug(f"  Found {len(present)}/{len(expected)} {name} features")

        check_coverage("Options", GLOBAL_OPTIONS_FEATURES, signals_df.columns)
        check_coverage("OFI", GLOBAL_OFI_FEATURES, signals_df.columns)
        check_coverage("Microstructure", GLOBAL_MICRO_FEATURES, signals_df.columns)
        check_coverage("Kinematics", GLOBAL_KINEMATICS_FEATURES, signals_df.columns)
        check_coverage("Walls", GLOBAL_WALL_FEATURES, signals_df.columns)
        
        # Summary stats
        logger.info(f"  Output columns: {len(signals_df.columns)}")
        if 'ofi_60s' in signals_df.columns:
            logger.info(f"  OFI_60s range: {signals_df['ofi_60s'].min():.0f} to {signals_df['ofi_60s'].max():.0f}")
        if 'spread' in signals_df.columns:
            logger.info(f"  Spread range: {signals_df['spread'].min():.4f} to {signals_df['spread'].max():.4f}")
        
        # Optional: persist to Silver
        signals_output_path = None
        if ctx.config.get("PIPELINE_WRITE_SIGNALS"):
            data_root = ctx.config.get("DATA_ROOT")
            canonical_version = ctx.config.get("PIPELINE_CANONICAL_VERSION")
            if data_root and canonical_version:
                base_dir = canonical_signals_dir(data_root, dataset="es_global", version=canonical_version)
                level_dir = base_dir / date_partition(ctx.date) / "market"
                
                if ctx.config.get("PIPELINE_OVERWRITE_PARTITIONS", True) and level_dir.exists():
                    shutil.rmtree(level_dir)
                level_dir.mkdir(parents=True, exist_ok=True)
                
                signals_output_path = level_dir / "signals.parquet"
                signals_df.to_parquet(
                    signals_output_path,
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )
                logger.info(f"  Wrote global signals: {signals_output_path}")
        
        return {
            'signals': signals_df,
            'signals_df': signals_df,
            'signals_output_path': str(signals_output_path) if signals_output_path else None,
        }

