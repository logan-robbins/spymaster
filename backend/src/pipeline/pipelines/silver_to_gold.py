"""
Silver → Gold Pipeline: Episode Construction

Transforms Silver features into Gold episode vectors for production retrieval.

Stages 0-1 (renumbered from original 17-18):
- Materializes 30s state table from RTH-filtered signals
- Constructs 149D episode vectors with DCT trajectory basis
- Generates raw 40×4 sequences for Transformer training

Input: Silver features from bronze_to_silver pipeline
  silver/features/es_pipeline/version={version}/date=YYYY-MM-DD/signals.parquet

Output: Gold episodes written to:
  gold/episodes/es_level_episodes/version={version}/vectors/date=YYYY-MM-DD/episodes.npy
  gold/episodes/es_level_episodes/version={version}/metadata/date=YYYY-MM-DD/metadata.parquet
  gold/episodes/es_level_episodes/version={version}/sequences/date=YYYY-MM-DD/sequences.npy

Consumers: FAISS index builder, ML training (Phase 5 Transformers)
"""

from src.pipeline.core.pipeline import Pipeline
from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.stages.materialize_state_table import MaterializeStateTableStage
from src.pipeline.stages.construct_episodes import ConstructEpisodesStage


class LoadSilverFeaturesStage(BaseStage):
    """Load Silver features from parquet for Gold episode construction."""
    
    name = "load_silver_features"
    required_inputs = []
    
    def execute(self, ctx: StageContext) -> dict:
        """Load Silver features for the given date."""
        import pandas as pd
        from pathlib import Path
        
        data_root = ctx.config.get("DATA_ROOT", "data")
        canonical_version = ctx.config.get("PIPELINE_CANONICAL_VERSION", "4.5.0")
        date = ctx.date
        
        # Path to Silver features
        silver_path = Path(data_root) / "silver" / "features" / "es_pipeline" / f"version={canonical_version}" / f"date={date}" / "signals.parquet"
        
        if not silver_path.exists():
            raise FileNotFoundError(f"Silver features not found: {silver_path}")
        
        signals_df = pd.read_parquet(silver_path)
        
        # Also load OHLCV for state table construction
        # This is a bit awkward - we need to re-load Bronze data
        # Alternative: Could save OHLCV to Silver or load from checkpoint
        from src.io.bronze import BronzeReader
        reader = BronzeReader(data_root=data_root)
        from src.pipeline.stages.load_bronze import futures_trades_from_df
        trades_df = reader.read_futures_trades('ES', date)
        trades_objs = futures_trades_from_df(trades_df)
        
        from src.pipeline.stages.build_spx_ohlcv import build_spx_ohlcv_from_es
        ohlcv_1min = build_spx_ohlcv_from_es(trades_objs, date=date, freq='1min')
        ohlcv_2min = build_spx_ohlcv_from_es(trades_objs, date=date, freq='2min')
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded Silver features: {len(signals_df)} signals")
        
        return {
            'signals_df': signals_df,
            'signals': signals_df,  # For pipeline.run() return
            'ohlcv_1min': ohlcv_1min,
            'ohlcv_2min': ohlcv_2min,
            'date': date
        }


def build_silver_to_gold_pipeline() -> Pipeline:
    """
    Build Silver → Gold pipeline (episode construction).
    
    Stage sequence (0-indexed, stages 0-1):
    0. LoadSilverFeatures (read from Silver layer)
    1. MaterializeStateTable (30s cadence state for episode construction)
    2. ConstructEpisodes (149D vectors with DCT trajectory basis)
    
    Input:
        silver/features/es_pipeline/version={version}/date=YYYY-MM-DD/signals.parquet
    
    Output:
        gold/episodes/es_level_episodes/version={version}/
        ├── vectors/date=YYYY-MM-DD/episodes.npy (149D)
        ├── metadata/date=YYYY-MM-DD/metadata.parquet
        └── sequences/date=YYYY-MM-DD/sequences.npy (40×4)
    
    Returns:
        Pipeline instance
    """
    return Pipeline(
        name="silver_to_gold",
        version="4.5.0",
        stages=[
            LoadSilverFeaturesStage(),
            MaterializeStateTableStage(),
            ConstructEpisodesStage()
        ]
    )

