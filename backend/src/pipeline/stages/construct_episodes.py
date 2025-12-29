"""Construct episode vectors - IMPLEMENTATION_READY.md Section 6 (Stage 18)."""
import logging
from typing import Any, Dict, List
from pathlib import Path
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.ml.episode_vector import construct_episodes_from_events, save_episodes
from src.ml.normalization import load_normalization_stats

logger = logging.getLogger(__name__)


class ConstructEpisodesStage(BaseStage):
    """
    Construct 111-dimensional episode vectors from events and state table.
    
    Per IMPLEMENTATION_READY.md Section 6 (Stage 18):
    - For each event (anchor), extract 5-bar history from state table
    - Construct raw 111-dim vector (5 sections: context, trajectory, history, physics, trends)
    - Normalize using precomputed statistics
    - Compute labels (outcome_2min/4min/8min) and emission weights
    - Output: vectors (npy) and metadata (parquet) partitioned by date
    
    Vector architecture:
    - Section A: Context State (26 dims)
    - Section B: Multi-Scale Trajectory (37 dims)
    - Section C: Micro-History (35 dims, 7 features × 5 bars)
    - Section D: Derived Physics (9 dims)
    - Section E: Cluster Trends (4 dims)
    
    Outputs:
        episodes_vectors: numpy array [N × 111]
        episodes_metadata: DataFrame with labels and metadata
    """
    
    def __init__(self, normalization_stats_path: str = None):
        """
        Initialize stage.
        
        Args:
            normalization_stats_path: Path to normalization stats JSON
                                     (defaults to gold/normalization/current.json)
        """
        self.normalization_stats_path = normalization_stats_path
    
    @property
    def name(self) -> str:
        return "construct_episodes"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'state_df', 'date']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        state_df = ctx.data['state_df']
        date = ctx.data.get('date', pd.Timestamp.now())
        
        n_events = len(signals_df)
        logger.info(f"  Constructing episode vectors from {n_events:,} events...")
        
        # Load normalization statistics
        if self.normalization_stats_path:
            stats_path = Path(self.normalization_stats_path)
        else:
            # Default: gold/normalization/current.json
            stats_path = Path('data/gold/normalization')
        
        try:
            normalization_stats = load_normalization_stats(stats_path)
        except FileNotFoundError:
            logger.warning("  Normalization stats not found, skipping episode construction")
            logger.warning("  Run normalization computation first (Stage 17)")
            return {
                'episodes_vectors': None,
                'episodes_metadata': None
            }
        
        # Construct episodes
        vectors, metadata = construct_episodes_from_events(
            events_df=signals_df,
            state_df=state_df,
            normalization_stats=normalization_stats
        )
        
        if len(vectors) == 0:
            logger.warning("  No episodes constructed (empty result)")
            return {
                'episodes_vectors': None,
                'episodes_metadata': None
            }
        
        # Log statistics
        logger.info(f"    Episode vectors: {vectors.shape}")
        logger.info(f"    Time bucket distribution: {metadata['time_bucket'].value_counts().to_dict()}")
        
        for horizon in ['2min', '4min', '8min']:
            col = f'outcome_{horizon}'
            if col in metadata.columns:
                dist = metadata[col].value_counts().to_dict()
                logger.info(f"    {horizon} outcomes: {dist}")
        
        avg_weight = metadata['emission_weight'].mean()
        logger.info(f"    Avg emission weight: {avg_weight:.3f}")
        
        return {
            'episodes_vectors': vectors,
            'episodes_metadata': metadata
        }

