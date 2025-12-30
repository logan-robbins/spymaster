"""Construct episode vectors - IMPLEMENTATION_READY.md Section 6 (Stage 17)."""
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
    Construct 144-dimensional episode vectors with DCT trajectory basis.
    
    Updated to Analyst Opinion specification (Dec 2025):
    - For each event (anchor), extract 5-bar micro-history from state table
    - Extract 40-bar (20-minute) trajectory window for DCT computation
    - Construct raw 144-dim vector (6 sections with DCT trajectory basis)
    - Normalize using precomputed statistics
    - Compute labels (outcome_2min/4min/8min) and emission weights
    - Output: vectors (npy) and metadata (parquet) partitioned by date
    
    Vector architecture (144D):
    - Section A: Context + Regime (25 dims) - removed redundant encodings
    - Section B: Multi-Scale Dynamics (37 dims)
    - Section C: Micro-History (35 dims, 7 features × 5 bars, LOG-TRANSFORMED)
    - Section D: Derived Physics (11 dims) - added mass_proxy, force_proxy, flow_alignment
    - Section E: Online Trends (4 dims)
    - Section F: Trajectory Basis (32 dims) - 4 series × 8 DCT coefficients
    
    Outputs:
        episodes_vectors: numpy array [N × 144]
        episodes_metadata: DataFrame with labels and metadata (5 time buckets)
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
        return ['signals_df', 'state_df']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        state_df = ctx.data['state_df']
        date = pd.Timestamp(ctx.date)
        
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
