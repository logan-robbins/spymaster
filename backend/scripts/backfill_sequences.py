import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.ml.episode_vector import construct_episodes_from_events, save_episodes
from src.ml.normalization import load_normalization_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")
VERSION = "3.1.0"

def backfill_date(date_str):
    logger.info(f"Processing {date_str}...")
    
    # Paths
    signals_path = DATA_ROOT / f"silver/features/es_pipeline/version={VERSION}/date={date_str}/signals.parquet"
    state_path = DATA_ROOT / f"silver/state/es_level_state/version={VERSION}/date={date_str}/state.parquet"
    
    if not signals_path.exists() or not state_path.exists():
        logger.warning(f"  Missing data for {date_str}, skipping.")
        return

    # Load Data
    signals_df = pd.read_parquet(signals_path)
    state_df = pd.read_parquet(state_path)
    
    # Load Stats
    stats_path = DATA_ROOT / "gold/normalization"
    stats = load_normalization_stats(stats_path)
    
    # Construct
    vectors, metadata, sequences = construct_episodes_from_events(
        events_df=signals_df,
        state_df=state_df,
        normalization_stats=stats
    )
    
    if len(vectors) == 0:
        logger.warning("  No vectors constructed.")
        return

    # Save
    output_dir = DATA_ROOT / f"gold/episodes/es_level_episodes/version={VERSION}"
    save_episodes(
        vectors=vectors,
        metadata=metadata,
        output_dir=output_dir,
        date=pd.Timestamp(date_str),
        sequences=sequences
    )

def main():
    # Iterate over existing dates in silver/features
    base_dir = DATA_ROOT / f"silver/features/es_pipeline/version={VERSION}"
    dates = sorted([d.name.split('=')[1] for d in base_dir.glob("date=*")])
    
    logger.info(f"Found {len(dates)} dates to backfill.")
    
    for d in dates:
        backfill_date(d)

if __name__ == "__main__":
    main()
