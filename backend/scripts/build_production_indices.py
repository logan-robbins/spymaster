
"""
Script to build production FAISS indices for SimilarityQueryEngine.
Uses existing episode data to populate data/gold/indices/es_level_indices.
"""
import logging
import sys
from pathlib import Path

# Add backend to path
import os
sys.path.append(os.getcwd())

from src.ml.index_builder import build_all_indices

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Define paths
    backend_root = Path(os.getcwd())
    data_root = backend_root / "data"
    
    # Input: Episode corpus
    episodes_dir = data_root / "gold/episodes/es_level_episodes/version=3.1.0"
    
    # Output: Indices
    output_dir = data_root / "gold/indices/es_level_indices"
    
    print(f"Build Production Indices")
    print(f"------------------------")
    print(f"Input:  {episodes_dir}")
    print(f"Output: {output_dir}")
    
    if not episodes_dir.exists():
        print(f"❌ Error: Episodes directory not found at {episodes_dir}")
        print("Please ensure episode generation (Step 18) has run.")
        sys.exit(1)
        
    print("\nStarting build process...")
    
    stats = build_all_indices(
        episodes_dir=episodes_dir,
        output_dir=output_dir,
        min_partition_size=50, # Lower threshold to ensure we get indices even for small days
        overwrite_output_dir=True
    )
    
    print("\n✅ Build complete!")
    print(f"Built {stats['n_partitions_built']} partitions.")
    print(f"Skipped {stats['n_partitions_skipped']} partitions.")

if __name__ == "__main__":
    main()
