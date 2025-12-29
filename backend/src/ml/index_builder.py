"""Index building for similarity search - IMPLEMENTATION_READY.md Section 8."""
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    logger.warning("FAISS not installed. Index building will not work.")
    faiss = None


# Constants per IMPLEMENTATION_READY.md Section 8
LEVEL_KINDS = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_200', 'SMA_400']
DIRECTIONS = ['UP', 'DOWN']
TIME_BUCKETS = ['T0_30', 'T30_60', 'T60_120', 'T120_180']
MIN_PARTITION_SIZE = 100  # Don't create index for tiny partitions


def select_index_type(n_vectors: int) -> str:
    """
    Select FAISS index type based on corpus size.
    
    Per IMPLEMENTATION_READY.md Section 8.2:
    - < 10K: IndexFlatIP (exact search)
    - 10K-100K: IndexIVFFlat (inverted file)
    - > 100K: IndexIVFPQ (product quantization)
    
    Args:
        n_vectors: Number of vectors in partition
    
    Returns:
        Index type string: 'Flat', 'IVF', or 'IVFPQ'
    """
    if n_vectors < 10_000:
        return 'Flat'
    elif n_vectors < 100_000:
        return 'IVF'
    else:
        return 'IVFPQ'


def build_faiss_index(
    vectors: np.ndarray,
    index_type: str = None
) -> 'faiss.Index':
    """
    Build FAISS index from vectors using cosine similarity.
    
    Per IMPLEMENTATION_READY.md Section 8.3:
    - L2-normalize vectors for cosine similarity via inner product
    - IndexFlatIP: Exact search
    - IndexIVFFlat: nlist = N/100, nprobe = 64
    - IndexIVFPQ: nlist = 4096, m = 8, nprobe = 64
    
    Args:
        vectors: Episode vectors [N × D]
        index_type: 'Flat', 'IVF', or 'IVFPQ' (auto-selected if None)
    
    Returns:
        FAISS index (trained and populated)
    """
    if faiss is None:
        raise ImportError("FAISS not installed. Run: uv add faiss-cpu")
    
    N, D = vectors.shape
    
    if index_type is None:
        index_type = select_index_type(N)
    
    logger.info(f"    Building {index_type} index for {N:,} vectors (dim={D})...")
    
    # L2-normalize for cosine similarity via inner product
    vectors_normalized = vectors.copy().astype(np.float32)
    faiss.normalize_L2(vectors_normalized)
    
    if index_type == 'Flat':
        # Exact search using inner product (cosine similarity)
        index = faiss.IndexFlatIP(D)
        index.add(vectors_normalized)
        logger.info(f"      Built Flat index ({N:,} vectors)")
    
    elif index_type == 'IVF':
        # Inverted file index
        nlist = min(4096, max(16, N // 100))
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors_normalized)
        index.add(vectors_normalized)
        index.nprobe = min(64, nlist // 4)
        logger.info(f"      Built IVF index (nlist={nlist}, nprobe={index.nprobe}, {N:,} vectors)")
    
    elif index_type == 'IVFPQ':
        # Product quantization for large corpora
        nlist = min(4096, max(16, N // 100))
        # m must divide D evenly. D=144, divisors: 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72
        # Use m=3 for minimal compression, or pad D to 112 for m=8
        # For now, use m=3 (37 subvectors)
        m = 3
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, m, 8)
        index.train(vectors_normalized)
        index.add(vectors_normalized)
        index.nprobe = 64
        logger.info(f"      Built IVFPQ index (nlist={nlist}, m={m}, nprobe={index.nprobe}, {N:,} vectors)")
    
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    return index


def load_episode_corpus(
    episodes_dir: Path,
    date_filter: List[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load all episode vectors and metadata from date-partitioned storage.
    
    Args:
        episodes_dir: Base directory (gold/episodes/es_level_episodes/)
        date_filter: List of dates to include (None = all dates)
    
    Returns:
        Tuple of (vectors array [N × 144], metadata DataFrame)
    """
    episodes_dir = Path(episodes_dir)
    
    vectors_dir = episodes_dir / 'vectors'
    metadata_dir = episodes_dir / 'metadata'
    
    if not vectors_dir.exists() or not metadata_dir.exists():
        raise FileNotFoundError(f"Episode directories not found: {episodes_dir}")
    
    logger.info(f"Loading episode corpus from {episodes_dir}...")
    
    # Find all date partitions
    date_dirs = sorted(vectors_dir.glob('date=*'))
    
    if date_filter:
        date_dirs = [d for d in date_dirs if d.name.split('=')[1] in date_filter]
    
    logger.info(f"  Found {len(date_dirs)} date partitions")
    
    all_vectors = []
    all_metadata = []
    
    for date_dir in date_dirs:
        date_str = date_dir.name.split('=')[1]
        
        # Load vectors
        vector_file = date_dir / 'episodes.npy'
        if not vector_file.exists():
            logger.warning(f"  Vectors not found for {date_str}, skipping")
            continue
        
        vectors = np.load(vector_file)
        all_vectors.append(vectors)
        
        # Load metadata
        metadata_file = metadata_dir / f'date={date_str}' / 'metadata.parquet'
        if not metadata_file.exists():
            logger.warning(f"  Metadata not found for {date_str}, skipping")
            continue
        
        metadata = pd.read_parquet(metadata_file)
        all_metadata.append(metadata)
    
    if not all_vectors:
        raise ValueError("No episode data found")
    
    # Concatenate all
    corpus_vectors = np.vstack(all_vectors)
    corpus_metadata = pd.concat(all_metadata, ignore_index=True)
    
    logger.info(f"  Loaded {len(corpus_vectors):,} episodes from {len(date_dirs)} dates")
    
    return corpus_vectors, corpus_metadata


def build_all_indices(
    episodes_dir: Path,
    output_dir: Path,
    date_filter: List[str] = None,
    min_partition_size: int = MIN_PARTITION_SIZE
) -> Dict[str, Any]:
    """
    Build FAISS indices for all 48 partitions.
    
    Per IMPLEMENTATION_READY.md Section 8.4:
    - Partition by (level_kind, direction, time_bucket) = 6 × 2 × 4 = 48
    - Build index per partition
    - Save index.faiss, vectors.npy, metadata.parquet per partition
    - Skip partitions with < min_partition_size vectors
    
    Args:
        episodes_dir: Episode corpus directory (gold/episodes/es_level_episodes/)
        output_dir: Index output directory (gold/indices/es_level_indices/)
        date_filter: Optional list of dates to include
        min_partition_size: Skip partitions smaller than this
    
    Returns:
        Dict with build statistics
    """
    if faiss is None:
        raise ImportError("FAISS not installed. Run: uv add faiss-cpu")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building FAISS indices for all partitions...")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Min partition size: {min_partition_size}")
    
    # Load full corpus
    corpus_vectors, corpus_metadata = load_episode_corpus(episodes_dir, date_filter)
    
    build_stats = {
        'timestamp': datetime.now().isoformat(),
        'n_total_episodes': len(corpus_vectors),
        'n_partitions_built': 0,
        'n_partitions_skipped': 0,
        'partitions': {}
    }
    
    # Build indices for each partition
    for level_kind in LEVEL_KINDS:
        for direction in DIRECTIONS:
            for time_bucket in TIME_BUCKETS:
                partition_key = f"{level_kind}/{direction}/{time_bucket}"
                
                # Filter to partition
                mask = (
                    (corpus_metadata['level_kind'] == level_kind) &
                    (corpus_metadata['direction'] == direction) &
                    (corpus_metadata['time_bucket'] == time_bucket)
                )
                
                partition_vectors = corpus_vectors[mask]
                partition_metadata = corpus_metadata[mask].copy()
                
                n_partition = len(partition_vectors)
                
                if n_partition < min_partition_size:
                    logger.info(f"  Skipping {partition_key}: only {n_partition} vectors (< {min_partition_size})")
                    build_stats['n_partitions_skipped'] += 1
                    continue
                
                logger.info(f"  Building {partition_key}: {n_partition:,} vectors")
                
                # Create partition directory
                partition_dir = output_dir / level_kind / direction / time_bucket
                partition_dir.mkdir(parents=True, exist_ok=True)
                
                # Build index
                try:
                    index = build_faiss_index(partition_vectors)
                    
                    # Save index
                    index_file = partition_dir / 'index.faiss'
                    faiss.write_index(index, str(index_file))
                    
                    # Save vectors
                    vectors_file = partition_dir / 'vectors.npy'
                    np.save(vectors_file, partition_vectors)
                    
                    # Save metadata
                    metadata_file = partition_dir / 'metadata.parquet'
                    partition_metadata.to_parquet(metadata_file, index=False)
                    
                    build_stats['partitions'][partition_key] = {
                        'n_vectors': n_partition,
                        'index_type': select_index_type(n_partition),
                        'outcome_dist': partition_metadata['outcome_4min'].value_counts().to_dict()
                    }
                    
                    build_stats['n_partitions_built'] += 1
                    
                except Exception as e:
                    logger.error(f"    Failed to build {partition_key}: {e}")
                    build_stats['n_partitions_skipped'] += 1
    
    # Save config
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(build_stats, f, indent=2)
    
    logger.info(f"Index building complete:")
    logger.info(f"  Built: {build_stats['n_partitions_built']} partitions")
    logger.info(f"  Skipped: {build_stats['n_partitions_skipped']} partitions")
    logger.info(f"  Config: {config_file}")
    
    return build_stats


class BuildIndicesStage:
    """
    Build FAISS indices for similarity search.
    
    Per IMPLEMENTATION_READY.md Section 8 (Stage 19):
    - Loads all episode vectors from gold/episodes/
    - Partitions by (level_kind, direction, time_bucket) = 48 indices
    - Builds appropriate FAISS index per partition size
    - Saves index.faiss, vectors.npy, metadata.parquet per partition
    - Runs daily at 17:00 ET (after episode construction)
    
    This is typically run offline/scheduled, not in the main pipeline.
    """
    
    def __init__(
        self,
        episodes_dir: Path,
        output_dir: Path,
        min_partition_size: int = MIN_PARTITION_SIZE
    ):
        self.episodes_dir = Path(episodes_dir)
        self.output_dir = Path(output_dir)
        self.min_partition_size = min_partition_size
    
    def execute(self, date_filter: List[str] = None) -> Dict[str, Any]:
        """
        Execute index building.
        
        Args:
            date_filter: Optional list of dates to include (None = all)
        
        Returns:
            Dict with build statistics
        """
        return build_all_indices(
            episodes_dir=self.episodes_dir,
            output_dir=self.output_dir,
            date_filter=date_filter,
            min_partition_size=self.min_partition_size
        )

