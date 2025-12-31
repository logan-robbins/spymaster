"""Retrieval pipeline - IMPLEMENTATION_READY.md Section 9."""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque, Counter
import numpy as np
import pandas as pd

from src.ml.constants import M_CANDIDATES, K_NEIGHBORS, MAX_PER_DAY, MAX_PER_EPISODE
from src.ml.episode_vector import (
    construct_episode_vector,
    get_feature_names,
    assign_time_bucket,
    compute_emission_weight
)
from src.ml.normalization import normalize_vector
from src.ml.outcome_aggregation import aggregate_query_results
from src.ml.attribution import compute_attribution
from src.ml.vector_compressor import VectorCompressor

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    logger.warning("FAISS not installed")
    faiss = None


@dataclass
class EpisodeQuery:
    """Query structure for episode retrieval."""
    level_kind: str
    level_price: float
    direction: str
    time_bucket: str
    vector: np.ndarray
    emission_weight: float
    timestamp: pd.Timestamp
    metadata: Dict[str, Any]



@dataclass
class QueryResult:
    """Result structure from similarity query."""
    outcome_probabilities: Dict[str, Any]
    context_metrics: Dict[str, Any]  # [NEW] Glass Box Metrics
    confidence_intervals: Dict[str, Any]
    multi_horizon: Dict[str, Any]
    attribution: Dict[str, Any]
    reliability: Dict[str, Any]
    neighbors: List[Dict[str, Any]]
    query_metadata: Dict[str, Any]


class IndexManager:
    """
    Manage FAISS indices for all partitions.
    
    Per IMPLEMENTATION_READY.md Section 8.5:
    - Lazy-loads indices per partition
    - Caches loaded indices in memory
    - Provides query interface
    """
    
    def __init__(self, index_dir: Path):
        """
        Initialize index manager.
        
        Args:
            index_dir: Base index directory (gold/indices/es_level_indices/)
        """
        if faiss is None:
            raise ImportError("FAISS not installed. Run: uv add faiss-cpu")
        
        self.index_dir = Path(index_dir)
        self.indices = {}      # {partition_key: FAISSIndex}
        self.metadata = {}     # {partition_key: DataFrame}
        self.vectors = {}      # {partition_key: ndarray}
        self.compressor: Optional[VectorCompressor] = None
        
        # Load compressor if available
        compressor_path = self.index_dir / 'compressor.pkl'
        if compressor_path.exists():
            try:
                self.compressor = VectorCompressor.load(compressor_path)
                logger.info(f"Loaded Vector Compressor: {self.compressor.strategy}")
            except Exception as e:
                logger.error(f"Failed to load compressor: {e}")
    
    def load_partition(
        self,
        level_kind: str,
        direction: str,
        time_bucket: str
    ) -> bool:
        """
        Load partition index, vectors, and metadata.
        
        Args:
            level_kind: Level kind
            direction: UP or DOWN
            time_bucket: Time bucket (T0_30, T30_60, etc.)
        
        Returns:
            True if loaded successfully, False if partition doesn't exist
        """
        key = f"{level_kind}/{direction}/{time_bucket}"
        
        if key in self.indices:
            # Already loaded
            return True
        
        partition_dir = self.index_dir / level_kind / direction / time_bucket
        
        if not partition_dir.exists():
            logger.warning(f"Partition not found: {key}")
            return False
        
        index_file = partition_dir / 'index.faiss'
        vectors_file = partition_dir / 'vectors.npy'
        metadata_file = partition_dir / 'metadata.parquet'
        
        if not all([f.exists() for f in [index_file, vectors_file, metadata_file]]):
            logger.warning(f"Incomplete partition: {key}")
            return False
        
        try:
            # Load index
            self.indices[key] = faiss.read_index(str(index_file))
            
            # Load vectors
            self.vectors[key] = np.load(vectors_file)
            
            # Load metadata
            self.metadata[key] = pd.read_parquet(metadata_file)
            
            logger.info(f"Loaded partition {key}: {len(self.vectors[key]):,} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load partition {key}: {e}")
            return False
    
    def query(
        self,
        level_kind: str,
        direction: str,
        time_bucket: str,
        query_vector: np.ndarray,
        k: int = 50
    ) -> Dict[str, Any]:
        """
        Query a partition index.
        
        Per IMPLEMENTATION_READY.md Section 8.5:
        - Load partition if not cached
        - Normalize query vector
        - Search index
        - Return similarities, indices, metadata, and vectors
        
        Args:
            level_kind: Level kind
            direction: Direction
            time_bucket: Time bucket
            query_vector: Query vector (raw or normalized)
            k: Number of neighbors to retrieve
        
        Returns:
            Dict with similarities, indices, metadata, and vectors
        """
        key = f"{level_kind}/{direction}/{time_bucket}"
        
        # Load partition if needed
        if key not in self.indices:
            if not self.load_partition(level_kind, direction, time_bucket):
                return {
                    'similarities': np.array([]),
                    'indices': np.array([]),
                    'metadata': pd.DataFrame(),
                    'vectors': None
                }
        
        # Normalize query vector for cosine similarity
        query = query_vector.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        try:
            similarities, indices = self.indices[key].search(query, k)
            similarities = similarities[0]
            indices = indices[0]
            
            # Filter invalid indices
            valid_mask = indices >= 0
            similarities = similarities[valid_mask]
            indices = indices[valid_mask]
            
            # Get metadata and vectors
            retrieved_metadata = self.metadata[key].iloc[indices].copy()
            retrieved_metadata['similarity'] = similarities
            
            retrieved_vectors = self.vectors[key][indices] if key in self.vectors else None
            
            return {
                'similarities': similarities,
                'indices': indices,
                'metadata': retrieved_metadata,
                'vectors': retrieved_vectors
            }
        
        except Exception as e:
            logger.error(f"Query failed for {key}: {e}")
            return {
                'similarities': np.array([]),
                'indices': np.array([]),
                'metadata': pd.DataFrame(),
                'vectors': None
            }


class LiveEpisodeBuilder:
    """
    Build episode queries from live state updates.
    
    Per IMPLEMENTATION_READY.md Section 9.1:
    - Maintains 5-bar history buffers per level
    - Emits queries when in approach zone
    - Constructs and normalizes vectors
    """

    def __init__(
        self,
        normalizer_stats: Dict[str, Any],
        state_cadence_seconds: int = 30,
        zone_threshold_atr: float = 2.0,  # Updated: compromise between 3.0 (original) and 1.25 (analyst)
        min_approach_velocity: float = 0.5
    ):
        """
        Initialize episode builder.
        
        Args:
            normalizer_stats: Normalization statistics
            state_cadence_seconds: State table cadence (30s)
            zone_threshold_atr: Distance threshold for zone entry
            min_approach_velocity: Minimum velocity to trigger query
        """
        self.normalizer_stats = normalizer_stats
        self.cadence = state_cadence_seconds
        self.zone_threshold = zone_threshold_atr
        self.min_velocity = min_approach_velocity
        self.buffers = {}  # {(level_kind, level_price): deque}
        self.buffer_size = 5
        self.feature_names = get_feature_names()
    
    def on_state_update(
        self,
        state_row: Dict[str, Any]
    ) -> List[EpisodeQuery]:
        """
        Process state update and emit queries if conditions met.
        
        Per IMPLEMENTATION_READY.md Section 9.1
        
        Args:
            state_row: State table row dict
        
        Returns:
            List of episode queries (may be empty)
        """
        queries = []
        
        level_kind = state_row.get('level_kind')
        level_price = state_row.get('level_price')
        level_key = (level_kind, level_price)
        
        # Initialize buffer if needed
        if level_key not in self.buffers:
            self.buffers[level_key] = deque(maxlen=self.buffer_size)
        
        # Add to buffer
        self.buffers[level_key].append(state_row)
        
        # Check if in approach zone
        distance_atr = abs(state_row.get('distance_signed_atr', 999))
        in_zone = distance_atr < self.zone_threshold
        
        # Check approach velocity
        approach_velocity = abs(state_row.get('approach_velocity', 0))
        has_velocity = approach_velocity > self.min_velocity
        
        # Emit query if conditions met
        if len(self.buffers[level_key]) >= self.buffer_size and in_zone and has_velocity:
            
            # Build vector
            try:
                raw_vector = construct_episode_vector(
                    current_bar=state_row,
                    history_buffer=list(self.buffers[level_key]),
                    level_price=level_price
                )
                
                # Normalize
                normalized_vector = normalize_vector(
                    raw_vector=raw_vector,
                    feature_names=self.feature_names,
                    stats=self.normalizer_stats
                )
                
                # Determine direction
                spot = state_row.get('spot', level_price)
                direction = 'UP' if spot < level_price else 'DOWN'
                
                # Assign time bucket
                minutes_since_open = state_row.get('minutes_since_open', 0)
                time_bucket = assign_time_bucket(minutes_since_open)
                
                # Compute emission weight
                emission_weight = compute_emission_weight(
                    spot=spot,
                    level_price=level_price,
                    atr=state_row.get('atr', 1.0),
                    approach_velocity=state_row.get('approach_velocity', 0),
                    ofi_60s=state_row.get('ofi_60s', 0)
                )
                
                queries.append(EpisodeQuery(
                    level_kind=level_kind,
                    level_price=level_price,
                    direction=direction,
                    time_bucket=time_bucket,
                    vector=normalized_vector,
                    emission_weight=emission_weight,
                    timestamp=state_row.get('timestamp'),
                    metadata={
                        'spot': spot,
                        'atr': state_row.get('atr', 1.0),
                        'minutes_since_open': minutes_since_open
                    }
                ))
            
            except Exception as e:
                logger.warning(f"Failed to build query for {level_key}: {e}")
        
        return queries


class SimilarityQueryEngine:
    """
    Execute similarity queries and compute outcomes.
    
    Per IMPLEMENTATION_READY.md Section 9.2:
    - Query appropriate partition
    - Retrieve M_CANDIDATES (500) from FAISS
    - Apply deduplication (max 2/day, max 1/episode)
    - Return top K_NEIGHBORS (50)
    - Compute outcome distributions
    - Return QueryResult
    """
    
    def __init__(
        self,
        index_manager: IndexManager,
        m_candidates: int = M_CANDIDATES,
        k_neighbors: int = K_NEIGHBORS,
        max_per_day: int = MAX_PER_DAY,
        max_per_episode: int = MAX_PER_EPISODE
    ):
        """
        Initialize query engine.
        
        Args:
            index_manager: IndexManager instance
            m_candidates: Number to retrieve from FAISS (over-fetch for dedup)
            k_neighbors: Final number of neighbors to return after dedup
            max_per_day: Maximum neighbors from same date
            max_per_episode: Maximum neighbors from same episode_id
        """
        self.index_manager = index_manager
        self.m_candidates = m_candidates
        self.k_neighbors = k_neighbors
        self.max_per_day = max_per_day
        self.max_per_episode = max_per_episode

    def query(
        self,
        episode_query: EpisodeQuery,
        filters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Execute similarity query with deduplication.
        
        Per IMPLEMENTATION_READY.md Section 9.2:
        - Retrieve M_CANDIDATES (500) from FAISS
        - Apply deduplication constraints (max 2/day, 1/episode)
        - Apply optional filters
        - Return top K_NEIGHBORS (50)
        
        Args:
            episode_query: Episode query structure
            filters: Optional metadata filters
        
        Returns:
            QueryResult with outcome distributions and neighbors
        """
        # Apply Vector Compression if configured (Phase 4)
        query_vector = episode_query.vector
        if self.index_manager.compressor:
            # Transform expects 2D array (samples, features)
            q_reshaped = query_vector.reshape(1, -1)
            query_vector = self.index_manager.compressor.transform(q_reshaped).flatten()

        # Retrieve M_CANDIDATES from appropriate partition
        result = self.index_manager.query(
            level_kind=episode_query.level_kind,
            direction=episode_query.direction,
            time_bucket=episode_query.time_bucket,
            query_vector=query_vector,
            k=self.m_candidates
        )
        
        retrieved_metadata = result['metadata']
        retrieved_vectors = result['vectors']
        
        if len(retrieved_metadata) == 0:
            return self._empty_result(episode_query)
        
        # Apply deduplication constraints
        retrieved_metadata = self._apply_deduplication(retrieved_metadata)
        
        if retrieved_vectors is not None and len(retrieved_metadata) < len(result['metadata']):
            # Update vectors to match deduplicated metadata
            dedup_indices = retrieved_metadata.index.tolist()
            original_indices = result['metadata'].index.tolist()
            mask = [i for i, idx in enumerate(original_indices) if idx in dedup_indices]
            retrieved_vectors = retrieved_vectors[mask]
        
        # Apply optional filters
        if filters:
            mask = np.ones(len(retrieved_metadata), dtype=bool)
            for key, value in filters.items():
                if key in retrieved_metadata.columns:
                    mask &= (retrieved_metadata[key] == value)
            retrieved_metadata = retrieved_metadata[mask]
            if retrieved_vectors is not None:
                retrieved_vectors = retrieved_vectors[mask]
        
        # Take top K_NEIGHBORS
        retrieved_metadata = retrieved_metadata.head(self.k_neighbors)
        if retrieved_vectors is not None:
            retrieved_vectors = retrieved_vectors[:self.k_neighbors]
        
        if len(retrieved_metadata) == 0:
            return self._empty_result(episode_query)
        
        # Compute outcome aggregations (Section 10) with query_date for recency weighting
        aggregated = aggregate_query_results(
            retrieved_metadata=retrieved_metadata,
            query_date=episode_query.timestamp,
            compute_ci=True,
            n_bootstrap=1000
        )
        
        # Compute attribution (Section 11)
        if retrieved_vectors is not None and len(retrieved_vectors) > 0:
            attribution = compute_attribution(
                query_vector=episode_query.vector,
                retrieved_vectors=retrieved_vectors,
                outcomes=retrieved_metadata['outcome_4min'].values,
                similarities=retrieved_metadata['similarity'].values
            )
        else:
            attribution = {}
        
        return QueryResult(
            outcome_probabilities=aggregated['outcome_probabilities'],
            context_metrics=aggregated.get('context_metrics', {}),
            confidence_intervals=aggregated['confidence_intervals'],
            multi_horizon=aggregated['multi_horizon'],
            attribution=attribution,
            reliability=aggregated['reliability'],
            neighbors=retrieved_metadata.to_dict('records'),
            query_metadata={
                'level_kind': episode_query.level_kind,
                'direction': episode_query.direction,
                'time_bucket': episode_query.time_bucket,
                'timestamp': episode_query.timestamp,
                'emission_weight': episode_query.emission_weight
            }
        )
    
    def _apply_deduplication(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Apply deduplication constraints to retrieved neighbors.
        
        Per IMPLEMENTATION_READY.md Section 9.2:
        - Maximum MAX_PER_DAY (2) neighbors from same date
        - Maximum MAX_PER_EPISODE (1) neighbor from same episode
        - Preserves similarity ordering
        
        Args:
            metadata: Retrieved metadata DataFrame (sorted by similarity desc)
        
        Returns:
            Deduplicated DataFrame
        """
        if len(metadata) == 0:
            return metadata
        
        date_counts = Counter()
        episode_counts = Counter()
        keep_indices = []
        
        for idx, row in metadata.iterrows():
            date = row.get('date')
            event_id = row.get('event_id')
            
            # Extract episode_id from event_id (format: "date_level_dir_attempt")
            # Or use event_id directly if no episode grouping exists
            episode_id = event_id
            
            # Check constraints
            date_ok = (date is None) or (date_counts[date] < self.max_per_day)
            episode_ok = (episode_id is None) or (episode_counts[episode_id] < self.max_per_episode)
            
            if date_ok and episode_ok:
                keep_indices.append(idx)
                if date is not None:
                    date_counts[date] += 1
                if episode_id is not None:
                    episode_counts[episode_id] += 1
        
        return metadata.loc[keep_indices]
    
    def _empty_result(self, episode_query: EpisodeQuery) -> QueryResult:
        """Return empty result structure."""
        return QueryResult(
            outcome_probabilities={'probabilities': {}, 'n_samples': 0},
            context_metrics={},
            confidence_intervals={},
            multi_horizon={},
            attribution={},
            reliability={'n_retrieved': 0},
            neighbors=[],
            query_metadata={
                'level_kind': episode_query.level_kind,
                'direction': episode_query.direction,
                'time_bucket': episode_query.time_bucket,
                'timestamp': episode_query.timestamp
            }
        )


class RealTimeQueryService:
    """
    Real-time query service with caching.
    
    Per IMPLEMENTATION_READY.md Section 9.3:
    - Integrates LiveEpisodeBuilder and SimilarityQueryEngine
    - Caches recent results (30s TTL)
    - Filters low-quality results
    """
    
    def __init__(
        self,
        normalizer_stats: Dict[str, Any],
        index_manager: IndexManager,
        cache_ttl_seconds: int = 30,
        min_similarity_threshold: float = 0.70,
        min_samples_threshold: int = 30
    ):
        """
        Initialize real-time service.
        
        Args:
            normalizer_stats: Normalization statistics
            index_manager: IndexManager instance
            cache_ttl_seconds: Cache TTL in seconds
            min_similarity_threshold: Minimum avg similarity for quality
            min_samples_threshold: Minimum neighbors for reliability
        """
        self.episode_builder = LiveEpisodeBuilder(normalizer_stats)
        self.query_engine = SimilarityQueryEngine(index_manager)
        self.result_cache = {}  # {level_key: (timestamp, result)}
        self.cache_ttl = cache_ttl_seconds
        self.min_similarity = min_similarity_threshold
        self.min_samples = min_samples_threshold
    
    def process_state_update(
        self,
        state_row: Dict[str, Any]
    ) -> List[QueryResult]:
        """
        Main entry point: process state update and return query results.
        
        Per IMPLEMENTATION_READY.md Section 9.3
        
        Args:
            state_row: State table row dict
        
        Returns:
            List of QueryResults (may be empty)
        """
        results = []
        
        # Build episode queries
        queries = self.episode_builder.on_state_update(state_row)
        
        for query in queries:
            level_key = (query.level_kind, query.level_price)
            
            # Check cache
            if self._is_cached(level_key, query.timestamp):
                    continue
            
            # Execute query
            result = self.query_engine.query(query)
            
            # Filter low-quality results
            if self._is_quality_result(result):
                results.append(result)
                self._cache_result(level_key, query.timestamp, result)
        
        return results
    
    def _is_cached(self, level_key: tuple, timestamp: pd.Timestamp) -> bool:
        """Check if result is cached and still valid."""
        if level_key not in self.result_cache:
            return False
        
        cached_ts, _ = self.result_cache[level_key]
        age_seconds = (timestamp - cached_ts).total_seconds()
        
        return age_seconds < self.cache_ttl
    
    def _cache_result(self, level_key: tuple, timestamp: pd.Timestamp, result: QueryResult):
        """Cache result."""
        self.result_cache[level_key] = (timestamp, result)
    
    def _is_quality_result(self, result: QueryResult) -> bool:
        """Check if result meets quality thresholds."""
        return (
            result.reliability.get('avg_similarity', 0) >= self.min_similarity and
            result.reliability.get('n_retrieved', 0) >= self.min_samples
        )
