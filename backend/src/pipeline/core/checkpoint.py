"""Pipeline checkpoint manager for incremental execution and validation.

Enables:
- Save context after each stage
- Resume from any stage
- Inspect intermediate outputs
- Validate stage-by-stage correctness
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline.core.stage import StageContext

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages pipeline checkpoints for incremental execution.
    
    Checkpoint Structure:
        checkpoints/
        └── {pipeline_name}/
            └── {date}/
                ├── manifest.json          # Stage sequence, config hash
                ├── stage_00_load_bronze/
                │   ├── metadata.json      # Stage name, outputs, timing
                │   ├── trades.parquet     # DataFrame outputs
                │   ├── mbp10_snapshots.pkl  # Complex object outputs
                │   └── ...
                ├── stage_01_build_ohlcv/
                │   └── ...
                └── ...
    
    Usage:
        manager = CheckpointManager(checkpoint_root="data/checkpoints")
        
        # Save after stage
        manager.save_checkpoint(
            pipeline_name="es_pipeline",
            date="2025-12-16",
            stage_idx=0,
            stage_name="load_bronze",
            ctx=ctx
        )
        
        # Resume from stage
        ctx = manager.load_checkpoint(
            pipeline_name="es_pipeline",
            date="2025-12-16",
            stage_idx=5
        )
    """
    
    def __init__(self, checkpoint_root: str):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_root: Root directory for checkpoints
        """
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
    
    def _get_checkpoint_dir(
        self,
        pipeline_name: str,
        date: str,
        stage_idx: Optional[int] = None
    ) -> Path:
        """Get checkpoint directory path.
        
        Args:
            pipeline_name: Pipeline name
            date: Date (YYYY-MM-DD)
            stage_idx: Optional stage index (0-based)
        
        Returns:
            Path to checkpoint directory
        """
        base = self.checkpoint_root / pipeline_name / date
        if stage_idx is not None:
            return base / f"stage_{stage_idx:02d}"
        return base
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of config for invalidation detection.
        
        Args:
            config: Config dict
        
        Returns:
            SHA256 hash (first 16 chars)
        """
        # Sort keys for deterministic hash
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _serialize_value(self, key: str, value: Any, output_dir: Path) -> Dict[str, str]:
        """Serialize a context value to disk.
        
        Args:
            key: Context key
            value: Context value
            output_dir: Directory to write to
        
        Returns:
            Dict with type and path info
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(value, pd.DataFrame):
            # Save DataFrame as parquet, preserving index
            filepath = output_dir / f"{key}.parquet"
            value.to_parquet(filepath, compression='zstd', index=True)
            return {
                'type': 'dataframe',
                'path': filepath.name,
                'rows': len(value),
                'columns': list(value.columns),
                'index_type': type(value.index).__name__
            }
        
        elif isinstance(value, pd.Series):
            # Save Series as parquet (convert to DataFrame)
            filepath = output_dir / f"{key}.parquet"
            value.to_frame().to_parquet(filepath, compression='zstd', index=True)
            return {
                'type': 'series',
                'path': filepath.name,
                'length': len(value)
            }
        
        elif isinstance(value, (list, dict, tuple)):
            # Save as pickle for complex Python objects
            filepath = output_dir / f"{key}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            size_info = len(value) if isinstance(value, (list, tuple)) else len(value.keys())
            return {
                'type': type(value).__name__,
                'path': filepath.name,
                'size': size_info
            }
        
        else:
            # Save scalar/simple types as pickle
            filepath = output_dir / f"{key}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            return {
                'type': type(value).__name__,
                'path': filepath.name
            }
    
    def _deserialize_value(self, key: str, metadata: Dict[str, str], input_dir: Path) -> Any:
        """Deserialize a context value from disk.
        
        Args:
            key: Context key
            metadata: Serialization metadata
            input_dir: Directory to read from
        
        Returns:
            Deserialized value
        """
        filepath = input_dir / metadata['path']
        
        if metadata['type'] == 'dataframe':
            return pd.read_parquet(filepath)
        
        elif metadata['type'] == 'series':
            df = pd.read_parquet(filepath)
            return df.iloc[:, 0]  # Extract first column as Series
        
        else:
            # Load pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    def save_checkpoint(
        self,
        pipeline_name: str,
        date: str,
        stage_idx: int,
        stage_name: str,
        ctx: StageContext,
        elapsed_time: float
    ) -> None:
        """Save checkpoint after stage execution.
        
        Args:
            pipeline_name: Pipeline name
            date: Date (YYYY-MM-DD)
            stage_idx: Stage index (0-based)
            stage_name: Stage name
            ctx: Stage context to save
            elapsed_time: Stage execution time (seconds)
        """
        checkpoint_dir = self._get_checkpoint_dir(pipeline_name, date, stage_idx)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"  Saving checkpoint to {checkpoint_dir}")
        
        # Serialize each context value
        serialized = {}
        for key, value in ctx.data.items():
            try:
                serialized[key] = self._serialize_value(key, value, checkpoint_dir)
            except Exception as e:
                logger.warning(f"  Failed to serialize '{key}': {e}")
        
        # Save metadata
        metadata = {
            'stage_idx': stage_idx,
            'stage_name': stage_name,
            'date': date,
            'elapsed_time': elapsed_time,
            'config_hash': self._compute_config_hash(ctx.config),
            'outputs': serialized
        }
        
        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"  Checkpoint saved: {len(serialized)} outputs")
    
    def load_checkpoint(
        self,
        pipeline_name: str,
        date: str,
        stage_idx: int
    ) -> Optional[StageContext]:
        """Load checkpoint from specified stage.
        
        Args:
            pipeline_name: Pipeline name
            date: Date (YYYY-MM-DD)
            stage_idx: Stage index to load (0-based)
        
        Returns:
            StageContext with loaded data, or None if not found
        """
        checkpoint_dir = self._get_checkpoint_dir(pipeline_name, date, stage_idx)
        metadata_path = checkpoint_dir / 'metadata.json'
        
        if not metadata_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_dir}")
            return None
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loading checkpoint from stage {stage_idx}: {metadata['stage_name']}")
        
        # Deserialize context data
        ctx_data = {}
        for key, value_meta in metadata['outputs'].items():
            try:
                ctx_data[key] = self._deserialize_value(key, value_meta, checkpoint_dir)
                logger.debug(f"  Loaded: {key} ({value_meta['type']})")
            except Exception as e:
                logger.error(f"  Failed to deserialize '{key}': {e}")
                raise
        
        # Create context
        ctx = StageContext(
            date=date,
            data=ctx_data,
            config={}  # Config will be set by pipeline
        )
        
        return ctx
    
    def list_checkpoints(
        self,
        pipeline_name: str,
        date: str
    ) -> List[Dict[str, Any]]:
        """List available checkpoints for a pipeline/date.
        
        Args:
            pipeline_name: Pipeline name
            date: Date (YYYY-MM-DD)
        
        Returns:
            List of checkpoint metadata dicts
        """
        base_dir = self._get_checkpoint_dir(pipeline_name, date)
        
        if not base_dir.exists():
            return []
        
        checkpoints = []
        for stage_dir in sorted(base_dir.glob("stage_*")):
            metadata_path = stage_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    checkpoints.append(metadata)
        
        return checkpoints
    
    def clear_checkpoints(
        self,
        pipeline_name: str,
        date: str
    ) -> None:
        """Clear all checkpoints for a pipeline/date.
        
        Args:
            pipeline_name: Pipeline name
            date: Date (YYYY-MM-DD)
        """
        import shutil
        
        base_dir = self._get_checkpoint_dir(pipeline_name, date)
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info(f"Cleared checkpoints: {base_dir}")

