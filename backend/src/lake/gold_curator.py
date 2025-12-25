"""
Gold Curator - Promote Best Silver Experiments to Production.

Curates production-ready ML datasets from Silver feature experiments,
ensuring Gold layer contains only validated, production-quality data.
"""

import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.common.config import CONFIG
from src.lake.silver_feature_builder import SilverFeatureBuilder


class GoldCurator:
    """
    Curates Gold layer from best Silver experiments.
    
    Gold layer serves as the production ML dataset source, containing:
    - Training datasets (curated from experiments)
    - Evaluation datasets (backtest results)
    - Streaming signals (real-time from Core Service)
    
    Usage:
        curator = GoldCurator()
        
        # Promote Silver experiment to Gold training
        curator.promote_to_training(
            silver_version='v2.0_full_ensemble',
            dataset_name='signals_production',
            notes='Best performing full ensemble model'
        )
        
        # Load Gold training data
        df = curator.load_training_data('signals_production')
    """
    
    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize Gold curator.
        
        Args:
            data_root: Root data directory (default: CONFIG.DATA_ROOT)
        """
        self.data_root = Path(data_root or CONFIG.DATA_ROOT)
        self.silver_builder = SilverFeatureBuilder(data_root=str(self.data_root))
        
        self.gold_root = self.data_root / "lake" / "gold"
        self.training_root = self.gold_root / "training"
        self.evaluation_root = self.gold_root / "evaluation"
        self.streaming_root = self.gold_root / "streaming"
        
        # Create directories
        self.training_root.mkdir(parents=True, exist_ok=True)
        self.evaluation_root.mkdir(parents=True, exist_ok=True)
        self.streaming_root.mkdir(parents=True, exist_ok=True)
        
        # Load catalog
        self.catalog_path = self.gold_root / "catalog.json"
        self.catalog = self._load_catalog()
    
    def _load_catalog(self) -> Dict[str, Any]:
        """Load Gold catalog."""
        if not self.catalog_path.exists():
            return {'datasets': [], 'last_updated': None}
        
        import json
        with open(self.catalog_path, 'r') as f:
            return json.load(f)
    
    def _save_catalog(self):
        """Save Gold catalog."""
        self.catalog['last_updated'] = datetime.utcnow().isoformat() + "Z"
        
        import json
        with open(self.catalog_path, 'w') as f:
            json.dump(self.catalog, f, indent=2)
    
    def promote_to_training(
        self,
        silver_version: str,
        dataset_name: str = 'signals_production',
        dates: Optional[List[str]] = None,
        notes: str = "",
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Promote a Silver feature version to Gold training.
        
        Args:
            silver_version: Silver feature version to promote
            dataset_name: Name for the Gold dataset
            dates: Optional specific dates to include (default: all)
            notes: Notes about this promotion
            force: If True, overwrite existing dataset
        
        Returns:
            Dictionary with promotion statistics
        """
        # Check if dataset exists
        dataset_path = self.training_root / f"{dataset_name}.parquet"
        if dataset_path.exists() and not force:
            return {
                'status': 'skipped',
                'reason': f'Dataset {dataset_name} already exists. Use force=True to overwrite.',
                'path': str(dataset_path)
            }
        
        # Load Silver features
        print(f"Loading Silver features from {silver_version}...")
        df = self.silver_builder.load_features(silver_version, dates=dates)
        
        if df.empty:
            return {
                'status': 'error',
                'reason': 'No data found in Silver version',
                'silver_version': silver_version
            }
        
        # Get manifest for metadata
        manifest = self.silver_builder.get_manifest(silver_version)
        if not manifest:
            return {
                'status': 'error',
                'reason': f'Manifest not found for {silver_version}',
                'silver_version': silver_version
            }
        
        # Write to Gold
        print(f"Writing {len(df)} signals to Gold training: {dataset_name}...")
        df.to_parquet(
            dataset_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        
        # Create metadata
        metadata = {
            'dataset_name': dataset_name,
            'silver_version': silver_version,
            'created_at': datetime.utcnow().isoformat() + "Z",
            'signal_count': len(df),
            'feature_count': len([c for c in df.columns if c not in ['event_id', 'ts_ns', 'date', 'symbol']]),
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None,
            },
            'manifest': {
                'version': manifest.version,
                'description': manifest.description,
                'parameters': manifest.parameters.to_dict() if manifest.parameters else {}
            },
            'notes': notes
        }
        
        # Save metadata
        metadata_path = self.training_root / f"{dataset_name}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update catalog
        self.catalog['datasets'].append({
            'name': dataset_name,
            'type': 'training',
            'path': str(dataset_path),
            'created_at': metadata['created_at'],
            'signal_count': metadata['signal_count'],
            'silver_version': silver_version
        })
        self._save_catalog()
        
        print(f"Successfully promoted {silver_version} to Gold training/{dataset_name}")
        
        return {
            'status': 'success',
            'dataset_name': dataset_name,
            'path': str(dataset_path),
            'signal_count': len(df),
            'feature_count': metadata['feature_count'],
            'silver_version': silver_version
        }
    
    def load_training_data(
        self,
        dataset_name: str = 'signals_production'
    ) -> pd.DataFrame:
        """
        Load Gold training dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            DataFrame with training data
        """
        dataset_path = self.training_root / f"{dataset_name}.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Gold training dataset '{dataset_name}' not found")
        
        return pd.read_parquet(dataset_path)
    
    def list_training_datasets(self) -> List[Dict[str, Any]]:
        """List all Gold training datasets."""
        datasets = []
        for item in self.catalog.get('datasets', []):
            if item.get('type') == 'training':
                datasets.append(item)
        return datasets
    
    def get_dataset_metadata(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a Gold dataset."""
        metadata_path = self.training_root / f"{dataset_name}_metadata.json"
        if not metadata_path.exists():
            return None
        
        import json
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def validate_dataset(
        self,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Validate a Gold training dataset.
        
        Returns:
            Dictionary with validation results
        """
        df = self.load_training_data(dataset_name)
        metadata = self.get_dataset_metadata(dataset_name)
        
        # Basic validation
        validation = {
            'dataset_name': dataset_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_summary': {},
            'label_distribution': {},
            'date_coverage': {}
        }
        
        # Check nulls
        null_counts = df.isnull().sum()
        validation['null_summary'] = {
            'total_nulls': int(null_counts.sum()),
            'high_null_columns': [
                col for col in df.columns
                if null_counts[col] / len(df) > 0.1
            ]
        }
        
        # Check label distribution (if outcome column exists)
        if 'outcome' in df.columns:
            validation['label_distribution'] = df['outcome'].value_counts().to_dict()
        
        # Check date coverage
        if 'date' in df.columns:
            date_counts = df['date'].value_counts().sort_index()
            validation['date_coverage'] = {
                'start_date': date_counts.index.min(),
                'end_date': date_counts.index.max(),
                'unique_dates': len(date_counts),
                'signals_per_date': date_counts.to_dict()
            }
        
        # Compare with metadata
        if metadata:
            validation['metadata_match'] = {
                'signal_count_match': metadata.get('signal_count') == len(df),
                'expected_count': metadata.get('signal_count'),
                'actual_count': len(df)
            }
        
        return validation


def promote_best_experiment(
    exp_id: str,
    dataset_name: str = 'signals_production',
    data_root: Optional[str] = None
):
    """
    Helper function to promote the best experiment to Gold.
    
    Args:
        exp_id: Experiment ID from Silver registry
        dataset_name: Name for the Gold dataset
        data_root: Optional data root
    """
    builder = SilverFeatureBuilder(data_root=data_root)
    curator = GoldCurator(data_root=data_root)
    
    # Find experiment
    experiment = builder.registry.get_by_id(exp_id)
    if not experiment:
        print(f"Error: Experiment {exp_id} not found")
        return
    
    print(f"Promoting experiment {exp_id} (version: {experiment.version}) to Gold...")
    
    result = curator.promote_to_training(
        silver_version=experiment.version,
        dataset_name=dataset_name,
        notes=f"Promoted from experiment {exp_id}: {experiment.notes}"
    )
    
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"  Dataset: {result['dataset_name']}")
        print(f"  Signals: {result['signal_count']}")
        print(f"  Features: {result['feature_count']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Curate Gold datasets from Silver")
    parser.add_argument('--action', choices=['promote', 'list', 'validate'],
                        default='list', help="Action to perform")
    parser.add_argument('--silver-version', help="Silver version to promote")
    parser.add_argument('--dataset-name', default='signals_production',
                        help="Gold dataset name")
    parser.add_argument('--exp-id', help="Experiment ID to promote")
    parser.add_argument('--force', action='store_true',
                        help="Force overwrite existing dataset")
    
    args = parser.parse_args()
    
    curator = GoldCurator()
    
    if args.action == 'promote':
        if args.exp_id:
            promote_best_experiment(args.exp_id, args.dataset_name)
        elif args.silver_version:
            result = curator.promote_to_training(
                silver_version=args.silver_version,
                dataset_name=args.dataset_name,
                force=args.force
            )
            print(f"Status: {result['status']}")
        else:
            print("Error: --silver-version or --exp-id required")
    
    elif args.action == 'list':
        datasets = curator.list_training_datasets()
        print(f"Gold Training Datasets ({len(datasets)}):")
        for ds in datasets:
            print(f"  - {ds['name']}")
            print(f"      Silver version: {ds['silver_version']}")
            print(f"      Signals: {ds['signal_count']}")
            print(f"      Created: {ds['created_at']}")
    
    elif args.action == 'validate':
        if not args.dataset_name:
            print("Error: --dataset-name required")
        else:
            validation = curator.validate_dataset(args.dataset_name)
            import json
            print(json.dumps(validation, indent=2, default=str))

