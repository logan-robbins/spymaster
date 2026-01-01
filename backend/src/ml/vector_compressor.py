"""
Vector Compressor for Phase 4 Optimization.
Handles dimensionality reduction and feature scaling to fix the 'Physics Noise' problem.
"""
import logging
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Canonical Section Boundaries (149D per RESEARCH.md Phase 4.5)
# Per src/ml/constants.py or episode_vector.py
SECTIONS = {
    'A_CONTEXT': range(0, 25),
    'B_DYNAMICS': range(25, 65), # 40 features (Physics)
    'C_HISTORY': range(65, 100),
    'D_DERIVED': range(100, 113),
    'E_TRENDS': range(113, 117),
    'F_GEOMETRY': range(117, 149) # 32 features (DCT)
}

class VectorCompressor:
    """
    Compresses the 149D raw vector into a 'Unified Field' vector.
    
    Strategies:
    - 'identity': No change (149D) -> Baseline
    - 'pca_physics': Compress Section B (40) -> 5 components. Keep rest.
    - 'geometry_only': Keep Section F only (32D per Phase 4).
    - 'weighted': Scale Physics * 0.2, Geometry * 1.0. 
    """
    
    def __init__(self, strategy: str = 'pca_physics', n_components: int = 5):
        self.strategy = strategy
        self.n_components = n_components
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray):
        """
        Fit the compression model (PCA) on a sample of vectors.
        """
        if self.strategy == 'identity' or self.strategy == 'geometry_only':
            self.is_fitted = True
            return
            
        if self.strategy == 'pca_physics':
            # Extract Physics section
            X_physics = X[:, SECTIONS['B_DYNAMICS']]
            
            # Pipeline: Scale -> PCA
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self.n_components))
            ])
            
            self.pipeline.fit(X_physics)
            explained_var = np.sum(self.pipeline.named_steps['pca'].explained_variance_ratio_)
            logger.info(f"Fitted PCA on Physics. Explained Variance (k={self.n_components}): {explained_var:.4f}")
            self.is_fitted = True
            
        elif self.strategy == 'weighted':
            # No fitting needed for simple scalar weighting
            self.is_fitted = True
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform vectors.
        """
        if not self.is_fitted:
            raise RuntimeError("Compressor not fitted.")
            
        if self.strategy == 'identity':
            return X
            
        if self.strategy == 'geometry_only':
            return X[:, SECTIONS['F_GEOMETRY']]
            
        if self.strategy == 'pca_physics':
            # 1. Transform Physics
            X_physics = X[:, SECTIONS['B_DYNAMICS']]
            X_phys_compressed = self.pipeline.transform(X_physics) # shape (N, k)
            
            # 2. Keep other sections (Context, History, Derived, Trends, Geometry)
            # Concatenate: [A, C, D, E, F] + [Physics_Compressed]
            # Wait, order matters?
            # Let's keep structure: [A] + [B_compressed] + [C] + [D] + [E] + [F]
            # This changes indices downstream! 
            # Ideally we append compressed features or replace.
            
            indices_others = []
            indices_others.extend(SECTIONS['A_CONTEXT'])
            indices_others.extend(SECTIONS['C_HISTORY'])
            indices_others.extend(SECTIONS['D_DERIVED'])
            indices_others.extend(SECTIONS['E_TRENDS'])
            indices_others.extend(SECTIONS['F_GEOMETRY'])
            
            X_others = X[:, indices_others]
            
            return np.hstack([X_others, X_phys_compressed])
            
        return X

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path: Path) -> 'VectorCompressor':
        with open(path, 'rb') as f:
            return pickle.load(f)
