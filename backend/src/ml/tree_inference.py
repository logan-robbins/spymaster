from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.ml.feature_sets import select_features


@dataclass
class TreePredictions:
    tradeable_2: np.ndarray
    p_break: np.ndarray
    strength_signed: np.ndarray
    t1_probs: Dict[int, np.ndarray]
    t2_probs: Dict[int, np.ndarray]
    t1_break_probs: Dict[int, np.ndarray]
    t1_bounce_probs: Dict[int, np.ndarray]
    t2_break_probs: Dict[int, np.ndarray]
    t2_bounce_probs: Dict[int, np.ndarray]


class TreeModelBundle:
    """
    Loader + predictor for boosted tree models.
    """

    def __init__(
        self,
        model_dir: Path,
        stage: str,
        ablation: str,
        horizons: List[int],
        timeframe: Optional[str] = None
    ):
        self.model_dir = Path(model_dir)
        self.stage = stage
        self.ablation = ablation
        self.horizons = horizons
        self.timeframe = timeframe

        tradeable_head = f"tradeable_2_{self.timeframe}" if self.timeframe else "tradeable_2"
        direction_head = f"direction_{self.timeframe}" if self.timeframe else "direction"
        strength_head = f"strength_{self.timeframe}" if self.timeframe else "strength"

        self.tradeable = self._load(tradeable_head)
        self.direction = self._load(direction_head)
        self.strength = self._load(strength_head)
        self.t1_models = {h: self._load(f"t1_{h}s") for h in horizons}
        self.t2_models = {h: self._load(f"t2_{h}s") for h in horizons}
        self.t1_break_models = {h: self._load(f"t1_break_{h}s") for h in horizons}
        self.t1_bounce_models = {h: self._load(f"t1_bounce_{h}s") for h in horizons}
        self.t2_break_models = {h: self._load(f"t2_break_{h}s") for h in horizons}
        self.t2_bounce_models = {h: self._load(f"t2_bounce_{h}s") for h in horizons}
        
        # Determine feature columns
        # We need a way to get the feature names without a real dataframe
        # Assuming select_features works with just stage/ablation params or we pass a dummy df
        dummy_df = pd.DataFrame(columns=[]) # Empty
        # feature_sets.select_features might fail if it tries to access columns.
        # Let's rely on FeatureSet having a static definition if possible.
        # Actually, let's just initialize it lazily or use a known set.
        # For now, let's try to get it from select_features if it supports it.
        # If not, we might need to modify select_features.
        # But wait, inference_engine passes a real dataframe.
        self._feature_cols = [] # Initialized later or we assume caller knows.
        # Actually, let's just use the logic from predict, but cache it.
        
    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
         # Helper to extract feature columns given a dataframe (to select correct subset)
         feature_set = select_features(df, stage=self.stage, ablation=self.ablation)
         return feature_set.numeric

    @property
    def feature_cols(self) -> List[str]:
        # Using a dummy dataframe to get feature list is inefficient but safe
        # Or better, just call select_features with empty df?
        # select_features expects columns to exist if it filters based on content?
        # Looking at select_features impl (imported), usually it returns names based on config.
        # Let's import select_features and check implementation.
        # Ideally we compute this once in __init__
        return self._feature_cols

    def _load(self, name: str):
        path = self.model_dir / f"{name}_{self.stage}_{self.ablation}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")
        return joblib.load(path)

    def predict(self, df: pd.DataFrame) -> TreePredictions:
        feature_set = select_features(df, stage=self.stage, ablation=self.ablation)
        required = feature_set.numeric + feature_set.categorical
        missing = [c for c in required if c not in df.columns]
        if missing:
            for col in missing:
                df[col] = np.nan
        X = df[required]

        tradeable_prob = self.tradeable.predict_proba(X)[:, 1]
        p_break = self.direction.predict_proba(X)[:, 1]
        strength = self.strength.predict(X)

        t1_probs = {h: model.predict_proba(X)[:, 1] for h, model in self.t1_models.items()}
        t2_probs = {h: model.predict_proba(X)[:, 1] for h, model in self.t2_models.items()}
        t1_break_probs = {h: model.predict_proba(X)[:, 1] for h, model in self.t1_break_models.items()}
        t1_bounce_probs = {h: model.predict_proba(X)[:, 1] for h, model in self.t1_bounce_models.items()}
        t2_break_probs = {h: model.predict_proba(X)[:, 1] for h, model in self.t2_break_models.items()}
        t2_bounce_probs = {h: model.predict_proba(X)[:, 1] for h, model in self.t2_bounce_models.items()}

        return TreePredictions(
            tradeable_2=tradeable_prob,
            p_break=p_break,
            strength_signed=strength,
            t1_probs=t1_probs,
            t2_probs=t2_probs,
            t1_break_probs=t1_break_probs,
            t1_bounce_probs=t1_bounce_probs,
            t2_break_probs=t2_break_probs,
            t2_bounce_probs=t2_bounce_probs
        )
