from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


class TreeModelBundle:
    """
    Loader + predictor for boosted tree models.
    """

    def __init__(self, model_dir: Path, stage: str, ablation: str, horizons: List[int]):
        self.model_dir = Path(model_dir)
        self.stage = stage
        self.ablation = ablation
        self.horizons = horizons

        self.tradeable = self._load("tradeable_2")
        self.direction = self._load("direction")
        self.strength = self._load("strength")
        self.t1_models = {h: self._load(f"t1_{h}s") for h in horizons}
        self.t2_models = {h: self._load(f"t2_{h}s") for h in horizons}

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

        return TreePredictions(
            tradeable_2=tradeable_prob,
            p_break=p_break,
            strength_signed=strength,
            t1_probs=t1_probs,
            t2_probs=t2_probs
        )
