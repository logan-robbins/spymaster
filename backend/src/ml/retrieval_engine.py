from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class RetrievalSummary:
    p_break: float
    p_bounce: float
    p_tradeable_2: float
    strength_signed_mean: float
    strength_abs_mean: float
    time_to_threshold_1_mean: float
    time_to_threshold_2_mean: float
    time_to_break_1_mean: float
    time_to_bounce_1_mean: float
    time_to_break_2_mean: float
    time_to_bounce_2_mean: float
    similarity: float
    entropy: float
    neighbors: pd.DataFrame


class RetrievalIndex:
    """
    kNN retrieval index over normalized engineered features.
    """

    def __init__(
        self,
        feature_cols: List[str],
        metadata_cols: Optional[List[str]] = None
    ):
        self.feature_cols = feature_cols
        self.metadata_cols = metadata_cols or ["level_kind_name", "direction"]
        self.scaler = StandardScaler()
        self._fit = False

        self._X = None
        self._metadata = None
        self._outcomes = None
        self._strength_signed = None
        self._strength_abs = None
        self._t1 = None
        self._t2 = None
        self._t1_break = None
        self._t1_bounce = None
        self._t2_break = None
        self._t2_bounce = None
        self._tradeable_2 = None
        self._feature_medians = None

    def fit(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for retrieval: {missing}")

        X = df[self.feature_cols].astype(np.float64).to_numpy()
        self._feature_medians = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, self._feature_medians)
        self._X = self.scaler.fit_transform(X)
        self._metadata = df[self.metadata_cols].copy()
        self._outcomes = df["outcome"].astype(str).to_numpy()
        self._strength_signed = df["strength_signed"].astype(float).to_numpy()
        if "strength_abs" in df.columns:
            self._strength_abs = df["strength_abs"].astype(float).to_numpy()
        else:
            self._strength_abs = np.abs(self._strength_signed)
        self._t1 = df.get("time_to_threshold_1", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
        self._t2 = df.get("time_to_threshold_2", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
        self._t1_break = df.get("time_to_break_1", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
        self._t1_bounce = df.get("time_to_bounce_1", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
        self._t2_break = df.get("time_to_break_2", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
        self._t2_bounce = df.get("time_to_bounce_2", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
        self._tradeable_2 = df.get("tradeable_2", pd.Series(0, index=df.index)).astype(int).to_numpy()

        if "gamma_bucket" in df.columns and "gamma_bucket" not in self._metadata.columns:
            self._metadata["gamma_bucket"] = df["gamma_bucket"].astype(str).to_numpy()

        self._fit = True

    def query(
        self,
        feature_vector: np.ndarray,
        filters: Optional[Dict[str, str]] = None,
        k: int = 20
    ) -> RetrievalSummary:
        if not self._fit:
            raise ValueError("Retrieval index not fit.")
        if feature_vector.shape[0] != len(self.feature_cols):
            raise ValueError("Feature vector shape mismatch.")

        if self._feature_medians is None:
            raise ValueError("Retrieval index missing medians.")
        x = feature_vector.copy()
        x = np.where(np.isfinite(x), x, self._feature_medians)
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        mask = np.ones(len(self._X), dtype=bool)
        if filters:
            for key, value in filters.items():
                if key not in self._metadata.columns:
                    continue
                mask &= (self._metadata[key].astype(str).to_numpy() == str(value))

        if not np.any(mask):
            raise ValueError("No retrieval candidates after filtering.")

        X_subset = self._X[mask]
        distances = np.linalg.norm(X_subset - x_scaled, axis=1)
        idx_sorted = np.argsort(distances)
        top_idx = idx_sorted[: min(k, len(idx_sorted))]

        top_dist = distances[top_idx]
        weights = 1.0 / (top_dist + 1e-6)
        weights = weights / weights.sum()

        outcomes = self._outcomes[mask][top_idx]
        strength_signed = self._strength_signed[mask][top_idx]
        strength_abs = self._strength_abs[mask][top_idx]
        t1 = self._t1[mask][top_idx]
        t2 = self._t2[mask][top_idx]
        t1_break = self._t1_break[mask][top_idx]
        t1_bounce = self._t1_bounce[mask][top_idx]
        t2_break = self._t2_break[mask][top_idx]
        t2_bounce = self._t2_bounce[mask][top_idx]
        tradeable_2 = self._tradeable_2[mask][top_idx]

        break_mask = outcomes == "BREAK"
        bounce_mask = outcomes == "BOUNCE"
        p_break = float(np.sum(weights * break_mask))
        p_bounce = float(np.sum(weights * bounce_mask))
        p_tradeable_2 = float(np.sum(weights * (tradeable_2 == 1)))

        strength_signed_mean = float(np.sum(weights * strength_signed))
        strength_abs_mean = float(np.sum(weights * strength_abs))
        time_to_threshold_1_mean = float(np.nanmean(t1)) if np.any(np.isfinite(t1)) else float("nan")
        time_to_threshold_2_mean = float(np.nanmean(t2)) if np.any(np.isfinite(t2)) else float("nan")
        time_to_break_1_mean = float(np.nanmean(t1_break)) if np.any(np.isfinite(t1_break)) else float("nan")
        time_to_bounce_1_mean = float(np.nanmean(t1_bounce)) if np.any(np.isfinite(t1_bounce)) else float("nan")
        time_to_break_2_mean = float(np.nanmean(t2_break)) if np.any(np.isfinite(t2_break)) else float("nan")
        time_to_bounce_2_mean = float(np.nanmean(t2_bounce)) if np.any(np.isfinite(t2_bounce)) else float("nan")

        similarity = float(np.mean(1.0 / (1.0 + top_dist)))
        probs = np.array([p_break, p_bounce, max(0.0, 1.0 - p_break - p_bounce)])
        probs = probs / max(probs.sum(), 1e-6)
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

        neighbors = self._metadata[mask].iloc[top_idx].copy()
        neighbors["distance"] = top_dist
        neighbors["weight"] = weights
        neighbors["outcome"] = outcomes

        return RetrievalSummary(
            p_break=p_break,
            p_bounce=p_bounce,
            p_tradeable_2=p_tradeable_2,
            strength_signed_mean=strength_signed_mean,
            strength_abs_mean=strength_abs_mean,
            time_to_threshold_1_mean=time_to_threshold_1_mean,
            time_to_threshold_2_mean=time_to_threshold_2_mean,
            time_to_break_1_mean=time_to_break_1_mean,
            time_to_bounce_1_mean=time_to_bounce_1_mean,
            time_to_break_2_mean=time_to_break_2_mean,
            time_to_bounce_2_mean=time_to_bounce_2_mean,
            similarity=similarity,
            entropy=entropy,
            neighbors=neighbors
        )
