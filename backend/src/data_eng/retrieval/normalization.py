from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

SCALE = 1.4826
EPS = 1e-6
CLIP = 8.0


@dataclass(frozen=True)
class RobustStats:
    median: np.ndarray
    mad: np.ndarray


def fit_robust_stats(vectors: np.ndarray) -> RobustStats:
    if vectors.ndim != 2:
        raise ValueError("Expected 2D array for vectors")
    median = np.median(vectors, axis=0)
    mad = np.median(np.abs(vectors - median), axis=0)
    return RobustStats(median=median, mad=mad)


def apply_robust_scaling(vectors: np.ndarray, stats: RobustStats) -> np.ndarray:
    scaled = (vectors - stats.median) / (SCALE * stats.mad + EPS)
    scaled = np.clip(scaled, -CLIP, CLIP)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def l2_normalize(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    normalized = np.zeros_like(vectors, dtype=np.float32)
    if np.any(valid):
        normalized[valid] = (vectors[valid] / norms[valid][:, None]).astype(np.float32)
    return normalized, valid
