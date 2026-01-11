from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .normalization import RobustStats, apply_robust_scaling

try:
    import faiss
except ImportError:
    faiss = None

APPROACH_DIRS = ["approach_up", "approach_down"]


@dataclass
class SimilarVector:
    index_id: int
    similarity: float
    metadata: Dict[str, object]


@dataclass
class OutcomeDistribution:
    probabilities: Dict[str, float]
    predicted: str
    total: int


class TriggerVectorRetriever:
    def __init__(self, indices_dir: Path):
        self.indices_dir = Path(indices_dir)
        self.stats = self._load_stats()
        self.indices: Dict[str, Dict[str, object]] = {}
        self.metadata: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self._load_indices()

    def _load_stats(self) -> RobustStats:
        stats_path = self.indices_dir / "norm_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing norm_stats.json in {self.indices_dir}")
        payload = json.loads(stats_path.read_text())
        median = np.array(payload.get("median", []), dtype=np.float64)
        mad = np.array(payload.get("mad", []), dtype=np.float64)
        if median.size == 0 or mad.size == 0:
            raise ValueError("Invalid norm_stats.json contents")
        return RobustStats(median=median, mad=mad)

    def _load_indices(self) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu or faiss-gpu required")

        for level_dir in sorted(self.indices_dir.iterdir()):
            if not level_dir.is_dir():
                continue
            level_id = level_dir.name
            self.indices[level_id] = {}
            self.metadata[level_id] = {}
            for approach_dir in APPROACH_DIRS:
                index_path = level_dir / f"{approach_dir}.index"
                meta_path = level_dir / f"metadata_{approach_dir}.npz"
                if not index_path.exists():
                    raise FileNotFoundError(f"Missing index: {index_path}")
                if not meta_path.exists():
                    raise FileNotFoundError(f"Missing metadata: {meta_path}")
                index = faiss.read_index(str(index_path))
                if hasattr(index, "hnsw"):
                    index.hnsw.efSearch = 64
                self.indices[level_id][approach_dir] = index
                self.metadata[level_id][approach_dir] = _load_metadata(meta_path)

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.stats.median.shape[0]:
            raise ValueError("Vector dimensions do not match normalization stats")
        scaled = apply_robust_scaling(v[None, :], self.stats)[0]
        norm = np.linalg.norm(scaled)
        if norm == 0:
            raise ValueError("Vector norm is zero after scaling")
        return (scaled / norm).astype(np.float32)

    def find_similar(
        self,
        level_id: str,
        approach_dir: str,
        vector: np.ndarray,
        k: int = 20,
    ) -> List[SimilarVector]:
        if level_id not in self.indices:
            raise ValueError(f"Unknown level_id: {level_id}")
        if approach_dir not in self.indices[level_id]:
            raise ValueError(f"Unknown approach_dir: {approach_dir}")

        index = self.indices[level_id][approach_dir]
        if index.ntotal == 0:
            return []

        query = self.normalize_vector(vector).reshape(1, -1)
        if query.shape[1] != index.d:
            raise ValueError("Query vector dimensions do not match index")

        k_fetch = min(k, index.ntotal)
        distances, indices = index.search(query, k_fetch)

        meta = self.metadata[level_id][approach_dir]
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                SimilarVector(
                    index_id=int(idx),
                    similarity=float(dist),
                    metadata=_metadata_row(meta, int(idx)),
                )
            )
        return results

    def outcome_distribution(
        self,
        neighbors: List[SimilarVector],
        approach_dir: str,
        horizon: int | None = None,
    ) -> OutcomeDistribution:
        break_label, reject_label = _labels_for_approach(approach_dir)
        label_key = "true_outcome" if horizon is None else f"true_outcome_h{horizon}"

        counts = {
            break_label: 0,
            reject_label: 0,
            "CHOP": 0,
        }
        for n in neighbors:
            outcome = str(n.metadata.get(label_key, ""))
            if outcome == "WHIPSAW":
                continue
            if outcome in counts:
                counts[outcome] += 1

        total = sum(counts.values())
        if total == 0:
            return OutcomeDistribution(probabilities={k: 0.0 for k in counts}, predicted="CHOP", total=0)

        probs = {k: counts[k] / total for k in counts}
        predicted = max(probs.items(), key=lambda kv: kv[1])[0]
        return OutcomeDistribution(probabilities=probs, predicted=predicted, total=total)


def _labels_for_approach(approach_dir: str) -> Tuple[str, str]:
    if approach_dir == "approach_up":
        return "BREAK_UP", "REJECT_DOWN"
    if approach_dir == "approach_down":
        return "BREAK_DOWN", "REJECT_UP"
    raise ValueError(f"Unexpected approach_dir: {approach_dir}")


def _load_metadata(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _metadata_row(meta: Dict[str, np.ndarray], idx: int) -> Dict[str, object]:
    row: Dict[str, object] = {}
    for key, values in meta.items():
        val = values[idx]
        if isinstance(val, np.generic):
            row[key] = val.item()
        else:
            row[key] = val
    return row
