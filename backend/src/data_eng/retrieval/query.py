from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

EPSILON = 1e-9
LEVEL_TYPES = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]


@dataclass
class SimilarSetup:
    episode_id: str
    distance: float
    similarity: float
    outcome: str
    outcome_score: float
    dt: str
    level_price: float
    approach_direction: int
    velocity_at_trigger: float = 0.0
    obi0_at_trigger: float = 0.0
    wall_imbal_at_trigger: float = 0.0


@dataclass
class OutcomeDistribution:
    unweighted: Dict[str, float]
    weighted: Dict[str, float]
    expected_score: float
    n_samples: int
    avg_similarity: float


@dataclass
class RetrievalResponse:
    query_level_type: str
    query_level_price: float
    query_approach_direction: int
    similar_setups: List[SimilarSetup]
    outcome_distribution: OutcomeDistribution
    n_retrieved: int
    avg_distance: float
    avg_similarity: float
    retrieval_confidence: float
    outcome_confidence: float

    def to_dict(self) -> dict:
        return {
            "level": f"{self.query_level_type} @ {self.query_level_price}",
            "approach": "from_below" if self.query_approach_direction == 1 else "from_above",
            "n_similar": self.n_retrieved,
            "outcome_probs": self.outcome_distribution.weighted,
            "expected_move": f"{self.outcome_distribution.expected_score:+.2f} pts",
            "retrieval_confidence": f"{self.retrieval_confidence:.0%}",
            "outcome_confidence": f"{self.outcome_confidence:.0%}",
            "top_matches": [
                {
                    "date": s.dt,
                    "outcome": s.outcome,
                    "similarity": f"{s.similarity:.0%}",
                    "outcome_score": s.outcome_score,
                }
                for s in self.similar_setups[:5]
            ],
        }


class SetupRetriever:
    def __init__(self, indices_dir: Path):
        self.indices_dir = Path(indices_dir)
        self.indices: Dict[str, Any] = {}
        self.norm_params: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._load_indices()
        self._load_norm_params()

    def _load_indices(self) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu or faiss-gpu required")

        for level_type in LEVEL_TYPES:
            lt_lower = level_type.lower()
            index_path = self.indices_dir / f"{lt_lower}_setups.index"
            if index_path.exists():
                self.indices[level_type] = faiss.read_index(str(index_path))

    def _load_norm_params(self) -> None:
        norm_path = self.indices_dir / "norm_params.json"
        if norm_path.exists():
            with open(norm_path) as f:
                self.norm_params = json.load(f)

    def _normalize_vector(
        self,
        vector: np.ndarray,
        level_type: str,
    ) -> np.ndarray:
        params = self.norm_params.get(level_type, {})
        mean = np.array(params.get("mean", []), dtype=np.float64)
        std = np.array(params.get("std", []), dtype=np.float64)

        if mean.size == 0 or std.size == 0:
            normalized = vector.astype(np.float32)
        else:
            if mean.size != vector.shape[0] or std.size != vector.shape[0]:
                raise ValueError("Normalization params do not match query vector dimensions")
            normalized = (vector - mean) / (std + EPSILON)
            normalized = np.clip(normalized, -10, 10)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)
            normalized = normalized.astype(np.float32)

        return normalized

    def _get_metadata(
        self,
        level_type: str,
        vector_ids: List[int],
    ) -> List[Dict[str, Any]]:
        lt_lower = level_type.lower()
        db_path = self.indices_dir / f"{lt_lower}_metadata.db"

        if not db_path.exists():
            return []

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(vector_ids))
        cursor.execute(
            f"SELECT * FROM setup_metadata WHERE vector_id IN ({placeholders})",
            vector_ids,
        )

        results = {row["vector_id"]: dict(row) for row in cursor.fetchall()}
        conn.close()

        return [results.get(vid, {}) for vid in vector_ids]

    def _get_metadata_dict(
        self,
        level_type: str,
        vector_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        lt_lower = level_type.lower()
        db_path = self.indices_dir / f"{lt_lower}_metadata.db"

        if not db_path.exists():
            return {}

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(vector_ids))
        cursor.execute(
            f"SELECT * FROM setup_metadata WHERE vector_id IN ({placeholders})",
            vector_ids,
        )

        results = {row["vector_id"]: dict(row) for row in cursor.fetchall()}
        conn.close()

        return results

    def find_similar_setups(
        self,
        query_vector: np.ndarray,
        level_type: str,
        k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        already_normalized: bool = False,
    ) -> List[SimilarSetup]:
        if level_type not in self.indices:
            return []

        index = self.indices[level_type]

        if already_normalized:
            query = query_vector.reshape(1, -1).astype(np.float32)
        else:
            normalized = self._normalize_vector(query_vector, level_type)
            query = normalized.reshape(1, -1)
        if query.shape[1] != index.d:
            raise ValueError("Query vector dimensions do not match index")

        k_fetch = min(k * 2, index.ntotal)
        if k_fetch == 0:
            return []

        distances, indices = index.search(query, k_fetch)

        all_indices = indices[0].tolist()
        metadata_dict = self._get_metadata_dict(level_type, [i for i in all_indices if i >= 0])

        results = []
        for dist, idx in zip(distances[0], all_indices):
            if idx < 0:
                continue

            meta = metadata_dict.get(idx)
            if meta:

                if filters:
                    if filters.get("min_date") and meta.get("dt", "") < filters["min_date"]:
                        continue
                    if filters.get("max_date") and meta.get("dt", "") > filters["max_date"]:
                        continue
                    if filters.get("exclude_chop") and meta.get("outcome") == "CHOP":
                        continue

                similarity = 1 / (1 + float(dist))

                results.append(SimilarSetup(
                    episode_id=meta.get("episode_id", ""),
                    distance=float(dist),
                    similarity=similarity,
                    outcome=meta.get("outcome", "UNKNOWN"),
                    outcome_score=float(meta.get("outcome_score", 0.0)),
                    dt=meta.get("dt", ""),
                    level_price=float(meta.get("level_price", 0.0)),
                    approach_direction=int(meta.get("approach_direction", 0)),
                    velocity_at_trigger=float(meta.get("velocity_at_trigger", 0.0) or 0.0),
                    obi0_at_trigger=float(meta.get("obi0_at_trigger", 0.0) or 0.0),
                    wall_imbal_at_trigger=float(meta.get("wall_imbal_at_trigger", 0.0) or 0.0),
                ))

                if len(results) >= k:
                    break

        return results

    def compute_outcome_distribution(
        self,
        similar_setups: List[SimilarSetup],
    ) -> OutcomeDistribution:
        outcomes = ["STRONG_BREAK", "WEAK_BREAK", "CHOP", "WEAK_BOUNCE", "STRONG_BOUNCE"]

        if not similar_setups:
            return OutcomeDistribution(
                unweighted={o: 0.0 for o in outcomes},
                weighted={o: 0.0 for o in outcomes},
                expected_score=0.0,
                n_samples=0,
                avg_similarity=0.0,
            )

        counts = Counter(s.outcome for s in similar_setups)
        total = len(similar_setups)

        unweighted = {o: counts.get(o, 0) / total for o in outcomes}

        weighted_counts: Dict[str, float] = defaultdict(float)
        total_weight = 0.0
        for s in similar_setups:
            weighted_counts[s.outcome] += s.similarity
            total_weight += s.similarity

        weighted = {o: weighted_counts.get(o, 0.0) / (total_weight + EPSILON) for o in outcomes}

        expected_score = sum(s.outcome_score * s.similarity for s in similar_setups) / (total_weight + EPSILON)

        return OutcomeDistribution(
            unweighted=unweighted,
            weighted=weighted,
            expected_score=expected_score,
            n_samples=total,
            avg_similarity=total_weight / total,
        )

    def query(
        self,
        query_vector: np.ndarray,
        level_type: str,
        level_price: float,
        approach_direction: int,
        k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        already_normalized: bool = False,
    ) -> RetrievalResponse:
        similar = self.find_similar_setups(query_vector, level_type, k, filters, already_normalized)
        distribution = self.compute_outcome_distribution(similar)

        if similar:
            avg_distance = np.mean([s.distance for s in similar])
            avg_similarity = np.mean([s.similarity for s in similar])

            retrieval_confidence = min(1.0, avg_similarity * 1.5)

            outcome_counts = Counter(s.outcome for s in similar)
            if outcome_counts:
                most_common_count = outcome_counts.most_common(1)[0][1]
                outcome_confidence = most_common_count / len(similar)
            else:
                outcome_confidence = 0.0
        else:
            avg_distance = 0.0
            avg_similarity = 0.0
            retrieval_confidence = 0.0
            outcome_confidence = 0.0

        return RetrievalResponse(
            query_level_type=level_type,
            query_level_price=level_price,
            query_approach_direction=approach_direction,
            similar_setups=similar,
            outcome_distribution=distribution,
            n_retrieved=len(similar),
            avg_distance=avg_distance,
            avg_similarity=avg_similarity,
            retrieval_confidence=retrieval_confidence,
            outcome_confidence=outcome_confidence,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query similar setups")
    parser.add_argument("--indices-dir", type=str, default="databases/indices")
    parser.add_argument("--level-type", type=str, required=True, choices=LEVEL_TYPES)
    parser.add_argument("--level-price", type=float, required=True)
    parser.add_argument("--direction", type=int, required=True, choices=[1, -1])
    parser.add_argument("--k", type=int, default=10)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    indices_dir = repo_root / args.indices_dir

    retriever = SetupRetriever(indices_dir)
    index = retriever.indices.get(args.level_type)
    if index is None:
        if retriever.indices:
            index = next(iter(retriever.indices.values()))
        else:
            raise ValueError("No indices found to infer vector dimensions")
    dummy_vector = np.random.randn(index.d).astype(np.float64)

    response = retriever.query(
        query_vector=dummy_vector,
        level_type=args.level_type,
        level_price=args.level_price,
        approach_direction=args.direction,
        k=args.k,
    )

    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    main()
