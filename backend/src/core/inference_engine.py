from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.ensemble import combine_probabilities
from src.ml.feasibility_gate import FeasibilityGate
from src.ml.retrieval_engine import RetrievalIndex
from src.ml.tree_inference import TreeModelBundle


class ViewportInferenceEngine:
    """
    Scores viewport targets using tree models + retrieval and ranks by utility.
    """

    def __init__(
        self,
        model_bundle: TreeModelBundle,
        retrieval_index: RetrievalIndex,
        feasibility_gate: Optional[FeasibilityGate] = None
    ):
        self.model_bundle = model_bundle
        self.retrieval_index = retrieval_index
        self.feasibility_gate = feasibility_gate or FeasibilityGate()

    def score_targets(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if features_df.empty:
            return []

        for col in self.retrieval_index.feature_cols:
            if col not in features_df.columns:
                features_df[col] = np.nan

        tree_preds = self.model_bundle.predict(features_df)
        results: List[Dict[str, Any]] = []

        for i, row in features_df.iterrows():
            feature_vector = row[self.retrieval_index.feature_cols].to_numpy(dtype=np.float64)
            filters = {
                "level_kind_name": row.get("level_kind_name"),
                "direction": row.get("direction")
            }
            if "gamma_bucket" in row:
                filters["gamma_bucket"] = row.get("gamma_bucket")

            retrieval = self.retrieval_index.query(feature_vector, filters=filters, k=20)

            ensemble = combine_probabilities(
                tree_prob=float(tree_preds.p_break[i]),
                knn_prob=retrieval.p_break,
                similarity=retrieval.similarity,
                entropy=retrieval.entropy
            )

            mask = self.feasibility_gate.compute_mask(
                direction=row.get("direction", "UP"),
                barrier_state=row.get("barrier_state", "NEUTRAL"),
                tape_imbalance=float(row.get("tape_imbalance", 0.0)),
                tape_velocity=float(row.get("tape_velocity", 0.0)),
                fuel_effect=row.get("fuel_effect", "NEUTRAL"),
                gamma_exposure=float(row.get("gamma_exposure", 0.0))
            )
            p_break = self.feasibility_gate.apply_mask(ensemble.p_break, mask)

            strength = float(tree_preds.strength_signed[i])
            p_tradeable_2 = float(tree_preds.tradeable_2[i])
            utility = p_tradeable_2 * abs(strength)

            level_id = row.get("level_id")
            if not level_id:
                level_id = f"{row.get('level_kind_name')}_{int(round(row.get('level_price', 0.0)))}"

            results.append({
                "level_id": level_id,
                "level_kind_name": row.get("level_kind_name"),
                "level_price": float(row.get("level_price", 0.0)),
                "direction": row.get("direction"),
                "distance": float(row.get("distance", 0.0)) if row.get("distance") is not None else None,
                "distance_signed": float(row.get("distance_signed", 0.0)) if row.get("distance_signed") is not None else None,
                "p_tradeable_2": p_tradeable_2,
                "p_break": p_break,
                "p_bounce": 1.0 - p_break,
                "strength_signed": strength,
                "strength_abs": abs(strength),
                "time_to_threshold": {
                    "t1": {k: float(v[i]) for k, v in tree_preds.t1_probs.items()},
                    "t2": {k: float(v[i]) for k, v in tree_preds.t2_probs.items()}
                },
                "retrieval": {
                    "p_break": retrieval.p_break,
                    "p_bounce": retrieval.p_bounce,
                    "p_tradeable_2": retrieval.p_tradeable_2,
                    "strength_signed_mean": retrieval.strength_signed_mean,
                    "strength_abs_mean": retrieval.strength_abs_mean,
                    "time_to_threshold_1_mean": retrieval.time_to_threshold_1_mean,
                    "time_to_threshold_2_mean": retrieval.time_to_threshold_2_mean,
                    "neighbors": retrieval.neighbors.to_dict(orient="records")
                },
                "ensemble": {
                    "mix_weight": ensemble.mix_weight,
                    "disagreement_penalty": ensemble.disagreement_penalty
                },
                "feasibility_mask": {
                    "allow_break": mask.allow_break,
                    "allow_bounce": mask.allow_bounce
                },
                "utility_score": utility
            })

        results.sort(key=lambda r: r["utility_score"], reverse=True)
        return results
