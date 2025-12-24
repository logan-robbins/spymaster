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

        features_matrix = features_df[self.retrieval_index.feature_cols].to_numpy(dtype=np.float64)
        count = len(features_df)

        def _col_or_none(name: str):
            return features_df[name].to_numpy() if name in features_df.columns else None

        def _float_or_default(value: Any, default: float) -> float:
            if value is None:
                return default
            try:
                if np.isnan(value):
                    return default
            except TypeError:
                pass
            return float(value)

        def _maybe_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                if np.isnan(value):
                    return None
            except TypeError:
                pass
            return float(value)

        level_kind_name = _col_or_none("level_kind_name")
        direction = _col_or_none("direction")
        gamma_bucket = _col_or_none("gamma_bucket")
        level_id = _col_or_none("level_id")
        level_price = _col_or_none("level_price")
        distance = _col_or_none("distance")
        distance_signed = _col_or_none("distance_signed")
        barrier_state = _col_or_none("barrier_state")
        tape_imbalance = _col_or_none("tape_imbalance")
        tape_velocity = _col_or_none("tape_velocity")
        fuel_effect = _col_or_none("fuel_effect")
        gamma_exposure = _col_or_none("gamma_exposure")

        for i in range(count):
            feature_vector = features_matrix[i]
            level_kind_name_val = level_kind_name[i] if level_kind_name is not None else None
            direction_val = direction[i] if direction is not None else "UP"

            filters = {
                "level_kind_name": level_kind_name_val,
                "direction": direction_val
            }
            if gamma_bucket is not None:
                filters["gamma_bucket"] = gamma_bucket[i]

            retrieval = self.retrieval_index.query(feature_vector, filters=filters, k=20)

            ensemble = combine_probabilities(
                tree_prob=float(tree_preds.p_break[i]),
                knn_prob=retrieval.p_break,
                similarity=retrieval.similarity,
                entropy=retrieval.entropy
            )

            mask = self.feasibility_gate.compute_mask(
                direction=direction_val if direction_val is not None else "UP",
                barrier_state=barrier_state[i] if barrier_state is not None else "NEUTRAL",
                tape_imbalance=_float_or_default(tape_imbalance[i] if tape_imbalance is not None else None, 0.0),
                tape_velocity=_float_or_default(tape_velocity[i] if tape_velocity is not None else None, 0.0),
                fuel_effect=fuel_effect[i] if fuel_effect is not None else "NEUTRAL",
                gamma_exposure=_float_or_default(gamma_exposure[i] if gamma_exposure is not None else None, 0.0)
            )
            p_break = self.feasibility_gate.apply_mask(ensemble.p_break, mask)

            strength = float(tree_preds.strength_signed[i])
            p_tradeable_2 = float(tree_preds.tradeable_2[i])
            utility = p_tradeable_2 * abs(strength)

            level_id_val = level_id[i] if level_id is not None else None
            level_price_val = _float_or_default(level_price[i] if level_price is not None else None, 0.0)
            if not level_id_val:
                kind_label = level_kind_name_val if level_kind_name_val is not None else "LEVEL"
                level_id_val = f"{kind_label}_{int(round(level_price_val))}"

            distance_val = _maybe_float(distance[i] if distance is not None else None)
            distance_signed_val = _maybe_float(distance_signed[i] if distance_signed is not None else None)

            results.append({
                "level_id": level_id_val,
                "level_kind_name": level_kind_name_val,
                "level_price": level_price_val,
                "direction": direction_val,
                "distance": distance_val,
                "distance_signed": distance_signed_val,
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
