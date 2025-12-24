from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnsembleResult:
    p_break: float
    mix_weight: float
    disagreement_penalty: float


def _mix_weight(similarity: float, entropy: float) -> float:
    """
    Weight toward tree model when similarity is low or kNN entropy is high.
    """
    weight = 0.7 - 0.4 * similarity + 0.4 * entropy
    return max(0.2, min(0.8, weight))


def combine_probabilities(
    tree_prob: float,
    knn_prob: float,
    similarity: float,
    entropy: float,
    disagreement_threshold: float = 0.4,
    similarity_threshold: float = 0.6
) -> EnsembleResult:
    mix_weight = _mix_weight(similarity, entropy)
    combined = mix_weight * tree_prob + (1.0 - mix_weight) * knn_prob

    disagreement = abs(tree_prob - knn_prob)
    penalty = 0.0
    if similarity >= similarity_threshold and disagreement >= disagreement_threshold:
        penalty = min(0.5, disagreement * 0.5)
        combined = 0.5 + (combined - 0.5) * (1.0 - penalty)

    return EnsembleResult(
        p_break=combined,
        mix_weight=mix_weight,
        disagreement_penalty=penalty
    )
