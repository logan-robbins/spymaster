from __future__ import annotations

from dataclasses import dataclass
from typing import List


IDENTITY_COLUMNS = {
    "event_id",
    "ts_ns",
    "confirm_ts_ns",
    "date",
    "symbol"
}

LABEL_COLUMNS = {
    "outcome",
    "future_price_5min",
    "excursion_max",
    "excursion_min",
    "strength_signed",
    "strength_abs",
    "time_to_threshold_1",
    "time_to_threshold_2",
    "tradeable_1",
    "tradeable_2",
    "anchor_spot"
}

LABEL_PREFIXES = (
    "outcome_",
    "future_price_",
    "excursion_",
    "strength_",
    "time_to_threshold_",
    "tradeable_",
    "anchor_spot_",
    "confirm_ts_ns_",
)

BASE_CATEGORICAL = ["level_kind_name", "direction"]
MECHANICS_CATEGORICAL = ["barrier_state", "fuel_effect", "gamma_bucket"]

STAGE_B_ONLY_PREFIXES = (
    "barrier_",
    "tape_",
    "wall_ratio",
    "liquidity_pressure",
    "tape_pressure",
    "net_break_pressure",
    "barrier_replenishment_trend",
    "barrier_delta_liq_trend",
    "tape_velocity_trend",
    "tape_imbalance_trend"
)

TA_PREFIXES = (
    "approach_",
    "dist_",
    "distance",
    "sma_",
    "mean_reversion_",
    "confluence_",
    "bars_since_open",
    "is_first_15m",
    "atr",
    "level_price_pct",
    "direction_sign",
    "attempt_",
    "prior_touches"
)

MECHANICS_PREFIXES = (
    "barrier_",
    "tape_",
    "fuel_",
    "gamma_",
    "wall_ratio",
    "liquidity_pressure",
    "tape_pressure",
    "gamma_pressure",
    "dealer_pressure",
    "net_break_pressure",
    "reversion_pressure"
)


@dataclass
class FeatureSet:
    numeric: List[str]
    categorical: List[str]


def select_features(df, stage: str, ablation: str = "full") -> FeatureSet:
    """
    Build feature lists for Stage A/B with ablation support.
    """
    stage = stage.lower()
    ablation = ablation.lower()

    exclude = set(IDENTITY_COLUMNS) | set(LABEL_COLUMNS)
    exclude.add("level_kind")

    candidate_cols = [
        c for c in df.columns
        if c not in exclude and not c.startswith(LABEL_PREFIXES)
    ]

    if stage == "stage_a":
        candidate_cols = [
            c for c in candidate_cols
            if not c.startswith(STAGE_B_ONLY_PREFIXES)
        ]
    elif stage != "stage_b":
        raise ValueError(f"Unknown stage: {stage}")

    def _is_ta(col: str) -> bool:
        return col.startswith(TA_PREFIXES) or col in {
            "spot",
            "level_price",
            "distance",
            "distance_signed",
            "level_price_pct"
        }

    def _is_mechanics(col: str) -> bool:
        return col.startswith(MECHANICS_PREFIXES)

    if ablation == "ta":
        candidate_cols = [c for c in candidate_cols if _is_ta(c)]
        categorical = list(BASE_CATEGORICAL)
    elif ablation == "mechanics":
        candidate_cols = [c for c in candidate_cols if _is_mechanics(c)]
        categorical = list(BASE_CATEGORICAL + MECHANICS_CATEGORICAL)
    elif ablation == "full":
        categorical = list(BASE_CATEGORICAL + MECHANICS_CATEGORICAL)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    numeric = [c for c in candidate_cols if c not in categorical]
    categorical = [c for c in categorical if c in df.columns]

    return FeatureSet(numeric=numeric, categorical=categorical)
