"""Canonical stage schema: explicit allow-lists for silver and gold columns.

Stage contracts are defined here as allow-lists, not as exclusion constants.
No model-choice features appear in the silver allow-list.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Silver stage: base cell tensors + EMA derivatives + BBO metadata
# ---------------------------------------------------------------------------

SILVER_FLOAT_COLS: tuple[str, ...] = (
    "add_mass",
    "pull_mass",
    "fill_mass",
    "rest_depth",
    "bid_depth",
    "ask_depth",
    "v_add",
    "v_pull",
    "v_fill",
    "v_rest_depth",
    "v_bid_depth",
    "v_ask_depth",
    "a_add",
    "a_pull",
    "a_fill",
    "a_rest_depth",
    "a_bid_depth",
    "a_ask_depth",
    "j_add",
    "j_pull",
    "j_fill",
    "j_rest_depth",
    "j_bid_depth",
    "j_ask_depth",
)

SILVER_INT_COL_DTYPES: dict[str, Any] = {
    "k": np.int32,
    "last_event_id": np.int64,
    "best_ask_move_ticks": np.int32,
    "best_bid_move_ticks": np.int32,
    "ask_reprice_sign": np.int8,
    "bid_reprice_sign": np.int8,
    "microstate_id": np.int8,
    "chase_up_flag": np.int8,
    "chase_down_flag": np.int8,
}

SILVER_COLS: frozenset[str] = frozenset(SILVER_FLOAT_COLS) | frozenset(SILVER_INT_COL_DTYPES)

# ---------------------------------------------------------------------------
# Gold stage: force block + spectrum/scoring block
# ---------------------------------------------------------------------------

GOLD_FORCE_FLOAT_COLS: tuple[str, ...] = (
    "pressure_variant",
    "vacuum_variant",
    "composite",
    "composite_d1",
    "composite_d2",
    "composite_d3",
)

GOLD_FORCE_INT_COL_DTYPES: dict[str, Any] = {
    "state5_code": np.int8,
}

GOLD_SCORING_FLOAT_COLS: tuple[str, ...] = ("flow_score",)

GOLD_SCORING_INT_COL_DTYPES: dict[str, Any] = {
    "flow_state_code": np.int8,
}

GOLD_COLS: frozenset[str] = (
    frozenset(GOLD_FORCE_FLOAT_COLS)
    | frozenset(GOLD_FORCE_INT_COL_DTYPES)
    | frozenset(GOLD_SCORING_FLOAT_COLS)
    | frozenset(GOLD_SCORING_INT_COL_DTYPES)
)
