"""Tests for the canonical stage schema allow-lists.

Verifies that silver and gold column sets are disjoint, contain the
expected fields, and that the stream wire schema matches silver.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.qmachina.stage_schema import (
    GOLD_COLS,
    GOLD_SCORING_FLOAT_COLS,
    GOLD_SCORING_INT_COL_DTYPES,
    GOLD_FORCE_FLOAT_COLS,
    GOLD_FORCE_INT_COL_DTYPES,
    SILVER_COLS,
    SILVER_FLOAT_COLS,
    SILVER_INT_COL_DTYPES,
)
from src.qmachina.stream_contract import grid_schema


# ---------------------------------------------------------------------------
# Disjointness
# ---------------------------------------------------------------------------

def test_silver_and_gold_cols_are_disjoint() -> None:
    overlap = SILVER_COLS & GOLD_COLS
    assert overlap == frozenset(), f"Overlap between silver and gold: {overlap}"


# ---------------------------------------------------------------------------
# Silver completeness
# ---------------------------------------------------------------------------

_EXPECTED_SILVER_FLOAT = {
    "add_mass", "pull_mass", "fill_mass", "rest_depth", "bid_depth", "ask_depth",
    "v_add", "v_pull", "v_fill", "v_rest_depth", "v_bid_depth", "v_ask_depth",
    "a_add", "a_pull", "a_fill", "a_rest_depth", "a_bid_depth", "a_ask_depth",
    "j_add", "j_pull", "j_fill", "j_rest_depth", "j_bid_depth", "j_ask_depth",
}

_EXPECTED_SILVER_INT = {
    "k", "last_event_id",
    "best_ask_move_ticks", "best_bid_move_ticks",
    "ask_reprice_sign", "bid_reprice_sign",
    "microstate_id", "chase_up_flag", "chase_down_flag",
}


def test_silver_float_cols_complete() -> None:
    assert _EXPECTED_SILVER_FLOAT == set(SILVER_FLOAT_COLS)


def test_silver_int_cols_complete() -> None:
    assert _EXPECTED_SILVER_INT == set(SILVER_INT_COL_DTYPES)


def test_silver_cols_union() -> None:
    assert SILVER_COLS == frozenset(SILVER_FLOAT_COLS) | frozenset(SILVER_INT_COL_DTYPES)


# ---------------------------------------------------------------------------
# Gold completeness
# ---------------------------------------------------------------------------

_EXPECTED_GOLD_VP_FLOAT = {
    "pressure_variant", "vacuum_variant", "composite",
    "composite_d1", "composite_d2", "composite_d3",
}

_EXPECTED_GOLD_VP_INT = {"state5_code"}
_EXPECTED_GOLD_SCORING_FLOAT = {"flow_score"}
_EXPECTED_GOLD_SCORING_INT = {"flow_state_code"}


def test_gold_vp_float_cols_complete() -> None:
    assert _EXPECTED_GOLD_VP_FLOAT == set(GOLD_FORCE_FLOAT_COLS)


def test_gold_vp_int_cols_complete() -> None:
    assert _EXPECTED_GOLD_VP_INT == set(GOLD_FORCE_INT_COL_DTYPES)


def test_gold_scoring_float_cols_complete() -> None:
    assert _EXPECTED_GOLD_SCORING_FLOAT == set(GOLD_SCORING_FLOAT_COLS)


def test_gold_scoring_int_cols_complete() -> None:
    assert _EXPECTED_GOLD_SCORING_INT == set(GOLD_SCORING_INT_COL_DTYPES)


def test_gold_cols_union() -> None:
    expected = (
        frozenset(GOLD_FORCE_FLOAT_COLS)
        | frozenset(GOLD_FORCE_INT_COL_DTYPES)
        | frozenset(GOLD_SCORING_FLOAT_COLS)
        | frozenset(GOLD_SCORING_INT_COL_DTYPES)
    )
    assert GOLD_COLS == expected


# ---------------------------------------------------------------------------
# Wire schema matches silver
# ---------------------------------------------------------------------------

def test_wire_schema_fields_are_subset_of_silver() -> None:
    schema = grid_schema()
    wire_fields = {f.name for f in schema}
    non_silver = wire_fields - SILVER_COLS
    assert non_silver == set(), f"Non-silver fields in wire schema: {non_silver}"


def test_wire_schema_k_is_int32() -> None:
    schema = grid_schema()
    k_field = next(f for f in schema if f.name == "k")
    assert k_field.type == pa.int32()


def test_wire_schema_last_event_id_is_int64() -> None:
    schema = grid_schema()
    field = next(f for f in schema if f.name == "last_event_id")
    assert field.type == pa.int64()


def test_wire_schema_excludes_all_gold_cols() -> None:
    schema = grid_schema()
    wire_fields = {f.name for f in schema}
    gold_in_wire = wire_fields & GOLD_COLS
    assert gold_in_wire == set(), f"Gold fields in wire schema: {gold_in_wire}"
