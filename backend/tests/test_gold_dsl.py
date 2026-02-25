"""Tests for Gold DSL schema, validation, preview, compat, and API endpoints."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.gold_dsl.compat import dsl_to_gold_config, gold_config_to_dsl
from src.gold_dsl.preview import execute_dsl_preview
from src.gold_dsl.schema import (
    ArithmeticExpr,
    GoldDslSpec,
    NormExpr,
    OutputNode,
    SilverRef,
    TemporalWindow,
)
from src.gold_dsl.validate import validate_dsl
from src.qmachina.gold_config import GoldFeatureConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_spec() -> GoldDslSpec:
    """Build a minimal valid spec: SilverRef -> NormExpr -> OutputNode."""
    return GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="v_add"),
            "normed": NormExpr(type="norm", source="src", method="log1p"),
            "out": OutputNode(type="output", source="normed", name="v_add_log1p"),
        },
    )


def _make_synthetic_parquets(tmp_dir: Path, n_bins: int = 20, n_k: int = 5) -> tuple[Path, Path]:
    """Create synthetic bins.parquet and grid_clean.parquet for preview tests.

    Args:
        tmp_dir: Directory to write parquets into.
        n_bins: Number of bins.
        n_k: Number of relative tick levels.

    Returns:
        Tuple of (bins_parquet_path, grid_parquet_path).
    """
    bin_seqs = list(range(n_bins))
    bins_df = pd.DataFrame({"bin_seq": bin_seqs})
    bins_path = tmp_dir / "bins.parquet"
    bins_df.to_parquet(bins_path, index=False)

    rows = []
    rng = np.random.default_rng(42)
    for bs in bin_seqs:
        for k in range(-n_k // 2, n_k // 2 + 1):
            rows.append({
                "bin_seq": bs,
                "k": k,
                "v_add": rng.normal(0.5, 0.1),
                "v_pull": rng.normal(0.3, 0.1),
                "v_fill": rng.normal(0.2, 0.05),
                "v_rest_depth": rng.normal(0.0, 0.3),
                "a_add": rng.normal(0.0, 0.05),
                "a_pull": rng.normal(0.0, 0.05),
                "rest_depth": rng.exponential(2.0),
                "add_mass": rng.exponential(1.0),
                "pull_mass": rng.exponential(0.8),
                "fill_mass": rng.exponential(0.5),
                "bid_depth": rng.exponential(3.0),
                "ask_depth": rng.exponential(3.0),
                "v_bid_depth": rng.normal(0.0, 0.1),
                "v_ask_depth": rng.normal(0.0, 0.1),
                "a_fill": rng.normal(0.0, 0.02),
                "a_rest_depth": rng.normal(0.0, 0.02),
                "a_bid_depth": rng.normal(0.0, 0.02),
                "a_ask_depth": rng.normal(0.0, 0.02),
                "j_add": rng.normal(0.0, 0.01),
                "j_pull": rng.normal(0.0, 0.01),
                "j_fill": rng.normal(0.0, 0.01),
                "j_rest_depth": rng.normal(0.0, 0.01),
                "j_bid_depth": rng.normal(0.0, 0.01),
                "j_ask_depth": rng.normal(0.0, 0.01),
                "last_event_id": bs * 100 + k,
                "best_ask_move_ticks": 0,
                "best_bid_move_ticks": 0,
                "ask_reprice_sign": 0,
                "bid_reprice_sign": 0,
                "microstate_id": 0,
                "chase_up_flag": 0,
                "chase_down_flag": 0,
            })

    grid_df = pd.DataFrame(rows)
    grid_path = tmp_dir / "grid_clean.parquet"
    grid_df.to_parquet(grid_path, index=False)

    return bins_path, grid_path


def _build_api_app():
    """Create a minimal FastAPI app with the gold DSL router."""
    from fastapi import FastAPI

    from src.qmachina.api_gold_dsl import router

    app = FastAPI()
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Test 1: Valid simple spec parses cleanly
# ---------------------------------------------------------------------------


def test_valid_simple_spec_parses() -> None:
    """A minimal SilverRef -> NormExpr -> OutputNode spec validates with no errors."""
    spec = _make_simple_spec()
    errors = validate_dsl(spec)
    assert errors == [], f"Expected no errors, got: {errors}"
    assert spec.spec_hash()


# ---------------------------------------------------------------------------
# Test 2: Cycle detection
# ---------------------------------------------------------------------------


def test_cycle_detection() -> None:
    """Cycle A -> B -> A returns a cycle error."""
    spec = GoldDslSpec(
        version=1,
        nodes={
            "a": TemporalWindow(type="temporal_window", source="b", window_bins=5, agg="mean"),
            "b": NormExpr(type="norm", source="a", method="tanh"),
            "out": OutputNode(type="output", source="a", name="cyclic_out"),
        },
    )
    errors = validate_dsl(spec)
    cycle_errors = [e for e in errors if "Cycle" in e or "cycle" in e.lower()]
    assert len(cycle_errors) >= 1, f"Expected cycle error, got: {errors}"


# ---------------------------------------------------------------------------
# Test 3: Unknown source reference
# ---------------------------------------------------------------------------


def test_unknown_source_reference() -> None:
    """Reference to a nonexistent node name returns an error."""
    spec = GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="v_add"),
            "normed": NormExpr(type="norm", source="nonexistent", method="log1p"),
            "out": OutputNode(type="output", source="normed", name="bad_ref_out"),
        },
    )
    errors = validate_dsl(spec)
    ref_errors = [e for e in errors if "nonexistent" in e]
    assert len(ref_errors) >= 1, f"Expected unknown ref error, got: {errors}"


# ---------------------------------------------------------------------------
# Test 4: Missing ewm alpha
# ---------------------------------------------------------------------------


def test_missing_ewm_alpha() -> None:
    """TemporalWindow with agg='ewm' but no alpha returns an error."""
    spec = GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="v_add"),
            "ewm_no_alpha": TemporalWindow(
                type="temporal_window", source="src", window_bins=10, agg="ewm"
            ),
            "out": OutputNode(type="output", source="ewm_no_alpha", name="ewm_out"),
        },
    )
    errors = validate_dsl(spec)
    alpha_errors = [e for e in errors if "alpha" in e.lower()]
    assert len(alpha_errors) >= 1, f"Expected ewm alpha error, got: {errors}"


# ---------------------------------------------------------------------------
# Test 5: SilverRef to nonexistent column
# ---------------------------------------------------------------------------


def test_silver_ref_nonexistent_column() -> None:
    """SilverRef to a column not in SILVER_COLS returns an error."""
    spec = GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="nonexistent_field"),
            "out": OutputNode(type="output", source="src", name="bad_col_out"),
        },
    )
    errors = validate_dsl(spec)
    col_errors = [e for e in errors if "nonexistent_field" in e]
    assert len(col_errors) >= 1, f"Expected silver col error, got: {errors}"


# ---------------------------------------------------------------------------
# Test 6: Preview executor returns stats dict
# ---------------------------------------------------------------------------


def test_preview_executor_returns_stats() -> None:
    """Preview executor returns dict with correct top-level keys."""
    spec = _make_simple_spec()

    with tempfile.TemporaryDirectory() as tmp_dir:
        bins_path, grid_path = _make_synthetic_parquets(Path(tmp_dir))
        result = execute_dsl_preview(spec, bins_path, grid_path, sample_bins=10)

    assert "output_stats" in result
    assert "n_bins_sampled" in result
    assert "execution_time_ms" in result
    assert "spec_hash" in result

    assert "v_add_log1p" in result["output_stats"]
    stats = result["output_stats"]["v_add_log1p"]
    for key in ("mean", "std", "pct25", "pct50", "pct75", "n_valid", "n_nan"):
        assert key in stats, f"Missing stat key: {key}"

    assert result["n_bins_sampled"] <= 10
    assert result["execution_time_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test 7: Legacy compat gold_config_to_dsl
# ---------------------------------------------------------------------------


def test_legacy_compat_gold_config_to_dsl() -> None:
    """gold_config_to_dsl produces a valid spec from GoldFeatureConfig."""
    config = GoldFeatureConfig(
        c1_v_add=1.0,
        c2_v_rest_pos=0.5,
        c3_a_add=0.3,
        c4_v_pull=1.0,
        c5_v_fill=1.5,
        c6_v_rest_neg=0.5,
        c7_a_pull=0.3,
        flow_windows=[10, 30, 60],
        flow_rollup_weights=[0.5, 0.3, 0.2],
    )
    dsl_spec = gold_config_to_dsl(config)
    errors = validate_dsl(dsl_spec)
    assert errors == [], f"Legacy-converted spec should be valid, got errors: {errors}"

    output_names = [
        node.name
        for node in dsl_spec.nodes.values()
        if isinstance(node, OutputNode)
    ]
    assert len(output_names) >= 1, "Must have at least one output"


# ---------------------------------------------------------------------------
# Test 8: Lineage - different hashes produce non-empty diff
# ---------------------------------------------------------------------------


def test_lineage_different_hashes() -> None:
    """Two specs with different nodes produce different hashes and non-empty diff."""
    spec_a = _make_simple_spec()

    spec_b = GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="v_pull"),
            "normed": NormExpr(type="norm", source="src", method="tanh"),
            "out": OutputNode(type="output", source="normed", name="v_pull_tanh"),
        },
    )

    hash_a = spec_a.spec_hash()
    hash_b = spec_b.spec_hash()
    assert hash_a != hash_b

    # Check structural diff
    names_a = set(spec_a.nodes.keys())
    names_b = set(spec_b.nodes.keys())
    common = names_a & names_b
    modified = [
        name
        for name in common
        if spec_a.nodes[name].model_dump() != spec_b.nodes[name].model_dump()
    ]
    assert len(modified) > 0, "Expected modified nodes for different specs"


# ---------------------------------------------------------------------------
# Test 9: API validate endpoint - valid payload
# ---------------------------------------------------------------------------


def test_api_validate_valid_payload() -> None:
    """POST /v1/gold/validate with valid spec returns {valid: true}."""
    app = _build_api_app()
    client = TestClient(app)

    spec_dict = _make_simple_spec().model_dump(mode="json")
    response = client.post("/v1/gold/validate", json={"spec": spec_dict})
    assert response.status_code == 200

    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == []
    assert body["spec_hash"] is not None


# ---------------------------------------------------------------------------
# Test 10: API validate endpoint - invalid payload (cycle)
# ---------------------------------------------------------------------------


def test_api_validate_invalid_cycle() -> None:
    """POST /v1/gold/validate with cyclic spec returns {valid: false, errors: [...]}."""
    app = _build_api_app()
    client = TestClient(app)

    spec_dict = GoldDslSpec(
        version=1,
        nodes={
            "a": TemporalWindow(
                type="temporal_window", source="b", window_bins=5, agg="mean"
            ),
            "b": NormExpr(type="norm", source="a", method="tanh"),
            "out": OutputNode(type="output", source="a", name="cyclic_out"),
        },
    ).model_dump(mode="json")

    response = client.post("/v1/gold/validate", json={"spec": spec_dict})
    assert response.status_code == 200

    body = response.json()
    assert body["valid"] is False
    assert len(body["errors"]) > 0
    assert any("ycle" in e for e in body["errors"])


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


def test_spec_hash_deterministic() -> None:
    """Same spec content always produces the same hash."""
    spec1 = _make_simple_spec()
    spec2 = _make_simple_spec()
    assert spec1.spec_hash() == spec2.spec_hash()


def test_from_dict_roundtrip() -> None:
    """Spec can roundtrip through dict serialization."""
    original = _make_simple_spec()
    raw = original.model_dump(mode="json")
    restored = GoldDslSpec.from_dict(raw)
    assert original.spec_hash() == restored.spec_hash()


def test_dsl_to_gold_config_reverse_mapping() -> None:
    """dsl_to_gold_config succeeds for compat-generated specs."""
    config = GoldFeatureConfig(
        c1_v_add=1.0,
        c2_v_rest_pos=0.5,
        c3_a_add=0.3,
        c4_v_pull=1.0,
        c5_v_fill=1.5,
        c6_v_rest_neg=0.5,
        c7_a_pull=0.3,
        flow_windows=[10],
        flow_rollup_weights=[1.0],
    )
    dsl_spec = gold_config_to_dsl(config)
    recovered = dsl_to_gold_config(dsl_spec)
    assert recovered is not None
    assert isinstance(recovered, GoldFeatureConfig)


def test_dsl_to_gold_config_returns_none_for_custom_spec() -> None:
    """dsl_to_gold_config returns None for a non-legacy spec."""
    spec = _make_simple_spec()
    result = dsl_to_gold_config(spec)
    assert result is None


def test_zscore_missing_window_bins() -> None:
    """NormExpr with method='zscore' but no window_bins returns error."""
    spec = GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="v_add"),
            "zs": NormExpr(type="norm", source="src", method="zscore"),
            "out": OutputNode(type="output", source="zs", name="zs_out"),
        },
    )
    errors = validate_dsl(spec)
    zs_errors = [e for e in errors if "zscore" in e.lower() and "window_bins" in e.lower()]
    assert len(zs_errors) >= 1


def test_no_output_node_error() -> None:
    """Spec with zero OutputNodes returns an error."""
    spec = GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="v_add"),
        },
    )
    errors = validate_dsl(spec)
    out_errors = [e for e in errors if "OutputNode" in e]
    assert len(out_errors) >= 1


def test_duplicate_output_names() -> None:
    """Two OutputNodes with the same name return an error."""
    spec = GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="v_add"),
            "out1": OutputNode(type="output", source="src", name="duplicated"),
            "out2": OutputNode(type="output", source="src", name="duplicated"),
        },
    )
    errors = validate_dsl(spec)
    dup_errors = [e for e in errors if "Duplicate" in e or "duplicate" in e.lower()]
    assert len(dup_errors) >= 1


def test_api_lineage_compare() -> None:
    """POST /v1/gold/lineage/compare returns structural diff."""
    app = _build_api_app()
    client = TestClient(app)

    spec_a = _make_simple_spec().model_dump(mode="json")
    spec_b = GoldDslSpec(
        version=1,
        nodes={
            "src": SilverRef(type="silver_ref", field="v_pull"),
            "normed": NormExpr(type="norm", source="src", method="tanh"),
            "out": OutputNode(type="output", source="normed", name="v_pull_tanh"),
        },
    ).model_dump(mode="json")

    response = client.post(
        "/v1/gold/lineage/compare",
        json={"spec_a": spec_a, "spec_b": spec_b},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hashes_match"] is False
    assert len(body["modified_nodes"]) > 0


def test_api_from_legacy() -> None:
    """GET /v1/gold/from_legacy returns a valid spec dict."""
    app = _build_api_app()
    client = TestClient(app)

    response = client.get("/v1/gold/from_legacy")
    assert response.status_code == 200
    body = response.json()
    assert "spec" in body
    assert "spec_hash" in body
    assert isinstance(body["spec"]["nodes"], dict)
