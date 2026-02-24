"""Tests for VisualizationConfig, CellShaderConfig, OverlaySpec models.

Covers:
- Model field validation
- VisualizationConfig validator: cell_shader required for heatmap, unique IDs, unique z_orders
- ServingSpec default visualization = VP default
- stream_contract payload includes gold_config + visualization
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.qmachina.serving_config import (
    CellShaderConfig,
    OverlaySpec,
    ServingSpec,
    VisualizationConfig,
)


# ------------------------------------------------------------------ CellShaderConfig


class TestCellShaderConfig:
    def test_defaults(self) -> None:
        cfg = CellShaderConfig()
        assert cfg.signal_field == "flow_score"
        assert cfg.depth_field == "rest_depth"
        assert cfg.color_scheme == "pressure_vacuum"
        assert cfg.normalization == "adaptive_running_max"
        assert cfg.gamma == pytest.approx(0.7)

    def test_custom_fields(self) -> None:
        cfg = CellShaderConfig(
            signal_field="momentum_score",
            depth_field="bid_depth",
            color_scheme="thermal",
            normalization="zscore_clamp",
            gamma=1.2,
        )
        assert cfg.signal_field == "momentum_score"
        assert cfg.color_scheme == "thermal"
        assert cfg.gamma == pytest.approx(1.2)

    def test_invalid_color_scheme(self) -> None:
        with pytest.raises(ValidationError):
            CellShaderConfig(color_scheme="neon_pink")  # type: ignore[arg-type]

    def test_invalid_normalization(self) -> None:
        with pytest.raises(ValidationError):
            CellShaderConfig(normalization="none")  # type: ignore[arg-type]

    def test_gamma_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CellShaderConfig(gamma=0.0)
        with pytest.raises(ValidationError):
            CellShaderConfig(gamma=5.1)
        # Boundary values
        CellShaderConfig(gamma=0.01)
        CellShaderConfig(gamma=5.0)

    def test_all_color_schemes_valid(self) -> None:
        for scheme in ("pressure_vacuum", "thermal", "mono_green", "mono_red"):
            cfg = CellShaderConfig(color_scheme=scheme)  # type: ignore[arg-type]
            assert cfg.color_scheme == scheme

    def test_all_normalizations_valid(self) -> None:
        for norm in ("adaptive_running_max", "zscore_clamp", "identity_clamp"):
            cfg = CellShaderConfig(normalization=norm)  # type: ignore[arg-type]
            assert cfg.normalization == norm


# ------------------------------------------------------------------ OverlaySpec


class TestOverlaySpec:
    def test_required_fields(self) -> None:
        o = OverlaySpec(id="my_overlay", type="spot_trail")
        assert o.id == "my_overlay"
        assert o.type == "spot_trail"
        assert o.enabled is True
        assert o.z_order == 0
        assert o.params == {}

    def test_with_params(self) -> None:
        o = OverlaySpec(
            id="proj",
            type="projection_bands",
            z_order=30,
            params={"signal_field": "flow_score", "composite_scale": 8.0},
        )
        assert o.params["signal_field"] == "flow_score"
        assert o.params["composite_scale"] == pytest.approx(8.0)

    def test_disabled(self) -> None:
        o = OverlaySpec(id="x", type="bbo_trail", enabled=False)
        assert o.enabled is False


# ------------------------------------------------------------------ VisualizationConfig


class TestVisualizationConfig:
    def test_heatmap_requires_cell_shader(self) -> None:
        with pytest.raises(ValidationError, match="cell_shader is required"):
            VisualizationConfig(display_mode="heatmap", cell_shader=None, overlays=[])

    def test_candle_without_cell_shader_ok(self) -> None:
        cfg = VisualizationConfig(display_mode="candle", cell_shader=None, overlays=[])
        assert cfg.display_mode == "candle"
        assert cfg.cell_shader is None

    def test_duplicate_overlay_ids(self) -> None:
        with pytest.raises(ValidationError, match="overlay IDs must be unique"):
            VisualizationConfig(
                display_mode="heatmap",
                cell_shader=CellShaderConfig(),
                overlays=[
                    OverlaySpec(id="dup", type="spot_trail", z_order=0),
                    OverlaySpec(id="dup", type="bbo_trail", z_order=10),
                ],
            )

    def test_duplicate_z_orders(self) -> None:
        with pytest.raises(ValidationError, match="z_order values must be unique"):
            VisualizationConfig(
                display_mode="heatmap",
                cell_shader=CellShaderConfig(),
                overlays=[
                    OverlaySpec(id="a", type="spot_trail", z_order=5),
                    OverlaySpec(id="b", type="bbo_trail", z_order=5),
                ],
            )

    def test_projection_bands_requires_signal_field(self) -> None:
        with pytest.raises(ValidationError, match="requires params.signal_field"):
            VisualizationConfig(
                display_mode="heatmap",
                cell_shader=CellShaderConfig(),
                overlays=[
                    OverlaySpec(
                        id="proj",
                        type="projection_bands",
                        z_order=0,
                        params={},  # missing signal_field
                    )
                ],
            )

    def test_projection_bands_with_signal_field_ok(self) -> None:
        cfg = VisualizationConfig(
            display_mode="heatmap",
            cell_shader=CellShaderConfig(),
            overlays=[
                OverlaySpec(
                    id="proj",
                    type="projection_bands",
                    z_order=0,
                    params={"signal_field": "flow_score"},
                )
            ],
        )
        assert cfg.overlays[0].params["signal_field"] == "flow_score"

    def test_valid_heatmap_config(self) -> None:
        cfg = VisualizationConfig(
            display_mode="heatmap",
            cell_shader=CellShaderConfig(),
            overlays=[
                OverlaySpec(id="vp_gold", type="vp_gold", z_order=0),
                OverlaySpec(id="spot_trail", type="spot_trail", z_order=10),
            ],
        )
        assert cfg.display_mode == "heatmap"
        assert len(cfg.overlays) == 2

    def test_default_heatmap_classmethod(self) -> None:
        cfg = VisualizationConfig.default_heatmap()
        assert cfg.display_mode == "heatmap"
        assert cfg.cell_shader is not None
        assert cfg.cell_shader.signal_field == "flow_score"
        overlay_ids = {o.id for o in cfg.overlays}
        assert "vp_gold" in overlay_ids
        assert "spot_trail" in overlay_ids
        assert "bbo_trail" in overlay_ids
        assert "projection_bands" in overlay_ids
        # projection_bands has signal_field
        pb = next(o for o in cfg.overlays if o.type == "projection_bands")
        assert pb.params["signal_field"] == "flow_score"
        assert pb.params["composite_scale"] == pytest.approx(8.0)

    def test_model_dump_roundtrip(self) -> None:
        cfg = VisualizationConfig.default_heatmap()
        data = cfg.model_dump()
        cfg2 = VisualizationConfig.model_validate(data)
        assert cfg2.display_mode == cfg.display_mode
        assert cfg2.cell_shader is not None
        assert cfg2.cell_shader.signal_field == cfg.cell_shader.signal_field  # type: ignore[union-attr]

    def test_empty_overlays_heatmap_ok(self) -> None:
        # No overlays but valid cell_shader — should pass
        cfg = VisualizationConfig(
            display_mode="heatmap",
            cell_shader=CellShaderConfig(),
            overlays=[],
        )
        assert cfg.overlays == []


# ------------------------------------------------------------------ ServingSpec default


class TestServingSpecVisualizationDefault:
    def test_serving_spec_gets_vp_default(self) -> None:
        spec = ServingSpec(name="test", pipeline="default_pipeline")
        viz = spec.visualization
        assert viz.display_mode == "heatmap"
        assert viz.cell_shader is not None
        assert viz.cell_shader.signal_field == "flow_score"
        overlay_types = {o.type for o in viz.overlays}
        assert "vp_gold" in overlay_types
        assert "projection_bands" in overlay_types

    def test_serving_spec_custom_visualization(self) -> None:
        spec = ServingSpec(
            name="candle_test",
            pipeline="default_pipeline",
            visualization=VisualizationConfig(
                display_mode="candle",
                cell_shader=None,
                overlays=[
                    OverlaySpec(id="spot_trail", type="spot_trail", z_order=10),
                ],
            ),
        )
        assert spec.visualization.display_mode == "candle"
        assert spec.visualization.cell_shader is None

    def test_serving_spec_yaml_roundtrip(self, tmp_path) -> None:
        spec = ServingSpec(name="test", pipeline="default_pipeline")
        yaml_path = tmp_path / "test.yaml"
        spec.to_yaml(yaml_path)
        loaded = ServingSpec.from_yaml(yaml_path)
        assert loaded.visualization.display_mode == "heatmap"
        assert loaded.visualization.cell_shader is not None
        assert loaded.visualization.cell_shader.signal_field == "flow_score"
        pb = next(o for o in loaded.visualization.overlays if o.type == "projection_bands")
        assert pb.params["signal_field"] == "flow_score"


# ------------------------------------------------------------------ Wire payload


class TestStreamContractVisualization:
    """Verify build_runtime_config_payload includes gold_config and visualization."""

    @pytest.fixture()
    def runtime_config(self):
        from src.qmachina.config import build_config_from_mapping  # noqa: PLC0415

        return build_config_from_mapping(
            {
                "product_type": "future_mbo",
                "symbol": "ESH6",
                "symbol_root": "ES",
                "price_scale": 1e-9,
                "tick_size": 0.25,
                "bucket_size_dollars": 0.25,
                "rel_tick_size": 0.25,
                "grid_radius_ticks": 40,
                "cell_width_ms": 1000,
                "n_absolute_ticks": 8192,
                "flow_windows": [1, 3, 5],
                "flow_rollup_weights": [0.5, 0.3, 0.2],
                "flow_derivative_weights": [0.55, 0.30, 0.15],
                "flow_tanh_scale": 3.0,
                "flow_neutral_threshold": 0.15,
                "flow_zscore_window_bins": 300,
                "flow_zscore_min_periods": 75,
                "projection_horizons_bins": [1, 2, 4],
                "contract_multiplier": 50,
                "qty_unit": "contracts",
                "price_decimals": 2,
                "tau_velocity": 2.0,
                "tau_acceleration": 5.0,
                "tau_jerk": 10.0,
                "tau_rest_decay": 30.0,
                "c1_v_add": 1.0,
                "c2_v_rest_pos": 0.5,
                "c3_a_add": 0.3,
                "c4_v_pull": 1.0,
                "c5_v_fill": 1.5,
                "c6_v_rest_neg": 0.5,
                "c7_a_pull": 0.3,
            },
            source="test",
        )

    def test_payload_includes_gold_config(self, runtime_config) -> None:
        from src.qmachina.stream_contract import build_runtime_config_payload, grid_schema

        schema = grid_schema()
        payload = build_runtime_config_payload(runtime_config, schema)
        assert "gold_config" in payload
        gc = payload["gold_config"]
        assert "c1_v_add" in gc
        assert "flow_windows" in gc
        assert gc["c1_v_add"] == pytest.approx(1.0)

    def test_payload_includes_visualization(self, runtime_config) -> None:
        from src.qmachina.stream_contract import build_runtime_config_payload, grid_schema

        schema = grid_schema()
        payload = build_runtime_config_payload(runtime_config, schema)
        assert "visualization" in payload
        viz = payload["visualization"]
        assert viz["display_mode"] == "heatmap"
        assert viz["cell_shader"] is not None
        assert viz["cell_shader"]["signal_field"] == "flow_score"

    def test_payload_visualization_with_resolved_serving(self, runtime_config) -> None:
        from unittest.mock import MagicMock

        from src.qmachina.stream_contract import build_runtime_config_payload, grid_schema

        viz = VisualizationConfig(
            display_mode="candle",
            cell_shader=None,
            overlays=[OverlaySpec(id="spot_trail", type="spot_trail", z_order=10)],
        )
        mock_spec = MagicMock()
        mock_spec.visualization = viz
        mock_spec.to_runtime_config_json.return_value = {"serving_name": "candle_alias"}

        resolved = MagicMock()
        resolved.spec = mock_spec
        resolved.alias = "candle_alias"
        resolved.serving_id = "candle_v1"

        schema = grid_schema()
        payload = build_runtime_config_payload(runtime_config, schema, resolved_serving=resolved)
        assert payload["visualization"]["display_mode"] == "candle"
        assert payload["visualization"]["cell_shader"] is None
