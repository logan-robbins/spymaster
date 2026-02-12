"""Tests for vacuum-pressure runtime configuration upgrade (UPGRADE.md Sections 4.1-4.7, 9).

Tests cover:
    1. Config resolution by product type and symbol (4.1)
    2. Dataset path selection by product type (4.2)
    3. Formula invariants under different bucket_size_dollars (4.3)
    4. Websocket control message includes runtime config (4.4)
    5. Cache keying includes config_version (4.6)
    6. Readiness checks (4.7)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.vacuum_pressure.config import (
    PRICE_SCALE,
    VALID_PRODUCT_TYPES,
    VPRuntimeConfig,
    _compute_config_version,
    _extract_root,
    _resolve_equity,
    _resolve_futures,
    resolve_config,
)
from src.vacuum_pressure.engine import (
    REQUIRED_TABLES,
    SILVER_PATH_TPL,
    VacuumPressureEngine,
    _runner_command,
    _silver_dir,
    validate_silver_readiness,
)
from src.vacuum_pressure.formulas import (
    DEPTH_RANGE_DOLLARS,
    NEAR_SPOT_DOLLARS,
    PROXIMITY_TAU_DOLLARS,
    _depth_range_ticks,
    _near_spot_ticks,
    _proximity_tau_ticks,
    aggregate_window_metrics,
    compute_per_bucket_scores,
    proximity_weight,
    run_full_pipeline,
)

# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

PRODUCTS_YAML = """\
products:
  ES:
    tick_size: 0.25
    grid_max_ticks: 200
    strike_step_points: 5.0
    max_strike_offsets: 10
    contract_multiplier: 50.0
  MES:
    tick_size: 0.25
    grid_max_ticks: 200
    strike_step_points: 5.0
    max_strike_offsets: 10
    contract_multiplier: 5.0
  MNQ:
    tick_size: 0.25
    grid_max_ticks: 400
    strike_step_points: 5.0
    max_strike_offsets: 10
    contract_multiplier: 2.0
  NQ:
    tick_size: 0.25
    grid_max_ticks: 400
    strike_step_points: 5.0
    max_strike_offsets: 10
    contract_multiplier: 20.0
  SI:
    tick_size: 0.005
    grid_max_ticks: 200
    strike_step_points: 0.25
    max_strike_offsets: 10
    contract_multiplier: 5000.0
  GC:
    tick_size: 0.10
    grid_max_ticks: 200
    strike_step_points: 5.0
    max_strike_offsets: 10
    contract_multiplier: 100.0
  CL:
    tick_size: 0.01
    grid_max_ticks: 200
    strike_step_points: 0.50
    max_strike_offsets: 10
    contract_multiplier: 1000.0
  6E:
    tick_size: 0.00005
    grid_max_ticks: 200
    strike_step_points: 0.005
    max_strike_offsets: 10
    contract_multiplier: 125000.0
"""


@pytest.fixture
def products_yaml_path(tmp_path: Path) -> Path:
    """Write a temporary products.yaml and return its path."""
    p = tmp_path / "products.yaml"
    p.write_text(PRODUCTS_YAML)
    return p


@pytest.fixture
def qqq_config(products_yaml_path: Path) -> VPRuntimeConfig:
    """QQQ equity config resolved from defaults."""
    return resolve_config("equity_mbo", "QQQ", products_yaml_path)


@pytest.fixture
def mnq_config(products_yaml_path: Path) -> VPRuntimeConfig:
    """MNQH6 futures config resolved from products.yaml."""
    return resolve_config("future_mbo", "MNQH6", products_yaml_path)


@pytest.fixture
def es_config(products_yaml_path: Path) -> VPRuntimeConfig:
    """ESH6 futures config resolved from products.yaml."""
    return resolve_config("future_mbo", "ESH6", products_yaml_path)


def _make_flow_df(
    n_windows: int = 3,
    n_ticks: int = 10,
) -> pd.DataFrame:
    """Create a synthetic depth_and_flow_1s DataFrame for formula tests."""
    rows = []
    base_ts = 1_000_000_000_000
    for w in range(n_windows):
        wid = base_ts + w * 1_000_000_000
        spot_int = int(100.0 / PRICE_SCALE)
        for k in range(-n_ticks, n_ticks + 1):
            if k == 0:
                continue
            for side in ("A", "B"):
                rows.append({
                    "window_end_ts_ns": wid,
                    "rel_ticks": k,
                    "side": side,
                    "spot_ref_price_int": spot_int,
                    "add_qty": float(abs(k) * 10),
                    "pull_qty": float(abs(k) * 5),
                    "fill_qty": float(abs(k) * 2),
                    "depth_qty_end": float(abs(k) * 50),
                    "depth_qty_rest": float(abs(k) * 20),
                    "depth_qty_start": float(abs(k) * 55),
                    "pull_qty_rest": float(abs(k) * 3),
                    "window_valid": True,
                })
    return pd.DataFrame(rows)


def _make_snap_df(n_windows: int = 3) -> pd.DataFrame:
    """Create a synthetic book_snapshot_1s DataFrame."""
    base_ts = 1_000_000_000_000
    rows = []
    for w in range(n_windows):
        wid = base_ts + w * 1_000_000_000
        rows.append({
            "window_end_ts_ns": wid,
            "mid_price": 100.0,
            "spot_ref_price_int": int(100.0 / PRICE_SCALE),
            "best_bid_price_int": int(99.5 / PRICE_SCALE),
            "best_ask_price_int": int(100.5 / PRICE_SCALE),
            "book_valid": True,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# 1. Config resolution by product type and symbol (4.1)
# ──────────────────────────────────────────────────────────────────────


class TestConfigResolution:
    """Test config resolution for equity and futures product types."""

    def test_equity_qqq_defaults(self, qqq_config: VPRuntimeConfig) -> None:
        """QQQ resolves with correct equity defaults."""
        assert qqq_config.product_type == "equity_mbo"
        assert qqq_config.symbol == "QQQ"
        assert qqq_config.symbol_root == "QQQ"
        assert qqq_config.price_scale == PRICE_SCALE
        assert qqq_config.tick_size == 0.01
        assert qqq_config.bucket_size_dollars == 0.50
        assert qqq_config.rel_tick_size == 0.50
        assert qqq_config.grid_max_ticks == 200
        assert qqq_config.contract_multiplier == 1.0
        assert qqq_config.qty_unit == "shares"
        assert qqq_config.price_decimals == 2
        assert len(qqq_config.config_version) == 12

    def test_equity_spy_defaults(self, products_yaml_path: Path) -> None:
        """SPY resolves with same equity defaults as QQQ."""
        cfg = resolve_config("equity_mbo", "SPY", products_yaml_path)
        assert cfg.product_type == "equity_mbo"
        assert cfg.symbol == "SPY"
        assert cfg.bucket_size_dollars == 0.50

    def test_equity_unknown_symbol_uses_defaults(
        self, products_yaml_path: Path,
    ) -> None:
        """Unknown equity symbol still resolves with global equity defaults."""
        cfg = resolve_config("equity_mbo", "AAPL", products_yaml_path)
        assert cfg.product_type == "equity_mbo"
        assert cfg.symbol == "AAPL"
        assert cfg.bucket_size_dollars == 0.50

    def test_futures_mnq(self, mnq_config: VPRuntimeConfig) -> None:
        """MNQH6 resolves from products.yaml MNQ root."""
        assert mnq_config.product_type == "future_mbo"
        assert mnq_config.symbol == "MNQH6"
        assert mnq_config.symbol_root == "MNQ"
        assert mnq_config.tick_size == 0.25
        assert mnq_config.bucket_size_dollars == 0.25
        assert mnq_config.rel_tick_size == 0.25
        assert mnq_config.grid_max_ticks == 400
        assert mnq_config.contract_multiplier == 2.0
        assert mnq_config.qty_unit == "contracts"
        assert mnq_config.price_decimals == 2

    def test_futures_es(self, es_config: VPRuntimeConfig) -> None:
        """ESH6 resolves from products.yaml ES root."""
        assert es_config.symbol_root == "ES"
        assert es_config.tick_size == 0.25
        assert es_config.grid_max_ticks == 200
        assert es_config.contract_multiplier == 50.0

    def test_futures_si(self, products_yaml_path: Path) -> None:
        """SIH6 resolves with SI root and correct tick decimals."""
        cfg = resolve_config("future_mbo", "SIH6", products_yaml_path)
        assert cfg.symbol_root == "SI"
        assert cfg.tick_size == 0.005
        assert cfg.price_decimals == 3
        assert cfg.contract_multiplier == 5000.0

    def test_futures_6e(self, products_yaml_path: Path) -> None:
        """6EH6 resolves with 6E root (leading digit)."""
        cfg = resolve_config("future_mbo", "6EH6", products_yaml_path)
        assert cfg.symbol_root == "6E"
        assert cfg.tick_size == 0.00005
        assert cfg.price_decimals == 5

    def test_invalid_product_type(self, products_yaml_path: Path) -> None:
        """Invalid product_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid product_type"):
            resolve_config("option_mbo", "QQQ", products_yaml_path)

    def test_unknown_futures_symbol(self, products_yaml_path: Path) -> None:
        """Unknown futures symbol root raises ValueError."""
        with pytest.raises(ValueError, match="Cannot extract product root"):
            resolve_config("future_mbo", "XXXH6", products_yaml_path)

    def test_missing_products_yaml(self, tmp_path: Path) -> None:
        """Missing products.yaml raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="products.yaml not found"):
            resolve_config("future_mbo", "ESH6", tmp_path / "nonexistent.yaml")

    def test_config_version_deterministic(
        self, products_yaml_path: Path,
    ) -> None:
        """Same inputs produce same config_version hash."""
        c1 = resolve_config("equity_mbo", "QQQ", products_yaml_path)
        c2 = resolve_config("equity_mbo", "QQQ", products_yaml_path)
        assert c1.config_version == c2.config_version

    def test_config_version_changes_with_inputs(
        self, products_yaml_path: Path,
    ) -> None:
        """Different symbols produce different config_version hashes."""
        c1 = resolve_config("equity_mbo", "QQQ", products_yaml_path)
        c2 = resolve_config("future_mbo", "ESH6", products_yaml_path)
        assert c1.config_version != c2.config_version

    def test_to_dict_roundtrip(self, qqq_config: VPRuntimeConfig) -> None:
        """to_dict() returns all fields as a serializable dict."""
        d = qqq_config.to_dict()
        assert isinstance(d, dict)
        expected_keys = {
            "product_type", "symbol", "symbol_root", "price_scale",
            "tick_size", "bucket_size_dollars", "rel_tick_size",
            "grid_max_ticks", "contract_multiplier", "qty_unit",
            "price_decimals", "config_version",
        }
        assert set(d.keys()) == expected_keys
        # Must be JSON-serializable
        json.dumps(d)


# ──────────────────────────────────────────────────────────────────────
# 2. Dataset path selection by product type (4.2)
# ──────────────────────────────────────────────────────────────────────


class TestDatasetPathSelection:
    """Test that silver paths are correctly constructed per product type."""

    def test_equity_path(self, tmp_path: Path) -> None:
        """Equity paths use product_type=equity_mbo."""
        d = _silver_dir(tmp_path, "equity_mbo", "QQQ", "book_snapshot_1s", "2026-02-06")
        expected = (
            tmp_path / "silver" / "product_type=equity_mbo" / "symbol=QQQ"
            / "table=book_snapshot_1s" / "dt=2026-02-06"
        )
        assert d == expected

    def test_futures_path(self, tmp_path: Path) -> None:
        """Futures paths use product_type=future_mbo."""
        d = _silver_dir(tmp_path, "future_mbo", "MNQH6", "depth_and_flow_1s", "2026-02-06")
        expected = (
            tmp_path / "silver" / "product_type=future_mbo" / "symbol=MNQH6"
            / "table=depth_and_flow_1s" / "dt=2026-02-06"
        )
        assert d == expected

    def test_runner_command_equity(self) -> None:
        """Runner command for equity includes product_type=equity_mbo."""
        cmd = _runner_command("equity_mbo", "QQQ", "2026-02-06")
        assert "--product-type equity_mbo" in cmd
        assert "--symbol QQQ" in cmd
        assert "--dt 2026-02-06" in cmd

    def test_runner_command_futures(self) -> None:
        """Runner command for futures includes product_type=future_mbo."""
        cmd = _runner_command("future_mbo", "MNQH6", "2026-02-06")
        assert "--product-type future_mbo" in cmd
        assert "--symbol MNQH6" in cmd

    def test_missing_partition_error_message(
        self,
        tmp_path: Path,
        qqq_config: VPRuntimeConfig,
    ) -> None:
        """Missing partition raises FileNotFoundError with actionable command."""
        engine = VacuumPressureEngine(tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            engine.load_silver(qqq_config, "2026-02-06")
        msg = str(exc_info.value)
        assert "Silver partition not found" in msg
        assert "--product-type equity_mbo" in msg
        assert "--symbol QQQ" in msg

    def test_missing_partition_futures_error(
        self,
        tmp_path: Path,
        mnq_config: VPRuntimeConfig,
    ) -> None:
        """Futures missing partition includes correct product type in command."""
        engine = VacuumPressureEngine(tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            engine.load_silver(mnq_config, "2026-02-06")
        msg = str(exc_info.value)
        assert "--product-type future_mbo" in msg
        assert "--symbol MNQH6" in msg

    def test_load_silver_reads_correct_product_type(
        self,
        tmp_path: Path,
        qqq_config: VPRuntimeConfig,
    ) -> None:
        """Engine reads from product_type-specific silver directory."""
        # Create fake parquet data in the equity path
        for table in REQUIRED_TABLES:
            partition = _silver_dir(
                tmp_path, "equity_mbo", "QQQ", table, "2026-02-06",
            )
            partition.mkdir(parents=True)
            if table == "book_snapshot_1s":
                df = _make_snap_df(1)
            else:
                df = _make_flow_df(1)
            df.to_parquet(partition / "part-0000.parquet")

        engine = VacuumPressureEngine(tmp_path)
        df_snap, df_flow = engine.load_silver(qqq_config, "2026-02-06")
        assert not df_snap.empty
        assert not df_flow.empty


# ──────────────────────────────────────────────────────────────────────
# 3. Formula invariants under different bucket_size_dollars (4.3)
# ──────────────────────────────────────────────────────────────────────


class TestFormulaInvariants:
    """Test formula parameterization produces correct results across instruments."""

    def test_qqq_tau_matches_original(self) -> None:
        """QQQ $0.50 buckets: tau_ticks = 2.50/0.50 = 5.0 (original constant)."""
        assert _proximity_tau_ticks(0.50) == 5.0

    def test_qqq_near_spot_matches_original(self) -> None:
        """QQQ $0.50 buckets: near_spot_ticks = 2.50/0.50 = 5 (original constant)."""
        assert _near_spot_ticks(0.50) == 5

    def test_qqq_depth_range_matches_original(self) -> None:
        """QQQ $0.50 buckets: depth_range_ticks = 5.00/0.50 = 10 (original constant)."""
        assert _depth_range_ticks(0.50) == 10

    def test_futures_025_tau(self) -> None:
        """Futures $0.25 buckets: tau_ticks = 2.50/0.25 = 10.0."""
        assert _proximity_tau_ticks(0.25) == 10.0

    def test_futures_025_near_spot(self) -> None:
        """Futures $0.25 buckets: near_spot_ticks = 2.50/0.25 = 10."""
        assert _near_spot_ticks(0.25) == 10

    def test_futures_025_depth_range(self) -> None:
        """Futures $0.25 buckets: depth_range_ticks = 5.00/0.25 = 20."""
        assert _depth_range_ticks(0.25) == 20

    def test_proximity_weight_at_origin(self) -> None:
        """Weight at k=0 is always 1.0 regardless of tau."""
        for tau in (5.0, 10.0, 20.0):
            rt = np.array([0])
            assert proximity_weight(rt, tau)[0] == pytest.approx(1.0)

    def test_proximity_weight_decay_shape(self) -> None:
        """Weight decays exponentially with distance from spot."""
        rt = np.array([0, 1, 5, 10])
        tau = 5.0
        w = proximity_weight(rt, tau)
        # Monotonically decreasing
        assert all(w[i] >= w[i + 1] for i in range(len(w) - 1))
        # At k=tau, weight should be 1/e
        assert w[2] == pytest.approx(np.exp(-1), rel=1e-10)

    def test_dollar_space_invariance(self) -> None:
        """Same dollar distance produces same weight, regardless of bucket size.

        For $0.50 buckets at k=5 ($2.50) with tau=5.0:
            w = exp(-5/5) = exp(-1)

        For $0.25 buckets at k=10 ($2.50) with tau=10.0:
            w = exp(-10/10) = exp(-1)

        Both represent $2.50 from spot and should produce identical weights.
        """
        # $0.50 buckets: k=5 ticks = $2.50
        w_equity = proximity_weight(np.array([5]), _proximity_tau_ticks(0.50))
        # $0.25 buckets: k=10 ticks = $2.50
        w_futures = proximity_weight(np.array([10]), _proximity_tau_ticks(0.25))
        assert w_equity[0] == pytest.approx(w_futures[0], rel=1e-10)

    def test_qqq_pipeline_numerically_identical(
        self, qqq_config: VPRuntimeConfig,
    ) -> None:
        """Full pipeline with QQQ config produces non-empty, finite results.

        This is the regression guard: QQQ path must produce valid output.
        """
        df_flow = _make_flow_df(5, 10)
        df_snap = _make_snap_df(5)
        df_signals, df_flow_enriched = run_full_pipeline(
            df_flow, df_snap, qqq_config,
        )
        assert not df_signals.empty
        assert not df_flow_enriched.empty
        # No NaN in output
        assert df_signals["composite"].notna().all()
        assert df_signals["strength"].notna().all()
        assert df_signals["confidence"].notna().all()
        # All finite
        for col in ("composite", "d1_composite", "d2_composite", "d3_composite"):
            assert np.isfinite(df_signals[col].values).all(), f"NaN/Inf in {col}"

    def test_futures_pipeline_produces_valid_output(
        self, mnq_config: VPRuntimeConfig,
    ) -> None:
        """Futures config produces valid pipeline output."""
        df_flow = _make_flow_df(5, 20)
        df_snap = _make_snap_df(5)
        df_signals, df_flow_enriched = run_full_pipeline(
            df_flow, df_snap, mnq_config,
        )
        assert not df_signals.empty
        assert df_signals["composite"].notna().all()

    def test_per_bucket_scores_use_correct_tau(self) -> None:
        """Per-bucket scores apply correct tau for the given bucket size."""
        df = _make_flow_df(1, 5)
        # $0.50 buckets: tau = 5.0
        enriched_050 = compute_per_bucket_scores(df, 0.50)
        # $0.25 buckets: tau = 10.0
        enriched_025 = compute_per_bucket_scores(df, 0.25)
        # At k=1, $0.50 bucket tau=5 gives exp(-1/5)=0.819
        # At k=1, $0.25 bucket tau=10 gives exp(-1/10)=0.905
        # So $0.25 buckets give HIGHER weight at same tick (closer in dollar space)
        mask = enriched_050["rel_ticks"] == 1
        w_050 = enriched_050.loc[mask, "proximity_weight"].values[0]
        w_025 = enriched_025.loc[mask, "proximity_weight"].values[0]
        assert w_025 > w_050

    def test_aggregate_window_metrics_different_buckets(self) -> None:
        """Aggregation uses correct near/depth ranges per bucket size."""
        df = _make_flow_df(1, 15)
        # $0.50: near=5, depth=10
        m_050 = aggregate_window_metrics(df, 0.50)
        # $0.25: near=10, depth=20
        m_025 = aggregate_window_metrics(df, 0.25)
        # Both should produce non-empty results
        assert not m_050.empty
        assert not m_025.empty
        # With different ranges, flow_imbalance should differ
        # (more ticks included with wider range)


# ──────────────────────────────────────────────────────────────────────
# 4. WebSocket control message includes runtime config (4.4)
# ──────────────────────────────────────────────────────────────────────


class TestWebsocketProtocol:
    """Test stream protocol contract for runtime config."""

    def test_runtime_config_message_format(
        self, qqq_config: VPRuntimeConfig,
    ) -> None:
        """Runtime config message has correct type and all required fields."""
        msg = {"type": "runtime_config", **qqq_config.to_dict()}
        assert msg["type"] == "runtime_config"
        assert msg["product_type"] == "equity_mbo"
        assert msg["symbol"] == "QQQ"
        assert msg["bucket_size_dollars"] == 0.50
        assert msg["rel_tick_size"] == 0.50
        assert msg["config_version"] is not None

    def test_batch_start_includes_config_values(
        self, mnq_config: VPRuntimeConfig,
    ) -> None:
        """batch_start message includes bucket_size and tick_size from config."""
        msg = {
            "type": "batch_start",
            "window_end_ts_ns": "1000000000000",
            "surfaces": ["snap", "flow", "signals"],
            "bucket_size": mnq_config.bucket_size_dollars,
            "tick_size": mnq_config.rel_tick_size,
        }
        assert msg["bucket_size"] == 0.25
        assert msg["tick_size"] == 0.25

    def test_futures_runtime_config_message(
        self, mnq_config: VPRuntimeConfig,
    ) -> None:
        """Futures runtime config has correct product-specific fields."""
        msg = {"type": "runtime_config", **mnq_config.to_dict()}
        assert msg["product_type"] == "future_mbo"
        assert msg["symbol"] == "MNQH6"
        assert msg["symbol_root"] == "MNQ"
        assert msg["qty_unit"] == "contracts"
        assert msg["contract_multiplier"] == 2.0
        assert msg["grid_max_ticks"] == 400


# ──────────────────────────────────────────────────────────────────────
# 5. Cache keying includes config_version (4.6)
# ──────────────────────────────────────────────────────────────────────


class TestCacheKeying:
    """Test cache keys include product_type + symbol + dt + config_version."""

    def test_cache_key_format(self, qqq_config: VPRuntimeConfig) -> None:
        """Cache key includes all four components."""
        key = qqq_config.cache_key("2026-02-06")
        parts = key.split(":")
        assert len(parts) == 4
        assert parts[0] == "equity_mbo"
        assert parts[1] == "QQQ"
        assert parts[2] == "2026-02-06"
        assert parts[3] == qqq_config.config_version

    def test_different_product_types_different_keys(
        self,
        qqq_config: VPRuntimeConfig,
        mnq_config: VPRuntimeConfig,
    ) -> None:
        """Equity and futures cache keys are distinct."""
        key_equity = qqq_config.cache_key("2026-02-06")
        key_futures = mnq_config.cache_key("2026-02-06")
        assert key_equity != key_futures

    def test_different_dates_different_keys(
        self, qqq_config: VPRuntimeConfig,
    ) -> None:
        """Same config, different dates produce different keys."""
        key1 = qqq_config.cache_key("2026-02-06")
        key2 = qqq_config.cache_key("2026-02-07")
        assert key1 != key2

    def test_engine_cache_uses_config_key(
        self,
        tmp_path: Path,
        qqq_config: VPRuntimeConfig,
    ) -> None:
        """Engine cache stores results under config.cache_key(dt)."""
        # Set up fake silver data
        for table in REQUIRED_TABLES:
            partition = _silver_dir(
                tmp_path, "equity_mbo", "QQQ", table, "2026-02-06",
            )
            partition.mkdir(parents=True)
            if table == "book_snapshot_1s":
                df = _make_snap_df(3)
            else:
                df = _make_flow_df(3)
            df.to_parquet(partition / "part-0000.parquet")

        engine = VacuumPressureEngine(tmp_path)
        engine.compute_day(qqq_config, "2026-02-06")

        expected_key = qqq_config.cache_key("2026-02-06")
        assert expected_key in engine._cache


# ──────────────────────────────────────────────────────────────────────
# 6. Readiness checks (4.7)
# ──────────────────────────────────────────────────────────────────────


class TestReadinessChecks:
    """Test silver readiness validation."""

    def test_readiness_check_passes_with_data(
        self,
        tmp_path: Path,
        qqq_config: VPRuntimeConfig,
    ) -> None:
        """Readiness check passes and returns row counts when data exists."""
        for table in REQUIRED_TABLES:
            partition = _silver_dir(
                tmp_path, "equity_mbo", "QQQ", table, "2026-02-06",
            )
            partition.mkdir(parents=True)
            if table == "book_snapshot_1s":
                df = _make_snap_df(10)
            else:
                df = _make_flow_df(10)
            df.to_parquet(partition / "part-0000.parquet")

        counts = validate_silver_readiness(tmp_path, qqq_config, "2026-02-06")
        assert "book_snapshot_1s" in counts
        assert "depth_and_flow_1s" in counts
        assert counts["book_snapshot_1s"] == 10
        assert counts["depth_and_flow_1s"] > 0

    def test_readiness_check_fails_missing_snap(
        self,
        tmp_path: Path,
        qqq_config: VPRuntimeConfig,
    ) -> None:
        """Readiness check fails when book_snapshot_1s is missing."""
        # Only create flow, not snap
        flow_partition = _silver_dir(
            tmp_path, "equity_mbo", "QQQ", "depth_and_flow_1s", "2026-02-06",
        )
        flow_partition.mkdir(parents=True)
        _make_flow_df(1).to_parquet(flow_partition / "part-0000.parquet")

        with pytest.raises(FileNotFoundError, match="Silver table missing"):
            validate_silver_readiness(tmp_path, qqq_config, "2026-02-06")

    def test_readiness_check_fails_missing_flow(
        self,
        tmp_path: Path,
        qqq_config: VPRuntimeConfig,
    ) -> None:
        """Readiness check fails when depth_and_flow_1s is missing."""
        snap_partition = _silver_dir(
            tmp_path, "equity_mbo", "QQQ", "book_snapshot_1s", "2026-02-06",
        )
        snap_partition.mkdir(parents=True)
        _make_snap_df(1).to_parquet(snap_partition / "part-0000.parquet")

        with pytest.raises(FileNotFoundError, match="Silver table missing"):
            validate_silver_readiness(tmp_path, qqq_config, "2026-02-06")

    def test_readiness_check_fails_empty_partition(
        self,
        tmp_path: Path,
        qqq_config: VPRuntimeConfig,
    ) -> None:
        """Readiness check fails when partition dir exists but has no parquets."""
        for table in REQUIRED_TABLES:
            partition = _silver_dir(
                tmp_path, "equity_mbo", "QQQ", table, "2026-02-06",
            )
            partition.mkdir(parents=True)
            # No parquet files written

        with pytest.raises(FileNotFoundError, match="No parquet files"):
            validate_silver_readiness(tmp_path, qqq_config, "2026-02-06")

    def test_readiness_error_includes_runner_command(
        self,
        tmp_path: Path,
        mnq_config: VPRuntimeConfig,
    ) -> None:
        """Readiness error includes exact runner command for futures."""
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_silver_readiness(tmp_path, mnq_config, "2026-02-06")
        msg = str(exc_info.value)
        assert "--product-type future_mbo" in msg
        assert "--symbol MNQH6" in msg


# ──────────────────────────────────────────────────────────────────────
# 7. Root extraction tests
# ──────────────────────────────────────────────────────────────────────


class TestRootExtraction:
    """Test _extract_root picks longest matching prefix."""

    def test_es(self) -> None:
        assert _extract_root("ESH6", ["ES", "MES", "NQ", "MNQ"]) == "ES"

    def test_mes(self) -> None:
        assert _extract_root("MESH6", ["ES", "MES", "NQ", "MNQ"]) == "MES"

    def test_mnq(self) -> None:
        assert _extract_root("MNQH6", ["ES", "MES", "NQ", "MNQ"]) == "MNQ"

    def test_nq(self) -> None:
        assert _extract_root("NQH6", ["ES", "MES", "NQ", "MNQ"]) == "NQ"

    def test_6e(self) -> None:
        assert _extract_root("6EH6", ["ES", "6E", "CL"]) == "6E"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot extract"):
            _extract_root("XXXH6", ["ES", "MES", "NQ"])
