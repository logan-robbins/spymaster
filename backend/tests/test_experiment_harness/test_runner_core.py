from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.config_schema import (
    ExperimentConfig,
    GridVariantConfig,
    SweepConfig,
)
from src.experiment_harness.runner import ExperimentRunner
from src.experiment_harness.signals import SIGNAL_REGISTRY, ensure_signals_loaded


def _lake_root() -> Path:
    return Path(__file__).resolve().parents[2] / "lake"


def test_signal_registry_loads_all_expected_signals() -> None:
    ensure_signals_loaded()
    expected = {
        "ads",
        "ads_pfp_svac",
        "spg",
        "erd",
        "pfp",
        "jad",
        "iirc",
        "msd",
        "derivative",
        "svm_sp",
        "gbm_mf",
        "knn_cl",
        "lsvm_der",
        "xgb_snap",
        "pca_ad",
    }
    assert expected.issubset(set(SIGNAL_REGISTRY.keys()))


def test_expand_grid_variant_specs_cartesian() -> None:
    cfg = GridVariantConfig(
        cell_width_ms=[50, 100],
        c1_v_add=[0.8, 1.0],
        bucket_size_dollars=[2.5, 5.0],
        tau_velocity=[1.5, 2.0],
    )
    specs = ExperimentRunner._expand_grid_variant_specs(cfg)
    assert len(specs) == 16
    for spec in specs:
        assert isinstance(spec.cell_width_ms, int)
        assert isinstance(spec.c1_v_add, float)
        assert isinstance(spec.bucket_size_dollars, float)
        assert isinstance(spec.tau_velocity, float)


def test_expand_signal_params_rejects_unknown_keys() -> None:
    runner = ExperimentRunner(_lake_root())
    cfg = ExperimentConfig(
        name="bad_params",
        datasets=["mnqh6_20260206_0925_1025"],
        signals=["ads"],
        sweep=SweepConfig(
            per_signal={"ads": {"definitely_not_a_param": [1, 2]}},
        ),
    )
    with pytest.raises(ValueError, match="Unknown sweep params"):
        runner._expand_signal_params("ads", cfg)
