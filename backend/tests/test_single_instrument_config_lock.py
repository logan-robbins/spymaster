from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.config import LOCKED_INSTRUMENT_CONFIG_ENV, resolve_config


def _write_locked_config(path: Path, symbol: str = "MNQH6") -> None:
    path.write_text(
        "\n".join(
            [
                "product_type: future_mbo",
                f"symbol: {symbol}",
                "symbol_root: MNQ",
                "price_scale: 1.0e-9",
                "tick_size: 0.25",
                "bucket_size_dollars: 0.25",
                "rel_tick_size: 0.25",
                "grid_radius_ticks: 40",
                "cell_width_ms: 100",
                "n_absolute_ticks: 500",
                "spectrum_windows: [5, 10, 20, 40]",
                "spectrum_rollup_weights: [1.0, 1.0, 1.0, 1.0]",
                "spectrum_derivative_weights: [0.55, 0.30, 0.15]",
                "spectrum_tanh_scale: 3.0",
                "spectrum_threshold_neutral: 0.15",
                "zscore_window_bins: 300",
                "zscore_min_periods: 75",
                "projection_horizons_bins: [1, 2, 3, 4]",
                "contract_multiplier: 2.0",
                "qty_unit: contracts",
                "price_decimals: 2",
            ]
        )
    )


def test_locked_config_is_single_source_of_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "instrument.yaml"
    _write_locked_config(cfg_path, symbol="MNQH6")
    monkeypatch.setenv(LOCKED_INSTRUMENT_CONFIG_ENV, str(cfg_path))

    cfg = resolve_config(
        product_type="future_mbo",
        symbol="MNQH6",
        products_yaml_path=tmp_path / "unused.yaml",
    )
    assert cfg.symbol == "MNQH6"
    assert cfg.tick_size == 0.25
    assert cfg.grid_radius_ticks == 40
    assert cfg.cell_width_ms == 100
    assert cfg.projection_horizons_bins == (1, 2, 3, 4)
    assert cfg.projection_horizons_ms == (100, 200, 300, 400)
    assert cfg.contract_multiplier == 2.0


def test_locked_config_rejects_symbol_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "instrument.yaml"
    _write_locked_config(cfg_path, symbol="MNQH6")
    monkeypatch.setenv(LOCKED_INSTRUMENT_CONFIG_ENV, str(cfg_path))

    with pytest.raises(ValueError, match="does not match locked single-instrument"):
        resolve_config(
            product_type="future_mbo",
            symbol="ESH6",
            products_yaml_path=tmp_path / "unused.yaml",
        )
