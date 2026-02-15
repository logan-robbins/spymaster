"""Single-instrument runtime configuration for vacuum-pressure.

Runtime contract:
    - One environment runs one instrument.
    - Instrument config is loaded from a single YAML file.
    - Requested (product_type, symbol) must match the locked config exactly.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml

LOCKED_INSTRUMENT_CONFIG_ENV = "VP_INSTRUMENT_CONFIG_PATH"
"""Optional override path for the single-instrument config YAML."""

PRICE_SCALE: float = 1e-9
"""System-wide price scale: price_dollars = price_int * PRICE_SCALE."""

VALID_PRODUCT_TYPES: frozenset[str] = frozenset({"equity_mbo", "future_mbo"})
"""Product types supported by vacuum-pressure runtime."""


@dataclass(frozen=True)
class VPRuntimeConfig:
    """Resolved runtime instrument configuration for vacuum-pressure."""

    product_type: str
    symbol: str
    symbol_root: str
    price_scale: float
    tick_size: float
    bucket_size_dollars: float
    rel_tick_size: float
    grid_radius_ticks: int
    cell_width_ms: int
    n_absolute_ticks: int
    spectrum_windows: Tuple[int, ...]
    spectrum_rollup_weights: Tuple[float, ...]
    spectrum_derivative_weights: Tuple[float, ...]
    spectrum_tanh_scale: float
    spectrum_threshold_neutral: float
    zscore_window_bins: int
    zscore_min_periods: int
    projection_horizons_ms: Tuple[int, ...]
    contract_multiplier: float
    qty_unit: str
    price_decimals: int
    config_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict for wire protocol."""
        return {
            "product_type": self.product_type,
            "symbol": self.symbol,
            "symbol_root": self.symbol_root,
            "price_scale": self.price_scale,
            "tick_size": self.tick_size,
            "bucket_size_dollars": self.bucket_size_dollars,
            "rel_tick_size": self.rel_tick_size,
            "grid_radius_ticks": self.grid_radius_ticks,
            "cell_width_ms": self.cell_width_ms,
            "n_absolute_ticks": self.n_absolute_ticks,
            "spectrum_windows": list(self.spectrum_windows),
            "spectrum_rollup_weights": list(self.spectrum_rollup_weights),
            "spectrum_derivative_weights": list(self.spectrum_derivative_weights),
            "spectrum_tanh_scale": self.spectrum_tanh_scale,
            "spectrum_threshold_neutral": self.spectrum_threshold_neutral,
            "zscore_window_bins": self.zscore_window_bins,
            "zscore_min_periods": self.zscore_min_periods,
            "projection_horizons_ms": list(self.projection_horizons_ms),
            "contract_multiplier": self.contract_multiplier,
            "qty_unit": self.qty_unit,
            "price_decimals": self.price_decimals,
            "config_version": self.config_version,
        }

    def cache_key(self, dt: str) -> str:
        """Return a composite cache key including config_version."""
        return f"{self.product_type}:{self.symbol}:{dt}:{self.config_version}"


def _compute_config_version(fields: Dict[str, Any]) -> str:
    """Compute short deterministic hash of config fields."""
    stable: Dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, tuple):
            stable[key] = list(value)
        else:
            stable[key] = value
    raw = "|".join(f"{k}={v}" for k, v in sorted(stable.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _default_locked_config_path() -> Path:
    """Default location for single-instrument runtime config."""
    return Path(__file__).resolve().with_name("instrument.yaml")


def _resolve_locked_config_path() -> Path:
    """Resolve locked instrument config path from env override or default."""
    override = os.getenv(LOCKED_INSTRUMENT_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_locked_config_path()


def _parse_int_sequence(raw: Any, field_name: str) -> Tuple[int, ...]:
    if isinstance(raw, str):
        items = [token.strip() for token in raw.split(",") if token.strip()]
    elif isinstance(raw, Iterable):
        items = list(raw)
    else:
        raise ValueError(f"'{field_name}' must be a list/tuple or comma-separated string.")
    if not items:
        raise ValueError(f"'{field_name}' must contain at least one value.")
    out: list[int] = []
    for item in items:
        try:
            val = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{field_name}' contains non-integer value: {item}") from exc
        if val <= 0:
            raise ValueError(f"'{field_name}' values must be > 0, got {val}.")
        out.append(val)
    return tuple(out)


def _parse_float_sequence(raw: Any, field_name: str) -> Tuple[float, ...]:
    if isinstance(raw, str):
        items = [token.strip() for token in raw.split(",") if token.strip()]
    elif isinstance(raw, Iterable):
        items = list(raw)
    else:
        raise ValueError(f"'{field_name}' must be a list/tuple or comma-separated string.")
    if not items:
        raise ValueError(f"'{field_name}' must contain at least one value.")
    out: list[float] = []
    for item in items:
        try:
            val = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{field_name}' contains non-float value: {item}") from exc
        if val <= 0.0:
            raise ValueError(f"'{field_name}' values must be > 0, got {val}.")
        out.append(val)
    return tuple(out)


def _load_locked_instrument_config(path: Path) -> VPRuntimeConfig:
    """Load single-instrument runtime config from YAML and validate it."""
    if not path.exists():
        raise FileNotFoundError(
            "Single-instrument config is required but was not found.\n"
            f"Expected: {path}\n"
            f"Override with env: {LOCKED_INSTRUMENT_CONFIG_ENV}"
        )

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid single-instrument config format at {path}.")

    required_fields = [
        "product_type",
        "symbol",
        "symbol_root",
        "price_scale",
        "tick_size",
        "bucket_size_dollars",
        "rel_tick_size",
        "grid_radius_ticks",
        "cell_width_ms",
        "n_absolute_ticks",
        "spectrum_windows",
        "spectrum_rollup_weights",
        "spectrum_derivative_weights",
        "spectrum_tanh_scale",
        "spectrum_threshold_neutral",
        "zscore_window_bins",
        "zscore_min_periods",
        "projection_horizons_ms",
        "contract_multiplier",
        "qty_unit",
        "price_decimals",
    ]
    missing = [f for f in required_fields if f not in raw]
    if missing:
        raise ValueError(
            f"Single-instrument config missing required fields: {missing} (path={path})"
        )

    fields = {
        "product_type": str(raw["product_type"]).strip(),
        "symbol": str(raw["symbol"]).strip(),
        "symbol_root": str(raw["symbol_root"]).strip(),
        "price_scale": float(raw["price_scale"]),
        "tick_size": float(raw["tick_size"]),
        "bucket_size_dollars": float(raw["bucket_size_dollars"]),
        "rel_tick_size": float(raw["rel_tick_size"]),
        "grid_radius_ticks": int(raw["grid_radius_ticks"]),
        "cell_width_ms": int(raw["cell_width_ms"]),
        "n_absolute_ticks": int(raw.get("n_absolute_ticks", 500)),
        "spectrum_windows": _parse_int_sequence(raw["spectrum_windows"], "spectrum_windows"),
        "spectrum_rollup_weights": _parse_float_sequence(
            raw["spectrum_rollup_weights"], "spectrum_rollup_weights"
        ),
        "spectrum_derivative_weights": _parse_float_sequence(
            raw["spectrum_derivative_weights"], "spectrum_derivative_weights"
        ),
        "spectrum_tanh_scale": float(raw["spectrum_tanh_scale"]),
        "spectrum_threshold_neutral": float(raw["spectrum_threshold_neutral"]),
        "zscore_window_bins": int(raw["zscore_window_bins"]),
        "zscore_min_periods": int(raw["zscore_min_periods"]),
        "projection_horizons_ms": _parse_int_sequence(
            raw["projection_horizons_ms"], "projection_horizons_ms"
        ),
        "contract_multiplier": float(raw["contract_multiplier"]),
        "qty_unit": str(raw["qty_unit"]).strip(),
        "price_decimals": int(raw["price_decimals"]),
    }
    if fields["product_type"] not in VALID_PRODUCT_TYPES:
        raise ValueError(
            f"Invalid product_type '{fields['product_type']}' in {path}. "
            f"Must be one of: {sorted(VALID_PRODUCT_TYPES)}"
        )
    if not fields["symbol"]:
        raise ValueError(f"'symbol' must be non-empty in {path}.")
    if fields["tick_size"] <= 0.0:
        raise ValueError(f"'tick_size' must be > 0 in {path}.")
    if fields["bucket_size_dollars"] <= 0.0:
        raise ValueError(f"'bucket_size_dollars' must be > 0 in {path}.")
    if fields["grid_radius_ticks"] < 1:
        raise ValueError(f"'grid_radius_ticks' must be >= 1 in {path}.")
    if fields["cell_width_ms"] < 1:
        raise ValueError(f"'cell_width_ms' must be >= 1 in {path}.")
    if fields["n_absolute_ticks"] < 3:
        raise ValueError(f"'n_absolute_ticks' must be >= 3 in {path}.")
    if len(fields["spectrum_windows"]) != len(fields["spectrum_rollup_weights"]):
        raise ValueError(
            "spectrum_windows and spectrum_rollup_weights must have identical lengths."
        )
    if len(fields["spectrum_derivative_weights"]) != 3:
        raise ValueError("spectrum_derivative_weights must contain exactly 3 weights (d1,d2,d3).")
    if fields["spectrum_tanh_scale"] <= 0.0:
        raise ValueError(f"'spectrum_tanh_scale' must be > 0 in {path}.")
    if not (0.0 < fields["spectrum_threshold_neutral"] < 1.0):
        raise ValueError(
            f"'spectrum_threshold_neutral' must be in (0,1), got {fields['spectrum_threshold_neutral']}."
        )
    if fields["zscore_window_bins"] < 2:
        raise ValueError("'zscore_window_bins' must be >= 2.")
    if fields["zscore_min_periods"] < 2:
        raise ValueError("'zscore_min_periods' must be >= 2.")
    if fields["zscore_min_periods"] > fields["zscore_window_bins"]:
        raise ValueError("'zscore_min_periods' cannot exceed 'zscore_window_bins'.")

    config_version = _compute_config_version(fields)
    return VPRuntimeConfig(**fields, config_version=config_version)


def resolve_config(
    product_type: str,
    symbol: str,
    products_yaml_path: Path,  # retained for call-site compatibility
) -> VPRuntimeConfig:
    """Resolve runtime config and enforce locked single-instrument contract."""
    del products_yaml_path

    locked_path = _resolve_locked_config_path()
    locked_cfg = _load_locked_instrument_config(locked_path)

    if product_type != locked_cfg.product_type or symbol != locked_cfg.symbol:
        raise ValueError(
            "Requested instrument does not match locked single-instrument runtime config.\n"
            f"Requested: {product_type}:{symbol}\n"
            f"Locked:    {locked_cfg.product_type}:{locked_cfg.symbol}\n"
            f"Config:    {locked_path}"
        )
    return locked_cfg
