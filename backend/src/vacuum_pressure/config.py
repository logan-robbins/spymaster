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
from typing import Any, Dict

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
    grid_max_ticks: int
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
            "grid_max_ticks": self.grid_max_ticks,
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
    raw = "|".join(f"{k}={v}" for k, v in sorted(fields.items()))
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
        "grid_max_ticks",
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
        "grid_max_ticks": int(raw["grid_max_ticks"]),
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
    if fields["grid_max_ticks"] < 1:
        raise ValueError(f"'grid_max_ticks' must be >= 1 in {path}.")

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
