"""Runtime configuration resolver for vacuum-pressure streaming.

Resolves instrument configuration by precedence:
    1. Explicit per-symbol override
    2. Product-root defaults (futures from products.yaml)
    3. Global product-type defaults (equity defaults)
    4. Fail fast if unresolved

Sources futures config from the canonical ``products.yaml`` without
duplication. Equity defaults are defined inline (phase 1: global defaults
only, with an override layer for symbol exceptions).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

PRICE_SCALE: float = 1e-9
"""System-wide price scale: price_dollars = price_int * PRICE_SCALE."""

VALID_PRODUCT_TYPES: frozenset[str] = frozenset({"equity_mbo", "future_mbo"})
"""Product types supported by vacuum-pressure runtime."""

# ──────────────────────────────────────────────────────────────────────
# Equity defaults (phase 1: global, with per-symbol overrides)
# ──────────────────────────────────────────────────────────────────────

_EQUITY_DEFAULTS: Dict[str, Any] = {
    "price_scale": PRICE_SCALE,
    "tick_size": 0.01,
    "bucket_size_dollars": 0.50,
    "grid_max_ticks": 200,
    "contract_multiplier": 1.0,
    "qty_unit": "shares",
    "price_decimals": 2,
}

_EQUITY_SYMBOL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # QQQ uses all defaults -- listed here for explicitness and extensibility
    "QQQ": {},
    "SPY": {},
}

# ──────────────────────────────────────────────────────────────────────
# Futures defaults (product-root config loaded from products.yaml)
# ──────────────────────────────────────────────────────────────────────

_FUTURES_GLOBAL_DEFAULTS: Dict[str, Any] = {
    "price_scale": PRICE_SCALE,
    "qty_unit": "contracts",
}

# Price decimals per tick_size resolution
_TICK_SIZE_TO_DECIMALS: Dict[float, int] = {
    0.25: 2,
    0.10: 2,
    0.01: 2,
    0.005: 3,
    0.00005: 5,
}


# ──────────────────────────────────────────────────────────────────────
# Runtime config dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VPRuntimeConfig:
    """Resolved runtime instrument configuration for a vacuum-pressure stream.

    All fields from UPGRADE.md section 3.2. This object is the single
    authoritative config used by server, engine, and formulas.
    """

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


# ──────────────────────────────────────────────────────────────────────
# Resolver
# ──────────────────────────────────────────────────────────────────────


def _compute_config_version(fields: Dict[str, Any]) -> str:
    """Compute a short deterministic hash of the config fields.

    Used for cache invalidation when config parameters change.
    """
    raw = "|".join(f"{k}={v}" for k, v in sorted(fields.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _load_products_yaml(products_yaml_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load products.yaml and return the raw products dict.

    Raises:
        FileNotFoundError: If products.yaml does not exist.
    """
    if not products_yaml_path.exists():
        raise FileNotFoundError(
            f"products.yaml not found: {products_yaml_path}\n"
            f"Expected at: backend/src/data_eng/config/products.yaml"
        )
    raw = yaml.safe_load(products_yaml_path.read_text())
    return raw.get("products", {})


def _extract_root(symbol: str, known_roots: list[str]) -> str:
    """Extract the longest matching product root from a futures symbol.

    Examples: ESH6 -> ES, MNQH6 -> MNQ, SIH6 -> SI, 6EH6 -> 6E

    Raises:
        ValueError: If no known root matches the symbol.
    """
    best = ""
    for root in known_roots:
        if symbol.startswith(root) and len(root) > len(best):
            best = root
    if not best:
        raise ValueError(
            f"Cannot extract product root from futures symbol '{symbol}'. "
            f"Known roots: {sorted(known_roots)}"
        )
    return best


def _price_decimals_for_tick(tick_size: float) -> int:
    """Determine price display decimals from tick size."""
    if tick_size in _TICK_SIZE_TO_DECIMALS:
        return _TICK_SIZE_TO_DECIMALS[tick_size]
    # Fall back: count decimals needed to represent tick_size
    s = f"{tick_size:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 2


def resolve_config(
    product_type: str,
    symbol: str,
    products_yaml_path: Path,
) -> VPRuntimeConfig:
    """Resolve runtime instrument configuration for a vacuum-pressure stream.

    Resolution precedence:
        1. Explicit per-symbol override (equity only, phase 1)
        2. Product-root defaults (futures from products.yaml)
        3. Global product-type defaults
        4. Fail fast if unresolved

    Args:
        product_type: ``"equity_mbo"`` or ``"future_mbo"``.
        symbol: Instrument symbol (e.g. ``"QQQ"``, ``"MNQH6"``).
        products_yaml_path: Path to ``products.yaml``.

    Returns:
        Fully resolved ``VPRuntimeConfig``.

    Raises:
        ValueError: If product_type is invalid or symbol cannot be resolved.
        FileNotFoundError: If products.yaml is missing (futures only).
    """
    if product_type not in VALID_PRODUCT_TYPES:
        raise ValueError(
            f"Invalid product_type '{product_type}'. "
            f"Must be one of: {sorted(VALID_PRODUCT_TYPES)}"
        )

    if product_type == "equity_mbo":
        return _resolve_equity(symbol)

    if product_type == "future_mbo":
        return _resolve_futures(symbol, products_yaml_path)

    # Unreachable due to validation above, but satisfies exhaustiveness
    raise ValueError(f"Unresolved product_type: {product_type}")


def _resolve_equity(symbol: str) -> VPRuntimeConfig:
    """Resolve equity config from defaults + optional per-symbol overrides."""
    base = dict(_EQUITY_DEFAULTS)

    # Apply per-symbol overrides if present
    overrides = _EQUITY_SYMBOL_OVERRIDES.get(symbol, {})
    base.update(overrides)

    fields = {
        "product_type": "equity_mbo",
        "symbol": symbol,
        "symbol_root": symbol,
        "price_scale": base["price_scale"],
        "tick_size": base["tick_size"],
        "bucket_size_dollars": base["bucket_size_dollars"],
        "rel_tick_size": base["bucket_size_dollars"],
        "grid_max_ticks": base["grid_max_ticks"],
        "contract_multiplier": base["contract_multiplier"],
        "qty_unit": base["qty_unit"],
        "price_decimals": base["price_decimals"],
    }

    config_version = _compute_config_version(fields)
    return VPRuntimeConfig(**fields, config_version=config_version)


def _resolve_futures(
    symbol: str,
    products_yaml_path: Path,
) -> VPRuntimeConfig:
    """Resolve futures config from products.yaml + global futures defaults.

    For futures, ``bucket_size_dollars`` equals ``tick_size`` because silver
    ``rel_ticks`` is native-tick-based per product config.
    """
    products_raw = _load_products_yaml(products_yaml_path)
    known_roots = list(products_raw.keys())
    root = _extract_root(symbol, known_roots)
    product_spec = products_raw[root]

    tick_size = float(product_spec["tick_size"])
    grid_max_ticks = int(product_spec["grid_max_ticks"])
    contract_multiplier = float(product_spec["contract_multiplier"])
    price_decimals = _price_decimals_for_tick(tick_size)

    fields = {
        "product_type": "future_mbo",
        "symbol": symbol,
        "symbol_root": root,
        "price_scale": _FUTURES_GLOBAL_DEFAULTS["price_scale"],
        "tick_size": tick_size,
        "bucket_size_dollars": tick_size,
        "rel_tick_size": tick_size,
        "grid_max_ticks": grid_max_ticks,
        "contract_multiplier": contract_multiplier,
        "qty_unit": _FUTURES_GLOBAL_DEFAULTS["qty_unit"],
        "price_decimals": price_decimals,
    }

    config_version = _compute_config_version(fields)
    return VPRuntimeConfig(**fields, config_version=config_version)
