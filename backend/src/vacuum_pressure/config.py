"""Single-instrument runtime configuration for vacuum-pressure.

Runtime contract:
    - One environment runs one instrument.
    - Instrument config is loaded from a single YAML file.
    - Requested (product_type, symbol) must match the locked config exactly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import yaml

from ..vp_shared.hashing import stable_short_hash

LOCKED_INSTRUMENT_CONFIG_ENV = "VP_INSTRUMENT_CONFIG_PATH"
"""Optional override path for the single-instrument config YAML."""

PRICE_SCALE: float = 1e-9
"""System-wide price scale: price_dollars = price_int * PRICE_SCALE."""

VALID_PRODUCT_TYPES: frozenset[str] = frozenset({"equity_mbo", "future_mbo"})
"""Product types supported by vacuum-pressure runtime."""

# Derivative-chain defaults (seconds)
DEFAULT_TAU_VELOCITY: float = 2.0
DEFAULT_TAU_ACCELERATION: float = 5.0
DEFAULT_TAU_JERK: float = 10.0
DEFAULT_TAU_REST_DECAY: float = 30.0

# Force-model defaults
DEFAULT_C1_V_ADD: float = 1.0
DEFAULT_C2_V_REST_POS: float = 0.5
DEFAULT_C3_A_ADD: float = 0.3
DEFAULT_C4_V_PULL: float = 1.0
DEFAULT_C5_V_FILL: float = 1.5
DEFAULT_C6_V_REST_NEG: float = 0.5
DEFAULT_C7_A_PULL: float = 0.3

# Derivative runtime model defaults (matching harness best row baseline)
DEFAULT_STATE_MODEL_ENABLED: bool = True
DEFAULT_STATE_MODEL_CENTER_EXCLUSION_RADIUS: int = 0
DEFAULT_STATE_MODEL_SPATIAL_DECAY_POWER: float = 0.0
DEFAULT_STATE_MODEL_ZSCORE_WINDOW_BINS: int = 240
DEFAULT_STATE_MODEL_ZSCORE_MIN_PERIODS: int = 60
DEFAULT_STATE_MODEL_TANH_SCALE: float = 3.0
DEFAULT_STATE_MODEL_D1_WEIGHT: float = 1.0
DEFAULT_STATE_MODEL_D2_WEIGHT: float = 0.0
DEFAULT_STATE_MODEL_D3_WEIGHT: float = 0.0
DEFAULT_STATE_MODEL_BULL_PRESSURE_WEIGHT: float = 1.0
DEFAULT_STATE_MODEL_BULL_VACUUM_WEIGHT: float = 1.0
DEFAULT_STATE_MODEL_BEAR_PRESSURE_WEIGHT: float = 1.0
DEFAULT_STATE_MODEL_BEAR_VACUUM_WEIGHT: float = 1.0
DEFAULT_STATE_MODEL_MIXED_WEIGHT: float = 0.0
DEFAULT_STATE_MODEL_ENABLE_WEIGHTED_BLEND: bool = False

_RUNTIME_REQUIRED_FIELDS: tuple[str, ...] = (
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
    "flow_windows",
    "flow_rollup_weights",
    "flow_derivative_weights",
    "flow_tanh_scale",
    "flow_neutral_threshold",
    "flow_zscore_window_bins",
    "flow_zscore_min_periods",
    "projection_horizons_bins",
    "contract_multiplier",
    "qty_unit",
    "price_decimals",
    "tau_velocity",
    "tau_acceleration",
    "tau_jerk",
    "tau_rest_decay",
    "c1_v_add",
    "c2_v_rest_pos",
    "c3_a_add",
    "c4_v_pull",
    "c5_v_fill",
    "c6_v_rest_neg",
    "c7_a_pull",
)


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
    flow_windows: Tuple[int, ...]
    flow_rollup_weights: Tuple[float, ...]
    flow_derivative_weights: Tuple[float, ...]
    flow_tanh_scale: float
    flow_neutral_threshold: float
    flow_zscore_window_bins: int
    flow_zscore_min_periods: int
    projection_horizons_bins: Tuple[int, ...]
    projection_horizons_ms: Tuple[int, ...]
    contract_multiplier: float
    qty_unit: str
    price_decimals: int
    config_version: str
    tau_velocity: float = DEFAULT_TAU_VELOCITY
    tau_acceleration: float = DEFAULT_TAU_ACCELERATION
    tau_jerk: float = DEFAULT_TAU_JERK
    tau_rest_decay: float = DEFAULT_TAU_REST_DECAY
    c1_v_add: float = DEFAULT_C1_V_ADD
    c2_v_rest_pos: float = DEFAULT_C2_V_REST_POS
    c3_a_add: float = DEFAULT_C3_A_ADD
    c4_v_pull: float = DEFAULT_C4_V_PULL
    c5_v_fill: float = DEFAULT_C5_V_FILL
    c6_v_rest_neg: float = DEFAULT_C6_V_REST_NEG
    c7_a_pull: float = DEFAULT_C7_A_PULL
    state_model_enabled: bool = DEFAULT_STATE_MODEL_ENABLED
    state_model_center_exclusion_radius: int = DEFAULT_STATE_MODEL_CENTER_EXCLUSION_RADIUS
    state_model_spatial_decay_power: float = DEFAULT_STATE_MODEL_SPATIAL_DECAY_POWER
    state_model_zscore_window_bins: int = DEFAULT_STATE_MODEL_ZSCORE_WINDOW_BINS
    state_model_zscore_min_periods: int = DEFAULT_STATE_MODEL_ZSCORE_MIN_PERIODS
    state_model_tanh_scale: float = DEFAULT_STATE_MODEL_TANH_SCALE
    state_model_d1_weight: float = DEFAULT_STATE_MODEL_D1_WEIGHT
    state_model_d2_weight: float = DEFAULT_STATE_MODEL_D2_WEIGHT
    state_model_d3_weight: float = DEFAULT_STATE_MODEL_D3_WEIGHT
    state_model_bull_pressure_weight: float = DEFAULT_STATE_MODEL_BULL_PRESSURE_WEIGHT
    state_model_bull_vacuum_weight: float = DEFAULT_STATE_MODEL_BULL_VACUUM_WEIGHT
    state_model_bear_pressure_weight: float = DEFAULT_STATE_MODEL_BEAR_PRESSURE_WEIGHT
    state_model_bear_vacuum_weight: float = DEFAULT_STATE_MODEL_BEAR_VACUUM_WEIGHT
    state_model_mixed_weight: float = DEFAULT_STATE_MODEL_MIXED_WEIGHT
    state_model_enable_weighted_blend: bool = DEFAULT_STATE_MODEL_ENABLE_WEIGHTED_BLEND

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
            "flow_windows": list(self.flow_windows),
            "flow_rollup_weights": list(self.flow_rollup_weights),
            "flow_derivative_weights": list(self.flow_derivative_weights),
            "flow_tanh_scale": self.flow_tanh_scale,
            "flow_neutral_threshold": self.flow_neutral_threshold,
            "flow_zscore_window_bins": self.flow_zscore_window_bins,
            "flow_zscore_min_periods": self.flow_zscore_min_periods,
            "projection_horizons_bins": list(self.projection_horizons_bins),
            "projection_horizons_ms": list(self.projection_horizons_ms),
            "contract_multiplier": self.contract_multiplier,
            "qty_unit": self.qty_unit,
            "price_decimals": self.price_decimals,
            "tau_velocity": self.tau_velocity,
            "tau_acceleration": self.tau_acceleration,
            "tau_jerk": self.tau_jerk,
            "tau_rest_decay": self.tau_rest_decay,
            "c1_v_add": self.c1_v_add,
            "c2_v_rest_pos": self.c2_v_rest_pos,
            "c3_a_add": self.c3_a_add,
            "c4_v_pull": self.c4_v_pull,
            "c5_v_fill": self.c5_v_fill,
            "c6_v_rest_neg": self.c6_v_rest_neg,
            "c7_a_pull": self.c7_a_pull,
            "state_model_enabled": self.state_model_enabled,
            "state_model_center_exclusion_radius": self.state_model_center_exclusion_radius,
            "state_model_spatial_decay_power": self.state_model_spatial_decay_power,
            "state_model_zscore_window_bins": self.state_model_zscore_window_bins,
            "state_model_zscore_min_periods": self.state_model_zscore_min_periods,
            "state_model_tanh_scale": self.state_model_tanh_scale,
            "state_model_d1_weight": self.state_model_d1_weight,
            "state_model_d2_weight": self.state_model_d2_weight,
            "state_model_d3_weight": self.state_model_d3_weight,
            "state_model_bull_pressure_weight": self.state_model_bull_pressure_weight,
            "state_model_bull_vacuum_weight": self.state_model_bull_vacuum_weight,
            "state_model_bear_pressure_weight": self.state_model_bear_pressure_weight,
            "state_model_bear_vacuum_weight": self.state_model_bear_vacuum_weight,
            "state_model_mixed_weight": self.state_model_mixed_weight,
            "state_model_enable_weighted_blend": self.state_model_enable_weighted_blend,
            "config_version": self.config_version,
        }

    def cache_key(self, dt: str) -> str:
        """Return a composite cache key including config_version."""
        return f"{self.product_type}:{self.symbol}:{dt}:{self.config_version}"


def _compute_config_version(fields: Dict[str, Any]) -> str:
    """Compute short deterministic hash of config fields."""
    return stable_short_hash(fields, length=12)


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


def _parse_bool(raw: Any, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(
        f"'{field_name}' must be a boolean (true/false), got {raw!r}."
    )


def _normalize_runtime_fields(raw: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    """Normalize and validate runtime fields from YAML/override payloads."""
    missing = [f for f in _RUNTIME_REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(
            f"Single-instrument config missing required fields: {missing} (path={source})"
        )

    fields: Dict[str, Any] = {
        "product_type": str(raw["product_type"]).strip(),
        "symbol": str(raw["symbol"]).strip(),
        "symbol_root": str(raw["symbol_root"]).strip(),
        "price_scale": float(raw["price_scale"]),
        "tick_size": float(raw["tick_size"]),
        "bucket_size_dollars": float(raw["bucket_size_dollars"]),
        "rel_tick_size": float(raw["rel_tick_size"]),
        "grid_radius_ticks": int(raw["grid_radius_ticks"]),
        "cell_width_ms": int(raw["cell_width_ms"]),
        "n_absolute_ticks": int(raw["n_absolute_ticks"]),
        "flow_windows": _parse_int_sequence(raw["flow_windows"], "flow_windows"),
        "flow_rollup_weights": _parse_float_sequence(
            raw["flow_rollup_weights"], "flow_rollup_weights"
        ),
        "flow_derivative_weights": _parse_float_sequence(
            raw["flow_derivative_weights"], "flow_derivative_weights"
        ),
        "flow_tanh_scale": float(raw["flow_tanh_scale"]),
        "flow_neutral_threshold": float(raw["flow_neutral_threshold"]),
        "projection_horizons_bins": _parse_int_sequence(
            raw["projection_horizons_bins"], "projection_horizons_bins"
        ),
        "contract_multiplier": float(raw["contract_multiplier"]),
        "qty_unit": str(raw["qty_unit"]).strip(),
        "price_decimals": int(raw["price_decimals"]),
        "tau_velocity": float(raw["tau_velocity"]),
        "tau_acceleration": float(raw["tau_acceleration"]),
        "tau_jerk": float(raw["tau_jerk"]),
        "tau_rest_decay": float(raw["tau_rest_decay"]),
        "c1_v_add": float(raw["c1_v_add"]),
        "c2_v_rest_pos": float(raw["c2_v_rest_pos"]),
        "c3_a_add": float(raw["c3_a_add"]),
        "c4_v_pull": float(raw["c4_v_pull"]),
        "c5_v_fill": float(raw["c5_v_fill"]),
        "c6_v_rest_neg": float(raw["c6_v_rest_neg"]),
        "c7_a_pull": float(raw["c7_a_pull"]),
        "state_model_enabled": _parse_bool(
            raw.get("state_model_enabled", DEFAULT_STATE_MODEL_ENABLED),
            "state_model_enabled",
        ),
        "state_model_center_exclusion_radius": int(
            raw.get(
                "state_model_center_exclusion_radius",
                DEFAULT_STATE_MODEL_CENTER_EXCLUSION_RADIUS,
            )
        ),
        "state_model_spatial_decay_power": float(
            raw.get("state_model_spatial_decay_power", DEFAULT_STATE_MODEL_SPATIAL_DECAY_POWER)
        ),
        "flow_zscore_window_bins": int(
            raw["flow_zscore_window_bins"]
        ),
        "flow_zscore_min_periods": int(
            raw["flow_zscore_min_periods"]
        ),
        "state_model_zscore_window_bins": int(
            raw.get("state_model_zscore_window_bins", DEFAULT_STATE_MODEL_ZSCORE_WINDOW_BINS)
        ),
        "state_model_zscore_min_periods": int(
            raw.get("state_model_zscore_min_periods", DEFAULT_STATE_MODEL_ZSCORE_MIN_PERIODS)
        ),
        "state_model_tanh_scale": float(raw.get("state_model_tanh_scale", DEFAULT_STATE_MODEL_TANH_SCALE)),
        "state_model_d1_weight": float(raw.get("state_model_d1_weight", DEFAULT_STATE_MODEL_D1_WEIGHT)),
        "state_model_d2_weight": float(raw.get("state_model_d2_weight", DEFAULT_STATE_MODEL_D2_WEIGHT)),
        "state_model_d3_weight": float(raw.get("state_model_d3_weight", DEFAULT_STATE_MODEL_D3_WEIGHT)),
        "state_model_bull_pressure_weight": float(
            raw.get(
                "state_model_bull_pressure_weight",
                DEFAULT_STATE_MODEL_BULL_PRESSURE_WEIGHT,
            )
        ),
        "state_model_bull_vacuum_weight": float(
            raw.get(
                "state_model_bull_vacuum_weight",
                DEFAULT_STATE_MODEL_BULL_VACUUM_WEIGHT,
            )
        ),
        "state_model_bear_pressure_weight": float(
            raw.get(
                "state_model_bear_pressure_weight",
                DEFAULT_STATE_MODEL_BEAR_PRESSURE_WEIGHT,
            )
        ),
        "state_model_bear_vacuum_weight": float(
            raw.get(
                "state_model_bear_vacuum_weight",
                DEFAULT_STATE_MODEL_BEAR_VACUUM_WEIGHT,
            )
        ),
        "state_model_mixed_weight": float(raw.get("state_model_mixed_weight", DEFAULT_STATE_MODEL_MIXED_WEIGHT)),
        "state_model_enable_weighted_blend": _parse_bool(
            raw.get(
                "state_model_enable_weighted_blend",
                DEFAULT_STATE_MODEL_ENABLE_WEIGHTED_BLEND,
            ),
            "state_model_enable_weighted_blend",
        ),
    }

    if fields["product_type"] not in VALID_PRODUCT_TYPES:
        raise ValueError(
            f"Invalid product_type '{fields['product_type']}' in {source}. "
            f"Must be one of: {sorted(VALID_PRODUCT_TYPES)}"
        )
    if not fields["symbol"]:
        raise ValueError(f"'symbol' must be non-empty in {source}.")
    if fields["tick_size"] <= 0.0:
        raise ValueError(f"'tick_size' must be > 0 in {source}.")
    if fields["bucket_size_dollars"] <= 0.0:
        raise ValueError(f"'bucket_size_dollars' must be > 0 in {source}.")
    if fields["grid_radius_ticks"] < 1:
        raise ValueError(f"'grid_radius_ticks' must be >= 1 in {source}.")
    if fields["cell_width_ms"] < 1:
        raise ValueError(f"'cell_width_ms' must be >= 1 in {source}.")
    if fields["n_absolute_ticks"] < 3:
        raise ValueError(f"'n_absolute_ticks' must be >= 3 in {source}.")
    if len(fields["flow_windows"]) != len(fields["flow_rollup_weights"]):
        raise ValueError(
            "flow_windows and flow_rollup_weights must have identical lengths."
        )
    if len(fields["flow_derivative_weights"]) != 3:
        raise ValueError(
            "flow_derivative_weights must contain exactly 3 weights (d1,d2,d3)."
        )
    if fields["flow_tanh_scale"] <= 0.0:
        raise ValueError(f"'flow_tanh_scale' must be > 0 in {source}.")
    if not (0.0 < fields["flow_neutral_threshold"] < 1.0):
        raise ValueError(
            f"'flow_neutral_threshold' must be in (0,1), got {fields['flow_neutral_threshold']}."
        )
    projection_horizons_bins = fields["projection_horizons_bins"]
    if len(set(projection_horizons_bins)) != len(projection_horizons_bins):
        raise ValueError("'projection_horizons_bins' values must be unique.")

    for tau_name in (
        "tau_velocity",
        "tau_acceleration",
        "tau_jerk",
        "tau_rest_decay",
    ):
        if fields[tau_name] <= 0.0:
            raise ValueError(f"'{tau_name}' must be > 0 in {source}.")

    for coeff_name in (
        "c1_v_add",
        "c2_v_rest_pos",
        "c3_a_add",
        "c4_v_pull",
        "c5_v_fill",
        "c6_v_rest_neg",
        "c7_a_pull",
    ):
        if fields[coeff_name] < 0.0:
            raise ValueError(f"'{coeff_name}' must be >= 0 in {source}.")

    if fields["state_model_center_exclusion_radius"] < 0:
        raise ValueError("'state_model_center_exclusion_radius' must be >= 0.")
    if fields["state_model_spatial_decay_power"] < 0.0:
        raise ValueError("'state_model_spatial_decay_power' must be >= 0.")
    if fields["flow_zscore_window_bins"] < 2:
        raise ValueError("'flow_zscore_window_bins' must be >= 2.")
    if fields["flow_zscore_min_periods"] < 2:
        raise ValueError("'flow_zscore_min_periods' must be >= 2.")
    if fields["flow_zscore_min_periods"] > fields["flow_zscore_window_bins"]:
        raise ValueError(
            "'flow_zscore_min_periods' cannot exceed 'flow_zscore_window_bins'."
        )
    if fields["state_model_zscore_window_bins"] < 2:
        raise ValueError("'state_model_zscore_window_bins' must be >= 2.")
    if fields["state_model_zscore_min_periods"] < 2:
        raise ValueError("'state_model_zscore_min_periods' must be >= 2.")
    if fields["state_model_zscore_min_periods"] > fields["state_model_zscore_window_bins"]:
        raise ValueError(
            "'state_model_zscore_min_periods' cannot exceed 'state_model_zscore_window_bins'."
        )
    if fields["state_model_tanh_scale"] <= 0.0:
        raise ValueError("'state_model_tanh_scale' must be > 0.")
    for name in (
        "state_model_d1_weight",
        "state_model_d2_weight",
        "state_model_d3_weight",
        "state_model_bull_pressure_weight",
        "state_model_bull_vacuum_weight",
        "state_model_bear_pressure_weight",
        "state_model_bear_vacuum_weight",
        "state_model_mixed_weight",
    ):
        if fields[name] < 0.0:
            raise ValueError(f"'{name}' must be >= 0.")
    if (
        abs(fields["state_model_d1_weight"])
        + abs(fields["state_model_d2_weight"])
        + abs(fields["state_model_d3_weight"])
        <= 0.0
    ):
        raise ValueError("At least one of state_model_d1_weight/state_model_d2_weight/state_model_d3_weight must be > 0.")

    projection_horizons_ms = tuple(
        int(bin_count) * int(fields["cell_width_ms"])
        for bin_count in projection_horizons_bins
    )
    if any(h_ms <= 0 for h_ms in projection_horizons_ms):
        raise ValueError(
            "'projection_horizons_bins' x 'cell_width_ms' must resolve to positive ms horizons."
        )
    fields["projection_horizons_ms"] = projection_horizons_ms

    return fields


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

    fields = _normalize_runtime_fields(raw, source=str(path))
    config_version = _compute_config_version(fields)
    return VPRuntimeConfig(**fields, config_version=config_version)


def build_config_with_overrides(
    base_cfg: VPRuntimeConfig,
    overrides: Mapping[str, Any] | None,
) -> VPRuntimeConfig:
    """Build a new validated runtime config by applying overrides to a base config.

    Unknown keys fail fast. Derived fields (``projection_horizons_ms``,
    ``config_version``) are recomputed from the final values.
    """
    if not overrides:
        return base_cfg

    merged: dict[str, Any] = base_cfg.to_dict()
    merged.pop("config_version", None)
    merged.pop("projection_horizons_ms", None)

    allowed = set(merged.keys())
    unknown = sorted(k for k in overrides.keys() if k not in allowed)
    if unknown:
        raise ValueError(
            "Unknown runtime override keys: "
            f"{unknown}. Allowed keys: {sorted(allowed)}"
        )

    for key, value in overrides.items():
        merged[key] = value

    fields = _normalize_runtime_fields(merged, source="runtime_overrides")
    config_version = _compute_config_version(fields)
    return VPRuntimeConfig(**fields, config_version=config_version)


def parse_projection_horizons_bins_override(raw: Any) -> Tuple[int, ...] | None:
    """Parse optional projection horizon override into canonical bin tuple.

    Accepts either:
    - ``None`` / empty string -> no override
    - comma-separated string (e.g. ``"1,2,4"``)
    - iterable of ints
    """
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    return _parse_int_sequence(raw, "projection_horizons_bins")


def resolve_config(
    product_type: str,
    symbol: str,
) -> VPRuntimeConfig:
    """Resolve runtime config and enforce locked single-instrument contract."""
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
