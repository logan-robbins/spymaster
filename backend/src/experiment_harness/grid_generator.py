"""Generate grid variants by re-running the VP engine with modified parameters.

Patches module-level constants in ``event_engine.py`` at runtime without
modifying source files.  Each unique parameter set produces a deterministic
variant_id (SHA256 of all params -> 12-char hex).

Output is stored in ``lake/research/vp_harness/generated_grids/{variant_id}/``
with ``bins.parquet``, ``grid_clean.parquet``, and ``manifest.json``.

Thread-safety: NOT thread-safe. The patching strategy mutates module-level
globals, so only one ``generate()`` call may be active per process at a time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_schema import GridVariantConfig
from .dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)

# Maps GridVariantConfig field names -> event_engine module-level constant names.
_COEFF_FIELD_MAP: list[tuple[str, str]] = [
    ("C1_V_ADD", "c1_v_add"),
    ("C2_V_REST_POS", "c2_v_rest_pos"),
    ("C3_A_ADD", "c3_a_add"),
    ("C4_V_PULL", "c4_v_pull"),
    ("C5_V_FILL", "c5_v_fill"),
    ("C6_V_REST_NEG", "c6_v_rest_neg"),
    ("C7_A_PULL", "c7_a_pull"),
]

_TAU_FIELD_MAP: list[tuple[str, str]] = [
    ("TAU_VELOCITY", "tau_velocity"),
    ("TAU_ACCELERATION", "tau_acceleration"),
    ("TAU_JERK", "tau_jerk"),
]

# Grid-clean columns extracted from each bucket row.
_GRID_SCALAR_COLS: list[str] = [
    "add_mass",
    "pull_mass",
    "fill_mass",
    "rest_depth",
    "v_add",
    "v_pull",
    "v_fill",
    "v_rest_depth",
    "a_add",
    "a_pull",
    "a_fill",
    "a_rest_depth",
    "j_add",
    "j_pull",
    "j_fill",
    "j_rest_depth",
    "pressure_variant",
    "vacuum_variant",
    "spectrum_score",
    "spectrum_state_code",
]


def _import_event_engine() -> Any:
    """Lazily import event_engine to avoid hard dependency at module load.

    Returns:
        The ``src.vacuum_pressure.event_engine`` module object.

    Raises:
        ImportError: If vacuum_pressure is not available in this environment.
    """
    try:
        import src.vacuum_pressure.event_engine as ee  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Cannot import vacuum_pressure.event_engine. "
            "Grid generation requires the full VP runtime to be installed."
        ) from exc
    return ee


def _import_stream_pipeline() -> Any:
    """Lazily import stream_pipeline to avoid hard dependency at module load.

    Returns:
        The ``src.vacuum_pressure.stream_pipeline`` module object.

    Raises:
        ImportError: If vacuum_pressure is not available in this environment.
    """
    try:
        import src.vacuum_pressure.stream_pipeline as sp  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Cannot import vacuum_pressure.stream_pipeline. "
            "Grid generation requires the full VP runtime to be installed."
        ) from exc
    return sp


def _import_vp_config() -> Any:
    """Lazily import VP config module.

    Returns:
        The ``src.vacuum_pressure.config`` module object.

    Raises:
        ImportError: If vacuum_pressure is not available in this environment.
    """
    try:
        import src.vacuum_pressure.config as vp_config  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Cannot import vacuum_pressure.config. "
            "Grid generation requires the full VP runtime to be installed."
        ) from exc
    return vp_config


class GridGenerator:
    """Generate grid variants by re-running the VP engine with modified parameters.

    Patches module-level constants in event_engine.py at runtime without
    modifying source files. Each unique parameter set produces a deterministic
    variant_id (SHA256 of all params -> 12-char hex).

    Output is stored in lake/research/vp_harness/generated_grids/{variant_id}/
    with bins.parquet, grid_clean.parquet, and manifest.json.

    Args:
        lake_root: Path to the data lake root directory.
    """

    def __init__(self, lake_root: Path) -> None:
        self.lake_root = Path(lake_root)
        self.output_root = self.lake_root / "research" / "vp_harness" / "generated_grids"

    def generate(self, spec: GridVariantConfig) -> str:
        """Generate a grid variant from spec. Returns the variant_id.

        Steps:
            1. Compute deterministic variant_id from all grid-dependent params.
            2. Check if variant already exists (idempotent skip).
            3. Patch module-level constants in event_engine.
            4. Construct VPRuntimeConfig directly (bypass resolve_config).
            5. Run stream_events() collecting grid bins.
            6. Save bins.parquet + grid_clean.parquet + manifest.json.
            7. Restore original constants (guaranteed via try/finally).

        Args:
            spec: Grid variant configuration with all grid-dependent parameters.

        Returns:
            The 12-character hex variant_id string.

        Raises:
            ImportError: If VP modules are not available.
            FileNotFoundError: If the .dbn data file for the specified
                symbol/date is not found.
            RuntimeError: If the stream pipeline yields zero bins.
        """
        variant_id = self._compute_variant_id(spec)
        variant_dir = self.output_root / variant_id

        # Idempotent: skip if all outputs already exist
        manifest_path = variant_dir / "manifest.json"
        bins_path = variant_dir / "bins.parquet"
        grid_path = variant_dir / "grid_clean.parquet"
        if manifest_path.exists() and bins_path.exists() and grid_path.exists():
            logger.info(
                "Variant %s already exists, skipping generation: %s",
                variant_id,
                variant_dir,
            )
            return variant_id

        logger.info("Generating grid variant %s from spec: %s", variant_id, spec.model_dump())

        ee = _import_event_engine()
        originals = self._patch_constants(spec, ee)
        try:
            runtime_config = self._build_runtime_config(spec)
            bins_records, grid_rows = self._run_pipeline(spec, runtime_config)
        finally:
            self._restore_constants(originals, ee)

        if not bins_records:
            raise RuntimeError(
                f"stream_events yielded zero bins for variant {variant_id}. "
                f"Check that .dbn data exists for {spec.product_type}/{spec.symbol}/{spec.dt}."
            )

        # Save outputs
        variant_dir.mkdir(parents=True, exist_ok=True)

        bins_df = pd.DataFrame(bins_records)
        bins_df.to_parquet(bins_path, index=False, engine="pyarrow")
        logger.info("Wrote %d bins to %s", len(bins_df), bins_path)

        grid_df = pd.DataFrame(grid_rows)
        grid_df.to_parquet(grid_path, index=False, engine="pyarrow")
        logger.info("Wrote %d grid rows to %s", len(grid_df), grid_path)

        manifest = self._build_manifest(spec, variant_id, len(bins_records))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote manifest to %s", manifest_path)

        # Register with dataset registry (currently a no-op hook)
        registry = DatasetRegistry(self.lake_root)
        registry.register_generated(variant_id, variant_dir)

        logger.info("Grid variant %s generation complete.", variant_id)
        return variant_id

    def _compute_variant_id(self, spec: GridVariantConfig) -> str:
        """Compute a deterministic 12-character hex variant ID from spec params.

        Only grid-dependent parameters are included in the hash. Fields that
        accept sweep lists are normalized: single values are used directly,
        lists are sorted for determinism.

        Args:
            spec: Grid variant configuration.

        Returns:
            12-character hex string derived from SHA256 of sorted param dict.
        """
        param_dict: dict[str, Any] = {}
        for field_name in [
            "cell_width_ms",
            "c1_v_add",
            "c2_v_rest_pos",
            "c3_a_add",
            "c4_v_pull",
            "c5_v_fill",
            "c6_v_rest_neg",
            "c7_a_pull",
            "bucket_size_dollars",
            "spectrum_windows",
            "spectrum_derivative_weights",
            "spectrum_tanh_scale",
            "tau_velocity",
            "tau_acceleration",
            "tau_jerk",
            "product_type",
            "symbol",
            "dt",
            "start_time",
        ]:
            val = getattr(spec, field_name)
            if isinstance(val, list):
                param_dict[field_name] = list(val)
            else:
                param_dict[field_name] = val

        raw = "|".join(f"{k}={v}" for k, v in sorted(param_dict.items()))
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _patch_constants(
        self,
        spec: GridVariantConfig,
        ee: Any,
    ) -> dict[str, Any]:
        """Patch event_engine module constants from spec. Returns originals.

        Only patches constants whose corresponding spec field is not None
        (for optional fields) or differs from the default (for required fields).
        All patched values are logged for auditability.

        Args:
            spec: Grid variant configuration.
            ee: The event_engine module object.

        Returns:
            Dict mapping constant names to their original values before patching.
        """
        originals: dict[str, Any] = {}

        for attr_name, spec_name in _COEFF_FIELD_MAP:
            originals[attr_name] = getattr(ee, attr_name)
            val = getattr(spec, spec_name)
            if val is not None and not isinstance(val, list):
                setattr(ee, attr_name, float(val))
                logger.info("Patched %s: %s -> %s", attr_name, originals[attr_name], val)

        for attr_name, spec_name in _TAU_FIELD_MAP:
            originals[attr_name] = getattr(ee, attr_name)
            val = getattr(spec, spec_name)
            if val is not None and not isinstance(val, list):
                setattr(ee, attr_name, float(val))
                logger.info("Patched %s: %s -> %s", attr_name, originals[attr_name], val)

        # TAU_REST_DECAY is not currently in GridVariantConfig but we capture it
        # for completeness in case future specs add it.
        originals["TAU_REST_DECAY"] = getattr(ee, "TAU_REST_DECAY")

        return originals

    def _restore_constants(self, originals: dict[str, Any], ee: Any) -> None:
        """Restore original event_engine module constants.

        Args:
            originals: Dict mapping constant names to their original values.
            ee: The event_engine module object.
        """
        for attr_name, original_val in originals.items():
            setattr(ee, attr_name, original_val)
        logger.info("Restored %d event_engine constants to original values.", len(originals))

    def _build_runtime_config(self, spec: GridVariantConfig) -> Any:
        """Construct a VPRuntimeConfig directly from spec and instrument.yaml defaults.

        Bypasses ``resolve_config()`` to avoid single-instrument lock enforcement.
        Reads the locked instrument.yaml for fields that GridVariantConfig does
        not override (e.g. price_scale, tick_size, contract_multiplier).

        Args:
            spec: Grid variant configuration.

        Returns:
            VPRuntimeConfig instance with spec overrides applied.

        Raises:
            ImportError: If VP config module is not available.
            FileNotFoundError: If instrument.yaml is not found.
        """
        vp_config_mod = _import_vp_config()

        # Load base config from instrument.yaml
        locked_path = vp_config_mod._resolve_locked_config_path()
        base_cfg = vp_config_mod._load_locked_instrument_config(locked_path)

        # Resolve cell_width_ms: spec may hold a single int or a sweep list.
        # For generation, we require a single value.
        cell_width_ms = spec.cell_width_ms
        if isinstance(cell_width_ms, list):
            raise ValueError(
                "GridGenerator.generate() requires a scalar cell_width_ms, "
                f"not a sweep list: {cell_width_ms}. "
                "The experiment runner must explode sweep axes before calling generate()."
            )
        if isinstance(spec.bucket_size_dollars, list):
            raise ValueError(
                "GridGenerator.generate() requires a scalar bucket_size_dollars, "
                f"not a sweep list: {spec.bucket_size_dollars}. "
                "The experiment runner must explode sweep axes before calling generate()."
            )

        # Override spectrum params if provided
        spectrum_windows = (
            tuple(spec.spectrum_windows)
            if spec.spectrum_windows is not None
            else base_cfg.spectrum_windows
        )
        spectrum_derivative_weights = (
            tuple(spec.spectrum_derivative_weights)
            if spec.spectrum_derivative_weights is not None
            else base_cfg.spectrum_derivative_weights
        )
        spectrum_tanh_scale = (
            spec.spectrum_tanh_scale
            if spec.spectrum_tanh_scale is not None
            else base_cfg.spectrum_tanh_scale
        )

        # Rollup weights must match spectrum_windows length
        if spec.spectrum_windows is not None:
            # If windows changed, use uniform weights unless base matches
            if len(spectrum_windows) == len(base_cfg.spectrum_rollup_weights):
                rollup_weights = base_cfg.spectrum_rollup_weights
            else:
                rollup_weights = tuple(1.0 for _ in spectrum_windows)
        else:
            rollup_weights = base_cfg.spectrum_rollup_weights

        projection_horizons_bins = (
            tuple(int(x) for x in spec.projection_horizons_bins)
            if spec.projection_horizons_bins is not None
            else tuple(int(x) for x in base_cfg.projection_horizons_bins)
        )
        if not projection_horizons_bins:
            raise ValueError("projection_horizons_bins must contain at least one bin count.")
        if any(int(x) <= 0 for x in projection_horizons_bins):
            raise ValueError("projection_horizons_bins values must be positive integers.")
        projection_horizons_ms = tuple(
            int(bin_count) * int(cell_width_ms)
            for bin_count in projection_horizons_bins
        )

        fields: dict[str, Any] = {
            "product_type": spec.product_type,
            "symbol": spec.symbol,
            "symbol_root": base_cfg.symbol_root,
            "price_scale": base_cfg.price_scale,
            "tick_size": base_cfg.tick_size,
            "bucket_size_dollars": (
                float(spec.bucket_size_dollars)
                if spec.bucket_size_dollars is not None
                else base_cfg.bucket_size_dollars
            ),
            "rel_tick_size": base_cfg.rel_tick_size,
            "grid_radius_ticks": base_cfg.grid_radius_ticks,
            "cell_width_ms": cell_width_ms,
            "n_absolute_ticks": base_cfg.n_absolute_ticks,
            "spectrum_windows": spectrum_windows,
            "spectrum_rollup_weights": rollup_weights,
            "spectrum_derivative_weights": spectrum_derivative_weights,
            "spectrum_tanh_scale": spectrum_tanh_scale,
            "spectrum_threshold_neutral": base_cfg.spectrum_threshold_neutral,
            "zscore_window_bins": base_cfg.zscore_window_bins,
            "zscore_min_periods": base_cfg.zscore_min_periods,
            "projection_horizons_bins": projection_horizons_bins,
            "projection_horizons_ms": projection_horizons_ms,
            "contract_multiplier": base_cfg.contract_multiplier,
            "qty_unit": base_cfg.qty_unit,
            "price_decimals": base_cfg.price_decimals,
        }

        config_version = vp_config_mod._compute_config_version(fields)
        return vp_config_mod.VPRuntimeConfig(**fields, config_version=config_version)

    def _run_pipeline(
        self,
        spec: GridVariantConfig,
        runtime_config: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run stream_events and collect bins + grid rows.

        Args:
            spec: Grid variant configuration (for dt and start_time).
            runtime_config: VPRuntimeConfig to pass to stream_events.

        Returns:
            Tuple of (bins_records, grid_rows) where:
                bins_records: list of dicts with bin-level metadata
                    (bin_seq, ts_ns, bin_start_ns, bin_end_ns, mid_price, etc.)
                grid_rows: list of dicts with per-tick per-bin grid values
                    (bin_seq, k, pressure_variant, vacuum_variant, etc.)
        """
        sp = _import_stream_pipeline()

        bins_records: list[dict[str, Any]] = []
        grid_rows: list[dict[str, Any]] = []

        t_start = time.monotonic()
        bin_count = 0

        for grid in sp.stream_events(
            lake_root=self.lake_root,
            config=runtime_config,
            dt=spec.dt,
            start_time=spec.start_time,
        ):
            bin_seq = int(grid["bin_seq"])
            ts_ns = int(grid["ts_ns"])
            bin_start_ns = int(grid["bin_start_ns"])
            bin_end_ns = int(grid["bin_end_ns"])
            mid_price = float(grid["mid_price"])
            event_id = int(grid["event_id"])
            bin_event_count = int(grid["bin_event_count"])
            book_valid = bool(grid["book_valid"])
            best_bid_int = int(grid["best_bid_price_int"])
            best_ask_int = int(grid["best_ask_price_int"])
            spot_ref_int = int(grid["spot_ref_price_int"])

            bins_records.append({
                "bin_seq": bin_seq,
                "ts_ns": ts_ns,
                "bin_start_ns": bin_start_ns,
                "bin_end_ns": bin_end_ns,
                "mid_price": mid_price,
                "event_id": event_id,
                "bin_event_count": bin_event_count,
                "book_valid": book_valid,
                "best_bid_price_int": best_bid_int,
                "best_ask_price_int": best_ask_int,
                "spot_ref_price_int": spot_ref_int,
            })

            # Extract per-tick data from buckets
            buckets: list[dict[str, Any]] = grid.get("buckets", [])
            for bucket in buckets:
                row: dict[str, Any] = {
                    "bin_seq": bin_seq,
                    "k": int(bucket["k"]),
                }
                for col in _GRID_SCALAR_COLS:
                    row[col] = float(bucket.get(col, 0.0))
                row["last_event_id"] = int(bucket.get("last_event_id", 0))

                # Include projection columns dynamically
                for key, val in bucket.items():
                    if key.startswith("proj_score_h"):
                        row[key] = float(val)

                grid_rows.append(row)

            bin_count += 1
            if bin_count % 500 == 0:
                elapsed = time.monotonic() - t_start
                logger.info(
                    "Grid generation progress: %d bins (%.1fs, %.0f bins/s)",
                    bin_count,
                    elapsed,
                    bin_count / elapsed if elapsed > 0 else 0.0,
                )

        elapsed = time.monotonic() - t_start
        logger.info(
            "Grid generation complete: %d bins, %d grid rows in %.2fs",
            bin_count,
            len(grid_rows),
            elapsed,
        )

        return bins_records, grid_rows

    def _build_manifest(
        self,
        spec: GridVariantConfig,
        variant_id: str,
        n_bins: int,
    ) -> dict[str, Any]:
        """Build manifest dict for reproducibility.

        Args:
            spec: Grid variant configuration used for generation.
            variant_id: The computed 12-char hex variant ID.
            n_bins: Number of bins produced.

        Returns:
            JSON-serializable manifest dict with all params and metadata.
        """
        return {
            "variant_id": variant_id,
            "n_bins": n_bins,
            "spec": spec.model_dump(),
            "grid_dependent_params": {
                "cell_width_ms": spec.cell_width_ms,
                "c1_v_add": spec.c1_v_add,
                "c2_v_rest_pos": spec.c2_v_rest_pos,
                "c3_a_add": spec.c3_a_add,
                "c4_v_pull": spec.c4_v_pull,
                "c5_v_fill": spec.c5_v_fill,
                "c6_v_rest_neg": spec.c6_v_rest_neg,
                "c7_a_pull": spec.c7_a_pull,
                "bucket_size_dollars": spec.bucket_size_dollars,
                "tau_velocity": spec.tau_velocity,
                "tau_acceleration": spec.tau_acceleration,
                "tau_jerk": spec.tau_jerk,
                "spectrum_windows": spec.spectrum_windows,
                "spectrum_derivative_weights": spec.spectrum_derivative_weights,
                "spectrum_tanh_scale": spec.spectrum_tanh_scale,
                "projection_horizons_bins": spec.projection_horizons_bins,
            },
            "files": [
                "bins.parquet",
                "grid_clean.parquet",
                "manifest.json",
            ],
        }
