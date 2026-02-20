"""Generate grid variants by re-running the VP engine with modified parameters.

Each unique parameter set produces a deterministic variant_id (SHA256 of all
relevant params -> 12-char hex).

Output is stored in ``lake/research/vp_harness/generated_grids/{variant_id}/``
with ``bins.parquet``, ``grid_clean.parquet``, and ``manifest.json``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.vp_shared.hashing import stable_short_hash

from .config_schema import GridVariantConfig
from .dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)

# Grid-clean columns extracted from each bucket row.
_GRID_SCALAR_COLS: list[str] = [
    "add_mass",
    "pull_mass",
    "fill_mass",
    "rest_depth",
    "bid_depth",
    "ask_depth",
    "v_add",
    "v_pull",
    "v_fill",
    "v_rest_depth",
    "v_bid_depth",
    "v_ask_depth",
    "a_add",
    "a_pull",
    "a_fill",
    "a_rest_depth",
    "a_bid_depth",
    "a_ask_depth",
    "j_add",
    "j_pull",
    "j_fill",
    "j_rest_depth",
    "j_bid_depth",
    "j_ask_depth",
    "pressure_variant",
    "vacuum_variant",
    "best_ask_move_ticks",
    "best_bid_move_ticks",
    "ask_reprice_sign",
    "bid_reprice_sign",
    "microstate_id",
    "state5_code",
    "chase_up_flag",
    "chase_down_flag",
]


def _import_stream_pipeline() -> Any:
    """Lazily import stream_pipeline to avoid hard dependency at module load."""
    try:
        import src.vacuum_pressure.stream_pipeline as sp  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Cannot import vacuum_pressure.stream_pipeline. "
            "Grid generation requires the full VP runtime to be installed."
        ) from exc
    return sp


def _import_vp_config() -> Any:
    """Lazily import VP config module."""
    try:
        import src.vacuum_pressure.config as vp_config  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Cannot import vacuum_pressure.config. "
            "Grid generation requires the full VP runtime to be installed."
        ) from exc
    return vp_config


def _require_scalar(value: Any, field_name: str) -> Any:
    """Fail fast when a sweep list leaks into one concrete generation spec."""
    if isinstance(value, list):
        raise ValueError(
            f"GridGenerator.generate() requires scalar {field_name}, "
            f"not a sweep list: {value}. "
            "The experiment runner must explode sweep axes before calling generate()."
        )
    return value


class GridGenerator:
    """Generate grid variants by re-running the VP engine with modified parameters."""

    def __init__(self, lake_root: Path) -> None:
        self.lake_root = Path(lake_root)
        self.output_root = self.lake_root / "research" / "vp_harness" / "generated_grids"

    def generate(self, spec: GridVariantConfig) -> str:
        """Generate a grid variant from spec. Returns the variant_id."""
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

        runtime_config = self._build_runtime_config(spec)
        bins_records, grid_rows = self._run_pipeline(spec, runtime_config)

        if not bins_records:
            raise RuntimeError(
                f"stream_events yielded zero bins for variant {variant_id}. "
                f"Check that .dbn data exists for {spec.product_type}/{spec.symbol}/{spec.dt}."
            )

        variant_dir.mkdir(parents=True, exist_ok=True)

        bins_df = pd.DataFrame(bins_records)
        bins_df.to_parquet(bins_path, index=False, engine="pyarrow")
        logger.info("Wrote %d bins to %s", len(bins_df), bins_path)

        grid_df = pd.DataFrame(grid_rows)
        grid_df.to_parquet(grid_path, index=False, engine="pyarrow")
        logger.info("Wrote %d grid rows to %s", len(grid_df), grid_path)

        manifest = self._build_manifest(
            spec,
            variant_id,
            len(bins_records),
            runtime_config=runtime_config,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote manifest to %s", manifest_path)

        registry = DatasetRegistry(self.lake_root)
        registry.register_generated(variant_id, variant_dir)

        logger.info("Grid variant %s generation complete.", variant_id)
        return variant_id

    def _compute_variant_id(self, spec: GridVariantConfig) -> str:
        """Compute deterministic 12-character hex variant ID from spec params."""
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
            "flow_windows",
            "flow_rollup_weights",
            "flow_derivative_weights",
            "flow_tanh_scale",
            "tau_velocity",
            "tau_acceleration",
            "tau_jerk",
            "tau_rest_decay",
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

        return stable_short_hash(param_dict, length=12)

    def _build_runtime_config(self, spec: GridVariantConfig) -> Any:
        """Construct a VPRuntimeConfig directly from spec + locked defaults."""
        vp_config_mod = _import_vp_config()

        locked_path = vp_config_mod._resolve_locked_config_path()
        base_cfg = vp_config_mod._load_locked_instrument_config(locked_path)

        cell_width_ms = int(_require_scalar(spec.cell_width_ms, "cell_width_ms"))
        bucket_size_dollars = spec.bucket_size_dollars
        if bucket_size_dollars is not None:
            bucket_size_dollars = float(
                _require_scalar(bucket_size_dollars, "bucket_size_dollars")
            )

        # Spectrum window overrides with stable rollup behavior.
        flow_windows = (
            list(spec.flow_windows)
            if spec.flow_windows is not None
            else list(base_cfg.flow_windows)
        )
        if spec.flow_rollup_weights is not None:
            flow_rollup_weights = list(spec.flow_rollup_weights)
        elif spec.flow_windows is not None:
            if len(flow_windows) == len(base_cfg.flow_rollup_weights):
                flow_rollup_weights = list(base_cfg.flow_rollup_weights)
            else:
                flow_rollup_weights = [1.0 for _ in flow_windows]
        else:
            flow_rollup_weights = list(base_cfg.flow_rollup_weights)

        flow_derivative_weights = (
            list(spec.flow_derivative_weights)
            if spec.flow_derivative_weights is not None
            else list(base_cfg.flow_derivative_weights)
        )
        flow_tanh_scale = (
            float(spec.flow_tanh_scale)
            if spec.flow_tanh_scale is not None
            else float(base_cfg.flow_tanh_scale)
        )

        def _optional_scalar(
            raw_value: float | list[float] | None,
            field_name: str,
            default_value: float,
        ) -> float:
            if raw_value is None:
                return float(default_value)
            return float(_require_scalar(raw_value, field_name))

        overrides: dict[str, Any] = {
            "product_type": spec.product_type,
            "symbol": spec.symbol,
            "cell_width_ms": cell_width_ms,
            "bucket_size_dollars": (
                bucket_size_dollars
                if bucket_size_dollars is not None
                else base_cfg.bucket_size_dollars
            ),
            "flow_windows": flow_windows,
            "flow_rollup_weights": flow_rollup_weights,
            "flow_derivative_weights": flow_derivative_weights,
            "flow_tanh_scale": flow_tanh_scale,
            "tau_velocity": _optional_scalar(
                spec.tau_velocity,
                "tau_velocity",
                base_cfg.tau_velocity,
            ),
            "tau_acceleration": _optional_scalar(
                spec.tau_acceleration,
                "tau_acceleration",
                base_cfg.tau_acceleration,
            ),
            "tau_jerk": _optional_scalar(
                spec.tau_jerk,
                "tau_jerk",
                base_cfg.tau_jerk,
            ),
            "tau_rest_decay": _optional_scalar(
                spec.tau_rest_decay,
                "tau_rest_decay",
                base_cfg.tau_rest_decay,
            ),
            "c1_v_add": float(_require_scalar(spec.c1_v_add, "c1_v_add")),
            "c2_v_rest_pos": float(_require_scalar(spec.c2_v_rest_pos, "c2_v_rest_pos")),
            "c3_a_add": float(_require_scalar(spec.c3_a_add, "c3_a_add")),
            "c4_v_pull": float(_require_scalar(spec.c4_v_pull, "c4_v_pull")),
            "c5_v_fill": float(_require_scalar(spec.c5_v_fill, "c5_v_fill")),
            "c6_v_rest_neg": float(_require_scalar(spec.c6_v_rest_neg, "c6_v_rest_neg")),
            "c7_a_pull": float(_require_scalar(spec.c7_a_pull, "c7_a_pull")),
        }

        return vp_config_mod.build_config_with_overrides(base_cfg, overrides)

    def _run_pipeline(
        self,
        spec: GridVariantConfig,
        runtime_config: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run stream_events and collect bins + grid rows."""
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

            cols: dict[str, Any] = grid["grid_cols"]
            n_rows = len(cols["k"])
            for row_idx in range(n_rows):
                row: dict[str, Any] = {
                    "bin_seq": bin_seq,
                    "k": int(cols["k"][row_idx]),
                }
                for col in _GRID_SCALAR_COLS:
                    row[col] = float(cols[col][row_idx])
                row["last_event_id"] = int(cols["last_event_id"][row_idx])

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
        *,
        runtime_config: Any,
    ) -> dict[str, Any]:
        """Build manifest dict for reproducibility."""
        return {
            "variant_id": variant_id,
            "n_bins": n_bins,
            "spec": spec.model_dump(),
            "flow_scoring": {
                "derivative_weights": list(runtime_config.flow_derivative_weights),
                "tanh_scale": float(runtime_config.flow_tanh_scale),
                "threshold_neutral": float(runtime_config.flow_neutral_threshold),
                "zscore_window_bins": int(runtime_config.flow_zscore_window_bins),
                "zscore_min_periods": int(runtime_config.flow_zscore_min_periods),
            },
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
                "tau_rest_decay": spec.tau_rest_decay,
                "flow_windows": spec.flow_windows,
                "flow_rollup_weights": spec.flow_rollup_weights,
                "flow_derivative_weights": spec.flow_derivative_weights,
                "flow_tanh_scale": spec.flow_tanh_scale,
            },
            "files": [
                "bins.parquet",
                "grid_clean.parquet",
                "manifest.json",
            ],
        }
