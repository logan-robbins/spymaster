"""Run canonical VP compute once and persist full emitted output for replay/data science.

Usage example:
    uv run scripts/cache_vp_output.py \
      --product-type future_mbo \
      --symbol MNQH6 \
      --dt 2026-02-06 \
      --capture-start-et 09:25:00 \
      --capture-end-et 10:25:00 \
      --output-dir /tmp/vp_cache_mnq_0925_1025
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

from src.vacuum_pressure.config import VPRuntimeConfig, resolve_config
from src.vacuum_pressure.stream_pipeline import stream_events

logger = logging.getLogger("cache_vp_output")

_BIN_FIELDS: list[tuple[str, pa.DataType]] = [
    ("ts_ns", pa.int64()),
    ("bin_seq", pa.int64()),
    ("bin_start_ns", pa.int64()),
    ("bin_end_ns", pa.int64()),
    ("bin_event_count", pa.int64()),
    ("event_id", pa.int64()),
    ("mid_price", pa.float64()),
    ("spot_ref_price_int", pa.int64()),
    ("best_bid_price_int", pa.int64()),
    ("best_ask_price_int", pa.int64()),
    ("book_valid", pa.bool_()),
]

_BUCKET_FLOAT_FIELDS: tuple[str, ...] = (
    "pressure_variant",
    "vacuum_variant",
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
    "spectrum_score",
)

_BUCKET_INT_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("k", pa.int32()),
    ("spectrum_state_code", pa.int8()),
    ("last_event_id", pa.int64()),
)

_BUCKET_PREFIX_FIELDS: list[tuple[str, pa.DataType]] = [
    ("ts_ns", pa.int64()),
    ("bin_seq", pa.int64()),
    ("bin_start_ns", pa.int64()),
    ("bin_end_ns", pa.int64()),
    ("bin_event_count", pa.int64()),
    ("event_id", pa.int64()),
    ("mid_price", pa.float64()),
    ("spot_ref_price_int", pa.int64()),
    ("best_bid_price_int", pa.int64()),
    ("best_ask_price_int", pa.int64()),
    ("book_valid", pa.bool_()),
]


def _parse_et_timestamp_ns(dt: str, raw_time: str, field_name: str) -> tuple[pd.Timestamp, int]:
    try:
        parsed = pd.to_datetime(f"{dt} {raw_time}", errors="raise")
    except Exception as exc:  # pragma: no cover - pandas exception type is unstable
        raise ValueError(
            f"{field_name} must be a valid ET wall-clock time (e.g. 09:25 or 09:25:00 AM). "
            f"Got: {raw_time}"
        ) from exc

    ts = pd.Timestamp(parsed)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    return ts, int(ts.tz_convert("UTC").value)


def _stream_start_hhmm(capture_start_et: pd.Timestamp) -> str:
    if capture_start_et.second != 0 or capture_start_et.microsecond != 0 or capture_start_et.nanosecond != 0:
        raise ValueError(
            "capture_start_et must align to a minute boundary because stream warmup currently uses HH:MM resolution."
        )
    return capture_start_et.strftime("%H:%M")


def _bin_schema() -> pa.Schema:
    return pa.schema([pa.field(name, dtype) for name, dtype in _BIN_FIELDS])


def _bucket_schema(projection_horizons_ms: Iterable[int]) -> pa.Schema:
    fields = [pa.field(name, dtype) for name, dtype in _BUCKET_PREFIX_FIELDS]
    for name, dtype in _BUCKET_INT_FIELDS:
        fields.append(pa.field(name, dtype))
    for name in _BUCKET_FLOAT_FIELDS:
        fields.append(pa.field(name, pa.float64()))
    for horizon_ms in projection_horizons_ms:
        fields.append(pa.field(f"proj_score_h{int(horizon_ms)}", pa.float64()))
    return pa.schema(fields)


def _grid_bin_row(grid: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ts_ns": int(grid["ts_ns"]),
        "bin_seq": int(grid["bin_seq"]),
        "bin_start_ns": int(grid["bin_start_ns"]),
        "bin_end_ns": int(grid["bin_end_ns"]),
        "bin_event_count": int(grid["bin_event_count"]),
        "event_id": int(grid["event_id"]),
        "mid_price": float(grid["mid_price"]),
        "spot_ref_price_int": int(grid["spot_ref_price_int"]),
        "best_bid_price_int": int(grid["best_bid_price_int"]),
        "best_ask_price_int": int(grid["best_ask_price_int"]),
        "book_valid": bool(grid["book_valid"]),
    }


def _grid_bucket_rows(grid: Dict[str, Any], projection_horizons_ms: tuple[int, ...]) -> list[Dict[str, Any]]:
    prefix: Dict[str, Any] = {
        "ts_ns": int(grid["ts_ns"]),
        "bin_seq": int(grid["bin_seq"]),
        "bin_start_ns": int(grid["bin_start_ns"]),
        "bin_end_ns": int(grid["bin_end_ns"]),
        "bin_event_count": int(grid["bin_event_count"]),
        "event_id": int(grid["event_id"]),
        "mid_price": float(grid["mid_price"]),
        "spot_ref_price_int": int(grid["spot_ref_price_int"]),
        "best_bid_price_int": int(grid["best_bid_price_int"]),
        "best_ask_price_int": int(grid["best_ask_price_int"]),
        "book_valid": bool(grid["book_valid"]),
    }
    rows: list[Dict[str, Any]] = []
    for bucket in grid["buckets"]:
        row: Dict[str, Any] = dict(prefix)
        for name, _dtype in _BUCKET_INT_FIELDS:
            row[name] = int(bucket[name])
        for name in _BUCKET_FLOAT_FIELDS:
            row[name] = float(bucket[name])
        for horizon_ms in projection_horizons_ms:
            key = f"proj_score_h{horizon_ms}"
            row[key] = float(bucket[key])
        rows.append(row)
    return rows


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def capture_stream_output(
    *,
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    stream_start_time_hhmm: str,
    capture_start_ns: int,
    capture_end_ns: int,
    output_dir: Path,
    projection_use_cubic: bool = False,
    projection_cubic_scale: float = 1.0 / 6.0,
    projection_damping_lambda: float = 0.0,
    flush_bins_every: int = 200,
) -> Dict[str, Any]:
    if capture_end_ns <= capture_start_ns:
        raise ValueError("capture_end_ns must be greater than capture_start_ns.")
    if flush_bins_every <= 0:
        raise ValueError(f"flush_bins_every must be > 0, got {flush_bins_every}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output_dir must be empty (or not exist): {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    bins_path = output_dir / "bins.parquet"
    buckets_path = output_dir / "buckets.parquet"
    manifest_path = output_dir / "manifest.json"

    projection_horizons = tuple(int(x) for x in config.projection_horizons_ms)
    bins_schema = _bin_schema()
    buckets_schema = _bucket_schema(projection_horizons)

    bins_buffer: list[Dict[str, Any]] = []
    buckets_buffer: list[Dict[str, Any]] = []

    n_bins = 0
    n_bucket_rows = 0
    first_ts_ns: int | None = None
    last_ts_ns: int | None = None
    t0 = time.monotonic()

    bins_writer: pq.ParquetWriter | None = None
    buckets_writer: pq.ParquetWriter | None = None

    def _flush() -> None:
        nonlocal bins_buffer, buckets_buffer
        if bins_buffer:
            assert bins_writer is not None
            bins_writer.write_table(pa.Table.from_pylist(bins_buffer, schema=bins_schema))
            bins_buffer = []
        if buckets_buffer:
            assert buckets_writer is not None
            buckets_writer.write_table(pa.Table.from_pylist(buckets_buffer, schema=buckets_schema))
            buckets_buffer = []

    try:
        bins_writer = pq.ParquetWriter(bins_path, bins_schema, compression="zstd")
        buckets_writer = pq.ParquetWriter(buckets_path, buckets_schema, compression="zstd")

        for grid in stream_events(
            lake_root=lake_root,
            config=config,
            dt=dt,
            start_time=stream_start_time_hhmm,
            projection_use_cubic=projection_use_cubic,
            projection_cubic_scale=projection_cubic_scale,
            projection_damping_lambda=projection_damping_lambda,
        ):
            ts_ns = int(grid["ts_ns"])
            if ts_ns < capture_start_ns:
                continue
            if ts_ns >= capture_end_ns:
                break

            bins_buffer.append(_grid_bin_row(grid))
            bucket_rows = _grid_bucket_rows(grid, projection_horizons)
            buckets_buffer.extend(bucket_rows)

            n_bins += 1
            n_bucket_rows += len(bucket_rows)
            if first_ts_ns is None:
                first_ts_ns = ts_ns
            last_ts_ns = ts_ns

            if len(bins_buffer) >= flush_bins_every:
                _flush()
                logger.info(
                    "captured %d bins / %d bucket rows so far (last_bin_ts_ns=%d)",
                    n_bins,
                    n_bucket_rows,
                    ts_ns,
                )

        if n_bins == 0:
            raise RuntimeError("No emitted bins captured in the requested window.")

        _flush()
    except Exception:
        for path in (bins_path, buckets_path, manifest_path):
            if path.exists():
                path.unlink()
        raise
    finally:
        if bins_writer is not None:
            bins_writer.close()
        if buckets_writer is not None:
            buckets_writer.close()

    elapsed_s = time.monotonic() - t0

    summary: Dict[str, Any] = {
        "product_type": config.product_type,
        "symbol": config.symbol,
        "dt": dt,
        "stream_start_time_hhmm": stream_start_time_hhmm,
        "capture_start_ns": capture_start_ns,
        "capture_end_ns": capture_end_ns,
        "projection_model": {
            "use_cubic": projection_use_cubic,
            "cubic_scale": projection_cubic_scale,
            "damping_lambda": projection_damping_lambda,
        },
        "config_version": config.config_version,
        "cell_width_ms": config.cell_width_ms,
        "grid_radius_ticks": config.grid_radius_ticks,
        "n_absolute_ticks": config.n_absolute_ticks,
        "projection_horizons_bins": [int(x) for x in config.projection_horizons_bins],
        "projection_horizons_ms": list(projection_horizons),
        "rows": {
            "bins": n_bins,
            "buckets": n_bucket_rows,
        },
        "first_bin_ts_ns": first_ts_ns,
        "last_bin_ts_ns": last_ts_ns,
        "elapsed_seconds": elapsed_s,
        "output": {
            "bins_parquet": str(bins_path),
            "buckets_parquet": str(buckets_path),
            "manifest_json": str(manifest_path),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical VP stream compute over a bounded ET window and persist "
            "full emitted output to parquet."
        ),
    )
    parser.add_argument(
        "--product-type",
        required=True,
        choices=["equity_mbo", "future_mbo"],
        help="Product type",
    )
    parser.add_argument("--symbol", required=True, help="Instrument symbol")
    parser.add_argument("--dt", required=True, help="Date YYYY-MM-DD")
    parser.add_argument(
        "--capture-start-et",
        required=True,
        help="Capture start wall time in ET (for example: 09:25 or 09:25:00 AM).",
    )
    parser.add_argument(
        "--capture-end-et",
        required=True,
        help="Capture end wall time in ET (exclusive).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for bins.parquet, buckets.parquet, and manifest.json.",
    )
    parser.add_argument(
        "--flush-bins-every",
        type=int,
        default=200,
        help="Flush parquet writers every N captured bins.",
    )
    parser.add_argument(
        "--projection-use-cubic",
        action="store_true",
        help="Enable cubic score projection term (score_d3 * h^3).",
    )
    parser.add_argument(
        "--projection-cubic-scale",
        type=float,
        default=1.0 / 6.0,
        help="Scale applied to cubic projection term when cubic mode is enabled.",
    )
    parser.add_argument(
        "--projection-damping-lambda",
        type=float,
        default=0.0,
        help="Exponential damping coefficient lambda for projection: proj *= exp(-lambda*h).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    if args.projection_cubic_scale < 0.0:
        parser.error("--projection-cubic-scale must be >= 0")
    if args.projection_damping_lambda < 0.0:
        parser.error("--projection-damping-lambda must be >= 0")
    if args.flush_bins_every <= 0:
        parser.error("--flush-bins-every must be > 0")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Replay can generate very high-volume out-of-range warnings; keep capture logs actionable.
    logging.getLogger("src.vacuum_pressure.event_engine").setLevel(logging.ERROR)

    capture_start_et, capture_start_ns = _parse_et_timestamp_ns(
        args.dt, args.capture_start_et, "--capture-start-et"
    )
    _capture_end_et, capture_end_ns = _parse_et_timestamp_ns(
        args.dt, args.capture_end_et, "--capture-end-et"
    )
    if capture_end_ns <= capture_start_ns:
        raise ValueError(
            "--capture-end-et must be later than --capture-start-et within the same session date."
        )

    stream_start_time_hhmm = _stream_start_hhmm(capture_start_et)

    products_yaml_path = backend_root / "src" / "data_eng" / "config" / "products.yaml"
    lake_root = backend_root / "lake"
    config = resolve_config(args.product_type, args.symbol, products_yaml_path)

    logger.info(
        "starting output capture: %s/%s dt=%s capture=[%s,%s) ET stream_start=%s output_dir=%s",
        args.product_type,
        args.symbol,
        args.dt,
        args.capture_start_et,
        args.capture_end_et,
        stream_start_time_hhmm,
        args.output_dir,
    )

    summary = capture_stream_output(
        lake_root=lake_root,
        config=config,
        dt=args.dt,
        stream_start_time_hhmm=stream_start_time_hhmm,
        capture_start_ns=capture_start_ns,
        capture_end_ns=capture_end_ns,
        output_dir=args.output_dir,
        projection_use_cubic=args.projection_use_cubic,
        projection_cubic_scale=args.projection_cubic_scale,
        projection_damping_lambda=args.projection_damping_lambda,
        flush_bins_every=args.flush_bins_every,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
