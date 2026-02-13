"""Validate the event-driven stream pipeline end-to-end on real data.

Connects stream_events() to real MNQH6 .dbn data and verifies:
    - Output has the right structure (grid_dict format)
    - G2: Dense grid (exactly 2K+1 buckets per emission)
    - G3: All numeric values are finite (no NaN/Inf)
    - G4: Untouched buckets persist prior values (when spot stable)
    - Arrow IPC serialization round-trips correctly

Usage:
    cd backend
    uv run python scripts/validate_stream_pipeline.py
"""
from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import pyarrow as pa

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vacuum_pressure.config import resolve_config
from src.vacuum_pressure.event_engine import PRICE_SCALE
from src.vacuum_pressure.server import GRID_SCHEMA, _grid_to_arrow_ipc
from src.vacuum_pressure.stream_pipeline import stream_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("validate_stream_pipeline")

LAKE_ROOT = Path(__file__).resolve().parents[1] / "lake"
PRODUCTS_YAML = (
    Path(__file__).resolve().parents[1]
    / "src" / "data_eng" / "config" / "products.yaml"
)

# Test parameters -- use available data
PRODUCT_TYPE = "future_mbo"
SYMBOL = "MNQH6"
DT = "2026-02-06"

# Maximum events to validate (0 = all)
MAX_EVENTS = 100_000

NUMERIC_FIELDS = [
    "add_mass", "pull_mass", "fill_mass", "rest_depth",
    "v_add", "v_pull", "v_fill", "v_rest_depth",
    "a_add", "a_pull", "a_fill", "a_rest_depth",
    "j_add", "j_pull", "j_fill", "j_rest_depth",
    "pressure_variant", "vacuum_variant", "resistance_variant",
]


def validate_grid_structure(
    grid: dict,
    K: int,
    prev_grid: dict | None,
) -> list[str]:
    """Validate a single grid emission against guarantees G2-G4.

    Returns list of violation strings (empty = pass).
    """
    violations: list[str] = []
    expected_count = 2 * K + 1

    # Check required top-level keys
    required_keys = {
        "ts_ns", "event_id", "spot_ref_price_int", "mid_price",
        "best_bid_price_int", "best_ask_price_int", "book_valid",
        "buckets", "touched_k",
    }
    missing_keys = required_keys - set(grid.keys())
    if missing_keys:
        violations.append(f"STRUCT: Missing top-level keys: {missing_keys}")
        return violations

    buckets = grid["buckets"]

    # G2: Cardinality
    if len(buckets) != expected_count:
        violations.append(
            f"G2: Expected {expected_count} buckets, got {len(buckets)}"
        )

    # G2: Completeness
    k_set = {b["k"] for b in buckets}
    expected_k = set(range(-K, K + 1))
    missing_k = expected_k - k_set
    if missing_k:
        violations.append(f"G2: Missing k values: {sorted(missing_k)}")

    # G3: Numeric invariant
    for b in buckets:
        for field in NUMERIC_FIELDS:
            val = b.get(field, None)
            if val is None:
                violations.append(
                    f"G3: k={b['k']} field={field} is None"
                )
            elif not math.isfinite(val):
                violations.append(
                    f"G3: k={b['k']} field={field} is {val} (not finite)"
                )

    # G4: Persistence (when spot didn't move)
    if prev_grid is not None:
        if grid["spot_ref_price_int"] == prev_grid["spot_ref_price_int"]:
            touched = grid["touched_k"]
            prev_by_k = {b["k"]: b for b in prev_grid["buckets"]}
            for b in buckets:
                k = b["k"]
                if k not in touched and k in prev_by_k:
                    for field in NUMERIC_FIELDS:
                        prev_val = prev_by_k[k].get(field, 0.0)
                        curr_val = b.get(field, 0.0)
                        if prev_val != curr_val:
                            violations.append(
                                f"G4: k={k} {field} changed "
                                f"({prev_val} -> {curr_val}) "
                                f"but bucket was not touched"
                            )
                            break  # one violation per bucket is enough

    return violations


def validate_arrow_ipc_roundtrip(grid: dict) -> list[str]:
    """Validate Arrow IPC serialization round-trips correctly."""
    violations: list[str] = []

    try:
        ipc_bytes = _grid_to_arrow_ipc(grid)
    except Exception as exc:
        violations.append(f"ARROW: Serialization failed: {exc}")
        return violations

    # Deserialize
    try:
        reader = pa.ipc.open_stream(ipc_bytes)
        table = reader.read_all()
    except Exception as exc:
        violations.append(f"ARROW: Deserialization failed: {exc}")
        return violations

    # Verify schema
    if table.schema != GRID_SCHEMA:
        violations.append(
            f"ARROW: Schema mismatch: got {table.schema} != expected {GRID_SCHEMA}"
        )

    # Verify row count
    expected_rows = len(grid["buckets"])
    if table.num_rows != expected_rows:
        violations.append(
            f"ARROW: Row count mismatch: {table.num_rows} != {expected_rows}"
        )

    # Spot-check a few values
    if table.num_rows > 0:
        k_col = table.column("k").to_pylist()
        pv_col = table.column("pressure_variant").to_pylist()
        buckets = grid["buckets"]

        for i in range(min(5, len(buckets))):
            if k_col[i] != buckets[i]["k"]:
                violations.append(
                    f"ARROW: k mismatch at row {i}: "
                    f"{k_col[i]} != {buckets[i]['k']}"
                )
            if abs(pv_col[i] - buckets[i]["pressure_variant"]) > 1e-12:
                violations.append(
                    f"ARROW: pressure_variant mismatch at row {i}: "
                    f"{pv_col[i]} != {buckets[i]['pressure_variant']}"
                )

    return violations


def main() -> None:
    # Resolve config
    config = resolve_config(PRODUCT_TYPE, SYMBOL, PRODUCTS_YAML)
    K = config.grid_max_ticks

    logger.info(
        "Validating stream_events: product=%s symbol=%s dt=%s K=%d",
        PRODUCT_TYPE, SYMBOL, DT, K,
    )

    total = 0
    total_violations = 0
    g2_violations = 0
    g3_violations = 0
    g4_violations = 0
    arrow_violations = 0
    struct_violations = 0

    prev_grid: dict | None = None
    first_valid_mid: float = 0.0
    last_mid: float = 0.0
    last_event_id: int = 0

    t_start = time.monotonic()

    for event_id, grid in stream_events(
        lake_root=LAKE_ROOT,
        config=config,
        dt=DT,
        start_time=None,
        throttle_ms=0,
    ):
        total += 1

        if MAX_EVENTS > 0 and total > MAX_EVENTS:
            break

        # Verify event_id matches
        if grid["event_id"] != event_id:
            logger.warning(
                "Event ID mismatch: yielded=%d grid=%d",
                event_id, grid["event_id"],
            )

        # Validate grid structure + guarantees
        violations = validate_grid_structure(grid, K, prev_grid)
        for v in violations:
            if v.startswith("G2"):
                g2_violations += 1
            elif v.startswith("G3"):
                g3_violations += 1
            elif v.startswith("G4"):
                g4_violations += 1
            elif v.startswith("STRUCT"):
                struct_violations += 1
            total_violations += 1

        # Validate Arrow IPC round-trip (every 10K events)
        if total % 10_000 == 1:
            arrow_issues = validate_arrow_ipc_roundtrip(grid)
            for v in arrow_issues:
                arrow_violations += 1
                total_violations += 1

        if violations and total_violations <= 20:
            for v in violations:
                logger.warning("Event %d: %s", total, v)

        # Track mid price
        if grid["mid_price"] > 0 and first_valid_mid == 0.0:
            first_valid_mid = grid["mid_price"]
        last_mid = grid["mid_price"]
        last_event_id = grid["event_id"]

        prev_grid = grid

        if total % 25_000 == 0:
            elapsed = time.monotonic() - t_start
            rate = total / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d events, %.0f evt/s, mid=$%.2f, "
                "violations=%d",
                total, rate, grid["mid_price"], total_violations,
            )

    elapsed = time.monotonic() - t_start
    rate = total / elapsed if elapsed > 0 else 0

    logger.info("=" * 70)
    logger.info("STREAM PIPELINE VALIDATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Events processed: %d", total)
    logger.info("Last event_id: %d", last_event_id)
    logger.info("Elapsed: %.2f s (%.0f events/s)", elapsed, rate)
    logger.info("First valid mid: $%.2f", first_valid_mid)
    logger.info("Last mid: $%.2f", last_mid)
    logger.info("")
    logger.info("Structure violations: %d", struct_violations)
    logger.info("G2 violations (grid density): %d", g2_violations)
    logger.info("G3 violations (NaN/Inf): %d", g3_violations)
    logger.info("G4 violations (persistence): %d", g4_violations)
    logger.info("Arrow IPC violations: %d", arrow_violations)
    logger.info("Total violations: %d", total_violations)

    if total_violations > 0:
        logger.error("FAIL: %d violations detected", total_violations)
        sys.exit(1)
    else:
        logger.info("PASS: All guarantees verified for stream pipeline")


if __name__ == "__main__":
    main()
