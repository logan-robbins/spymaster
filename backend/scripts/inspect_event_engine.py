"""Inspect EventDrivenVPEngine output on real MNQH6 data.

Quick diagnostic: process events up to a specific point and dump
grid state to verify pressure_variant values are non-trivial.

Usage:
    cd backend
    uv run python scripts/inspect_event_engine.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vacuum_pressure.event_engine import EventDrivenVPEngine, PRICE_SCALE
from src.vacuum_pressure.replay_source import iter_mbo_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("inspect")

LAKE_ROOT = Path(__file__).resolve().parents[1] / "lake"
TICK_INT = 250_000_000
K = 40

# Process 100K events then inspect
TARGET_EVENTS = 100_000


def main() -> None:
    engine = EventDrivenVPEngine(K=K, tick_int=TICK_INT, bucket_size_dollars=0.25)

    events = iter_mbo_events(LAKE_ROOT, "future_mbo", "MNQH6", "2026-02-06")
    grid = None
    count = 0

    for ts_ns, action, side, price_int, size, order_id, flags in events:
        count += 1
        grid = engine.update(ts_ns, action, side, price_int, size, order_id, flags)
        if count >= TARGET_EVENTS:
            break

    if grid is None:
        logger.error("No events processed")
        return

    # Get array snapshot
    arrays = engine.grid_snapshot_arrays()

    logger.info("After %d events:", count)
    logger.info("  spot_ref = $%.2f", engine.spot_ref_price_int * PRICE_SCALE)
    logger.info("  orders in book = %d", engine.order_count)
    logger.info("  best_bid = $%.2f", grid["best_bid_price_int"] * PRICE_SCALE)
    logger.info("  best_ask = $%.2f", grid["best_ask_price_int"] * PRICE_SCALE)
    logger.info("")

    # Pressure variant statistics
    pv = arrays["pressure_variant"]
    logger.info("pressure_variant stats:")
    logger.info("  min    = %.6f", np.min(pv))
    logger.info("  max    = %.6f", np.max(pv))
    logger.info("  mean   = %.6f", np.mean(pv))
    logger.info("  std    = %.6f", np.std(pv))
    logger.info("  nonzero = %d / %d", np.count_nonzero(pv), len(pv))
    logger.info("")

    # Vacuum variant statistics
    vv = arrays["vacuum_variant"]
    logger.info("vacuum_variant stats:")
    logger.info("  min    = %.6f", np.min(vv))
    logger.info("  max    = %.6f", np.max(vv))
    logger.info("  nonzero = %d / %d", np.count_nonzero(vv), len(vv))
    logger.info("")

    # Resistance variant statistics
    rv = arrays["resistance_variant"]
    logger.info("resistance_variant stats:")
    logger.info("  min    = %.6f", np.min(rv))
    logger.info("  max    = %.6f", np.max(rv))
    logger.info("  nonzero = %d / %d", np.count_nonzero(rv), len(rv))
    logger.info("")

    # Rest depth statistics
    rd = arrays["rest_depth"]
    logger.info("rest_depth stats:")
    logger.info("  min    = %.1f", np.min(rd))
    logger.info("  max    = %.1f", np.max(rd))
    logger.info("  total  = %.1f", np.sum(rd))
    logger.info("  nonzero = %d / %d", np.count_nonzero(rd), len(rd))
    logger.info("")

    # Velocity add stats
    va = arrays["v_add"]
    logger.info("v_add stats:")
    logger.info("  min    = %.6f", np.min(va))
    logger.info("  max    = %.6f", np.max(va))
    logger.info("  nonzero = %d / %d", np.count_nonzero(va), len(va))
    logger.info("")

    # Print a few sample buckets near spot
    logger.info("Sample buckets near spot (k=-3..+3):")
    for k in range(-3, 4):
        b = grid["buckets"][k + K]  # k=0 is at index K in sorted list
        logger.info(
            "  k=%+d: rest_depth=%.0f add_mass=%.2f pull_mass=%.2f "
            "v_add=%.4f v_pull=%.4f pv=%.4f vac=%.4f res=%.4f "
            "last_eid=%d",
            k, b["rest_depth"], b["add_mass"], b["pull_mass"],
            b["v_add"], b["v_pull"],
            b["pressure_variant"], b["vacuum_variant"],
            b["resistance_variant"], b["last_event_id"],
        )

    # Check all finite
    all_finite = True
    for name, arr in arrays.items():
        if name == "k" or name == "last_event_id":
            continue
        if not np.all(np.isfinite(arr)):
            logger.error("NON-FINITE values in %s!", name)
            all_finite = False

    if all_finite:
        logger.info("\nAll values are finite. PASS.")
    else:
        logger.error("\nNON-FINITE values detected. FAIL.")
        sys.exit(1)


if __name__ == "__main__":
    main()
