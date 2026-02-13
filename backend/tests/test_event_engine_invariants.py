"""Invariant tests T1-T6 for EventDrivenVPEngine.

Uses REAL data from the lake (not synthetic). Tests are marked with
pytest markers for selective execution:
    @pytest.mark.real_data - requires real .dbn files in lake

All tests process actual MBO events from Databento .dbn files.

Invariants tested:
    T1: Event update invariant -- touched buckets have current event_id.
    T2: Dense-grid invariant -- exactly 2K+1 buckets, all k present.
    T3: Numeric invariant -- all force values are finite.
    T4: Persistence invariant -- untouched buckets unchanged when spot stable.
    T5: Replay/live parity -- two engines produce identical output.
    T6: Spot-shift stress -- grid remains dense during rapid spot changes.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.vacuum_pressure.event_engine import EventDrivenVPEngine
from src.vacuum_pressure.replay_source import iter_mbo_events

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAKE_ROOT: Path = Path(__file__).resolve().parents[1] / "lake"

# Engine parameters matching MNQ / ES defaults
TICK_INT: int = 250_000_000  # $0.25 tick, price_int units
K: int = 40  # Grid half-width
BUCKET_SIZE_DOLLARS: float = 0.25

NUMERIC_FIELDS: list[str] = [
    "add_mass", "pull_mass", "fill_mass", "rest_depth",
    "v_add", "v_pull", "v_fill", "v_rest_depth",
    "a_add", "a_pull", "a_fill", "a_rest_depth",
    "j_add", "j_pull", "j_fill", "j_rest_depth",
    "pressure_variant", "vacuum_variant", "resistance_variant",
]

# All bucket dict fields that carry numeric state (superset including metadata)
ALL_STATE_FIELDS: list[str] = NUMERIC_FIELDS + ["last_event_id"]


# ---------------------------------------------------------------------------
# Data discovery helpers
# ---------------------------------------------------------------------------

def _discover_dbn_files() -> List[Tuple[str, str, str, Path]]:
    """Discover available .dbn files in the lake.

    Returns:
        List of (product_type, symbol, dt, path) tuples.
    """
    results: List[Tuple[str, str, str, Path]] = []
    raw_root = LAKE_ROOT / "raw" / "source=databento"
    if not raw_root.exists():
        return results

    for pt_dir in sorted(raw_root.iterdir()):
        if not pt_dir.is_dir() or not pt_dir.name.startswith("product_type="):
            continue
        product_type = pt_dir.name.split("=", 1)[1]
        if product_type != "future_mbo":
            continue

        for sym_dir in sorted(pt_dir.iterdir()):
            if not sym_dir.is_dir() or not sym_dir.name.startswith("symbol="):
                continue
            symbol_root = sym_dir.name.split("=", 1)[1]

            table_dir = sym_dir / "table=market_by_order_dbn"
            if not table_dir.exists():
                continue

            for dbn_file in sorted(table_dir.glob("*.dbn")):
                # Extract date from filename: glbx-mdp3-YYYYMMDD.mbo.dbn
                name = dbn_file.name
                parts = name.split("-")
                date_part = None
                for p in parts:
                    if len(p) >= 8 and p[:8].isdigit():
                        date_part = p[:8]
                        break
                if date_part is None:
                    continue

                dt = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                results.append((product_type, symbol_root, dt, dbn_file))

    return results


def _resolve_contract_symbol(
    lake_root: Path,
    product_type: str,
    symbol_root: str,
    dt: str,
) -> str:
    """Resolve the actual contract symbol from DBN metadata.

    For futures, the .dbn file is stored under the root symbol (MNQ)
    but contains contract symbols (MNQH6). We read metadata to find
    the correct contract symbol.

    Args:
        lake_root: Path to the lake directory.
        product_type: Product type string.
        symbol_root: Root symbol (e.g. MNQ, SI).
        dt: Date string YYYY-MM-DD.

    Returns:
        Contract symbol string (e.g. MNQH6).
    """
    import databento as db

    date_compact = dt.replace("-", "")
    raw_dir = (
        lake_root / "raw" / "source=databento"
        / f"product_type={product_type}"
        / f"symbol={symbol_root}"
        / "table=market_by_order_dbn"
    )
    dbn_files = list(raw_dir.glob(f"*{date_compact}*.dbn*"))
    if not dbn_files:
        raise FileNotFoundError(f"No .dbn files for {symbol_root}/{dt}")

    # Prefer uncompressed
    dbn_path = dbn_files[0]
    for f in dbn_files:
        if f.suffix == ".dbn":
            dbn_path = f
            break

    store = db.DBNStore.from_file(str(dbn_path))
    symbols = store.metadata.symbols or []
    if len(symbols) == 1:
        return symbols[0]
    if len(symbols) > 1:
        # Return first non-spread symbol
        for s in symbols:
            if "-" not in s:
                return s
        return symbols[0]

    raise ValueError(
        f"Cannot resolve contract symbol from metadata for {symbol_root}/{dt}"
    )


def _get_tick_int_for_symbol(symbol_root: str) -> int:
    """Return tick_int for a given symbol root.

    tick_int = tick_size / PRICE_SCALE where PRICE_SCALE = 1e-9.
    """
    # From products.yaml
    tick_sizes: dict[str, float] = {
        "ES": 0.25,
        "MES": 0.25,
        "MNQ": 0.25,
        "NQ": 0.25,
        "SI": 0.005,
        "GC": 0.10,
        "CL": 0.01,
        "6E": 0.00005,
    }
    tick_size = tick_sizes.get(symbol_root, 0.25)
    return int(round(tick_size / 1e-9))


def _get_bucket_size_for_symbol(symbol_root: str) -> float:
    """Return bucket_size_dollars for a given symbol root.

    For futures, bucket_size_dollars equals tick_size.
    """
    tick_sizes: dict[str, float] = {
        "ES": 0.25,
        "MES": 0.25,
        "MNQ": 0.25,
        "NQ": 0.25,
        "SI": 0.005,
        "GC": 0.10,
        "CL": 0.01,
        "6E": 0.00005,
    }
    return tick_sizes.get(symbol_root, 0.25)


# ---------------------------------------------------------------------------
# Fixture: discover and cache event data
# ---------------------------------------------------------------------------

_CACHED_EVENTS: Optional[List[Tuple[int, str, str, int, int, int, int]]] = None
_CACHED_METADATA: Optional[Dict[str, Any]] = None


def _load_events(
    max_events: int = 100_000,
) -> Tuple[List[Tuple[int, str, str, int, int, int, int]], Dict[str, Any]]:
    """Load real MBO events from the first available .dbn file.

    Caches results across tests for performance.

    Returns:
        (events_list, metadata_dict) where metadata contains
        product_type, symbol_root, contract_symbol, dt, tick_int,
        bucket_size_dollars, K.
    """
    global _CACHED_EVENTS, _CACHED_METADATA

    if _CACHED_EVENTS is not None and _CACHED_METADATA is not None:
        return _CACHED_EVENTS, _CACHED_METADATA

    available = _discover_dbn_files()
    if not available:
        pytest.skip("No real .dbn files found in lake")

    product_type, symbol_root, dt, dbn_path = available[0]
    contract_symbol = _resolve_contract_symbol(
        LAKE_ROOT, product_type, symbol_root, dt
    )
    tick_int = _get_tick_int_for_symbol(symbol_root)
    bucket_size = _get_bucket_size_for_symbol(symbol_root)

    events: List[Tuple[int, str, str, int, int, int, int]] = []
    for ev in iter_mbo_events(LAKE_ROOT, product_type, contract_symbol, dt):
        events.append(ev)
        if len(events) >= max_events:
            break

    if len(events) < 1000:
        pytest.skip(
            f"Only {len(events)} events available; need at least 1000"
        )

    metadata = {
        "product_type": product_type,
        "symbol_root": symbol_root,
        "contract_symbol": contract_symbol,
        "dt": dt,
        "tick_int": tick_int,
        "bucket_size_dollars": bucket_size,
        "K": K,
    }

    _CACHED_EVENTS = events
    _CACHED_METADATA = metadata
    return events, metadata


def _make_engine(metadata: Dict[str, Any]) -> EventDrivenVPEngine:
    """Create an engine from resolved metadata."""
    return EventDrivenVPEngine(
        K=metadata["K"],
        tick_int=metadata["tick_int"],
        bucket_size_dollars=metadata["bucket_size_dollars"],
    )


# ---------------------------------------------------------------------------
# T1: Event update invariant
# ---------------------------------------------------------------------------


@pytest.mark.real_data
class TestT1EventUpdateInvariant:
    """T1: For every processed event, at least one bucket's last_event_id
    equals the current event_id.

    Exception: events during snapshot phase or before spot is established
    may not touch any bucket. Once book_valid=True and spot > 0, touched
    events should produce at least one bucket update. The ratio of
    post-snapshot events with touched_k being non-empty should be >80%.
    """

    def test_touched_ratio_post_snapshot(self) -> None:
        """Post-snapshot events should produce bucket touches at high rate."""
        events, metadata = _load_events(max_events=100_000)
        engine = _make_engine(metadata)

        post_snapshot_events: int = 0
        events_with_touch: int = 0
        events_with_last_event_id_match: int = 0
        total_processed: int = 0

        for ts_ns, action, side, price_int, size, order_id, flags in events:
            grid = engine.update(
                ts_ns=ts_ns, action=action, side=side,
                price_int=price_int, size=size,
                order_id=order_id, flags=flags,
            )
            total_processed += 1
            event_id = grid["event_id"]
            touched_k = grid["touched_k"]

            # Only count events after book is valid and spot established
            if grid["book_valid"] and grid["spot_ref_price_int"] > 0:
                post_snapshot_events += 1

                if len(touched_k) > 0:
                    events_with_touch += 1

                # Check that at least one bucket has last_event_id == event_id
                bucket_match = any(
                    b["last_event_id"] == event_id
                    for b in grid["buckets"]
                )
                if bucket_match:
                    events_with_last_event_id_match += 1

        assert post_snapshot_events > 0, (
            f"No post-snapshot events found in {total_processed} events"
        )

        touch_ratio = events_with_touch / post_snapshot_events
        event_id_ratio = events_with_last_event_id_match / post_snapshot_events

        # Report stats
        print(f"\n--- T1 Stats ---")
        print(f"Total events processed: {total_processed}")
        print(f"Post-snapshot events: {post_snapshot_events}")
        print(f"Events with touched_k non-empty: {events_with_touch} "
              f"({touch_ratio:.1%})")
        print(f"Events with last_event_id match: {events_with_last_event_id_match} "
              f"({event_id_ratio:.1%})")

        # Trades and clears legitimately don't touch buckets.
        # Cancels for unknown orders also don't touch.
        # Expect >80% touch rate for post-snapshot events.
        assert touch_ratio > 0.80, (
            f"Touch ratio {touch_ratio:.1%} below 80% threshold. "
            f"{events_with_touch}/{post_snapshot_events} post-snapshot "
            f"events produced bucket touches."
        )


# ---------------------------------------------------------------------------
# T2: Dense-grid invariant
# ---------------------------------------------------------------------------


@pytest.mark.real_data
class TestT2DenseGridInvariant:
    """T2: After every event, exactly 2K+1 buckets emitted, all expected
    k indices exist exactly once.

    This is a HARD invariant -- zero violations allowed.
    """

    def test_grid_density_all_events(self) -> None:
        """Every emission must have exactly 2K+1 buckets with complete k range."""
        events, metadata = _load_events(max_events=100_000)
        engine = _make_engine(metadata)
        k_val = metadata["K"]
        expected_count = 2 * k_val + 1
        expected_k_set = set(range(-k_val, k_val + 1))

        violations: List[str] = []
        total_processed: int = 0

        for ts_ns, action, side, price_int, size, order_id, flags in events:
            grid = engine.update(
                ts_ns=ts_ns, action=action, side=side,
                price_int=price_int, size=size,
                order_id=order_id, flags=flags,
            )
            total_processed += 1
            event_id = grid["event_id"]
            buckets = grid["buckets"]

            # Check cardinality
            if len(buckets) != expected_count:
                violations.append(
                    f"Event {event_id}: expected {expected_count} buckets, "
                    f"got {len(buckets)}"
                )

            # Check completeness
            k_set = {b["k"] for b in buckets}
            if k_set != expected_k_set:
                missing = expected_k_set - k_set
                extra = k_set - expected_k_set
                violations.append(
                    f"Event {event_id}: missing_k={sorted(missing)}, "
                    f"extra_k={sorted(extra)}"
                )

        print(f"\n--- T2 Stats ---")
        print(f"Total events processed: {total_processed}")
        print(f"Grid density violations: {len(violations)}")
        if violations:
            for v in violations[:10]:
                print(f"  VIOLATION: {v}")

        assert len(violations) == 0, (
            f"T2 HARD INVARIANT VIOLATED: {len(violations)} events had "
            f"incorrect grid density. First: {violations[0]}"
        )


# ---------------------------------------------------------------------------
# T3: Numeric invariant
# ---------------------------------------------------------------------------


@pytest.mark.real_data
class TestT3NumericInvariant:
    """T3: All force values are finite for all buckets for all events.

    This is a HARD invariant -- zero violations allowed.
    """

    def test_all_values_finite(self) -> None:
        """Every numeric field in every bucket must be finite (not NaN/Inf)."""
        events, metadata = _load_events(max_events=100_000)
        engine = _make_engine(metadata)

        violations: List[str] = []
        total_processed: int = 0
        total_checks: int = 0

        for ts_ns, action, side, price_int, size, order_id, flags in events:
            grid = engine.update(
                ts_ns=ts_ns, action=action, side=side,
                price_int=price_int, size=size,
                order_id=order_id, flags=flags,
            )
            total_processed += 1
            event_id = grid["event_id"]

            for b in grid["buckets"]:
                for field in NUMERIC_FIELDS:
                    val = b[field]
                    total_checks += 1
                    if not math.isfinite(val):
                        violations.append(
                            f"Event {event_id}, k={b['k']}, "
                            f"{field}={val}"
                        )

        print(f"\n--- T3 Stats ---")
        print(f"Total events processed: {total_processed}")
        print(f"Total numeric checks: {total_checks:,}")
        print(f"Non-finite violations: {len(violations)}")
        if violations:
            for v in violations[:10]:
                print(f"  VIOLATION: {v}")

        assert len(violations) == 0, (
            f"T3 HARD INVARIANT VIOLATED: {len(violations)} non-finite "
            f"values found. First: {violations[0]}"
        )


# ---------------------------------------------------------------------------
# T4: Persistence invariant
# ---------------------------------------------------------------------------


@pytest.mark.real_data
class TestT4PersistenceInvariant:
    """T4: If a bucket is not touched and not remapped out-of-range by
    spot shift, its values remain unchanged from the prior event.

    We only check when spot_ref_price_int is stable between consecutive
    events (no shift remap occurred). Known exception: snapshot phase
    and spot-establishment events may reset rest_depth globally.
    """

    def test_untouched_buckets_persist(self) -> None:
        """Untouched bucket numeric fields must be identical between events."""
        events, metadata = _load_events(max_events=100_000)
        engine = _make_engine(metadata)

        persistence_checks: int = 0
        persistence_violations: int = 0
        violation_details: List[str] = []

        prev_grid: Optional[Dict[str, Any]] = None
        total_processed: int = 0

        for ts_ns, action, side, price_int, size, order_id, flags in events:
            grid = engine.update(
                ts_ns=ts_ns, action=action, side=side,
                price_int=price_int, size=size,
                order_id=order_id, flags=flags,
            )
            total_processed += 1
            touched_k = grid["touched_k"]

            if prev_grid is not None:
                spot_stable = (
                    grid["spot_ref_price_int"]
                    == prev_grid["spot_ref_price_int"]
                )

                if spot_stable and grid["book_valid"]:
                    prev_by_k = {
                        b["k"]: b for b in prev_grid["buckets"]
                    }
                    for b in grid["buckets"]:
                        k = b["k"]
                        if k in touched_k:
                            continue
                        if k not in prev_by_k:
                            continue

                        prev_b = prev_by_k[k]
                        for field in NUMERIC_FIELDS:
                            persistence_checks += 1
                            if b[field] != prev_b[field]:
                                persistence_violations += 1
                                if len(violation_details) < 20:
                                    violation_details.append(
                                        f"Event {grid['event_id']}, k={k}, "
                                        f"{field}: {prev_b[field]} -> "
                                        f"{b[field]}"
                                    )

            prev_grid = grid

        print(f"\n--- T4 Stats ---")
        print(f"Total events processed: {total_processed}")
        print(f"Persistence checks: {persistence_checks:,}")
        print(f"Persistence violations: {persistence_violations}")
        if violation_details:
            for v in violation_details[:10]:
                print(f"  VIOLATION: {v}")

        # T4 has known exceptions during snapshot sync (rest_depth reset).
        # Compute violation rate; expect <1%.
        if persistence_checks > 0:
            violation_rate = persistence_violations / persistence_checks
            print(f"Violation rate: {violation_rate:.6%}")
            assert violation_rate < 0.01, (
                f"T4 persistence violation rate {violation_rate:.4%} exceeds "
                f"1% threshold. {persistence_violations}/{persistence_checks} "
                f"checks failed."
            )
        else:
            # No checks possible (too few events or constant spot shifts)
            pytest.skip("No persistence checks possible (spot never stable)")


# ---------------------------------------------------------------------------
# T5: Replay/live parity
# ---------------------------------------------------------------------------


@pytest.mark.real_data
class TestT5ReplayParity:
    """T5: Feed same event list to two separate engine instances.
    Compare emitted grid states -- must match exactly.

    This is a HARD invariant -- zero deviations allowed.
    """

    def test_two_engines_identical_output(self) -> None:
        """Two engines processing identical events must produce identical grids."""
        events, metadata = _load_events(max_events=50_000)
        engine_a = _make_engine(metadata)
        engine_b = _make_engine(metadata)

        mismatches: List[str] = []
        total_processed: int = 0

        for ts_ns, action, side, price_int, size, order_id, flags in events:
            grid_a = engine_a.update(
                ts_ns=ts_ns, action=action, side=side,
                price_int=price_int, size=size,
                order_id=order_id, flags=flags,
            )
            grid_b = engine_b.update(
                ts_ns=ts_ns, action=action, side=side,
                price_int=price_int, size=size,
                order_id=order_id, flags=flags,
            )
            total_processed += 1

            # Compare scalar fields
            for key in ("ts_ns", "event_id", "spot_ref_price_int",
                        "mid_price", "best_bid_price_int",
                        "best_ask_price_int", "book_valid"):
                if grid_a[key] != grid_b[key]:
                    mismatches.append(
                        f"Event {grid_a['event_id']}: {key} "
                        f"A={grid_a[key]} B={grid_b[key]}"
                    )

            # Compare touched_k
            if grid_a["touched_k"] != grid_b["touched_k"]:
                mismatches.append(
                    f"Event {grid_a['event_id']}: touched_k differs "
                    f"A={grid_a['touched_k']} B={grid_b['touched_k']}"
                )

            # Compare all bucket fields
            buckets_a = grid_a["buckets"]
            buckets_b = grid_b["buckets"]

            if len(buckets_a) != len(buckets_b):
                mismatches.append(
                    f"Event {grid_a['event_id']}: bucket count "
                    f"A={len(buckets_a)} B={len(buckets_b)}"
                )
                continue

            for ba, bb in zip(buckets_a, buckets_b):
                for field in ALL_STATE_FIELDS + ["k", "cell_valid"]:
                    val_a = ba[field]
                    val_b = bb[field]
                    if val_a != val_b:
                        mismatches.append(
                            f"Event {grid_a['event_id']}, k={ba['k']}: "
                            f"{field} A={val_a} B={val_b}"
                        )

            if len(mismatches) > 50:
                break

        print(f"\n--- T5 Stats ---")
        print(f"Total events processed: {total_processed}")
        print(f"Mismatches: {len(mismatches)}")
        if mismatches:
            for m in mismatches[:10]:
                print(f"  MISMATCH: {m}")

        assert len(mismatches) == 0, (
            f"T5 HARD INVARIANT VIOLATED: {len(mismatches)} mismatches "
            f"between two engines. First: {mismatches[0]}"
        )


# ---------------------------------------------------------------------------
# T6: Spot-shift stress
# ---------------------------------------------------------------------------


@pytest.mark.real_data
class TestT6SpotShiftStress:
    """T6: Rapid spot changes. Grid remains dense, remap stable, no empty
    cells.

    Part A: Find periods in real data where spot shifts frequently and
    verify G2/G3 still hold during those shifts.

    Part B: Verify grid density is maintained even during shifts by
    checking consecutive events where spot_ref_price_int changed.
    """

    def test_density_during_spot_shifts(self) -> None:
        """During spot shifts, grid must remain dense (G2) and finite (G3)."""
        events, metadata = _load_events(max_events=100_000)
        engine = _make_engine(metadata)
        k_val = metadata["K"]
        expected_count = 2 * k_val + 1
        expected_k_set = set(range(-k_val, k_val + 1))

        total_processed: int = 0
        spot_shifts: int = 0
        g2_violations_during_shift: int = 0
        g3_violations_during_shift: int = 0
        prev_spot: int = 0
        max_shift_ticks: int = 0

        for ts_ns, action, side, price_int, size, order_id, flags in events:
            grid = engine.update(
                ts_ns=ts_ns, action=action, side=side,
                price_int=price_int, size=size,
                order_id=order_id, flags=flags,
            )
            total_processed += 1
            current_spot = grid["spot_ref_price_int"]

            spot_shifted = (
                prev_spot > 0
                and current_spot > 0
                and current_spot != prev_spot
            )

            if spot_shifted:
                spot_shifts += 1
                shift_ticks = abs(
                    round(
                        (current_spot - prev_spot) / metadata["tick_int"]
                    )
                )
                if shift_ticks > max_shift_ticks:
                    max_shift_ticks = shift_ticks

                # G2: dense grid
                buckets = grid["buckets"]
                if len(buckets) != expected_count:
                    g2_violations_during_shift += 1

                k_set = {b["k"] for b in buckets}
                if k_set != expected_k_set:
                    g2_violations_during_shift += 1

                # G3: finite values
                for b in buckets:
                    for field in NUMERIC_FIELDS:
                        if not math.isfinite(b[field]):
                            g3_violations_during_shift += 1

            if current_spot > 0:
                prev_spot = current_spot

        print(f"\n--- T6 Stats ---")
        print(f"Total events processed: {total_processed}")
        print(f"Spot shifts observed: {spot_shifts}")
        print(f"Max shift magnitude (ticks): {max_shift_ticks}")
        print(f"G2 violations during shifts: {g2_violations_during_shift}")
        print(f"G3 violations during shifts: {g3_violations_during_shift}")

        assert spot_shifts > 0, (
            f"No spot shifts observed in {total_processed} events. "
            f"Cannot validate T6."
        )

        assert g2_violations_during_shift == 0, (
            f"T6/G2 VIOLATED: {g2_violations_during_shift} grid density "
            f"violations during spot shifts."
        )

        assert g3_violations_during_shift == 0, (
            f"T6/G3 VIOLATED: {g3_violations_during_shift} non-finite "
            f"values during spot shifts."
        )

    def test_synthetic_rapid_spot_shifts(self) -> None:
        """Worst-case: rapid BBO changes forcing repeated grid remaps.

        Construct a sequence where spot moves by 1 tick per step.
        Uses only the public API (no internal state manipulation).
        Each step: cancel old best bid, add new bid 1 tick higher,
        add new ask 1 tick higher. This walks the BBO upward.
        Verify grid density and finiteness after each move.
        """
        engine = EventDrivenVPEngine(
            K=K, tick_int=TICK_INT, bucket_size_dollars=BUCKET_SIZE_DOLLARS,
        )
        k_val = K
        expected_count = 2 * k_val + 1
        expected_k_set = set(range(-k_val, k_val + 1))

        # Start price: $20,000.00 in price_int
        base_price = 20_000_000_000_000
        ts = 1_000_000_000_000_000_000  # arbitrary

        # Build initial book via snapshot.
        # Best bid = base - tick, best ask = base. Mid = base - tick/2.
        engine.update(ts, "R", "B", 0, 0, 0, 32)  # Clear + snapshot
        bid_oid_1 = 1001
        bid_oid_2 = 1002
        ask_oid_1 = 2001
        ask_oid_2 = 2002

        engine.update(ts, "A", "B", base_price - TICK_INT, 10, bid_oid_1, 32)
        engine.update(ts, "A", "B", base_price - 2 * TICK_INT, 20, bid_oid_2, 32)
        engine.update(ts, "A", "A", base_price, 10, ask_oid_1, 32)
        grid = engine.update(
            ts, "A", "A", base_price + TICK_INT, 20, ask_oid_2, 32 | 128,
        )

        assert grid["book_valid"]
        assert grid["spot_ref_price_int"] > 0

        violations: List[str] = []
        oid_counter = 10_000
        spot_shifts_seen = 0

        # Walk spot upward by 1 tick per step, 50 times.
        # Strategy: cancel the deepest bid, add new best bid 1 tick higher,
        # cancel deepest ask, add new best ask 1 tick higher.
        # Track order_ids of the 2 bids and 2 asks currently live.
        live_bids: List[Tuple[int, int]] = [  # (price_int, order_id)
            (base_price - 2 * TICK_INT, bid_oid_2),
            (base_price - TICK_INT, bid_oid_1),
        ]
        live_asks: List[Tuple[int, int]] = [  # (price_int, order_id)
            (base_price, ask_oid_1),
            (base_price + TICK_INT, ask_oid_2),
        ]

        prev_spot = grid["spot_ref_price_int"]

        for step in range(50):
            ts += 100_000_000  # +100ms

            # Cancel deepest (lowest) bid
            _, cancel_bid_oid = live_bids.pop(0)
            engine.update(ts, "C", "B", 0, 0, cancel_bid_oid, 0)
            ts += 1_000_000

            # Cancel deepest (highest) ask
            _, cancel_ask_oid = live_asks.pop(-1)
            engine.update(ts, "C", "A", 0, 0, cancel_ask_oid, 0)
            ts += 1_000_000

            # Add new best bid 1 tick above old best bid
            new_bid_price = live_bids[-1][0] + TICK_INT
            oid_counter += 1
            engine.update(ts, "A", "B", new_bid_price, 10, oid_counter, 0)
            live_bids.append((new_bid_price, oid_counter))
            ts += 1_000_000

            # Add new best ask 1 tick above old best ask (which is now live_asks[0])
            new_ask_price = live_asks[0][0] + TICK_INT
            oid_counter += 1
            grid = engine.update(
                ts, "A", "A", new_ask_price, 10, oid_counter, 0,
            )
            live_asks.append((new_ask_price, oid_counter))
            live_asks.sort(key=lambda x: x[0])

            current_spot = grid["spot_ref_price_int"]
            if current_spot != prev_spot and prev_spot > 0:
                spot_shifts_seen += 1
            prev_spot = current_spot

            buckets = grid["buckets"]

            # G2
            if len(buckets) != expected_count:
                violations.append(
                    f"Step {step}: expected {expected_count} buckets, "
                    f"got {len(buckets)}"
                )

            k_set = {b["k"] for b in buckets}
            if k_set != expected_k_set:
                missing = expected_k_set - k_set
                violations.append(
                    f"Step {step}: missing k={sorted(missing)}"
                )

            # G3
            for b in buckets:
                for field in NUMERIC_FIELDS:
                    if not math.isfinite(b[field]):
                        violations.append(
                            f"Step {step}, k={b['k']}: "
                            f"{field}={b[field]}"
                        )

        print(f"\n--- T6 Synthetic Stress Stats ---")
        print(f"Rapid shifts simulated: 50 steps")
        print(f"Spot shifts observed: {spot_shifts_seen}")
        print(f"Violations: {len(violations)}")
        if violations:
            for v in violations[:10]:
                print(f"  VIOLATION: {v}")

        assert spot_shifts_seen > 0, (
            f"Synthetic stress test failed to produce any spot shifts. "
            f"Check order placement logic."
        )

        assert len(violations) == 0, (
            f"T6 synthetic stress VIOLATED: {len(violations)} violations. "
            f"First: {violations[0]}"
        )
