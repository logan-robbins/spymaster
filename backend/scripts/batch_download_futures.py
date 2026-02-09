"""
Batch-download GLBX futures + futures options with strict flat-file day slices.

Pipeline per session date and symbol (generic futures root, e.g. ES/SI/CL):
1) Resolve front-month contract via continuous definition (SI.v.0 -> SIH6)
2) Discover options via FREE symbology.resolve + tiny streaming definition:
   a) Compute candidate parents (0DTE daily/weekly, then nearest-expiry fallbacks)
   b) Resolve ALL candidates at once via symbology.resolve (FREE, one API call)
   c) Pick the first resolving candidate in priority order
   d) Stream a tiny targeted definition for that parent's instruments
   e) Filter by underlying == front-month, instrument_class in {C,P}, nearest expiry
3) Submit futures MBO batch job for ONLY the front-month contract
4) Submit options data jobs (mbo, statistics) for discovered contracts

Daemon 2-phase flow:
- Phase 1: Resolve front month + discover options (FREE) + submit data batch jobs
- Phase 2: Poll/download data batch jobs

All batch requests are forced to Databento flat-file delivery with one-day requests:
- delivery="download"
- split_duration="day"
- start=<session_date>, end=<session_date + 1 day>
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import databento as db
import pandas as pd
from databento.common.enums import PriceType
from dotenv import load_dotenv

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
load_dotenv(backend_dir / ".env")
load_dotenv(backend_dir.parent / ".env")

JOB_TRACKER_FILE = backend_dir / "logs" / "futures_jobs.json"

FUTURES_DATASET = "GLBX.MDP3"
SUPPORTED_OPTIONS_SCHEMAS = {"definition", "mbo", "statistics"}

FLAT_FILE_DELIVERY = "download"
FLAT_FILE_SPLIT_DURATION = "day"

PRODUCT_FUTURES = "futures"
PRODUCT_FUTURES_OPTIONS = "futures_options"
PRODUCT_FUTURES_OPTIONS_DEF = "futures_options_def"

# ---------------------------------------------------------------------------
# Per-product options parent configuration for CME Globex (GLBX.MDP3)
# ---------------------------------------------------------------------------
# Each futures root maps to its complete set of daily/weekly/monthly option
# asset codes.  Fields:
#   quarterly  - Asset code for quarterly options (3rd Friday of Mar/Jun/Sep/Dec).
#                None for products where monthly IS the standard option (GC/SI/CL/6E).
#   eom        - End-of-month (equity index) or monthly (metals/energy/FX) option code.
#   friday     - Friday weekly codes, 1-indexed by week-of-month occurrence.
#   daily      - Weekday 0=Mon..3=Thu -> list of codes, 1-indexed by week occurrence.
#   max_weeks  - Maximum weekly occurrences per month (4 or 5).
#
# Naming conventions vary by product family:
#   Equity Index (ES/NQ/MES/MNQ): A/B/C/D suffix for Mon-Thu
#   Metals (GC/SI):               M/T/W/R suffix for Mon-Thu
#   Energy (CL):                  ML/NL/WL/XL prefix for Mon-Thu
#   FX (6E):                      MO/TU/WE/SU prefix for Mon-Thu

OPTIONS_CONFIG: dict[str, dict] = {
    "ES": {
        "quarterly": "ES",
        "eom": "EW",
        "friday": ["EW1", "EW2", "EW3", "EW4"],
        "daily": {
            0: ["E1A", "E2A", "E3A", "E4A", "E5A"],
            1: ["E1B", "E2B", "E3B", "E4B", "E5B"],
            2: ["E1C", "E2C", "E3C", "E4C", "E5C"],
            3: ["E1D", "E2D", "E3D", "E4D", "E5D"],
        },
        "max_weeks": 5,
    },
    "NQ": {
        "quarterly": "NQ",
        "eom": "QNE",
        "friday": ["QN1", "QN2", "QN3", "QN4"],
        "daily": {
            0: ["Q1A", "Q2A", "Q3A", "Q4A", "Q5A"],
            1: ["Q1B", "Q2B", "Q3B", "Q4B", "Q5B"],
            2: ["Q1C", "Q2C", "Q3C", "Q4C", "Q5C"],
            3: ["Q1D", "Q2D", "Q3D", "Q4D", "Q5D"],
        },
        "max_weeks": 5,
    },
    "MES": {
        "quarterly": "MES",
        "eom": "EX",
        "friday": ["EX1", "EX2", "EX3", "EX4"],
        "daily": {
            0: ["X1A", "X2A", "X3A", "X4A", "X5A"],
            1: ["X1B", "X2B", "X3B", "X4B", "X5B"],
            2: ["X1C", "X2C", "X3C", "X4C", "X5C"],
            3: ["X1D", "X2D", "X3D", "X4D", "X5D"],
        },
        "max_weeks": 5,
    },
    "MNQ": {
        "quarterly": "MNQ",
        "eom": "MQE",
        "friday": ["MQ1", "MQ2", "MQ3", "MQ4"],
        "daily": {
            0: ["D1A", "D2A", "D3A", "D4A", "D5A"],
            1: ["D1B", "D2B", "D3B", "D4B", "D5B"],
            2: ["D1C", "D2C", "D3C", "D4C", "D5C"],
            3: ["D1D", "D2D", "D3D", "D4D", "D5D"],
        },
        "max_weeks": 5,
    },
    "GC": {
        "quarterly": None,
        "eom": "OG",
        "friday": ["OG1", "OG2", "OG3", "OG4", "OG5"],
        "daily": {
            0: ["G1M", "G2M", "G3M", "G4M", "G5M"],
            1: ["G1T", "G2T", "G3T", "G4T", "G5T"],
            2: ["G1W", "G2W", "G3W", "G4W", "G5W"],
            3: ["G1R", "G2R", "G3R", "G4R", "G5R"],
        },
        "max_weeks": 5,
    },
    "SI": {
        "quarterly": None,
        "eom": "SO",
        "friday": ["SO1", "SO2", "SO3", "SO4", "SO5"],
        "daily": {
            0: ["M1S", "M2S", "M3S", "M4S", "M5S"],
            1: ["S1T", "S2T", "S3T", "S4T", "S5T"],
            2: ["W1S", "W2S", "W3S", "W4S", "W5S"],
            3: ["R1S", "R2S", "R3S", "R4S", "R5S"],
        },
        "max_weeks": 5,
    },
    "CL": {
        "quarterly": None,
        "eom": "LO",
        "friday": ["LO1", "LO2", "LO3", "LO4", "LO5"],
        "daily": {
            0: ["ML1", "ML2", "ML3", "ML4", "ML5"],
            1: ["NL1", "NL2", "NL3", "NL4", "NL5"],
            2: ["WL1", "WL2", "WL3", "WL4", "WL5"],
            3: ["XL1", "XL2", "XL3", "XL4", "XL5"],
        },
        "max_weeks": 5,
    },
    "6E": {
        "quarterly": None,
        "eom": "EUU",
        "friday": ["1EU", "2EU", "3EU", "4EU", "5EU"],
        "daily": {
            0: ["MO1", "MO2", "MO3", "MO4", "MO5"],
            1: ["TU1", "TU2", "TU3", "TU4", "TU5"],
            2: ["WE1", "WE2", "WE3", "WE4", "WE5"],
            3: ["SU1", "SU2", "SU3", "SU4", "SU5"],
        },
        "max_weeks": 5,
    },
}


def _weekday_occurrence(dt: datetime) -> int:
    """Return the nth occurrence of this weekday in its month (1-indexed)."""
    return (dt.day - 1) // 7 + 1


def _is_third_friday(dt: datetime) -> bool:
    """True if dt is the 3rd Friday of its month."""
    return dt.weekday() == 4 and 15 <= dt.day <= 21


def _is_quarterly_month(dt: datetime) -> bool:
    """True if dt falls in a quarterly expiration month (Mar/Jun/Sep/Dec)."""
    return dt.month in (3, 6, 9, 12)


def _is_last_business_day(dt: datetime) -> bool:
    """True if the next weekday falls in a different month."""
    nxt = dt + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.month != dt.month


def options_parents_for(symbol: str, session_date: str) -> list[str]:
    """Return GLBX parent symbol(s) for options on a given futures root.

    Uses OPTIONS_CONFIG for data-driven parent resolution. For configured
    products, returns the targeted 1-3 parent symbols that could have 0DTE
    expirations on session_date. For unknown products, falls back to
    {symbol}.OPT (single quarterly parent).

    Rules:
    - Friday: friday weekly code for that week occurrence
    - Mon-Thu: daily code for that weekday and week occurrence
    - 3rd Friday of quarterly month: also include quarterly parent (if configured)
    - Last business day: also include EOM/monthly parent
    - 3rd Friday of quarterly month for products where quarterly IS the monthly
      (quarterly=None): include EOM parent instead
    - Saturday/Sunday: empty list (no options expire)
    """
    sym = symbol.upper()
    cfg = OPTIONS_CONFIG.get(sym)
    if cfg is None:
        return [f"{sym}.OPT"]

    dt = datetime.strptime(session_date, "%Y-%m-%d")
    weekday = dt.weekday()  # 0=Mon .. 4=Fri, 5=Sat, 6=Sun
    occ = _weekday_occurrence(dt)
    parents: list[str] = []

    if weekday == 4:  # Friday
        friday_codes: list[str] = cfg["friday"]
        if occ <= len(friday_codes):
            parents.append(f"{friday_codes[occ - 1]}.OPT")
        # Quarterly parent on 3rd Friday of quarterly months
        if _is_third_friday(dt) and _is_quarterly_month(dt):
            quarterly_code: str | None = cfg["quarterly"]
            if quarterly_code is not None:
                parents.append(f"{quarterly_code}.OPT")
    elif weekday in cfg["daily"]:  # Mon-Thu
        day_codes: list[str] = cfg["daily"][weekday]
        if occ <= len(day_codes):
            parents.append(f"{day_codes[occ - 1]}.OPT")
    # else: Saturday/Sunday -- no options expire

    # EOM/monthly parent on last business day of month.
    # Also on 3rd Friday of quarterly months for products where
    # quarterly=None (monthly IS the standard option).
    eom_code: str = cfg["eom"]
    is_eom = _is_last_business_day(dt)
    is_quarterly_friday = (
        _is_third_friday(dt) and _is_quarterly_month(dt) and cfg["quarterly"] is None
    )
    if is_eom or is_quarterly_friday:
        eom_parent = f"{eom_code}.OPT"
        if eom_parent not in parents:
            parents.append(eom_parent)

    return parents


# ---------------------------------------------------------------------------
# Smart Options Resolution (FREE symbology.resolve + targeted streaming)
# ---------------------------------------------------------------------------

def _symbology_resolve_raw_symbols(
    client: db.Historical,
    parents: list[str],
    session_date: str,
    log_path: Path,
) -> dict[str, list[str]]:
    """Resolve parent symbols to raw symbols via FREE symbology.resolve API.

    Uses stype_out="instrument_id" (the only combination GLBX.MDP3 supports
    for parent symbology). Raw symbols come back as the **keys** of the
    result dict. We filter out user-defined spread instruments (UD:* prefix)
    and group by parent.

    Returns dict mapping each parent that has instruments to its raw symbols.
    Parents with no instruments on the given date are omitted.
    """
    if not parents:
        return {}

    # Resolve one parent at a time to attribute raw symbols correctly.
    # Each call is FREE and fast (<1s), and we have at most ~15 candidates.
    resolved: dict[str, list[str]] = {}
    for parent in parents:
        try:
            result = client.symbology.resolve(
                dataset=FUTURES_DATASET,
                symbols=[parent],
                stype_in="parent",
                stype_out="instrument_id",
                start_date=session_date,
                end_date=_next_day(session_date),
            )
            # Response keys are raw_symbol names; values are instrument_id mappings.
            # Filter out user-defined spreads (UD:*) which are not individual options.
            raw_syms = [
                sym for sym in result.get("result", {}).keys()
                if not sym.startswith("UD:")
            ]
            if raw_syms:
                resolved[parent] = raw_syms
        except Exception as e:
            log_msg(log_path, f"SYMBOLOGY_RESOLVE {parent}: {e}")
            # A 422 for one parent is not fatal — try the next candidate
            continue

    return resolved


def _candidate_parents_ordered(
    symbol: str,
    session_date: str,
    max_lookahead_days: int = 7,
) -> list[tuple[str, list[str]]]:
    """Compute candidate option parent symbols in priority order.

    Returns list of (label, parents) tuples:
      - "0dte": primary parents for the exact session date
      - "Ndte_Day": parents for each subsequent trading day
      - "monthly": the monthly/EOM parent as last resort

    All candidates can be resolved in a single FREE symbology.resolve call.
    """
    candidates: list[tuple[str, list[str]]] = []
    seen_parents: set[str] = set()

    # 1. Primary: 0DTE parents for the exact session date
    primary = options_parents_for(symbol, session_date)
    if primary:
        candidates.append(("0dte", primary))
        seen_parents.update(primary)

    # 2. Next trading days' parents (ordered by proximity)
    dt = datetime.strptime(session_date, "%Y-%m-%d")
    for offset in range(1, max_lookahead_days + 1):
        next_dt = dt + timedelta(days=offset)
        if next_dt.weekday() in (5, 6):  # skip weekends
            continue
        next_date = next_dt.strftime("%Y-%m-%d")
        parents = options_parents_for(symbol, next_date)
        new_parents = [p for p in parents if p not in seen_parents]
        if new_parents:
            day_name = next_dt.strftime("%a")
            candidates.append((f"{offset}dte_{day_name}", new_parents))
            seen_parents.update(new_parents)

    # 3. Monthly/EOM parent as last resort
    cfg = OPTIONS_CONFIG.get(symbol.upper())
    if cfg:
        monthly_parent = f"{cfg['eom']}.OPT"
        if monthly_parent not in seen_parents:
            candidates.append(("monthly", [monthly_parent]))
    else:
        generic_parent = f"{symbol.upper()}.OPT"
        if generic_parent not in seen_parents:
            candidates.append(("generic", [generic_parent]))

    return candidates


def discover_options_raw_symbols(
    client: db.Historical,
    symbol: str,
    session_date: str,
    active_contract: str,
    log_path: Path,
) -> tuple[list[str], str, str | None]:
    """Discover option raw symbols for data download.

    Strategy:
    1. Compute candidate parents: primary (0DTE) + fallbacks (nearest expiry)
    2. Resolve each candidate via symbology.resolve (FREE, per-parent calls)
    3. Pick the first resolving candidate in priority order
    4. Stream a targeted definition using the parent (stype_in=parent)
    5. Filter by underlying == active_contract, instrument_class in {C, P}
    6. For 0DTE candidates: also filter by expiration == session_date
    7. For fallback candidates: pick the nearest expiration >= session_date

    Returns (raw_symbols, expiry_label, resolved_parent).
    The resolved_parent is used for batch data submission via stype_in=parent
    when the symbol count exceeds Databento's 2,000-symbol limit.
    """
    # Check cache
    cache_dir = (
        backend_dir / "lake" / "raw" / "source=databento"
        / "dataset=definition" / "venue=glbx" / "type=futures_options"
        / f"symbol={symbol}"
    )
    cache_file = cache_dir / f"resolved_{session_date}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        log_msg(
            log_path,
            f"CACHED_RESOLVE {symbol} {session_date}: "
            f"{len(cached['raw_symbols'])} symbols ({cached['label']})",
        )
        return cached["raw_symbols"], cached["label"], cached.get("parent")

    candidates = _candidate_parents_ordered(symbol, session_date)
    all_parents = [p for _, plist in candidates for p in plist]

    if not all_parents:
        log_msg(log_path, f"NO_PARENTS {symbol} {session_date}: no candidate parents computed")
        return [], "none", None

    log_msg(
        log_path,
        f"SYMBOLOGY {symbol} {session_date}: resolving {len(all_parents)} "
        f"candidate parents (FREE)",
    )
    resolved_map = _symbology_resolve_raw_symbols(
        client, all_parents, session_date, log_path,
    )

    resolved_count = sum(len(v) for v in resolved_map.values())
    log_msg(
        log_path,
        f"SYMBOLOGY {symbol} {session_date}: "
        f"{len(resolved_map)}/{len(all_parents)} parents resolved "
        f"({resolved_count} total instruments)",
    )

    if not resolved_map:
        log_msg(
            log_path,
            f"NO_INSTRUMENTS {symbol} {session_date}: "
            f"none of {len(all_parents)} candidate parents resolved",
        )
        return [], "none", None

    # Try each candidate in priority order
    session_dt = datetime.strptime(session_date, "%Y-%m-%d").date()

    for label, parents in candidates:
        for parent in parents:
            if parent not in resolved_map:
                continue

            instruments = resolved_map[parent]
            log_msg(
                log_path,
                f"TRYING {parent}: {len(instruments)} instruments ({label})",
            )

            # Stream targeted definition using parent symbology.
            # Uses stype_in="parent" so we send ONE parent symbol — the
            # server returns definitions for just that series (not all options).
            try:
                store = client.timeseries.get_range(
                    dataset=FUTURES_DATASET,
                    symbols=[parent],
                    schema="definition",
                    stype_in="parent",
                    start=session_date,
                    end=_next_day(session_date),
                )
                df = store.to_df(
                    price_type=PriceType.FIXED,
                    pretty_ts=False,
                    map_symbols=True,
                )
            except Exception as e:
                log_msg(log_path, f"DEF_STREAM_ERROR {parent}: {e}")
                continue

            if df.empty:
                continue

            # Filter: instrument_class in {C, P}, underlying == active_contract
            df = df[df["instrument_class"].isin(["C", "P"])].copy()
            df["underlying"] = df["underlying"].astype(str).str.strip()
            df = df[df["underlying"] == active_contract].copy()

            if df.empty:
                log_msg(
                    log_path,
                    f"SKIP {parent}: no instruments with underlying={active_contract}",
                )
                continue

            df = df[df["expiration"].notna()].copy()
            exp_dates = pd.to_datetime(
                df["expiration"].astype("int64"), utc=True,
            ).dt.date

            if label == "0dte":
                # Exact session date expiration
                zero_dte = df[exp_dates == session_dt]
                if zero_dte.empty:
                    log_msg(
                        log_path,
                        f"SKIP {parent}: instruments exist but none expire on {session_date}",
                    )
                    continue
                raw_syms = sorted(
                    {str(s).strip() for s in zero_dte["raw_symbol"].dropna().unique()
                     if str(s).strip()}
                )
                log_msg(
                    log_path,
                    f"0DTE {symbol} {session_date}: {len(raw_syms)} options via {parent}",
                )
                _save_resolve_cache(cache_dir, cache_file, raw_syms, "0dte", parent)
                return raw_syms, "0dte", parent
            else:
                # Nearest expiry: closest expiration >= session_date
                future_mask = exp_dates >= session_dt
                if not future_mask.any():
                    continue
                nearest_exp = exp_dates[future_mask].min()
                nearest_df = df[exp_dates == nearest_exp]
                raw_syms = sorted(
                    {str(s).strip() for s in nearest_df["raw_symbol"].dropna().unique()
                     if str(s).strip()}
                )
                exp_str = str(nearest_exp)
                result_label = f"nearest_{exp_str}"
                log_msg(
                    log_path,
                    f"NEAREST {symbol} {session_date}: {len(raw_syms)} options "
                    f"expiring {exp_str} via {parent} ({label})",
                )
                _save_resolve_cache(cache_dir, cache_file, raw_syms, result_label, parent)
                return raw_syms, result_label, parent

    log_msg(log_path, f"NO_OPTIONS {symbol} {session_date}: all candidates exhausted")
    return [], "none", None


def _save_resolve_cache(
    cache_dir: Path,
    cache_file: Path,
    raw_symbols: list[str],
    label: str,
    parent: str,
) -> None:
    """Save resolved symbols to JSON cache for re-run efficiency."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {"raw_symbols": raw_symbols, "label": label, "parent": parent},
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# Symbol / schema parsing
# ---------------------------------------------------------------------------

def parse_symbols(raw_symbols: str) -> list[str]:
    symbols = []
    seen: set[str] = set()
    for raw in raw_symbols.split(","):
        sym = raw.strip().upper()
        if not sym:
            continue
        if not re.fullmatch(r"[A-Z0-9]{1,8}", sym):
            raise ValueError(
                f"Invalid futures root symbol '{sym}'. "
                "Expected 1-8 uppercase letters/digits (examples: ES, SI, CL, 6E)."
            )
        if sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
    if not symbols:
        raise ValueError("At least one symbol is required")
    return symbols


def parse_options_schemas(raw_schemas: str) -> list[str]:
    schemas = [s.strip() for s in raw_schemas.split(",") if s.strip()]
    if not schemas:
        raise ValueError("At least one options schema is required")
    unsupported = sorted(set(schemas) - SUPPORTED_OPTIONS_SCHEMAS)
    if unsupported:
        allowed = ",".join(sorted(SUPPORTED_OPTIONS_SCHEMAS))
        raise ValueError(f"Unsupported options schemas: {unsupported}. Allowed: {allowed}")
    return schemas


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def _decompress_zst_files(directory: Path, log_path: Path) -> None:
    """Decompress all .dbn.zst files in directory, remove originals."""
    for zst_file in sorted(directory.glob("*.dbn.zst")):
        out_file = zst_file.with_suffix("")  # strip .zst
        subprocess.run(["zstd", "-d", "--rm", str(zst_file)], check=True)
        log_msg(log_path, f"DECOMPRESS {zst_file.name} -> {out_file.name}")


# ---------------------------------------------------------------------------
# Target path helpers
# ---------------------------------------------------------------------------

def target_path_futures_mbo(symbol: str, date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    out_dir = base / "product_type=future_mbo" / f"symbol={symbol}" / "table=market_by_order_dbn"
    return out_dir / f"glbx-mdp3-{date_compact}.mbo.dbn.zst"


def target_path_options_definition(symbol: str, date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    out_dir = base / "dataset=definition" / "venue=glbx" / "type=futures_options" / f"symbol={symbol}"
    return out_dir / f"glbx-mdp3-{date_compact}.definition.dbn.zst"


def target_path_options_data(schema: str, symbol: str, date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    if schema == "mbo":
        out_dir = base / "product_type=future_option_mbo" / f"symbol={symbol}" / "table=market_by_order_dbn"
        name = f"glbx-mdp3-{date_compact}.mbo.dbn"
    elif schema == "statistics":
        out_dir = base / "product_type=future_option_mbo" / f"symbol={symbol}" / "table=statistics"
        name = f"glbx-mdp3-{date_compact}.statistics.dbn"
    else:
        raise ValueError(f"Unsupported options schema: {schema}")
    return out_dir / name


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

def job_key(dataset: str, schema: str, symbol: str, date_str: str, product_type: str) -> str:
    return f"{dataset}|{schema}|{symbol}|{date_str}|{product_type}"


def load_job_tracker() -> dict[str, Any]:
    if JOB_TRACKER_FILE.exists():
        with open(JOB_TRACKER_FILE) as f:
            return json.load(f)
    return {"jobs": {}, "pending_downloads": []}


def save_job_tracker(tracker: dict[str, Any]) -> None:
    JOB_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(JOB_TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_msg(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat() + "Z"
    line = f"{ts} {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def date_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("end date before start date")

    out: list[str] = []
    curr = start_dt
    while curr <= end_dt:
        # Globex has a Sunday session. Only skip Saturday.
        if curr.weekday() != 5:
            out.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return out


def _next_day(date_str: str) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Phase 1: Front-month resolution via continuous definition stream
# ---------------------------------------------------------------------------

def resolve_front_month(
    client: db.Historical,
    symbol: str,
    session_date: str,
    log_path: Path,
) -> str:
    """Resolve front-month contract by streaming its definition record.

    Downloads a single definition record for the volume-ranked continuous
    front-month symbol (SI.v.0) via timeseries.get_range. Volume ranking
    ensures we get the most actively traded contract, not just the
    nearest-to-expire (which may be nearly illiquid near expiration).
    The raw_symbol field of the definition is the actual contract name
    (e.g. SIH6). This is a tiny request — one definition record.
    """
    continuous_symbol = f"{symbol}.v.0"
    end = _next_day(session_date)

    log_msg(log_path, f"RESOLVING front month: {continuous_symbol} on {session_date}")
    store = client.timeseries.get_range(
        dataset=FUTURES_DATASET,
        symbols=[continuous_symbol],
        stype_in="continuous",
        schema="definition",
        start=session_date,
        end=end,
    )

    df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if df.empty:
        raise ValueError(
            f"No definition for {continuous_symbol} on {session_date}"
        )

    raw_symbol = str(df["raw_symbol"].iloc[0]).strip()
    if not raw_symbol:
        raise ValueError(
            f"Empty raw_symbol in definition for {continuous_symbol} on {session_date}"
        )

    log_msg(log_path, f"RESOLVED {continuous_symbol} -> {raw_symbol} on {session_date}")
    return raw_symbol


# ---------------------------------------------------------------------------
# Phase 2: Submit options definition as batch job
# ---------------------------------------------------------------------------

def submit_definition_batch_job(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    session_date: str,
    log_path: Path,
    pause_seconds: int,
) -> str | None:
    """Submit a batch job for options definition download.

    Uses batch.submit_job with parent symbology instead of streaming
    timeseries.get_range. This avoids re-billing when re-downloading on
    a new system. If the definition file already exists on disk, skips
    submission entirely.

    Returns the batch job ID, or None if cached/skipped.
    """
    date_compact = session_date.replace("-", "")
    out_path = target_path_options_definition(symbol, date_compact)

    if out_path.exists():
        log_msg(log_path, f"CACHED options definition: {out_path}")
        return None

    parents = options_parents_for(symbol, session_date)
    if not parents:
        raise ValueError(f"No options parents for {symbol} on {session_date}")

    job_id = submit_job(
        client=client,
        tracker=tracker,
        dataset=FUTURES_DATASET,
        schema="definition",
        symbol=symbol,
        date_str=session_date,
        symbols=parents,
        stype_in="parent",
        log_path=log_path,
        product_type=PRODUCT_FUTURES_OPTIONS_DEF,
    )
    time.sleep(pause_seconds)
    return job_id


# ---------------------------------------------------------------------------
# Phase 3: 0DTE filtering (local processing)
# ---------------------------------------------------------------------------

def filter_0dte_futures_option_raw_symbols_from_definitions(
    definitions_df: pd.DataFrame,
    active_underlying_raw_symbol: str,
    session_date: str,
) -> list[str]:
    """Return 0DTE futures option raw symbols for one active underlying contract."""
    required = {"raw_symbol", "underlying", "instrument_class", "expiration"}
    missing = sorted(required - set(definitions_df.columns))
    if missing:
        raise ValueError(f"futures options definitions missing required columns: {missing}")

    df = definitions_df.copy()
    df = df[df["instrument_class"].isin(["C", "P"])].copy()
    df["underlying"] = df["underlying"].astype(str).str.strip()
    df = df[df["underlying"] == active_underlying_raw_symbol].copy()
    df = df[df["expiration"].notna()].copy()

    exp_dates = pd.to_datetime(df["expiration"].astype("int64"), utc=True).dt.date.astype(str)
    df = df[exp_dates == session_date].copy()

    raw_symbols = sorted({str(s).strip() for s in df["raw_symbol"].dropna().unique() if str(s).strip()})
    if not raw_symbols:
        raise ValueError(
            "No 0DTE futures options found for "
            f"underlying={active_underlying_raw_symbol} session={session_date}"
        )
    return raw_symbols


def load_0dte_option_raw_symbols(
    definition_path: Path,
    active_contract: str,
    session_date: str,
) -> list[str]:
    """Load options definition from disk and filter to 0DTE contracts."""
    store = db.DBNStore.from_file(str(definition_path))
    df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if df.empty:
        raise ValueError(f"Options definition file empty: {definition_path}")
    return filter_0dte_futures_option_raw_symbols_from_definitions(
        definitions_df=df,
        active_underlying_raw_symbol=active_contract,
        session_date=session_date,
    )


# ---------------------------------------------------------------------------
# Batch job submission
# ---------------------------------------------------------------------------

def submit_job(
    client: db.Historical,
    tracker: dict[str, Any],
    dataset: str,
    schema: str,
    symbol: str,
    date_str: str,
    symbols: list[str],
    stype_in: str,
    log_path: Path,
    product_type: str,
) -> str | None:
    """Submit one strict one-day flat-file batch job."""
    key = job_key(dataset, schema, symbol, date_str, product_type)
    symbols_digest = hashlib.sha256("\x1f".join(sorted(symbols)).encode("utf-8")).hexdigest()

    if key in tracker["jobs"]:
        existing = tracker["jobs"][key]
        payload_changed = (
            existing.get("stype_in") != stype_in
            or existing.get("symbols_count") != len(symbols)
            or existing.get("symbols_digest") != symbols_digest
        )
        if existing["state"] in ("done", "downloaded") and not payload_changed:
            log_msg(log_path, f"SKIP {key} already {existing['state']}")
            return None
        if existing["state"] != "error" and not payload_changed:
            log_msg(log_path, f"REUSE {key} job={existing['job_id']} state={existing['state']}")
            return existing["job_id"]
        if payload_changed:
            log_msg(log_path, f"RESUBMIT {key} (payload changed)")
        else:
            log_msg(log_path, f"RESUBMIT {key} (was error)")

    end_date = _next_day(date_str)

    try:
        job = client.batch.submit_job(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=date_str,
            end=end_date,
            stype_in=stype_in,
            delivery=FLAT_FILE_DELIVERY,
            split_duration=FLAT_FILE_SPLIT_DURATION,
        )
        job_id = job["id"]
        tracker["jobs"][key] = {
            "job_id": job_id,
            "dataset": dataset,
            "schema": schema,
            "symbol": symbol,
            "date_str": date_str,
            "product_type": product_type,
            "symbols_count": len(symbols),
            "symbols_digest": symbols_digest,
            "stype_in": stype_in,
            "state": "submitted",
            "submitted_at": datetime.utcnow().isoformat(),
        }
        save_job_tracker(tracker)
        log_msg(
            log_path,
            f"SUBMIT {key} job={job_id} symbols={len(symbols)} "
            f"delivery={FLAT_FILE_DELIVERY} split_duration={FLAT_FILE_SPLIT_DURATION}",
        )
        return job_id
    except Exception as e:
        log_msg(log_path, f"ERROR submit {key}: {e}")
        return None


# ---------------------------------------------------------------------------
# Batch polling and download
# ---------------------------------------------------------------------------

def poll_jobs(client: db.Historical, tracker: dict[str, Any], log_path: Path) -> int:
    pending_keys = [
        k for k, v in tracker["jobs"].items()
        if v["state"] not in ("done", "downloaded", "expired", "error")
    ]
    if not pending_keys:
        return 0

    try:
        api_jobs = client.batch.list_jobs(
            states=["queued", "processing", "done", "expired"],
            since=(datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
        )
        job_states = {j["id"]: j for j in api_jobs}
    except Exception as e:
        log_msg(log_path, f"ERROR polling jobs: {e}")
        return 0

    done_count = 0
    for key in pending_keys:
        job_info = tracker["jobs"][key]
        job_id = job_info["job_id"]
        if job_id not in job_states:
            continue

        api_state = job_states[job_id].get("state", "unknown")
        old_state = job_info["state"]
        if api_state != old_state:
            job_info["state"] = api_state
            job_info["updated_at"] = datetime.utcnow().isoformat()
            log_msg(log_path, f"STATE {key} {old_state} -> {api_state}")

        if api_state == "done":
            done_count += 1
            if key not in tracker["pending_downloads"]:
                tracker["pending_downloads"].append(key)

    save_job_tracker(tracker)
    return done_count


def target_path_for_job(schema: str, symbol: str, date_compact: str, product_type: str) -> Path:
    if product_type == PRODUCT_FUTURES:
        return target_path_futures_mbo(symbol, date_compact)
    if product_type == PRODUCT_FUTURES_OPTIONS:
        return target_path_options_data(schema, symbol, date_compact)
    if product_type == PRODUCT_FUTURES_OPTIONS_DEF:
        return target_path_options_definition(symbol, date_compact)
    raise ValueError(f"Unsupported product_type: {product_type}")


def download_completed_jobs(client: db.Historical, tracker: dict[str, Any], log_path: Path) -> int:
    downloaded = 0

    for key in list(tracker.get("pending_downloads", [])):
        if key not in tracker["jobs"]:
            tracker["pending_downloads"].remove(key)
            continue

        job_info = tracker["jobs"][key]
        if job_info["state"] != "done":
            continue

        job_id = job_info["job_id"]
        schema = job_info["schema"]
        symbol = job_info["symbol"]
        date_str = job_info["date_str"]
        product_type = job_info.get("product_type", PRODUCT_FUTURES_OPTIONS)
        date_compact = date_str.replace("-", "")

        out_path = target_path_for_job(schema, symbol, date_compact, product_type)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        log_msg(log_path, f"DOWNLOAD {key} job={job_id} -> {out_dir}")

        try:
            downloaded_files = client.batch.download(job_id=job_id, output_dir=out_dir)
            # batch.download creates a job-ID subdirectory — flatten .dbn* files up
            job_subdir = out_dir / job_id
            if job_subdir.is_dir():
                for f in job_subdir.iterdir():
                    if f.suffix in (".dbn", ".zst"):
                        dest = out_dir / f.name
                        f.rename(dest)
                        log_msg(log_path, f"MOVED {f.name} -> {dest}")
                shutil.rmtree(job_subdir, ignore_errors=True)
            # Decompress .dbn.zst -> .dbn, remove originals
            _decompress_zst_files(out_dir, log_path)
            job_info["state"] = "downloaded"
            job_info["downloaded_at"] = datetime.utcnow().isoformat()
            job_info["files"] = [str(f) for f in downloaded_files]
            tracker["pending_downloads"].remove(key)
            save_job_tracker(tracker)
            log_msg(log_path, f"COMPLETE {key} files={len(downloaded_files)}")
            downloaded += 1
        except Exception as e:
            log_msg(log_path, f"ERROR download {key}: {e}")
            job_info["state"] = "error"
            job_info["error"] = str(e)
            tracker["pending_downloads"].remove(key)
            save_job_tracker(tracker)

    return downloaded


# ---------------------------------------------------------------------------
# Job filtering helpers
# ---------------------------------------------------------------------------

def has_pending_jobs(tracker: dict[str, Any], product_type: str | None = None) -> bool:
    """Check if there are pending jobs, optionally filtered by product type."""
    for v in tracker["jobs"].values():
        if v["state"] in ("downloaded", "expired", "error"):
            continue
        if product_type is not None and v.get("product_type") != product_type:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Per-day processing: 3-phase architecture
# ---------------------------------------------------------------------------

def process_session_day_phase1(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    date_str: str,
    log_path: Path,
    pause_seconds: int,
) -> tuple[str, list[str], str, str | None]:
    """Phase 1: Resolve front-month contract and discover options symbols.

    Uses FREE symbology.resolve to find the right parent, then streams a
    tiny targeted definition to filter for 0DTE or nearest-expiry options.

    Returns (active_contract, option_raw_symbols, expiry_label, resolved_parent).
    """
    # Step 1: Resolve front-month contract (synchronous, one definition record)
    active_contract = resolve_front_month(client, symbol, date_str, log_path)

    # Step 2: Discover options via FREE symbology.resolve + tiny streaming def
    option_raw_symbols, expiry_label, resolved_parent = discover_options_raw_symbols(
        client=client,
        symbol=symbol,
        session_date=date_str,
        active_contract=active_contract,
        log_path=log_path,
    )

    return active_contract, option_raw_symbols, expiry_label, resolved_parent


def process_session_day_phase2(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    date_str: str,
    active_contract: str,
    option_raw_symbols: list[str],
    expiry_label: str,
    resolved_parent: str | None,
    include_futures: bool,
    options_schemas: list[str],
    log_path: Path,
    pause_seconds: int,
) -> None:
    """Phase 2: Submit data batch jobs for pre-discovered options.

    Accepts option raw symbols discovered in phase 1 (via FREE symbology
    resolution + tiny streaming definition). No definition file needed.

    Uses stype_in=parent when the discovered symbol count exceeds Databento's
    2,000-symbol limit per batch job. The parent symbol targets the same
    weekly/daily series, so the download is nearly identical.

    1. Submit futures MBO batch job for front-month contract only
    2. Submit options MBO/statistics batch jobs for discovered contracts
    """
    # Step 1: Submit futures MBO batch job (front-month only)
    if include_futures:
        submit_job(
            client=client,
            tracker=tracker,
            dataset=FUTURES_DATASET,
            schema="mbo",
            symbol=symbol,
            date_str=date_str,
            symbols=[active_contract],
            stype_in="raw_symbol",
            log_path=log_path,
            product_type=PRODUCT_FUTURES,
        )
        time.sleep(pause_seconds)

    # Step 2: Submit options data batch jobs
    if not option_raw_symbols:
        log_msg(
            log_path,
            f"SKIP options data for {symbol} {date_str}: "
            f"no options discovered ({expiry_label})",
        )
        return

    # Choose symbology: use parent when symbols exceed Databento's 2,000 limit.
    # Parent symbology targets the same weekly/daily series (e.g. SO1.OPT)
    # so the download covers the same instruments with one symbol.
    MAX_SYMBOLS = 2000
    if len(option_raw_symbols) > MAX_SYMBOLS and resolved_parent:
        submit_symbols = [resolved_parent]
        submit_stype = "parent"
        log_msg(
            log_path,
            f"PARENT_STYPE {symbol} {date_str}: {len(option_raw_symbols)} symbols "
            f"> {MAX_SYMBOLS} limit, using {resolved_parent} (stype_in=parent)",
        )
    else:
        submit_symbols = option_raw_symbols
        submit_stype = "raw_symbol"

    for schema in options_schemas:
        if schema == "definition":
            continue
        submit_job(
            client=client,
            tracker=tracker,
            dataset=FUTURES_DATASET,
            schema=schema,
            symbol=symbol,
            date_str=date_str,
            symbols=submit_symbols,
            stype_in=submit_stype,
            log_path=log_path,
            product_type=PRODUCT_FUTURES_OPTIONS,
        )
        time.sleep(pause_seconds)


def process_session_day(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    date_str: str,
    include_futures: bool,
    options_schemas: list[str],
    log_path: Path,
    pause_seconds: int,
) -> None:
    """Process one session date: resolve front month, discover options, submit batch jobs.

    Convenience wrapper that runs phase 1 + phase 2 sequentially.
    Phase 1 uses FREE symbology.resolve to discover options — no separate
    definition download needed.
    """
    active_contract, option_raw_symbols, expiry_label, resolved_parent = (
        process_session_day_phase1(
            client=client,
            tracker=tracker,
            symbol=symbol,
            date_str=date_str,
            log_path=log_path,
            pause_seconds=pause_seconds,
        )
    )

    process_session_day_phase2(
        client=client,
        tracker=tracker,
        symbol=symbol,
        date_str=date_str,
        active_contract=active_contract,
        option_raw_symbols=option_raw_symbols,
        expiry_label=expiry_label,
        resolved_parent=resolved_parent,
        include_futures=include_futures,
        options_schemas=options_schemas,
        log_path=log_path,
        pause_seconds=pause_seconds,
    )


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set")

    client = db.Historical(key=api_key)
    tracker = load_job_tracker()
    log_path = Path(args.log_file)

    symbols = parse_symbols(args.symbols)
    options_schemas = parse_options_schemas(args.options_schemas)

    for date_str in date_range(args.start, args.end):
        for symbol in symbols:
            try:
                process_session_day(
                    client=client,
                    tracker=tracker,
                    symbol=symbol,
                    date_str=date_str,
                    include_futures=args.include_futures,
                    options_schemas=options_schemas,
                    log_path=log_path,
                    pause_seconds=args.pause_seconds,
                )
            except Exception as e:
                log_msg(log_path, f"ERROR processing {symbol} {date_str}: {e}")


def cmd_poll(args: argparse.Namespace) -> None:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set")

    client = db.Historical(key=api_key)
    tracker = load_job_tracker()
    log_path = Path(args.log_file)

    done = poll_jobs(client, tracker, log_path)
    log_msg(log_path, f"POLL complete: {done} jobs ready")

    downloaded = download_completed_jobs(client, tracker, log_path)
    log_msg(log_path, f"DOWNLOADED {downloaded} jobs")


def _poll_until_complete(
    client: db.Historical,
    tracker: dict[str, Any],
    log_path: Path,
    poll_interval: int,
    label: str,
) -> None:
    """Poll and download batch jobs until none remain pending."""
    while True:
        tracker = load_job_tracker()
        poll_jobs(client, tracker, log_path)
        download_completed_jobs(client, tracker, log_path)

        tracker = load_job_tracker()
        total = len(tracker["jobs"])
        done = sum(1 for v in tracker["jobs"].values() if v["state"] == "downloaded")
        pending = sum(
            1
            for v in tracker["jobs"].values()
            if v["state"] not in ("downloaded", "expired", "error")
        )

        log_msg(log_path, f"STATUS [{label}]: {done}/{total} downloaded, {pending} pending")
        if pending == 0:
            if done > 0:
                log_msg(log_path, f"COMPLETE [{label}]: all jobs finished")
            break

        log_msg(log_path, f"Sleeping {poll_interval}s...")
        time.sleep(poll_interval)


def cmd_daemon(args: argparse.Namespace) -> None:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set")

    client = db.Historical(key=api_key)
    log_path = Path(args.log_file)

    symbols = parse_symbols(args.symbols)
    options_schemas = parse_options_schemas(args.options_schemas)
    dates = date_range(args.start, args.end)

    log_msg(log_path, f"DAEMON start: {len(dates)} dates, {len(symbols)} symbols")

    # Phase 1: Resolve front month + discover options (FREE) + submit data batch jobs
    # Uses symbology.resolve (FREE) to find the right parent, then streams a
    # tiny definition to filter. No definition batch job or polling needed.
    tracker = load_job_tracker()
    for date_str in dates:
        for symbol in symbols:
            try:
                active_contract, option_syms, label, parent = (
                    process_session_day_phase1(
                        client=client,
                        tracker=tracker,
                        symbol=symbol,
                        date_str=date_str,
                        log_path=log_path,
                        pause_seconds=args.pause_seconds,
                    )
                )
                process_session_day_phase2(
                    client=client,
                    tracker=tracker,
                    symbol=symbol,
                    date_str=date_str,
                    active_contract=active_contract,
                    option_raw_symbols=option_syms,
                    expiry_label=label,
                    resolved_parent=parent,
                    include_futures=args.include_futures,
                    options_schemas=options_schemas,
                    log_path=log_path,
                    pause_seconds=args.pause_seconds,
                )
            except Exception as e:
                log_msg(log_path, f"ERROR {symbol} {date_str}: {e}")

    # Phase 2: Poll and download data batch jobs
    tracker = load_job_tracker()
    if has_pending_jobs(tracker, PRODUCT_FUTURES) or has_pending_jobs(tracker, PRODUCT_FUTURES_OPTIONS):
        log_msg(log_path, "PHASE2: polling data batch jobs...")
        _poll_until_complete(client, tracker, log_path, args.poll_interval, "data")

    log_msg(log_path, "DAEMON complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch download futures + futures options (front-month + 0DTE only)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit batch jobs")
    submit_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    submit_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    submit_parser.add_argument(
        "--symbols",
        default="ES",
        help="Comma-separated futures root symbols (examples: ES,SI,CL,6E)",
    )
    submit_parser.add_argument("--include-futures", action="store_true", help="Also download futures MBO")
    submit_parser.add_argument("--options-schemas", default="mbo,statistics", help="Options schemas")
    submit_parser.add_argument("--pause-seconds", type=int, default=5, help="Pause between submissions")
    submit_parser.add_argument("--log-file", required=True, help="Log file path")

    poll_parser = subparsers.add_parser("poll", help="Poll jobs and download completed")
    poll_parser.add_argument("--log-file", required=True, help="Log file path")

    daemon_parser = subparsers.add_parser("daemon", help="Run submit + poll in loop until complete")
    daemon_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    daemon_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    daemon_parser.add_argument(
        "--symbols",
        default="ES",
        help="Comma-separated futures root symbols (examples: ES,SI,CL,6E)",
    )
    daemon_parser.add_argument("--include-futures", action="store_true", help="Also download futures MBO")
    daemon_parser.add_argument("--options-schemas", default="mbo,statistics", help="Options schemas")
    daemon_parser.add_argument("--pause-seconds", type=int, default=5, help="Pause between submissions")
    daemon_parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between poll iterations")
    daemon_parser.add_argument("--log-file", required=True, help="Log file path")

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "poll":
        cmd_poll(args)
    elif args.command == "daemon":
        cmd_daemon(args)


if __name__ == "__main__":
    main()
