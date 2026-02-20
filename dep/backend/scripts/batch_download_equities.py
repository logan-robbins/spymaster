"""
Batch Download Equity + Equity Options Data (0DTE only, flat-file day slices)

Downloads from two datasets:
- XNAS.ITCH: Underlying equity MBO data
- OPRA.PILLAR: Equity options data via Databento CMBP-1 schema (0DTE filtered)

Options definition files are downloaded via batch API (not streaming) to avoid
re-billing when re-running on a new system without cached files.

Daemon 3-phase flow:
- Phase 1: Submit definition batch jobs
- Phase 2: Poll/download definition jobs -> filter 0DTE -> submit data batch jobs
- Phase 3: Poll/download data batch jobs

All batch requests are strict flat-file requests:
- delivery="download"
- split_duration="day"
- start=<session_date>, end=<session_date + 1 day>

For equity options (<SYMBOL>), 0DTE filtering is by:
- instrument_class in {C, P}
- underlying == <SYMBOL>
- expiration UTC date == session date
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

JOB_TRACKER_FILE = backend_dir / "logs" / "equity_options_jobs.json"

SUPPORTED_EQUITY_SCHEMAS = {"mbo", "definition"}
SUPPORTED_OPTIONS_SCHEMAS = {"definition", "cmbp-1", "statistics"}

FLAT_FILE_DELIVERY = "download"
FLAT_FILE_SPLIT_DURATION = "day"

PRODUCT_EQUITY = "equity"
PRODUCT_EQUITY_OPTIONS = "equity_options"
PRODUCT_EQUITY_OPTIONS_DEF = "equity_options_def"


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
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,14}", sym):
            raise ValueError(
                f"Invalid equity symbol '{sym}'. "
                "Expected 1-15 chars: letters/digits plus optional . _ - (examples: QQQ, AAPL, BRK.B)."
            )
        if sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
    if not symbols:
        raise ValueError("At least one symbol is required")
    return symbols


def parse_schema_list(raw_schemas: str, allowed: set[str], label: str) -> list[str]:
    schemas = [s.strip() for s in raw_schemas.split(",") if s.strip()]
    if not schemas:
        raise ValueError(f"At least one {label} schema is required")
    unsupported = sorted(set(schemas) - allowed)
    if unsupported:
        allowed_str = ",".join(sorted(allowed))
        raise ValueError(f"Unsupported {label} schemas: {unsupported}. Allowed: {allowed_str}")
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

def target_path_equity(schema: str, symbol: str, date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    if schema == "mbo":
        out_dir = base / "product_type=equity_mbo" / f"symbol={symbol}" / "table=market_by_order_dbn"
        name = f"xnas-itch-{date_compact}.mbo.dbn.zst"
    elif schema == "definition":
        out_dir = base / "dataset=definition" / "venue=xnas"
        name = f"xnas-itch-{date_compact}.definition.dbn"
    else:
        raise ValueError(f"Unsupported equity schema: {schema}")
    return out_dir / name


def target_path_options_definition(symbol: str, date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    out_dir = base / "dataset=definition" / "venue=opra" / f"symbol={symbol}"
    return out_dir / f"opra-pillar-{date_compact}.definition.dbn.zst"


def target_path_options(schema: str, symbol: str, date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    if schema == "cmbp-1":
        out_dir = base / "product_type=equity_option_cmbp_1" / f"symbol={symbol}" / "table=cmbp_1"
        name = f"opra-pillar-{date_compact}.cmbp-1.dbn"
    elif schema == "statistics":
        out_dir = base / "product_type=equity_option_statistics" / f"symbol={symbol}" / "table=statistics"
        name = f"opra-pillar-{date_compact}.statistics.dbn"
    elif schema == "definition":
        out_dir = base / "dataset=definition" / "venue=opra"
        name = f"opra-pillar-{date_compact}.definition.dbn"
    else:
        raise ValueError(f"Unsupported options schema: {schema}")
    return out_dir / name


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

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
# Logging & dates
# ---------------------------------------------------------------------------

def log_msg(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat() + "Z"
    line = f"{ts} {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def date_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("end date before start date")
    out = []
    curr = start_dt
    while curr <= end_dt:
        if curr.weekday() not in (5, 6):
            out.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return out


def _next_day(date_str: str) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Submit options definition as batch job
# ---------------------------------------------------------------------------

def submit_definition_batch_job(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    session_date: str,
    log_path: Path,
    pause_seconds: int,
) -> str | None:
    """Submit a batch job for OPRA options definition download.

    Uses batch.submit_job with parent symbology (QQQ.OPT) instead of
    streaming timeseries.get_range. This avoids re-billing when
    re-downloading on a new system. If the definition file already
    exists on disk, skips submission entirely.

    Returns the batch job ID, or None if cached/skipped.
    """
    date_compact = session_date.replace("-", "")
    out_path = target_path_options_definition(symbol, date_compact)

    if out_path.exists():
        log_msg(log_path, f"CACHED options definition: {out_path}")
        return None

    job_id = submit_job(
        client=client,
        tracker=tracker,
        dataset="OPRA.PILLAR",
        schema="definition",
        symbol=symbol,
        date_str=session_date,
        symbols=[f"{symbol}.OPT"],
        stype_in="parent",
        log_path=log_path,
        product_type=PRODUCT_EQUITY_OPTIONS_DEF,
    )
    time.sleep(pause_seconds)
    return job_id


# ---------------------------------------------------------------------------
# 0DTE filtering
# ---------------------------------------------------------------------------

def filter_0dte_equity_option_raw_symbols_from_definitions(
    definitions_df: pd.DataFrame, symbol: str, date_str: str
) -> list[str]:
    required = {"raw_symbol", "underlying", "instrument_class", "expiration"}
    missing = sorted(required - set(definitions_df.columns))
    if missing:
        raise ValueError(f"Definitions missing required columns: {missing}")

    df = definitions_df.copy()
    df = df[df["instrument_class"].isin(["C", "P"])].copy()
    df = df[df["underlying"].astype(str).str.upper() == symbol.upper()].copy()
    df = df[df["expiration"].notna()].copy()

    exp_dates = pd.to_datetime(df["expiration"].astype("int64"), utc=True).dt.date.astype(str)
    df = df[exp_dates == date_str].copy()

    if df.empty:
        raise ValueError(f"No {symbol} 0DTE definitions for {date_str}")

    raw_symbols = sorted({str(s).strip() for s in df["raw_symbol"].dropna().unique() if str(s).strip()})
    if not raw_symbols:
        raise ValueError(f"No 0DTE raw_symbols for {symbol} on {date_str}")
    return raw_symbols


def load_0dte_assets(
    definition_path: Path,
    symbol: str,
    date_str: str,
) -> list[str]:
    """Load OPRA definition from disk and filter to 0DTE contracts."""
    store = db.DBNStore.from_file(str(definition_path))
    df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if df.empty:
        raise ValueError(f"Definition file empty: {definition_path}")
    return filter_0dte_equity_option_raw_symbols_from_definitions(df, symbol, date_str)


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
    product_type: str = PRODUCT_EQUITY_OPTIONS,
) -> str | None:
    """Submit one strict one-day flat-file batch job."""
    key = f"{dataset}|{schema}|{symbol}|{date_str}|{product_type}"
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

def poll_jobs(client: db.Historical, tracker: dict, log_path: Path) -> int:
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

        if job_id in job_states:
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


def target_path_for_job(
    dataset: str, schema: str, symbol: str, date_compact: str, product_type: str,
) -> Path:
    """Resolve download target path based on product type."""
    if product_type == PRODUCT_EQUITY:
        return target_path_equity(schema, symbol, date_compact)
    if product_type == PRODUCT_EQUITY_OPTIONS_DEF:
        return target_path_options_definition(symbol, date_compact)
    if product_type == PRODUCT_EQUITY_OPTIONS:
        return target_path_options(schema, symbol, date_compact)
    # Fallback for jobs tracked before product_type was added
    if dataset == "XNAS.ITCH":
        return target_path_equity(schema, symbol, date_compact)
    return target_path_options(schema, symbol, date_compact)


def download_completed_jobs(client: db.Historical, tracker: dict, log_path: Path) -> int:
    downloaded = 0

    pending = list(tracker.get("pending_downloads", []))
    for key in pending:
        if key not in tracker["jobs"]:
            tracker["pending_downloads"].remove(key)
            continue

        job_info = tracker["jobs"][key]
        if job_info["state"] != "done":
            continue

        job_id = job_info["job_id"]
        dataset = job_info["dataset"]
        schema = job_info["schema"]
        symbol = job_info["symbol"]
        date_str = job_info["date_str"]
        date_compact = date_str.replace("-", "")
        product_type = job_info.get("product_type", PRODUCT_EQUITY_OPTIONS)

        out_path = target_path_for_job(dataset, schema, symbol, date_compact, product_type)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        log_msg(log_path, f"DOWNLOAD {key} job={job_id} -> {out_dir}")

        try:
            downloaded_files = client.batch.download(
                job_id=job_id,
                output_dir=out_dir,
            )
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
) -> None:
    """Phase 1: Submit definition batch job for OPRA options.

    Submits definition download via batch API. Skipped if the definition
    file already exists on disk (cached from a prior run).
    """
    submit_definition_batch_job(
        client=client,
        tracker=tracker,
        symbol=symbol,
        session_date=date_str,
        log_path=log_path,
        pause_seconds=pause_seconds,
    )


def process_session_day_phase2(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    date_str: str,
    equity_schemas: list[str],
    options_schemas: list[str],
    log_path: Path,
    pause_seconds: int,
) -> None:
    """Phase 2: Filter 0DTE from downloaded definitions, submit data batch jobs.

    Requires definition file to exist on disk (downloaded in phase 1).

    1. Load definition file and filter to 0DTE contracts
    2. Submit equity MBO batch jobs (XNAS.ITCH)
    3. Submit options CMBP-1/statistics batch jobs for 0DTE contracts only
    """
    date_compact = date_str.replace("-", "")
    def_path = target_path_options_definition(symbol, date_compact)

    if not def_path.exists():
        log_msg(log_path, f"SKIP {symbol} {date_str}: definition file not yet available")
        return

    # Step 1: Filter 0DTE options (local processing)
    try:
        raw_symbols = load_0dte_assets(def_path, symbol, date_str)
        log_msg(log_path, f"0DTE {symbol} {date_str}: options={len(raw_symbols)}")
    except ValueError as e:
        # No 0DTE options for this symbol/date — still proceed with equity data.
        raw_symbols = []
        log_msg(log_path, f"NO_0DTE {symbol} {date_str}: {e}")

    # Step 2: Submit equity MBO batch jobs (XNAS.ITCH)
    for schema in equity_schemas:
        submit_job(
            client=client,
            tracker=tracker,
            dataset="XNAS.ITCH",
            schema=schema,
            symbol=symbol,
            date_str=date_str,
            symbols=[symbol],
            stype_in="raw_symbol",
            log_path=log_path,
            product_type=PRODUCT_EQUITY,
        )
        time.sleep(pause_seconds)

    # Step 3: Submit options data batch jobs (OPRA.PILLAR, 0DTE only)
    if not raw_symbols:
        log_msg(log_path, f"SKIP options data for {symbol} {date_str}: no 0DTE contracts")
        return

    for schema in options_schemas:
        if schema == "definition":
            continue
        submit_job(
            client=client,
            tracker=tracker,
            dataset="OPRA.PILLAR",
            schema=schema,
            symbol=symbol,
            date_str=date_str,
            symbols=raw_symbols,
            stype_in="raw_symbol",
            log_path=log_path,
            product_type=PRODUCT_EQUITY_OPTIONS,
        )
        time.sleep(pause_seconds)


def process_session_day(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    date_str: str,
    equity_schemas: list[str],
    options_schemas: list[str],
    log_path: Path,
    pause_seconds: int,
) -> None:
    """Process one session date (convenience wrapper).

    Runs phase 1 + phase 2 sequentially. Used by the 'submit' CLI command.
    In daemon mode, the 3-phase architecture is used directly instead.
    """
    process_session_day_phase1(
        client=client,
        tracker=tracker,
        symbol=symbol,
        date_str=date_str,
        log_path=log_path,
        pause_seconds=pause_seconds,
    )

    date_compact = date_str.replace("-", "")
    def_path = target_path_options_definition(symbol, date_compact)
    if def_path.exists():
        process_session_day_phase2(
            client=client,
            tracker=tracker,
            symbol=symbol,
            date_str=date_str,
            equity_schemas=equity_schemas,
            options_schemas=options_schemas,
            log_path=log_path,
            pause_seconds=pause_seconds,
        )
    else:
        log_msg(log_path, f"DEF_PENDING {symbol} {date_str}: definition batch job submitted, poll to download")


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
    equity_schemas = parse_schema_list(args.equity_schemas, SUPPORTED_EQUITY_SCHEMAS, "equity")
    options_schemas = parse_schema_list(args.options_schemas, SUPPORTED_OPTIONS_SCHEMAS, "options")

    for date_str in date_range(args.start, args.end):
        for symbol in symbols:
            try:
                process_session_day(
                    client=client,
                    tracker=tracker,
                    symbol=symbol,
                    date_str=date_str,
                    equity_schemas=equity_schemas,
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
    equity_schemas = parse_schema_list(args.equity_schemas, SUPPORTED_EQUITY_SCHEMAS, "equity")
    options_schemas = parse_schema_list(args.options_schemas, SUPPORTED_OPTIONS_SCHEMAS, "options")
    dates = date_range(args.start, args.end)

    log_msg(log_path, f"DAEMON start: {len(dates)} dates, {len(symbols)} symbols")

    # Phase 1: Submit definition batch jobs
    tracker = load_job_tracker()
    for date_str in dates:
        for symbol in symbols:
            try:
                process_session_day_phase1(
                    client=client,
                    tracker=tracker,
                    symbol=symbol,
                    date_str=date_str,
                    log_path=log_path,
                    pause_seconds=args.pause_seconds,
                )
            except Exception as e:
                log_msg(log_path, f"ERROR phase1 {symbol} {date_str}: {e}")

    # Phase 2: Poll/download definition jobs -> filter 0DTE -> submit data batch jobs
    if has_pending_jobs(tracker, PRODUCT_EQUITY_OPTIONS_DEF):
        log_msg(log_path, "PHASE2: polling definition batch jobs...")
        _poll_until_complete(client, tracker, log_path, args.poll_interval, "definitions")

    tracker = load_job_tracker()
    for date_str in dates:
        for symbol in symbols:
            try:
                process_session_day_phase2(
                    client=client,
                    tracker=tracker,
                    symbol=symbol,
                    date_str=date_str,
                    equity_schemas=equity_schemas,
                    options_schemas=options_schemas,
                    log_path=log_path,
                    pause_seconds=args.pause_seconds,
                )
            except Exception as e:
                log_msg(log_path, f"ERROR phase2 {symbol} {date_str}: {e}")

    # Phase 3: Poll and download data batch jobs
    if has_pending_jobs(tracker, PRODUCT_EQUITY) or has_pending_jobs(tracker, PRODUCT_EQUITY_OPTIONS):
        log_msg(log_path, "PHASE3: polling data batch jobs...")
        _poll_until_complete(client, tracker, log_path, args.poll_interval, "data")

    log_msg(log_path, "DAEMON complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch download equity + equity options data (0DTE only)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit batch jobs")
    submit_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    submit_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    submit_parser.add_argument(
        "--symbols",
        default="QQQ",
        help="Comma-separated equity symbols (examples: QQQ,AAPL,SPY)",
    )
    submit_parser.add_argument("--equity-schemas", default="mbo", help="Equity schemas (mbo,definition)")
    submit_parser.add_argument("--options-schemas", default="cmbp-1,statistics", help="Options schemas")
    submit_parser.add_argument("--pause-seconds", type=int, default=5, help="Pause between submissions")
    submit_parser.add_argument("--log-file", required=True, help="Log file path")

    poll_parser = subparsers.add_parser("poll", help="Poll jobs and download completed")
    poll_parser.add_argument("--log-file", required=True, help="Log file path")

    daemon_parser = subparsers.add_parser("daemon", help="Run submit + poll in loop until complete")
    daemon_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    daemon_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    daemon_parser.add_argument(
        "--symbols",
        default="QQQ",
        help="Comma-separated equity symbols (examples: QQQ,AAPL,SPY)",
    )
    daemon_parser.add_argument("--equity-schemas", default="mbo", help="Equity schemas")
    daemon_parser.add_argument("--options-schemas", default="cmbp-1,statistics", help="Options schemas")
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
