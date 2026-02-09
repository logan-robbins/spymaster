"""
Batch Download Equity + Equity Options Data (0DTE only, flat-file day slices)

Downloads from two datasets:
- XNAS.ITCH: Underlying equity MBO data
- OPRA.PILLAR: Equity options data via Databento CMBP-1 schema (0DTE filtered)

Prerequisites (options definition) are synchronous:
- timeseries.get_range: streaming download to disk, immediate

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
import json
import os
import re
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
# Stream options definition via timeseries API
# ---------------------------------------------------------------------------

def stream_options_definition(
    client: db.Historical,
    symbol: str,
    session_date: str,
    log_path: Path,
) -> Path:
    """Stream OPRA options definition file directly to disk.

    Uses timeseries.get_range with parent symbology (QQQ.OPT).
    Returns path to the saved file.
    """
    date_compact = session_date.replace("-", "")
    out_path = target_path_options_definition(symbol, date_compact)

    if out_path.exists():
        log_msg(log_path, f"CACHED options definition: {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    end = _next_day(session_date)

    log_msg(log_path, f"STREAMING options definition for {symbol}.OPT {session_date}")
    client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        symbols=[f"{symbol}.OPT"],
        stype_in="parent",
        schema="definition",
        start=session_date,
        end=end,
        path=str(out_path),
    )
    log_msg(log_path, f"DONE options definition: {out_path}")
    return out_path


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
    tracker: dict,
    dataset: str,
    schema: str,
    symbol: str,
    date_str: str,
    symbols: list[str],
    stype_in: str,
    log_path: Path,
) -> str | None:
    job_key = f"{dataset}|{schema}|{symbol}|{date_str}"

    if job_key in tracker["jobs"]:
        existing = tracker["jobs"][job_key]
        if existing["state"] in ("done", "downloaded"):
            log_msg(log_path, f"SKIP {job_key} already {existing['state']}")
            return None
        if existing["state"] != "error":
            log_msg(log_path, f"REUSE {job_key} job={existing['job_id']} state={existing['state']}")
            return existing["job_id"]
        log_msg(log_path, f"RESUBMIT {job_key} (was error)")

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
        tracker["jobs"][job_key] = {
            "job_id": job_id,
            "dataset": dataset,
            "schema": schema,
            "symbol": symbol,
            "date_str": date_str,
            "symbols_count": len(symbols),
            "stype_in": stype_in,
            "state": "submitted",
            "submitted_at": datetime.utcnow().isoformat(),
        }
        save_job_tracker(tracker)
        log_msg(
            log_path,
            f"SUBMIT {job_key} job={job_id} symbols={len(symbols)} "
            f"delivery={FLAT_FILE_DELIVERY} split_duration={FLAT_FILE_SPLIT_DURATION}",
        )
        return job_id
    except Exception as e:
        log_msg(log_path, f"ERROR submit {job_key}: {e}")
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

        if dataset == "XNAS.ITCH":
            out_path = target_path_equity(schema, symbol, date_compact)
        else:
            out_path = target_path_options(schema, symbol, date_compact)

        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        log_msg(log_path, f"DOWNLOAD {key} job={job_id} -> {out_dir}")

        try:
            downloaded_files = client.batch.download(
                job_id=job_id,
                output_dir=out_dir,
            )
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
# Per-day processing: synchronous prereqs + batch data submission
# ---------------------------------------------------------------------------

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
    """Process one session date: stream options definition, filter 0DTE, submit batch jobs.

    Steps 1-2 are synchronous (no batch queue wait):
    1. Stream options definition via timeseries.get_range
    2. Filter 0DTE options locally

    Steps 3-4 submit batch jobs (async, polled later):
    3. Submit equity MBO jobs (XNAS.ITCH)
    4. Submit options CMBP-1/statistics jobs (OPRA.PILLAR) for 0DTE contracts only
    """
    # Step 1: Stream options definition (synchronous)
    def_path = stream_options_definition(client, symbol, date_str, log_path)

    # Step 2: Filter 0DTE options (local processing)
    raw_symbols = load_0dte_assets(def_path, symbol, date_str)
    log_msg(log_path, f"0DTE {symbol} {date_str}: options={len(raw_symbols)}")

    # Step 3: Submit equity MBO batch jobs (XNAS.ITCH)
    for schema in equity_schemas:
        submit_job(
            client, tracker, "XNAS.ITCH", schema, symbol, date_str,
            [symbol], "raw_symbol", log_path
        )
        time.sleep(pause_seconds)

    # Step 4: Submit options data batch jobs (OPRA.PILLAR, 0DTE only)
    for schema in options_schemas:
        if schema == "definition":
            continue
        submit_job(
            client, tracker, "OPRA.PILLAR", schema, symbol, date_str,
            raw_symbols, "raw_symbol", log_path
        )
        time.sleep(pause_seconds)


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

    # Phase 1: Synchronous setup — stream definitions + filter 0DTE + submit batch jobs
    tracker = load_job_tracker()
    for date_str in dates:
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

    # Phase 2: Poll and download batch data jobs
    while True:
        tracker = load_job_tracker()
        poll_jobs(client, tracker, log_path)
        download_completed_jobs(client, tracker, log_path)

        tracker = load_job_tracker()
        total = len(tracker["jobs"])
        done = sum(1 for v in tracker["jobs"].values() if v["state"] == "downloaded")
        pending = sum(1 for v in tracker["jobs"].values() if v["state"] not in ("downloaded", "expired", "error"))

        log_msg(log_path, f"STATUS: {done}/{total} downloaded, {pending} pending")

        if pending == 0:
            if done > 0:
                log_msg(log_path, "DAEMON complete: all jobs finished")
            break

        log_msg(log_path, f"Sleeping {args.poll_interval}s...")
        time.sleep(args.poll_interval)


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
