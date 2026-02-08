"""
Batch Download Equity + Equity Options Data (0DTE only, flat-file day slices)

Downloads from two datasets:
- XNAS.ITCH: Underlying equity MBO data
- OPRA.PILLAR: Equity options data via Databento CMBP-1 schema (0DTE filtered)

All requests are strict batch flat-file requests:
- delivery=\"download\"
- split_duration=\"day\"
- start=<session_date>, end=<session_date + 1 day>

For equity options (QQQ), 0DTE filtering is by:
- instrument_class in {C, P}
- underlying == QQQ
- expiration UTC date == session date

This differs from futures options, where 0DTE is filtered against the active
underlying futures contract for the date.

Usage:
    # Submit jobs for a date range
    uv run python scripts/batch_download_equities.py submit \
        --start 2026-01-06 --end 2026-01-10 --symbols QQQ \
        --log-file logs/equities.log

    # Poll and download completed jobs
    uv run python scripts/batch_download_equities.py poll \
        --log-file logs/equities.log

    # Run both in a loop (background process)
    nohup uv run python scripts/batch_download_equities.py daemon \
        --start 2026-01-06 --end 2026-01-10 --symbols QQQ \
        --log-file logs/equities.log > logs/equities_daemon.out 2>&1 &
"""
import argparse
import json
import os
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

# Job tracking file
JOB_TRACKER_FILE = backend_dir / "logs" / "equity_options_jobs.json"

SUPPORTED_SYMBOLS = {"QQQ"}
SUPPORTED_EQUITY_SCHEMAS = {"mbo", "definition"}
SUPPORTED_OPTIONS_SCHEMAS = {"definition", "cmbp-1", "statistics"}

FLAT_FILE_DELIVERY = "download"
FLAT_FILE_SPLIT_DURATION = "day"


def parse_symbols(raw_symbols: str) -> list[str]:
    symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    unsupported = sorted(set(symbols) - SUPPORTED_SYMBOLS)
    if unsupported:
        allowed = ",".join(sorted(SUPPORTED_SYMBOLS))
        raise ValueError(
            f"Unsupported symbols for this equity 0DTE pipeline: {unsupported}. Allowed: {allowed}"
        )
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


def target_path_equity(schema: str, symbol: str, date_compact: str) -> Path:
    """Paths for underlying equity MBO data (XNAS.ITCH)"""
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


def target_path_options(schema: str, symbol: str, date_compact: str) -> Path:
    """Paths for equity options data (OPRA.PILLAR)"""
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


def load_job_tracker() -> dict[str, Any]:
    if JOB_TRACKER_FILE.exists():
        with open(JOB_TRACKER_FILE) as f:
            return json.load(f)
    return {"jobs": {}, "pending_downloads": []}


def save_job_tracker(tracker: dict[str, Any]) -> None:
    JOB_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(JOB_TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2, default=str)


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


def resolve_definition_file_options(symbol: str, date_str: str) -> Path:
    """Resolve OPRA options definition file path for a given session date.

    Databento batch downloads place files under a job-id subdirectory (e.g.
    `.../venue=opra/OPRA-YYYYMMDD-XXXX/...`). Some historical runs also leave a
    convenient root-level copy at `.../venue=opra/opra-pillar-YYYYMMDD.definition.dbn`.

    This resolver prefers the root-level file when present, otherwise it falls
    back to the newest matching file under job directories.
    """
    date_compact = date_str.replace("-", "")
    base = backend_dir / "lake" / "raw" / "source=databento" / "dataset=definition" / "venue=opra"

    # 1) Prefer canonical root-level definition file(s)
    root_dbn = base / f"opra-pillar-{date_compact}.definition.dbn"
    if root_dbn.exists():
        return root_dbn
    root_dbn_zst = base / f"opra-pillar-{date_compact}.definition.dbn.zst"
    if root_dbn_zst.exists():
        return root_dbn_zst

    # 2) Fall back to any batch job output (nested under OPRA-*/ directories)
    nested = [p for p in base.glob(f"**/opra-pillar-{date_compact}.definition.dbn*") if p.is_file()]
    if not nested:
        raise FileNotFoundError(
            f"Definition file missing for {symbol} {date_str}: "
            f"expected {root_dbn} (or .zst), or a nested batch output under {base}"
        )

    def sort_key(p: Path) -> tuple[int, float, int]:
        # Prefer uncompressed .dbn over .dbn.zst, then newest, then largest.
        prefer = 0 if p.name.endswith(".definition.dbn") else 1
        st = p.stat()
        return (prefer, -st.st_mtime, -st.st_size)

    return sorted(nested, key=sort_key)[0]


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

    # OPRA expiration is date-granularity and represented at UTC midnight.
    exp_dates = pd.to_datetime(df["expiration"].astype("int64"), utc=True).dt.date.astype(str)
    df = df[exp_dates == date_str].copy()

    if df.empty:
        raise ValueError(f"No {symbol} 0DTE definitions for {date_str}")

    raw_symbols = sorted({str(s).strip() for s in df["raw_symbol"].dropna().unique() if str(s).strip()})
    if not raw_symbols:
        raise ValueError(f"No 0DTE raw_symbols for {symbol} on {date_str}")
    return raw_symbols


def load_0dte_assets(symbol: str, date_str: str) -> list[str]:
    """Load 0DTE option raw symbols from downloaded OPRA definitions."""
    def_path = resolve_definition_file_options(symbol, date_str)
    store = db.DBNStore.from_file(str(def_path))
    df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if df.empty:
        raise ValueError(f"Definition file empty: {def_path}")
    return filter_0dte_equity_option_raw_symbols_from_definitions(df, symbol, date_str)


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
    """Submit a batch job and track it. Returns job_id or None if already exists."""
    job_key = f"{dataset}|{schema}|{symbol}|{date_str}"

    # Check if already submitted
    if job_key in tracker["jobs"]:
        existing = tracker["jobs"][job_key]
        if existing["state"] in ("done", "downloaded"):
            log_msg(log_path, f"SKIP {job_key} already {existing['state']}")
            return None
        # Resubmit on error, otherwise reuse existing job
        if existing["state"] != "error":
            log_msg(log_path, f"REUSE {job_key} job={existing['job_id']} state={existing['state']}")
            return existing["job_id"]
        log_msg(log_path, f"RESUBMIT {job_key} (was error)")

    end_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

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


def poll_jobs(client: db.Historical, tracker: dict, log_path: Path) -> int:
    """Poll all pending jobs and update states. Returns count of done jobs ready for download."""
    pending_keys = [
        k for k, v in tracker["jobs"].items()
        if v["state"] not in ("done", "downloaded", "expired", "error")
    ]

    if not pending_keys:
        return 0

    # Get all jobs from API
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
    """Download all completed jobs. Returns count of successful downloads."""
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

        # Determine output directory
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


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit batch jobs for all dates and symbols."""
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
            # 1. Submit EQUITY MBO jobs (XNAS.ITCH)
            for schema in equity_schemas:
                submit_job(
                    client, tracker, "XNAS.ITCH", schema, symbol, date_str,
                    [symbol], "raw_symbol", log_path
                )
                time.sleep(args.pause_seconds)

            # 2. Submit OPTIONS DEFINITION job (OPRA.PILLAR)
            parent = f"{symbol}.OPT"
            submit_job(
                client, tracker, "OPRA.PILLAR", "definition", symbol, date_str,
                [parent], "parent", log_path
            )
            time.sleep(args.pause_seconds)

            # 3. Check if definition is downloaded, then submit 0DTE data jobs
            def_key = f"OPRA.PILLAR|definition|{symbol}|{date_str}"
            def_info = tracker["jobs"].get(def_key, {})
            if def_info.get("state") == "downloaded":
                try:
                    raw_symbols = load_0dte_assets(symbol, date_str)
                    log_msg(log_path, f"0DTE {symbol} {date_str}: {len(raw_symbols)} contracts")

                    for schema in options_schemas:
                        if schema == "definition":
                            continue
                        submit_job(
                            client, tracker, "OPRA.PILLAR", schema, symbol, date_str,
                            raw_symbols, "raw_symbol", log_path
                        )
                        time.sleep(args.pause_seconds)
                except Exception as e:
                    log_msg(log_path, f"SKIP 0DTE {symbol} {date_str}: {e}")


def cmd_poll(args: argparse.Namespace) -> None:
    """Poll job status and download completed jobs."""
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
    """Run submit + poll in a loop until all jobs complete."""
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

    iteration = 0
    while True:
        iteration += 1
        tracker = load_job_tracker()
        log_msg(log_path, f"--- Iteration {iteration} ---")

        # Phase 1: Submit any missing jobs
        for date_str in dates:
            for symbol in symbols:
                # Equity MBO
                for schema in equity_schemas:
                    key = f"XNAS.ITCH|{schema}|{symbol}|{date_str}"
                    if key not in tracker["jobs"] or tracker["jobs"][key]["state"] == "error":
                        submit_job(
                            client, tracker, "XNAS.ITCH", schema, symbol, date_str,
                            [symbol], "raw_symbol", log_path
                        )
                        time.sleep(args.pause_seconds)

                # Options definition
                def_key = f"OPRA.PILLAR|definition|{symbol}|{date_str}"
                if def_key not in tracker["jobs"] or tracker["jobs"][def_key]["state"] == "error":
                    parent = f"{symbol}.OPT"
                    submit_job(
                        client, tracker, "OPRA.PILLAR", "definition", symbol, date_str,
                        [parent], "parent", log_path
                    )
                    time.sleep(args.pause_seconds)

        # Phase 2: Poll and download
        tracker = load_job_tracker()
        poll_jobs(client, tracker, log_path)
        download_completed_jobs(client, tracker, log_path)

        # Phase 3: Submit 0DTE data jobs for completed definitions
        tracker = load_job_tracker()
        for date_str in dates:
            for symbol in symbols:
                def_key = f"OPRA.PILLAR|definition|{symbol}|{date_str}"
                def_info = tracker["jobs"].get(def_key, {})
                if def_info.get("state") != "downloaded":
                    continue

                try:
                    raw_symbols = load_0dte_assets(symbol, date_str)

                    for schema in options_schemas:
                        if schema == "definition":
                            continue
                        key = f"OPRA.PILLAR|{schema}|{symbol}|{date_str}"
                        if key not in tracker["jobs"] or tracker["jobs"][key]["state"] == "error":
                            submit_job(
                                client, tracker, "OPRA.PILLAR", schema, symbol, date_str,
                                raw_symbols, "raw_symbol", log_path
                            )
                            time.sleep(args.pause_seconds)
                except Exception as e:
                    log_msg(log_path, f"SKIP 0DTE {symbol} {date_str}: {e}")

        # Phase 4: Check completion
        tracker = load_job_tracker()
        total = len(tracker["jobs"])
        done = sum(1 for v in tracker["jobs"].values() if v["state"] == "downloaded")
        pending = sum(1 for v in tracker["jobs"].values() if v["state"] not in ("downloaded", "expired", "error"))

        log_msg(log_path, f"STATUS: {done}/{total} downloaded, {pending} pending")

        if pending == 0 and done > 0:
            log_msg(log_path, "DAEMON complete: all jobs finished")
            break

        # Wait before next iteration
        log_msg(log_path, f"Sleeping {args.poll_interval}s...")
        time.sleep(args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch download equity + equity options data (flat-file day-by-day 0DTE pipeline)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit batch jobs")
    submit_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    submit_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    submit_parser.add_argument("--symbols", default="QQQ", help="Comma-separated symbols (QQQ only)")
    submit_parser.add_argument("--equity-schemas", default="mbo", help="Equity schemas (mbo,definition)")
    submit_parser.add_argument("--options-schemas", default="cmbp-1,statistics", help="Options schemas")
    submit_parser.add_argument("--pause-seconds", type=int, default=5, help="Pause between submissions")
    submit_parser.add_argument("--log-file", required=True, help="Log file path")

    # Poll command
    poll_parser = subparsers.add_parser("poll", help="Poll jobs and download completed")
    poll_parser.add_argument("--log-file", required=True, help="Log file path")

    # Daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Run submit + poll in loop until complete")
    daemon_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    daemon_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    daemon_parser.add_argument("--symbols", default="QQQ", help="Comma-separated symbols (QQQ only)")
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
