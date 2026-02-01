"""
Batch Download Futures MBO + Futures Options MBO Data (0DTE only)

Downloads from GLBX.MDP3 dataset:
- Futures MBO: ES, NQ underlying (using ES.FUT, NQ.FUT parent symbols)
- Futures Options MBO: 0DTE filtered by raw_symbol from definitions
- Instrument Definitions: For 0DTE filtering
- Statistics: Optional

0DTE Parent Symbol Logic (ES):
- 3rd Friday (Monthly): ES.OPT
- Other Fridays (Weekly): EW.OPT, EW1-4.OPT
- Mon-Thu (Daily): E1-E5.OPT + E1A-E5E.OPT variants

Usage:
    # Submit jobs for a date range
    uv run python scripts/batch_download_futures.py submit \
        --start 2026-01-06 --end 2026-01-10 --symbols ES,NQ \
        --log-file logs/futures.log

    # Poll and download completed jobs
    uv run python scripts/batch_download_futures.py poll \
        --log-file logs/futures.log

    # Run both in a loop (background process)
    nohup uv run python scripts/batch_download_futures.py daemon \
        --start 2026-01-06 --end 2026-01-10 --symbols ES,NQ \
        --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &
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

# Job tracking file
JOB_TRACKER_FILE = backend_dir / "logs" / "futures_jobs.json"

# 0DTE parent symbol patterns
WEEKLY_ASSETS = {"EW", "EW1", "EW2", "EW3", "EW4"}
STANDARD_ASSETS = {"ES"}
DAILY_PATTERN = re.compile(r"^E\d")


def is_third_friday(day: datetime) -> bool:
    """Check if day is the 3rd Friday of the month."""
    if day.weekday() != 4:
        return False
    return 15 <= day.day <= 21


def is_quarterly_month(day: datetime) -> bool:
    """Check if day is in a quarterly expiration month (Mar, Jun, Sep, Dec)."""
    return day.month in (3, 6, 9, 12)


def category_for_date(day: datetime) -> str:
    """Categorize date for 0DTE parent symbol selection.
    
    - standard: 3rd Friday of quarterly month (Mar/Jun/Sep/Dec) -> ES.OPT
    - weekly: Any Friday that's NOT standard -> EW family
    - daily: Mon-Thu -> E1-E5 family
    """
    if is_third_friday(day) and is_quarterly_month(day):
        return "standard"
    if day.weekday() == 4:  # Friday (any Friday not in quarterly 3rd Friday)
        return "weekly"
    return "daily"


def parents_for_definition(day: datetime, symbol: str) -> list[str]:
    """Get parent symbols for definition download based on date type."""
    if symbol.upper() != "ES":
        # NQ options have simpler structure
        return [f"{symbol}.OPT"]
    
    category = category_for_date(day)
    if category == "standard":
        return ["ES.OPT"]
    if category == "weekly":
        return ["EW.OPT", "EW1.OPT", "EW2.OPT", "EW3.OPT", "EW4.OPT"]
    # Daily
    parents = []
    for i in range(1, 6):
        parents.append(f"E{i}.OPT")
        for char in ["A", "B", "C", "D", "E"]:
            parents.append(f"E{i}{char}.OPT")
    return parents


def validate_assets(day: datetime, assets: list[str]) -> None:
    """Validate extracted assets match expected pattern for the date."""
    category = category_for_date(day)
    if category == "standard":
        bad = [a for a in assets if a not in STANDARD_ASSETS]
        if bad:
            raise ValueError(f"Unexpected assets for standard (quarterly 3rd Fri) day {day.date()}: {bad}")
        return
    if category == "weekly":
        bad = [a for a in assets if a not in WEEKLY_ASSETS]
        if bad:
            raise ValueError(f"Unexpected assets for weekly (Friday) day {day.date()}: {bad}")
        return
    # Daily (Mon-Thu)
    bad = [a for a in assets if not DAILY_PATTERN.match(a)]
    if bad:
        raise ValueError(f"Unexpected assets for daily (Mon-Thu) day {day.date()}: {bad}")


def target_path_futures(schema: str, symbol: str, date_compact: str) -> Path:
    """Paths for futures MBO data."""
    base = backend_dir / "lake" / "raw" / "source=databento"
    if schema == "mbo":
        out_dir = base / "product_type=future_mbo" / f"symbol={symbol}" / "table=market_by_order_dbn"
        name = f"glbx-mdp3-{date_compact}.mbo.dbn.zst"
    else:
        raise ValueError(f"Unsupported futures schema: {schema}")
    return out_dir / name


def target_path_options(schema: str, symbol: str, date_compact: str) -> Path:
    """Paths for futures options data."""
    base = backend_dir / "lake" / "raw" / "source=databento"
    if schema == "mbo":
        out_dir = base / "product_type=future_option_mbo" / f"symbol={symbol}" / "table=market_by_order_dbn"
        name = f"glbx-mdp3-{date_compact}.mbo.dbn"
    elif schema == "statistics":
        out_dir = base / "product_type=future_option_mbo" / f"symbol={symbol}" / "table=statistics"
        name = f"glbx-mdp3-{date_compact}.statistics.dbn"
    elif schema == "definition":
        out_dir = base / "dataset=definition"
        name = f"glbx-mdp3-{date_compact}.definition.dbn"
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
        # Skip Saturday only (Sunday has ES/NQ globex data)
        if curr.weekday() != 5:
            out.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return out


def definition_path(date_str: str) -> Path:
    date_compact = date_str.replace("-", "")
    return target_path_options("definition", "", date_compact)


def load_0dte_contracts(symbol: str, date_str: str) -> tuple[list[str], list[str]]:
    """Load 0DTE option raw_symbols (contracts) and asset roots from definitions."""
    def_path = definition_path(date_str)
    if not def_path.exists():
        raise FileNotFoundError(f"Definition file missing: {def_path}")
    store = db.DBNStore.from_file(str(def_path))
    df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if df.empty:
        raise ValueError(f"Definition file empty: {def_path}")

    df = df[df["instrument_class"].isin(["C", "P"])].copy()
    df = df[df["underlying"].astype(str).str.startswith(symbol.upper())].copy()
    df = df[df["expiration"].notna()].copy()

    exp_dates = (
        pd.to_datetime(df["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("Etc/GMT+5")
        .dt.date.astype(str)
    )
    df = df[exp_dates == date_str].copy()

    if df.empty:
        raise ValueError(f"No {symbol} 0DTE definitions for {date_str}")

    assets = sorted({str(a).strip() for a in df["asset"].dropna().unique()})
    raw_symbols = sorted({str(s).strip() for s in df["raw_symbol"].dropna().unique()})
    if not raw_symbols:
        raise ValueError(f"No raw_symbols for {symbol} on {date_str}")
    return raw_symbols, assets


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
    product_type: str = "futures",
) -> str | None:
    """Submit a batch job and track it. Returns job_id or None if already exists."""
    job_key = f"{dataset}|{schema}|{symbol}|{date_str}|{product_type}"

    # Check if already submitted
    if job_key in tracker["jobs"]:
        existing = tracker["jobs"][job_key]
        if existing["state"] in ("done", "downloaded"):
            log_msg(log_path, f"SKIP {job_key} already {existing['state']}")
            return None
        log_msg(log_path, f"REUSE {job_key} job={existing['job_id']} state={existing['state']}")
        return existing["job_id"]

    end_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        job = client.batch.submit_job(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=date_str,
            end=end_date,
            stype_in=stype_in,
            delivery="download",
            split_duration="day",
        )
        job_id = job["id"]
        tracker["jobs"][job_key] = {
            "job_id": job_id,
            "dataset": dataset,
            "schema": schema,
            "symbol": symbol,
            "date_str": date_str,
            "product_type": product_type,
            "symbols_count": len(symbols),
            "stype_in": stype_in,
            "state": "submitted",
            "submitted_at": datetime.utcnow().isoformat(),
        }
        save_job_tracker(tracker)
        log_msg(log_path, f"SUBMIT {job_key} job={job_id} symbols={len(symbols)}")
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
        schema = job_info["schema"]
        symbol = job_info["symbol"]
        date_str = job_info["date_str"]
        product_type = job_info.get("product_type", "futures_options")
        date_compact = date_str.replace("-", "")

        # Determine output directory
        if product_type == "futures":
            out_path = target_path_futures(schema, symbol, date_compact)
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

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    options_schemas = [s.strip() for s in args.options_schemas.split(",") if s.strip()]

    for date_str in date_range(args.start, args.end):
        day = datetime.strptime(date_str, "%Y-%m-%d")
        
        for symbol in symbols:
            # 1. Submit FUTURES MBO job (using .FUT parent)
            if args.include_futures:
                submit_job(
                    client, tracker, "GLBX.MDP3", "mbo", symbol, date_str,
                    [f"{symbol}.FUT"], "parent", log_path, product_type="futures"
                )
                time.sleep(args.pause_seconds)

            # 2. Submit OPTIONS DEFINITION job
            definition_parents = parents_for_definition(day, symbol)
            submit_job(
                client, tracker, "GLBX.MDP3", "definition", symbol, date_str,
                definition_parents, "parent", log_path, product_type="futures_options"
            )
            time.sleep(args.pause_seconds)

            # 3. Check if definition is downloaded, then submit 0DTE data jobs
            def_key = f"GLBX.MDP3|definition|{symbol}|{date_str}|futures_options"
            def_info = tracker["jobs"].get(def_key, {})
            if def_info.get("state") == "downloaded":
                try:
                    raw_symbols, assets = load_0dte_contracts(symbol, date_str)
                    if symbol == "ES":
                        validate_assets(day, assets)
                    log_msg(log_path, f"0DTE {symbol} {date_str}: {len(raw_symbols)} contracts")

                    for schema in options_schemas:
                        if schema == "definition":
                            continue
                        submit_job(
                            client, tracker, "GLBX.MDP3", schema, symbol, date_str,
                            raw_symbols, "raw_symbol", log_path, product_type="futures_options"
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

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    options_schemas = [s.strip() for s in args.options_schemas.split(",") if s.strip()]
    dates = date_range(args.start, args.end)

    log_msg(log_path, f"DAEMON start: {len(dates)} dates, {len(symbols)} symbols")

    iteration = 0
    while True:
        iteration += 1
        tracker = load_job_tracker()
        log_msg(log_path, f"--- Iteration {iteration} ---")

        # Phase 1: Submit any missing jobs
        for date_str in dates:
            day = datetime.strptime(date_str, "%Y-%m-%d")
            
            for symbol in symbols:
                # Futures MBO
                if args.include_futures:
                    key = f"GLBX.MDP3|mbo|{symbol}|{date_str}|futures"
                    if key not in tracker["jobs"] or tracker["jobs"][key]["state"] == "error":
                        submit_job(
                            client, tracker, "GLBX.MDP3", "mbo", symbol, date_str,
                            [f"{symbol}.FUT"], "parent", log_path, product_type="futures"
                        )
                        time.sleep(args.pause_seconds)

                # Options definition
                def_key = f"GLBX.MDP3|definition|{symbol}|{date_str}|futures_options"
                if def_key not in tracker["jobs"] or tracker["jobs"][def_key]["state"] == "error":
                    definition_parents = parents_for_definition(day, symbol)
                    submit_job(
                        client, tracker, "GLBX.MDP3", "definition", symbol, date_str,
                        definition_parents, "parent", log_path, product_type="futures_options"
                    )
                    time.sleep(args.pause_seconds)

        # Phase 2: Poll and download
        tracker = load_job_tracker()
        poll_jobs(client, tracker, log_path)
        download_completed_jobs(client, tracker, log_path)

        # Phase 3: Submit 0DTE data jobs for completed definitions
        tracker = load_job_tracker()
        for date_str in dates:
            day = datetime.strptime(date_str, "%Y-%m-%d")
            
            for symbol in symbols:
                def_key = f"GLBX.MDP3|definition|{symbol}|{date_str}|futures_options"
                def_info = tracker["jobs"].get(def_key, {})
                if def_info.get("state") != "downloaded":
                    continue

                try:
                    raw_symbols, assets = load_0dte_contracts(symbol, date_str)
                    if symbol == "ES":
                        validate_assets(day, assets)
                    for schema in options_schemas:
                        if schema == "definition":
                            continue
                        key = f"GLBX.MDP3|{schema}|{symbol}|{date_str}|futures_options"
                        if key not in tracker["jobs"] or tracker["jobs"][key]["state"] == "error":
                            submit_job(
                                client, tracker, "GLBX.MDP3", schema, symbol, date_str,
                                raw_symbols, "raw_symbol", log_path, product_type="futures_options"
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
    parser = argparse.ArgumentParser(description="Batch download futures MBO + futures options MBO data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit batch jobs")
    submit_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    submit_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    submit_parser.add_argument("--symbols", default="ES", help="Comma-separated symbols (ES,NQ)")
    submit_parser.add_argument("--include-futures", action="store_true", help="Also download futures MBO")
    submit_parser.add_argument("--options-schemas", default="definition,mbo,statistics", help="Options schemas")
    submit_parser.add_argument("--pause-seconds", type=int, default=5, help="Pause between submissions")
    submit_parser.add_argument("--log-file", required=True, help="Log file path")

    # Poll command
    poll_parser = subparsers.add_parser("poll", help="Poll jobs and download completed")
    poll_parser.add_argument("--log-file", required=True, help="Log file path")

    # Daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Run submit + poll in loop until complete")
    daemon_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    daemon_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    daemon_parser.add_argument("--symbols", default="ES", help="Comma-separated symbols (ES,NQ)")
    daemon_parser.add_argument("--include-futures", action="store_true", help="Also download futures MBO")
    daemon_parser.add_argument("--options-schemas", default="definition,mbo,statistics", help="Options schemas")
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
