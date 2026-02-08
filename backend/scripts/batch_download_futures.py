"""
Batch-download GLBX futures + futures options with strict flat-file day slices.

Pipeline per session date and symbol (ES only):
1) Download futures definitions (`ES.FUT`, schema=definition)
2) Download futures daily OHLCV (`ES.FUT`, schema=ohlcv-1d) for front-month mapping
3) Resolve active futures contract from highest day volume (front-month-like selection)
4) Download options definitions (`ALL_SYMBOLS`, schema=definition)
5) Filter options to 0DTE contracts where underlying == active futures contract
6) Submit options data jobs (`mbo`, `statistics`) using `stype_in=raw_symbol`

All batch requests are forced to Databento flat-file delivery with one-day requests:
- `delivery="download"`
- `split_duration="day"`
- `start=<session_date>`, `end=<session_date + 1 day>`
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

JOB_TRACKER_FILE = backend_dir / "logs" / "futures_jobs.json"

FUTURES_DATASET = "GLBX.MDP3"
SUPPORTED_SYMBOLS = {"ES"}
SUPPORTED_OPTIONS_SCHEMAS = {"definition", "mbo", "statistics"}

FLAT_FILE_DELIVERY = "download"
FLAT_FILE_SPLIT_DURATION = "day"

PRODUCT_FUTURES = "futures"
PRODUCT_FUTURES_DEFINITION = "futures_definition"
PRODUCT_FUTURES_CONTRACT_MAP = "futures_contract_map"
PRODUCT_FUTURES_OPTIONS_DEFINITION = "futures_options_definition"
PRODUCT_FUTURES_OPTIONS = "futures_options"


def parse_symbols(raw_symbols: str) -> list[str]:
    symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    unsupported = sorted(set(symbols) - SUPPORTED_SYMBOLS)
    if unsupported:
        allowed = ",".join(sorted(SUPPORTED_SYMBOLS))
        raise ValueError(
            f"Unsupported symbols for this 0DTE pipeline: {unsupported}. Allowed: {allowed}"
        )
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


def target_path_futures(schema: str, symbol: str, date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    if schema == "mbo":
        out_dir = base / "product_type=future_mbo" / f"symbol={symbol}" / "table=market_by_order_dbn"
        name = f"glbx-mdp3-{date_compact}.mbo.dbn.zst"
    elif schema == "ohlcv-1d":
        out_dir = base / "product_type=future_contract_map" / f"symbol={symbol}" / "table=ohlcv_1d"
        name = f"glbx-mdp3-{date_compact}.ohlcv-1d.dbn"
    else:
        raise ValueError(f"Unsupported futures schema: {schema}")
    return out_dir / name


def target_path_futures_definition(date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    out_dir = base / "dataset=definition" / "venue=glbx" / "type=futures"
    return out_dir / f"glbx-mdp3-{date_compact}.definition.dbn"


def target_path_options_definition(date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    out_dir = base / "dataset=definition" / "venue=glbx" / "type=futures_options"
    return out_dir / f"glbx-mdp3-{date_compact}.definition.dbn"


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

    out: list[str] = []
    curr = start_dt
    while curr <= end_dt:
        # Globex has a Sunday session. Only skip Saturday.
        if curr.weekday() != 5:
            out.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return out


def _best_matching_download(base_dir: Path, canonical_name: str) -> Path:
    direct = base_dir / canonical_name
    if direct.exists():
        return direct

    direct_zst = Path(str(direct) + ".zst")
    if direct_zst.exists():
        return direct_zst

    nested = [p for p in base_dir.glob(f"**/{canonical_name}*") if p.is_file()]
    if not nested:
        raise FileNotFoundError(
            f"Missing file {canonical_name} (or .zst) under {base_dir}"
        )

    def sort_key(p: Path) -> tuple[int, float, int]:
        prefer_uncompressed = 0 if not p.name.endswith(".zst") else 1
        st = p.stat()
        return (prefer_uncompressed, -st.st_mtime, -st.st_size)

    return sorted(nested, key=sort_key)[0]


def resolve_futures_definition_file(date_str: str) -> Path:
    date_compact = date_str.replace("-", "")
    canonical = target_path_futures_definition(date_compact)
    return _best_matching_download(canonical.parent, canonical.name)


def resolve_options_definition_file(date_str: str) -> Path:
    date_compact = date_str.replace("-", "")
    canonical = target_path_options_definition(date_compact)
    return _best_matching_download(canonical.parent, canonical.name)


def resolve_futures_contract_map_file(symbol: str, date_str: str) -> Path:
    date_compact = date_str.replace("-", "")
    canonical = target_path_futures("ohlcv-1d", symbol, date_compact)
    return _best_matching_download(canonical.parent, canonical.name)


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def select_active_futures_contract_from_frames(
    symbol: str,
    session_date: str,
    definitions_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> tuple[str, str]:
    """Resolve the active contract for a session date using daily volume."""
    _require_columns(definitions_df, {"raw_symbol", "instrument_class", "expiration"}, "futures definitions")

    session_day = datetime.strptime(session_date, "%Y-%m-%d").date()

    defs = definitions_df.copy()
    defs["raw_symbol"] = defs["raw_symbol"].astype(str).str.strip()
    defs = defs[defs["instrument_class"].astype(str) == "F"].copy()
    defs = defs[defs["raw_symbol"].str.startswith(symbol.upper())].copy()
    defs = defs[defs["expiration"].notna()].copy()

    if defs.empty:
        raise ValueError(f"No futures definitions for {symbol} on {session_date}")

    defs["expiration_ts_utc"] = pd.to_datetime(defs["expiration"].astype("int64"), utc=True)
    defs["expiration_date_utc"] = defs["expiration_ts_utc"].dt.date

    # Keep contracts that have not expired as of the session date.
    defs = defs[defs["expiration_date_utc"] >= session_day].copy()
    if defs.empty:
        raise ValueError(f"No non-expired futures contracts for {symbol} on {session_date}")

    def_expiries = defs[["raw_symbol", "expiration_ts_utc"]].drop_duplicates("raw_symbol")
    expiry_lookup = {
        row.raw_symbol: row.expiration_ts_utc
        for row in def_expiries.itertuples(index=False)
    }

    symbol_col = "symbol" if "symbol" in ohlcv_df.columns else "raw_symbol"
    _require_columns(ohlcv_df, {symbol_col, "volume"}, "futures ohlcv-1d")

    ohlcv = ohlcv_df.copy()
    ohlcv[symbol_col] = ohlcv[symbol_col].astype(str).str.strip()
    ohlcv = ohlcv[ohlcv[symbol_col].isin(set(expiry_lookup))].copy()
    ohlcv["volume"] = pd.to_numeric(ohlcv["volume"], errors="coerce").fillna(0)

    if not ohlcv.empty and (ohlcv["volume"] > 0).any():
        grouped = ohlcv.groupby(symbol_col, as_index=False)["volume"].sum()
        top_volume = grouped["volume"].max()
        top_symbols = sorted(grouped[grouped["volume"] == top_volume][symbol_col].tolist())
        if len(top_symbols) == 1:
            return top_symbols[0], "max_daily_volume"

        # Tie-break by nearest expiration, then lexicographic order for determinism.
        top_symbols.sort(key=lambda s: (expiry_lookup[s], s))
        return top_symbols[0], "max_daily_volume_tie_breaker"

    # Fallback if OHLCV has no volume for the date.
    fallback = sorted(expiry_lookup, key=lambda s: (expiry_lookup[s], s))[0]
    return fallback, "nearest_expiration_fallback"


def filter_0dte_futures_option_raw_symbols_from_definitions(
    definitions_df: pd.DataFrame,
    active_underlying_raw_symbol: str,
    session_date: str,
) -> list[str]:
    """Return 0DTE futures option raw symbols for one active underlying contract."""
    _require_columns(
        definitions_df,
        {"raw_symbol", "underlying", "instrument_class", "expiration"},
        "futures options definitions",
    )

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


def load_active_futures_contract(symbol: str, date_str: str) -> tuple[str, str]:
    futures_def_path = resolve_futures_definition_file(date_str)
    contract_map_path = resolve_futures_contract_map_file(symbol, date_str)

    futures_def_store = db.DBNStore.from_file(str(futures_def_path))
    futures_def_df = futures_def_store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if futures_def_df.empty:
        raise ValueError(f"Futures definition file empty: {futures_def_path}")

    contract_map_store = db.DBNStore.from_file(str(contract_map_path))
    contract_map_df = contract_map_store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if contract_map_df.empty:
        raise ValueError(f"Futures contract map file empty: {contract_map_path}")

    return select_active_futures_contract_from_frames(symbol, date_str, futures_def_df, contract_map_df)


def load_0dte_option_raw_symbols(symbol: str, active_contract: str, date_str: str) -> list[str]:
    options_def_path = resolve_options_definition_file(date_str)

    store = db.DBNStore.from_file(str(options_def_path))
    df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if df.empty:
        raise ValueError(f"Options definition file empty: {options_def_path}")

    return filter_0dte_futures_option_raw_symbols_from_definitions(
        definitions_df=df,
        active_underlying_raw_symbol=active_contract,
        session_date=date_str,
    )


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

    if key in tracker["jobs"]:
        existing = tracker["jobs"][key]
        if existing["state"] in ("done", "downloaded"):
            log_msg(log_path, f"SKIP {key} already {existing['state']}")
            return None
        if existing["state"] != "error":
            log_msg(log_path, f"REUSE {key} job={existing['job_id']} state={existing['state']}")
            return existing["job_id"]
        log_msg(log_path, f"RESUBMIT {key} (was error)")

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
        tracker["jobs"][key] = {
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
        log_msg(
            log_path,
            f"SUBMIT {key} job={job_id} symbols={len(symbols)} delivery={FLAT_FILE_DELIVERY} split_duration={FLAT_FILE_SPLIT_DURATION}",
        )
        return job_id
    except Exception as e:
        log_msg(log_path, f"ERROR submit {key}: {e}")
        return None


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
        return target_path_futures(schema, symbol, date_compact)
    if product_type == PRODUCT_FUTURES_CONTRACT_MAP:
        return target_path_futures(schema, symbol, date_compact)
    if product_type == PRODUCT_FUTURES_DEFINITION:
        return target_path_futures_definition(date_compact)
    if product_type == PRODUCT_FUTURES_OPTIONS_DEFINITION:
        return target_path_options_definition(date_compact)
    if product_type == PRODUCT_FUTURES_OPTIONS:
        return target_path_options_data(schema, symbol, date_compact)
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


def _submit_prereq_jobs_for_day(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    date_str: str,
    include_futures: bool,
    log_path: Path,
    pause_seconds: int,
) -> None:
    if include_futures:
        submit_job(
            client=client,
            tracker=tracker,
            dataset=FUTURES_DATASET,
            schema="mbo",
            symbol=symbol,
            date_str=date_str,
            symbols=[f"{symbol}.FUT"],
            stype_in="parent",
            log_path=log_path,
            product_type=PRODUCT_FUTURES,
        )
        time.sleep(pause_seconds)

    submit_job(
        client=client,
        tracker=tracker,
        dataset=FUTURES_DATASET,
        schema="definition",
        symbol=symbol,
        date_str=date_str,
        symbols=[f"{symbol}.FUT"],
        stype_in="parent",
        log_path=log_path,
        product_type=PRODUCT_FUTURES_DEFINITION,
    )
    time.sleep(pause_seconds)

    submit_job(
        client=client,
        tracker=tracker,
        dataset=FUTURES_DATASET,
        schema="ohlcv-1d",
        symbol=symbol,
        date_str=date_str,
        symbols=[f"{symbol}.FUT"],
        stype_in="parent",
        log_path=log_path,
        product_type=PRODUCT_FUTURES_CONTRACT_MAP,
    )
    time.sleep(pause_seconds)

    # Pull all options definitions, then filter to active-contract 0DTE locally.
    submit_job(
        client=client,
        tracker=tracker,
        dataset=FUTURES_DATASET,
        schema="definition",
        symbol=symbol,
        date_str=date_str,
        symbols=["ALL_SYMBOLS"],
        stype_in="raw_symbol",
        log_path=log_path,
        product_type=PRODUCT_FUTURES_OPTIONS_DEFINITION,
    )
    time.sleep(pause_seconds)


def _submit_0dte_option_jobs_if_ready(
    client: db.Historical,
    tracker: dict[str, Any],
    symbol: str,
    date_str: str,
    options_schemas: list[str],
    log_path: Path,
    pause_seconds: int,
) -> None:
    futures_def_key = job_key(FUTURES_DATASET, "definition", symbol, date_str, PRODUCT_FUTURES_DEFINITION)
    contract_map_key = job_key(FUTURES_DATASET, "ohlcv-1d", symbol, date_str, PRODUCT_FUTURES_CONTRACT_MAP)
    options_def_key = job_key(FUTURES_DATASET, "definition", symbol, date_str, PRODUCT_FUTURES_OPTIONS_DEFINITION)

    prereq_states = {
        "futures_definition": tracker["jobs"].get(futures_def_key, {}).get("state"),
        "futures_contract_map": tracker["jobs"].get(contract_map_key, {}).get("state"),
        "options_definition": tracker["jobs"].get(options_def_key, {}).get("state"),
    }

    if any(state != "downloaded" for state in prereq_states.values()):
        return

    try:
        active_contract, active_reason = load_active_futures_contract(symbol, date_str)
        raw_symbols = load_0dte_option_raw_symbols(symbol, active_contract, date_str)
        log_msg(
            log_path,
            f"0DTE {symbol} {date_str}: active_contract={active_contract} reason={active_reason} contracts={len(raw_symbols)}",
        )
    except Exception as e:
        log_msg(log_path, f"SKIP 0DTE {symbol} {date_str}: {e}")
        return

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
            symbols=raw_symbols,
            stype_in="raw_symbol",
            log_path=log_path,
            product_type=PRODUCT_FUTURES_OPTIONS,
        )
        time.sleep(pause_seconds)


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
            _submit_prereq_jobs_for_day(
                client=client,
                tracker=tracker,
                symbol=symbol,
                date_str=date_str,
                include_futures=args.include_futures,
                log_path=log_path,
                pause_seconds=args.pause_seconds,
            )
            _submit_0dte_option_jobs_if_ready(
                client=client,
                tracker=tracker,
                symbol=symbol,
                date_str=date_str,
                options_schemas=options_schemas,
                log_path=log_path,
                pause_seconds=args.pause_seconds,
            )


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
    options_schemas = parse_options_schemas(args.options_schemas)
    dates = date_range(args.start, args.end)

    log_msg(log_path, f"DAEMON start: {len(dates)} dates, {len(symbols)} symbols")

    iteration = 0
    while True:
        iteration += 1
        tracker = load_job_tracker()
        log_msg(log_path, f"--- Iteration {iteration} ---")

        # Phase 1: enforce one-day prerequisite flat-file jobs per date.
        for date_str in dates:
            for symbol in symbols:
                _submit_prereq_jobs_for_day(
                    client=client,
                    tracker=tracker,
                    symbol=symbol,
                    date_str=date_str,
                    include_futures=args.include_futures,
                    log_path=log_path,
                    pause_seconds=args.pause_seconds,
                )

        # Phase 2: poll and download completed files.
        tracker = load_job_tracker()
        poll_jobs(client, tracker, log_path)
        download_completed_jobs(client, tracker, log_path)

        # Phase 3: once prerequisites are downloaded, submit active-contract 0DTE option jobs.
        tracker = load_job_tracker()
        for date_str in dates:
            for symbol in symbols:
                _submit_0dte_option_jobs_if_ready(
                    client=client,
                    tracker=tracker,
                    symbol=symbol,
                    date_str=date_str,
                    options_schemas=options_schemas,
                    log_path=log_path,
                    pause_seconds=args.pause_seconds,
                )

        # Phase 4: completion check.
        tracker = load_job_tracker()
        total = len(tracker["jobs"])
        done = sum(1 for v in tracker["jobs"].values() if v["state"] == "downloaded")
        pending = sum(
            1
            for v in tracker["jobs"].values()
            if v["state"] not in ("downloaded", "expired", "error")
        )

        log_msg(log_path, f"STATUS: {done}/{total} downloaded, {pending} pending")
        if pending == 0 and done > 0:
            log_msg(log_path, "DAEMON complete: all jobs finished")
            break

        log_msg(log_path, f"Sleeping {args.poll_interval}s...")
        time.sleep(args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch download futures + futures options (flat-file day-by-day active-contract 0DTE pipeline)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit batch jobs")
    submit_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    submit_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    submit_parser.add_argument("--symbols", default="ES", help="Comma-separated symbols (ES only)")
    submit_parser.add_argument("--include-futures", action="store_true", help="Also download futures MBO")
    submit_parser.add_argument("--options-schemas", default="mbo,statistics", help="Options schemas")
    submit_parser.add_argument("--pause-seconds", type=int, default=5, help="Pause between submissions")
    submit_parser.add_argument("--log-file", required=True, help="Log file path")

    poll_parser = subparsers.add_parser("poll", help="Poll jobs and download completed")
    poll_parser.add_argument("--log-file", required=True, help="Log file path")

    daemon_parser = subparsers.add_parser("daemon", help="Run submit + poll in loop until complete")
    daemon_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    daemon_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    daemon_parser.add_argument("--symbols", default="ES", help="Comma-separated symbols (ES only)")
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
