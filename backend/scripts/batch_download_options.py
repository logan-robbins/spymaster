import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import databento as db
import pandas as pd
from databento.common.enums import PriceType
from dotenv import load_dotenv

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
load_dotenv(backend_dir / ".env")

WEEKLY_ASSETS = {"EW", "EW1", "EW2", "EW3", "EW4"}
STANDARD_ASSETS = {"ES"}
DAILY_PATTERN = re.compile(r"^E\d")


def is_third_friday(day: datetime) -> bool:
    if day.weekday() != 4:
        return False
    return 15 <= day.day <= 21


def category_for_date(day: datetime) -> str:
    if is_third_friday(day):
        return "standard"
    if day.weekday() == 4:
        return "weekly"
    return "daily"


def parents_for_definition(day: datetime) -> list[str]:
    category = category_for_date(day)
    if category == "standard":
        return ["ES.OPT"]
    if category == "weekly":
        return ["EW.OPT", "EW1.OPT", "EW2.OPT", "EW3.OPT", "EW4.OPT"]
    parents = []
    for i in range(1, 6):
        parents.append(f"E{i}.OPT")
        for char in ["A", "B", "C", "D", "E"]:
            parents.append(f"E{i}{char}.OPT")
    return parents


def target_path(schema: str, date_compact: str) -> Path:
    base = backend_dir / "lake" / "raw" / "source=databento"
    if schema == "mbo":
        out_dir = base / "product_type=future_option_mbo" / "symbol=ES" / "table=market_by_order_dbn"
        name = f"glbx-mdp3-{date_compact}.mbo.dbn"
    elif schema == "statistics":
        out_dir = base / "product_type=future_option_mbo" / "symbol=ES" / "table=statistics"
        name = f"glbx-mdp3-{date_compact}.statistics.dbn"
    elif schema == "definition":
        out_dir = base / "dataset=definition"
        name = f"glbx-mdp3-{date_compact}.definition.dbn"
    else:
        raise ValueError(f"Unsupported schema: {schema}")
    return out_dir / name


def normalize_symbols(symbols: object) -> tuple[str, ...]:
    if isinstance(symbols, str):
        return tuple(sorted([symbols]))
    if isinstance(symbols, list):
        return tuple(sorted([str(s) for s in symbols]))
    return tuple()


def find_existing_job(
    client: db.Historical,
    schema: str,
    date_str: str,
    end_date: str,
    parents: list[str],
) -> dict | None:
    jobs = client.batch.list_jobs(since=date_str)
    target_symbols = normalize_symbols(parents)
    for job in jobs:
        if job.get("dataset") != "GLBX.MDP3":
            continue
        if job.get("schema") != schema:
            continue
        if job.get("start") != date_str:
            continue
        if job.get("end") != end_date:
            continue
        job_symbols = normalize_symbols(job.get("symbols"))
        if job_symbols != target_symbols:
            continue
        return job
    return None


def definition_path(date_str: str) -> Path:
    date_compact = date_str.replace("-", "")
    return target_path("definition", date_compact)


def load_assets_for_date(date_str: str) -> list[str]:
    def_path = definition_path(date_str)
    if not def_path.exists():
        raise FileNotFoundError(f"Definition file missing: {def_path}")
    store = db.DBNStore.from_file(str(def_path))
    df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
    if df.empty:
        raise ValueError(f"Definition file empty: {def_path}")
    df = df[df["instrument_class"].isin(["C", "P"])].copy()
    df = df[df["underlying"].astype(str).str.startswith("ES")].copy()
    df = df[df["expiration"].notna()].copy()
    exp_dates = (
        pd.to_datetime(df["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("Etc/GMT+5")
        .dt.date.astype(str)
    )
    df = df[exp_dates == date_str].copy()
    if df.empty:
        raise ValueError(f"No ES 0DTE definitions for {date_str}")
    assets = sorted({str(a).strip() for a in df["asset"].dropna().unique()})
    if not assets:
        raise ValueError(f"No assets for {date_str}")
    return assets


def validate_assets(day: datetime, assets: list[str]) -> None:
    category = category_for_date(day)
    if category == "standard":
        bad = [a for a in assets if a not in STANDARD_ASSETS]
        if bad:
            raise ValueError(f"Unexpected assets for standard day {day.date()}: {bad}")
        return
    if category == "weekly":
        bad = [a for a in assets if a not in WEEKLY_ASSETS]
        if bad:
            raise ValueError(f"Unexpected assets for weekly day {day.date()}: {bad}")
        return
    bad = [a for a in assets if not DAILY_PATTERN.match(a)]
    if bad:
        raise ValueError(f"Unexpected assets for daily day {day.date()}: {bad}")


def date_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("end date before start date")
    out = []
    curr = start_dt
    while curr <= end_dt:
        if curr.weekday() != 5:
            out.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return out


def log_job(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def submit_request(
    client: db.Historical,
    schema: str,
    date_str: str,
    parents: list[str],
    pause_seconds: int,
    log_path: Path,
) -> None:
    end_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    existing = find_existing_job(client, schema, date_str, end_date, parents)
    if existing is not None:
        job_id = existing.get("id")
        state = existing.get("state") or existing.get("status")
        log_job(
            log_path,
            f"{datetime.utcnow().isoformat()}Z reuse {date_str} {schema} job={job_id} state={state} parents={len(parents)}",
        )
        print(f"reuse {schema} {date_str} job={job_id} state={state}", flush=True)
        return

    job = client.batch.submit_job(
        dataset="GLBX.MDP3",
        symbols=parents,
        schema=schema,
        start=date_str,
        end=end_date,
        stype_in="parent",
        delivery="download",
    )
    job_id = job["id"]
    log_job(
        log_path,
        f"{datetime.utcnow().isoformat()}Z submit {date_str} {schema} job={job_id} parents={len(parents)}",
    )
    print(f"submit {schema} {date_str} job={job_id}", flush=True)
    time.sleep(pause_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--schemas", default="mbo")
    parser.add_argument("--pause-seconds", type=int, default=60)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set")

    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    log_path = Path(args.log_file)
    client = db.Historical(key=api_key)

    for date_str in date_range(args.start, args.end):
        day = datetime.strptime(date_str, "%Y-%m-%d")
        definition_parents = parents_for_definition(day)
        submit_request(client, "definition", date_str, definition_parents, args.pause_seconds, log_path)
        def_path = definition_path(date_str)
        if not def_path.exists():
            raise FileNotFoundError(f"Definition file missing after submit: {def_path}")
        assets = load_assets_for_date(date_str)
        validate_assets(day, assets)
        parents = [f"{a}.OPT" for a in assets]
        for schema in schemas:
            if schema == "definition":
                continue
            submit_request(client, schema, date_str, parents, args.pause_seconds, log_path)


if __name__ == "__main__":
    main()
