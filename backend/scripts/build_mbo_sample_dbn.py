import argparse
from pathlib import Path
import sys

import databento as db
import databento_dbn as dbn
import hashlib
import uuid

import pandas as pd
import pyarrow as pa
import zstandard as zstd
from deltalake import write_deltalake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small MBO DBN sample for a date and time window."
    )
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--symbol", default="ES", help="Root symbol (default: ES)")
    parser.add_argument("--start-time", default="09:30", help="ET start time (HH:MM)")
    parser.add_argument("--end-time", default="11:30", help="ET end time (HH:MM)")
    return parser.parse_args()


def build_sample(
    date: str,
    symbol: str,
    start_time: str,
    end_time: str,
) -> tuple[Path, Path, int]:
    backend_dir = Path(__file__).resolve().parents[1]
    raw_dir = (
        backend_dir
        / "lake"
        / "raw"
        / "source=databento"
        / "product_type=future"
        / f"symbol={symbol}"
        / "table=market_by_order_dbn"
    )
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw DBN directory not found: {raw_dir}")

    date_compact = date.replace("-", "")
    dbn_files = sorted(raw_dir.glob(f"*{date_compact}*.dbn*"))
    if len(dbn_files) != 1:
        raise ValueError(
            f"Expected exactly one DBN file for {date} in {raw_dir}, found {len(dbn_files)}"
        )
    source_file = dbn_files[0]

    start_local = pd.Timestamp(f"{date} {start_time}:00", tz="America/New_York")
    end_local = pd.Timestamp(f"{date} {end_time}:00", tz="America/New_York")
    start_ns = int(start_local.tz_convert("UTC").value)
    end_ns = int(end_local.tz_convert("UTC").value)
    if end_ns <= start_ns:
        raise ValueError("End time must be after start time")

    out_dir = (
        backend_dir
        / "lake"
        / "raw"
        / "sample"
        / "source=databento"
        / "product_type=future"
        / f"symbol={symbol}"
        / "table=market_by_order_dbn"
        / f"dt={date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    time_tag = f"{start_time.replace(':', '')}-{end_time.replace(':', '')}"
    out_dbn = out_dir / f"{source_file.stem}.sample_{time_tag}.dbn"
    out_dbn_zst = out_dir / f"{source_file.stem}.sample_{time_tag}.dbn.zst"
    if out_dbn.exists() or out_dbn_zst.exists():
        raise FileExistsError(f"Sample output already exists in {out_dir}")

    store = db.DBNStore.from_file(str(source_file))
    meta_bytes = bytes(store.metadata)

    sample_count = 0
    first_ts = None
    last_ts = None

    with out_dbn.open("wb") as f_out:
        f_out.write(meta_bytes)
        for record in store:
            ts_event = getattr(record, "ts_event", None)
            if ts_event is None:
                continue
            if ts_event < start_ns:
                continue
            if ts_event >= end_ns:
                break
            f_out.write(bytes(record))
            if first_ts is None:
                first_ts = ts_event
            last_ts = ts_event
            sample_count += 1

    if sample_count == 0:
        raise ValueError("No records found in the requested window")

    with out_dbn.open("r+b") as f_out:
        dbn.update_encoded_metadata(
            f_out,
            start=first_ts,
            end=last_ts,
            limit=sample_count,
        )

    compressor = zstd.ZstdCompressor(level=3)
    with out_dbn.open("rb") as f_in, out_dbn_zst.open("wb") as f_out:
        with compressor.stream_writer(f_out) as writer:
            while True:
                chunk = f_in.read(2**20)
                if not chunk:
                    break
                writer.write(chunk)

    return out_dbn, out_dbn_zst, sample_count


def write_manifest(
    sample_file: Path,
    date: str,
    underlier: str,
    instrument_type: str,
) -> Path:
    backend_dir = Path(__file__).resolve().parents[1]
    manifest_dir = backend_dir / "lake" / "bronze" / "dbn_manifest"

    checksum = hashlib.sha256(sample_file.read_bytes()).hexdigest()
    date_compact = date.replace("-", "")
    batch_id = uuid.uuid4().hex
    adls_path = f"raw-dbn/{underlier}/{instrument_type}/{date_compact}/{sample_file.name}"

    table = pa.Table.from_pylist(
        [
            {
                "file_path": adls_path,
                "checksum_sha256": checksum,
                "session_date": date,
                "underlier": underlier,
                "instrument_type": instrument_type,
                "ingestion_batch_id": batch_id,
            }
        ]
    )

    write_deltalake(
        manifest_dir,
        table,
        mode="append",
    )
    return manifest_dir


def main() -> int:
    args = parse_args()
    try:
        out_dbn, out_dbn_zst, sample_count = build_sample(
            date=args.date,
            symbol=args.symbol,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        manifest_dir = write_manifest(
            sample_file=out_dbn_zst,
            date=args.date,
            underlier=args.symbol,
            instrument_type="FUT",
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Sample records: {sample_count:,}")
    print(f"Sample DBN: {out_dbn}")
    print(f"Sample DBN (zst): {out_dbn_zst}")
    print(f"Manifest Delta: {manifest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
