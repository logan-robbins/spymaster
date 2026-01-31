from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import pyarrow.dataset as ds

from ....config import AppConfig
from ....contracts import load_avro_contract
from ....io import is_partition_complete, partition_ref

MBO_COLUMNS = [
    "ts_event",
    "action",
    "side",
    "price",
    "size",
    "order_id",
    "sequence",
    "flags",
]


def first_hour_window_ns(session_date: str) -> Tuple[int, int]:
    """Return (start_ns, end_ns) for the session output window.
    
    Stage F calibration requires the first 3 hours of RTH (09:30-12:30 ET).
    """
    start = pd.Timestamp(f"{session_date} 09:30:00", tz="US/Eastern")
    end = pd.Timestamp(f"{session_date} 12:30:00", tz="US/Eastern")
    return int(start.tz_convert("UTC").value), int(end.tz_convert("UTC").value)


def iter_mbo_batches(
    cfg: AppConfig,
    repo_root: Path,
    symbol: str,
    dt: str,
    batch_size: int = 1_000_000,
    start_buffer_ns: int = 0,
) -> Iterable[pd.DataFrame]:
    dataset_key = "bronze.future_mbo.mbo"
    ref = partition_ref(cfg, dataset_key, symbol, dt)
    if not is_partition_complete(ref):
        raise FileNotFoundError(f"Input not ready: {dataset_key} dt={dt}")

    files = [str(path) for path in ref.list_data_files()]
    if not files:
        raise FileNotFoundError(f"No data files found in: {ref.dir}")

    contract = load_avro_contract(repo_root / cfg.dataset(dataset_key).contract)

    dataset = ds.dataset(files, format="parquet")
    schema_names = set(dataset.schema.names)
    contract_names = set(contract.fields)

    missing = contract_names.difference(schema_names)
    extra = schema_names.difference(contract_names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected extra columns not in contract: {sorted(extra)}")

    start_ns, end_ns = first_hour_window_ns(dt)
    filt = (ds.field("ts_event") >= (start_ns - start_buffer_ns)) & (ds.field("ts_event") < end_ns)

    scanner = dataset.scanner(columns=MBO_COLUMNS, filter=filt, batch_size=batch_size)

    last_ts = None
    last_seq = None

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue
        if not _is_sorted_batch(df, last_ts, last_seq):
            raise ValueError("MBO events are not ordered by ts_event, sequence")
        last_ts = int(df["ts_event"].iloc[-1])
        last_seq = int(df["sequence"].iloc[-1])
        yield df


def _is_sorted_batch(df: pd.DataFrame, last_ts: int | None, last_seq: int | None) -> bool:
    ts = df["ts_event"].to_numpy()
    seq = df["sequence"].to_numpy()
    if ts.size <= 1:
        return _check_boundary(ts, seq, last_ts, last_seq)

    ts_diff = ts[1:] - ts[:-1]
    if (ts_diff < 0).any():
        return False
    eq_mask = ts_diff == 0
    if eq_mask.any():
        seq_diff = seq[1:] - seq[:-1]
        if (seq_diff[eq_mask] < 0).any():
            return False

    return _check_boundary(ts, seq, last_ts, last_seq)


def _check_boundary(ts: pd.Series | list, seq: pd.Series | list, last_ts: int | None, last_seq: int | None) -> bool:
    if last_ts is None:
        return True
    first_ts = int(ts[0])
    first_seq = int(seq[0])
    if first_ts < last_ts:
        return False
    if first_ts == last_ts and first_seq < last_seq:
        return False
    return True
