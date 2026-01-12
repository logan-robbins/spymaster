from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import load_config
from ..contracts import enforce_contract, load_avro_contract
from ..io import is_partition_complete, partition_ref, read_manifest_hash, read_partition
from ..utils import expand_date_range
from .normalization import RobustStats, apply_robust_scaling, l2_normalize
from .mbo_contract_day_selector import load_selection
from ..vector_schema import VECTOR_DIM, vector_feature_names

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss-cpu or faiss-gpu required") from exc

DATASET_KEY = "gold.future_mbo.mbo_trigger_vectors"
APPROACH_DIRS = ["approach_up", "approach_down"]

METADATA_FIELDS = [
    "vector_id",
    "ts_end_ns",
    "session_date",
    "symbol",
    "level_id",
    "P_ref",
    "P_REF_INT",
    "approach_dir",
    "first_hit",
    "first_hit_bar_offset",
    "whipsaw_flag",
    "true_outcome",
    "true_outcome_h0",
    "true_outcome_h1",
    "true_outcome_h2",
    "true_outcome_h3",
    "true_outcome_h4",
    "true_outcome_h5",
    "true_outcome_h6",
    "mfe_up_ticks",
    "mfe_down_ticks",
    "mae_before_upper_ticks",
    "mae_before_lower_ticks",
]

METADATA_TYPES = {
    "vector_id": "int",
    "ts_end_ns": "int",
    "session_date": "str",
    "symbol": "str",
    "level_id": "str",
    "P_ref": "float",
    "P_REF_INT": "int",
    "approach_dir": "str",
    "first_hit": "str",
    "first_hit_bar_offset": "int_null",
    "whipsaw_flag": "int",
    "true_outcome": "str",
    "true_outcome_h0": "str",
    "true_outcome_h1": "str",
    "true_outcome_h2": "str",
    "true_outcome_h3": "str",
    "true_outcome_h4": "str",
    "true_outcome_h5": "str",
    "true_outcome_h6": "str",
    "mfe_up_ticks": "float",
    "mfe_down_ticks": "float",
    "mae_before_upper_ticks": "float",
    "mae_before_lower_ticks": "float",
}

INVARIANT_DIM = VECTOR_DIM
FEATURE_NAMES = vector_feature_names()
METADATA_SCHEMA_HASH = hashlib.sha256(
    json.dumps(METADATA_TYPES, sort_keys=True).encode("utf-8")
).hexdigest()
if set(METADATA_FIELDS) != set(METADATA_TYPES.keys()):
    raise ValueError("Metadata type mapping mismatch")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_feature_list(output_dir: Path) -> Path:
    payload = {"vector_dim": VECTOR_DIM, "features": FEATURE_NAMES}
    path = output_dir / "feature_list.json"
    path.write_text(json.dumps(payload))
    return path


def _git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_indices(
    df: pd.DataFrame,
    output_dir: Path,
    stats: RobustStats,
    repo_root: Path,
    input_partitions: List[Dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if VECTOR_DIM != INVARIANT_DIM:
        raise ValueError("Vector dim invariant violated")
    if len(FEATURE_NAMES) != VECTOR_DIM:
        raise ValueError("Feature list size mismatch")
    faiss.omp_set_num_threads(1)

    _write_stats(output_dir, stats)
    feature_path = _write_feature_list(output_dir)
    stats_hash = _hash_file(output_dir / "norm_stats.json")
    feature_hash = _hash_file(feature_path)
    build_time = datetime.now(timezone.utc).isoformat()
    git_sha = _git_sha(repo_root)

    if len(df) == 0:
        raise ValueError("No vectors available for index build")

    if "vector_dim" not in df.columns:
        raise ValueError("Missing vector_dim column in vectors")
    if df["vector_dim"].nunique() != 1 or int(df["vector_dim"].iloc[0]) != VECTOR_DIM:
        raise ValueError("Vector dim column mismatch")

    vectors = np.array(df["vector"].tolist(), dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_DIM:
        raise ValueError("Vector payload dim mismatch")
    dims = vectors.shape[1]
    sort_cols = ["level_id", "approach_dir", "session_date", "symbol", "vector_id"]
    missing = [col for col in sort_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing sort columns: {missing}")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df = _filter_rth(df)
    if len(df) == 0:
        raise ValueError("No vectors inside RTH window")

    for level_id in sorted(df["level_id"].unique()):
        level_dir = output_dir / level_id
        level_dir.mkdir(parents=True, exist_ok=True)
        level_df = df[df["level_id"] == level_id].reset_index(drop=True)

        for approach_dir in APPROACH_DIRS:
            approach_df = level_df[level_df["approach_dir"] == approach_dir].reset_index(drop=True)
            index_path = level_dir / f"{approach_dir}.index"
            meta_path = level_dir / f"metadata_{approach_dir}.npz"
            vectors_path = level_dir / f"vectors_{approach_dir}.npy"
            manifest_path = level_dir / f"manifest_{approach_dir}.json"

            index = faiss.IndexHNSWFlat(dims, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64

            if len(approach_df) == 0:
                faiss.write_index(index, str(index_path))
                _write_metadata(meta_path, pd.DataFrame(columns=METADATA_FIELDS))
                np.save(vectors_path, np.empty((0, VECTOR_DIM), dtype=np.float32))
                _write_manifest(
                    manifest_path=manifest_path,
                    level_id=level_id,
                    approach_dir=approach_dir,
                    count=0,
                    dims=dims,
                    index=index,
                    index_path=index_path,
                    meta_path=meta_path,
                    vectors_path=vectors_path,
                    build_time=build_time,
                    git_sha=git_sha,
                    stats_hash=stats_hash,
                    feature_hash=feature_hash,
                    input_partitions=input_partitions,
                    absent=True,
                )
                continue

            v_group = np.array(approach_df["vector"].tolist(), dtype=np.float64)
            scaled = apply_robust_scaling(v_group, stats)
            normalized, valid = l2_normalize(scaled)

            if not np.all(valid):
                raise ValueError("Zero norm vectors after scaling")
            index.add(normalized)
            meta_df = _coerce_metadata(approach_df.reset_index(drop=True))

            faiss.write_index(index, str(index_path))
            _write_metadata(meta_path, meta_df)
            np.save(vectors_path, normalized.astype(np.float32))
            _write_manifest(
                manifest_path=manifest_path,
                level_id=level_id,
                approach_dir=approach_dir,
                count=len(meta_df),
                dims=dims,
                index=index,
                index_path=index_path,
                meta_path=meta_path,
                vectors_path=vectors_path,
                build_time=build_time,
                git_sha=git_sha,
                stats_hash=stats_hash,
                feature_hash=feature_hash,
                input_partitions=input_partitions,
                absent=False,
            )


def _write_stats(output_dir: Path, stats: RobustStats) -> None:
    payload = {
        "median": stats.median.tolist(),
        "mad": stats.mad.tolist(),
        "vector_dim": int(stats.median.shape[0]),
    }
    (output_dir / "norm_stats.json").write_text(json.dumps(payload))


def _write_metadata(path: Path, df: pd.DataFrame) -> None:
    if len(df) == 0:
        data: Dict[str, np.ndarray] = {"id": np.array([], dtype=np.int64)}
        for field in METADATA_FIELDS:
            data[field] = np.array([], dtype=object)
        np.savez_compressed(path, **data)
        return

    metadata: Dict[str, np.ndarray] = {
        "id": np.arange(len(df), dtype=np.int64),
    }
    for field in METADATA_FIELDS:
        metadata[field] = df[field].to_numpy()
    np.savez_compressed(path, **metadata)


def _load_stats(path: Path) -> RobustStats:
    if not path.exists():
        raise FileNotFoundError(f"Missing norm stats: {path}")
    payload = json.loads(path.read_text())
    median = np.array(payload.get("median", []), dtype=np.float64)
    mad = np.array(payload.get("mad", []), dtype=np.float64)
    if median.size != INVARIANT_DIM or mad.size != INVARIANT_DIM:
        raise ValueError("Norm stats dim mismatch")
    if not np.all(np.isfinite(median)) or not np.all(np.isfinite(mad)):
        raise ValueError("Non-finite norm stats")
    return RobustStats(median=median, mad=mad)


def _coerce_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for field, kind in METADATA_TYPES.items():
        if field not in out.columns:
            raise ValueError(f"Missing metadata field: {field}")
        if kind == "int":
            out[field] = out[field].astype("int64")
        elif kind == "float":
            out[field] = out[field].astype("float64")
        elif kind == "str":
            out[field] = out[field].astype(str)
        elif kind == "int_null":
            vals = [None if pd.isna(v) else int(v) for v in out[field].tolist()]
            out[field] = pd.Series(vals, dtype=object)
        else:
            raise ValueError(f"Unexpected metadata type: {kind}")
    return out


def _rth_window_ns(session_date: str) -> tuple[int, int]:
    start = pd.Timestamp(f"{session_date} 09:30:00", tz="America/New_York")
    end = pd.Timestamp(f"{session_date} 12:30:00", tz="America/New_York")
    return int(start.tz_convert("UTC").value), int(end.tz_convert("UTC").value)


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    if "session_date" not in df.columns or "ts_end_ns" not in df.columns:
        raise ValueError("Missing session_date or ts_end_ns for time filter")
    masks = []
    for session_date in sorted(df["session_date"].astype(str).unique()):
        start_ns, end_ns = _rth_window_ns(session_date)
        mask = (df["session_date"] == session_date) & (
            (df["ts_end_ns"] >= start_ns) & (df["ts_end_ns"] <= end_ns)
        )
        masks.append(mask.to_numpy())
    if not masks:
        return df.iloc[0:0].copy()
    keep = np.logical_or.reduce(masks)
    return df.loc[keep].reset_index(drop=True)


def _write_manifest(
    manifest_path: Path,
    level_id: str,
    approach_dir: str,
    count: int,
    dims: int,
    index: faiss.Index,
    index_path: Path,
    meta_path: Path,
    vectors_path: Path,
    build_time: str,
    git_sha: str,
    stats_hash: str,
    feature_hash: str,
    input_partitions: List[Dict[str, str]],
    absent: bool,
) -> None:
    payload = {
        "level_id": level_id,
        "approach_dir": approach_dir,
        "vector_dim": int(dims),
        "count": int(count),
        "absent": bool(absent),
        "index_type": type(index).__name__,
        "metric_type": int(index.metric_type),
        "index_file": index_path.name,
        "metadata_file": meta_path.name,
        "vectors_file": vectors_path.name,
        "norm_stats_sha256": stats_hash,
        "feature_list_sha256": feature_hash,
        "metadata_schema_sha256": METADATA_SCHEMA_HASH,
        "build_time": build_time,
        "git_sha": git_sha,
        "input_partitions": input_partitions,
        "ef_construction": int(index.hnsw.efConstruction) if hasattr(index, "hnsw") else None,
        "ef_search": int(index.hnsw.efSearch) if hasattr(index, "hnsw") else None,
    }
    manifest_path.write_text(json.dumps(payload, sort_keys=True))


def _load_selection_rows(selection_path: Path, dates: List[str]) -> pd.DataFrame:
    df = load_selection(selection_path)
    df = df.loc[df["session_date"].isin(dates)].copy()
    df = df.loc[df["selected_symbol"] != ""]
    if len(df) == 0:
        raise ValueError("No included sessions in selection map")
    return df


def _load_vectors(repo_root: Path, selection: pd.DataFrame) -> tuple[pd.DataFrame, List[Dict[str, str]]]:
    cfg = load_config(repo_root, repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
    contract_path = repo_root / cfg.dataset(DATASET_KEY).contract
    contract = load_avro_contract(contract_path)

    frames = []
    lineage: List[Dict[str, str]] = []
    for row in selection.itertuples(index=False):
        session_date = str(getattr(row, "session_date"))
        symbol = str(getattr(row, "selected_symbol"))
        ref = partition_ref(cfg, DATASET_KEY, symbol, session_date)
        if not is_partition_complete(ref):
            raise FileNotFoundError(f"Missing partition: {DATASET_KEY} symbol={symbol} dt={session_date}")
        df = read_partition(ref)
        if len(df) > 0:
            df = enforce_contract(df, contract)
            frames.append(df)
            lineage.append(
                {
                    "dataset": DATASET_KEY,
                    "symbol": symbol,
                    "session_date": session_date,
                    "manifest_sha256": read_manifest_hash(ref),
                }
            )

    if not frames:
        return pd.DataFrame(columns=contract.fields), []
    df = pd.concat(frames, ignore_index=True)
    sort_cols = ["level_id", "approach_dir", "session_date", "symbol", "vector_id"]
    missing = [col for col in sort_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing sort columns: {missing}")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df, lineage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indices for trigger vectors.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--dates", required=True)
    parser.add_argument("--selection-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dates = expand_date_range(dates=args.dates)
    if not dates:
        raise ValueError("No dates provided")

    if args.selection_path is None:
        args.selection_path = args.repo_root / "lake" / "selection" / "mbo_contract_day_selection.parquet"

    stats = _load_stats(args.norm_stats_path)
    output_dir = args.output_dir.resolve()
    norm_stats_path = args.norm_stats_path.resolve()
    seed_payload = None
    try:
        norm_stats_path.relative_to(output_dir)
        seed_payload = {
            "median": stats.median.tolist(),
            "mad": stats.mad.tolist(),
            "vector_dim": int(stats.median.shape[0]),
        }
    except ValueError:
        seed_payload = None

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    if seed_payload is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        norm_stats_path.write_text(json.dumps(seed_payload))

    selection = _load_selection_rows(args.selection_path, dates)
    df, lineage = _load_vectors(args.repo_root, selection)
    if len(df) == 0:
        raise ValueError("No vectors loaded from lake")
    build_indices(df, args.output_dir, stats, args.repo_root, lineage)


if __name__ == "__main__":
    main()
