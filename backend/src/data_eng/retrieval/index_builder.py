from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import load_config
from ..contracts import enforce_contract, load_avro_contract
from ..io import is_partition_complete, partition_ref, read_partition
from ..utils import expand_date_range
from .normalization import RobustStats, apply_robust_scaling, fit_robust_stats, l2_normalize
from .mbo_contract_day_selector import load_selection

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


def build_indices(
    df: pd.DataFrame,
    output_dir: Path,
    stats: RobustStats,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_stats(output_dir, stats)

    if len(df) == 0:
        raise ValueError("No vectors available for index build")

    vectors = np.array(df["vector"].tolist(), dtype=np.float64)
    dims = vectors.shape[1]

    for level_id in sorted(df["level_id"].unique()):
        level_dir = output_dir / level_id
        level_dir.mkdir(parents=True, exist_ok=True)
        level_df = df[df["level_id"] == level_id].reset_index(drop=True)

        for approach_dir in APPROACH_DIRS:
            approach_df = level_df[level_df["approach_dir"] == approach_dir].reset_index(drop=True)
            index_path = level_dir / f"{approach_dir}.index"
            meta_path = level_dir / f"metadata_{approach_dir}.npz"

            index = faiss.IndexHNSWFlat(dims, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64

            if len(approach_df) == 0:
                faiss.write_index(index, str(index_path))
                _write_metadata(meta_path, pd.DataFrame(columns=METADATA_FIELDS))
                continue

            v_group = np.array(approach_df["vector"].tolist(), dtype=np.float64)
            scaled = apply_robust_scaling(v_group, stats)
            normalized, valid = l2_normalize(scaled)

            if np.any(valid):
                index.add(normalized[valid])
                meta_df = approach_df.loc[valid].reset_index(drop=True)
            else:
                meta_df = pd.DataFrame(columns=METADATA_FIELDS)

            faiss.write_index(index, str(index_path))
            _write_metadata(meta_path, meta_df)


def _write_stats(output_dir: Path, stats: RobustStats) -> None:
    payload = {
        "median": stats.median.tolist(),
        "mad": stats.mad.tolist(),
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


def _load_selection_rows(selection_path: Path, dates: List[str]) -> pd.DataFrame:
    df = load_selection(selection_path)
    df = df.loc[df["session_date"].isin(dates)].copy()
    df = df.loc[(df["include_flag"] == 1) & (df["selected_symbol"] != "")]
    if len(df) == 0:
        raise ValueError("No included sessions in selection map")
    return df


def _load_vectors(repo_root: Path, selection: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config(repo_root, repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
    contract_path = repo_root / cfg.dataset(DATASET_KEY).contract
    contract = load_avro_contract(contract_path)

    frames = []
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

    if not frames:
        return pd.DataFrame(columns=contract.fields)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indices for trigger vectors.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--dates", required=True)
    parser.add_argument("--selection-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dates = expand_date_range(dates=args.dates)
    if not dates:
        raise ValueError("No dates provided")

    selection = _load_selection_rows(args.selection_path, dates)
    df = _load_vectors(args.repo_root, selection)
    if len(df) == 0:
        raise ValueError("No vectors loaded from lake")

    vectors = np.array(df["vector"].tolist(), dtype=np.float64)
    stats = fit_robust_stats(vectors)
    build_indices(df, args.output_dir, stats)


if __name__ == "__main__":
    main()
