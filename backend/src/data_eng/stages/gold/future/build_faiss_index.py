from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    faiss = None

from ....config import AppConfig, load_config
from ....io import (
    is_partition_complete,
    partition_ref,
    read_partition,
)
from ....utils import expand_date_range

TARGET_DIM = 256
LEVEL_TYPES = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]
EPSILON = 1e-9


def load_all_setup_vectors(
    cfg: AppConfig,
    symbol: str,
    dates: List[str],
) -> pd.DataFrame:
    all_dfs = []

    for dt in dates:
        ref = partition_ref(cfg, "gold.future.setup_vectors", symbol, dt)
        if not is_partition_complete(ref):
            continue
        df = read_partition(ref)
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def compute_normalization_params(
    df: pd.DataFrame,
    level_type: str,
) -> Dict[str, Dict[str, float]]:
    subset = df[df["level_type"] == level_type]
    if len(subset) == 0:
        return {}

    params = {}
    vector_cols = [f"v_{i}" for i in range(TARGET_DIM)]

    for col in vector_cols:
        if col in subset.columns:
            vals = subset[col].dropna()
            if len(vals) > 0:
                params[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if vals.std() > EPSILON else 1.0,
                }

    return params


def normalize_vectors(
    vectors: np.ndarray,
    norm_params: Dict[str, Dict[str, float]],
) -> np.ndarray:
    normalized = vectors.copy()

    for i in range(vectors.shape[1]):
        col = f"v_{i}"
        if col in norm_params:
            mean = norm_params[col]["mean"]
            std = norm_params[col]["std"]
            normalized[:, i] = (normalized[:, i] - mean) / (std + EPSILON)

    normalized = np.clip(normalized, -10, 10)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)

    return normalized.astype(np.float32)


def build_faiss_index(
    vectors: np.ndarray,
    index_type: str = "flat",
) -> Any:
    if faiss is None:
        raise ImportError("faiss-cpu or faiss-gpu required. Install with: uv add faiss-cpu")

    d = vectors.shape[1]
    n = vectors.shape[0]

    if index_type == "flat":
        index = faiss.IndexFlatL2(d)
    elif index_type == "ivf_flat":
        nlist = min(100, max(1, n // 10))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        if n > 0:
            index.train(vectors)
    elif index_type == "ivf_pq":
        nlist = min(1000, max(1, n // 100))
        m = 32
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        if n > 0:
            index.train(vectors)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    if n > 0:
        index.add(vectors)

    return index


def create_metadata_db(
    db_path: Path,
    df: pd.DataFrame,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE setup_metadata (
            vector_id       INTEGER PRIMARY KEY,
            episode_id      TEXT NOT NULL,
            dt              TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            level_type      TEXT NOT NULL,
            level_price     REAL NOT NULL,
            trigger_bar_ts  INTEGER NOT NULL,
            approach_direction INTEGER NOT NULL,
            outcome         TEXT NOT NULL,
            outcome_score   REAL NOT NULL,
            velocity_at_trigger     REAL,
            obi0_at_trigger         REAL,
            wall_imbal_at_trigger   REAL
        )
    """)

    cursor.execute("CREATE INDEX idx_level_type ON setup_metadata(level_type)")
    cursor.execute("CREATE INDEX idx_outcome ON setup_metadata(outcome)")
    cursor.execute("CREATE INDEX idx_dt ON setup_metadata(dt)")
    cursor.execute("CREATE INDEX idx_symbol ON setup_metadata(symbol)")

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO setup_metadata (
                vector_id, episode_id, dt, symbol, level_type,
                level_price, trigger_bar_ts, approach_direction,
                outcome, outcome_score, velocity_at_trigger,
                obi0_at_trigger, wall_imbal_at_trigger
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["vector_id"]),
            str(row["episode_id"]),
            str(row["dt"]),
            str(row["symbol"]),
            str(row["level_type"]),
            float(row["level_price"]),
            int(row["trigger_bar_ts"]),
            int(row["approach_direction"]),
            str(row["outcome"]),
            float(row["outcome_score"]),
            float(row.get("velocity_at_trigger", 0.0) or 0.0),
            float(row.get("obi0_at_trigger", 0.0) or 0.0),
            float(row.get("wall_imbal_at_trigger", 0.0) or 0.0),
        ))

    conn.commit()
    conn.close()


def build_all_indices(
    cfg: AppConfig,
    repo_root: Path,
    symbol: str,
    dates: List[str],
    output_dir: Path,
    index_type: str = "flat",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_setup_vectors(cfg, symbol, dates)

    if len(df) == 0:
        print("No setup vectors found for the specified dates.")
        return {}

    print(f"Loaded {len(df)} setup vectors across {len(dates)} dates.")

    vector_cols = [f"v_{i}" for i in range(TARGET_DIM)]

    all_norm_params = {}

    for level_type in LEVEL_TYPES:
        level_df = df[df["level_type"] == level_type].copy()

        if len(level_df) == 0:
            print(f"No vectors for {level_type}, skipping...")
            continue

        print(f"Building index for {level_type} with {len(level_df)} vectors...")

        level_df = level_df.reset_index(drop=True)
        level_df["vector_id"] = range(len(level_df))

        vectors = level_df[vector_cols].values.astype(np.float64)

        norm_params = compute_normalization_params(level_df, level_type)
        all_norm_params[level_type] = norm_params

        normalized_vectors = normalize_vectors(vectors, norm_params)

        index = build_faiss_index(normalized_vectors, index_type)

        index_path = output_dir / f"{level_type.lower()}_setups.index"
        faiss.write_index(index, str(index_path))
        print(f"  Wrote index to {index_path}")

        vectors_path = output_dir / f"{level_type.lower()}_vectors.npy"
        np.save(str(vectors_path), normalized_vectors)
        print(f"  Wrote vectors to {vectors_path}")

        metadata_path = output_dir / f"{level_type.lower()}_metadata.db"
        create_metadata_db(metadata_path, level_df)
        print(f"  Wrote metadata to {metadata_path}")

        episode_ids_path = output_dir / f"{level_type.lower()}_episode_ids.json"
        with open(episode_ids_path, "w") as f:
            json.dump(level_df["episode_id"].tolist(), f)
        print(f"  Wrote episode IDs to {episode_ids_path}")

    norm_params_path = output_dir / "norm_params.json"
    with open(norm_params_path, "w") as f:
        json.dump(all_norm_params, f, indent=2)
    print(f"Wrote normalization params to {norm_params_path}")

    combined_metadata_path = output_dir / "setup_metadata.db"

    df_with_global_ids = df.copy()
    df_with_global_ids["vector_id"] = range(len(df_with_global_ids))
    create_metadata_db(combined_metadata_path, df_with_global_ids)
    print(f"Wrote combined metadata to {combined_metadata_path}")

    return {
        "total_vectors": len(df),
        "level_counts": {lt: len(df[df["level_type"] == lt]) for lt in LEVEL_TYPES},
        "output_dir": str(output_dir),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS indices and metadata from setup vectors")
    parser.add_argument("--symbol", type=str, default="ESU5", help="Symbol to process")
    parser.add_argument("--dates", type=str, help="Comma-separated dates or range (2025-06-05:2025-06-10)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="databases/indices", help="Output directory")
    parser.add_argument("--index-type", type=str, default="flat", choices=["flat", "ivf_flat", "ivf_pq"])

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[5]
    config_path = repo_root / "src" / "data_eng" / "config" / "datasets.yaml"
    cfg = load_config(repo_root, config_path)

    dates = expand_date_range(
        dates=args.dates,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not dates:
        raise ValueError("Must provide --dates or --start-date/--end-date")
    output_dir = repo_root / args.output_dir

    result = build_all_indices(
        cfg=cfg,
        repo_root=repo_root,
        symbol=args.symbol,
        dates=dates,
        output_dir=output_dir,
        index_type=args.index_type,
    )

    print("\nBuild complete:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
