from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_eng.config import load_config
from src.data_eng.io import is_partition_complete, partition_ref, read_partition
from src.data_eng.vector_schema import VECTOR_BLOCKS, VECTOR_DIM, X_COLUMNS

DATASET_KEY = "gold.future_mbo.mbo_trigger_vectors"
SELECTION_PATH = Path("lake/selection/mbo_contract_day_selection.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trigger vectors to validator feature columns")
    parser.add_argument("--dt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")

    selection_path = repo_root / SELECTION_PATH
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selection map: {selection_path}")

    selection_df = pd.read_parquet(selection_path)
    selection_row = selection_df.loc[selection_df["session_date"] == args.dt]
    if selection_row.empty:
        raise ValueError(f"Selection map missing session_date: {args.dt}")
    symbol = str(selection_row["selected_symbol"].iloc[0])

    ref = partition_ref(cfg, DATASET_KEY, symbol, args.dt)
    if not is_partition_complete(ref):
        raise FileNotFoundError(f"Input not ready: {DATASET_KEY} dt={args.dt} symbol={symbol}")

    df = read_partition(ref)
    if df.empty:
        raise ValueError(f"No rows in {DATASET_KEY} dt={args.dt} symbol={symbol}")

    if "vector_dim" not in df.columns:
        raise ValueError("Missing vector_dim column in trigger vectors")
    if df["vector_dim"].nunique() != 1:
        raise ValueError("Mixed vector_dim values in trigger vectors")
    if int(df["vector_dim"].iloc[0]) != VECTOR_DIM:
        raise ValueError("Vector dim mismatch")

    feature_names = _vector_feature_names()
    features = pd.DataFrame(df["vector"].tolist(), columns=feature_names)

    meta_cols = ["ts_end_ns", "session_date", "symbol", "level_id", "approach_dir", "P_ref"]
    missing_meta = [col for col in meta_cols if col not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing metadata columns: {missing_meta}")

    out_df = pd.concat([df[meta_cols].reset_index(drop=True), features], axis=1)
    out_df = out_df.rename(columns={"ts_end_ns": "window_start_ts_ns"})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    p_refs = df["P_ref"].unique()
    if len(p_refs) != 1:
        raise ValueError(f"Expected single P_ref, found {len(p_refs)}")

    print(f"symbol={symbol}")
    print(f"P_ref={p_refs[0]}")
    print(f"rows={len(out_df)}")
    print(f"output={out_path}")


def _vector_feature_names() -> list[str]:
    names: list[str] = []
    for base in X_COLUMNS:
        for block in VECTOR_BLOCKS:
            names.append(f"{block}_{base}")
    if len(names) != VECTOR_DIM:
        raise ValueError("Vector feature name length mismatch")
    return names


if __name__ == "__main__":
    main()
