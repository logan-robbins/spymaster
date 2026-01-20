from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    df = pd.read_parquet(input_dir)
    if df.empty:
        raise ValueError("Input dataset is empty")

    if "session_date" not in df.columns:
        raise ValueError("Input dataset missing session_date")

    sort_cols = ["session_date"]
    if "vector_time" in df.columns:
        sort_cols.append("vector_time")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    unique_dates = sorted(df["session_date"].dropna().unique().tolist())
    if not unique_dates:
        raise ValueError("Input dataset has no session_date values")

    if len(unique_dates) == 1:
        n = len(df)
        train_end = int(n * args.train_frac)
        val_end = train_end + int(n * args.val_frac)
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
    else:
        n_dates = len(unique_dates)
        train_end = max(1, int(n_dates * args.train_frac))
        val_end = min(n_dates - 1, train_end + int(n_dates * args.val_frac))
        train_dates = set(unique_dates[:train_end])
        val_dates = set(unique_dates[train_end:val_end])
        test_dates = set(unique_dates[val_end:])
        train_df = df[df["session_date"].isin(train_dates)].copy()
        val_df = df[df["session_date"].isin(val_dates)].copy()
        test_df = df[df["session_date"].isin(test_dates)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Split produced empty partition")

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_dir / "part-00000.parquet", index=False)
    val_df.to_parquet(val_dir / "part-00000.parquet", index=False)
    test_df.to_parquet(test_dir / "part-00000.parquet", index=False)


if __name__ == "__main__":
    main()
