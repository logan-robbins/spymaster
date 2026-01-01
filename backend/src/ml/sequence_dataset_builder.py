"""
Sequence Dataset Builder for PatchTST.

Builds fixed-length sequences from 1-minute OHLCV bars and aligns them to
touch events in the signals dataset.

Outputs a compressed NPZ with:
- X: (n_samples, seq_len, n_features)
- mask: (n_samples, seq_len)
- static: (n_samples, n_static_features)
- y_break: (n_samples,)
- y_strength: (n_samples,)
- event_id, ts_ns
- feature name arrays for sequence + static features
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.common.config import CONFIG
from src.ingestion.databento.dbn_reader import DBNIngestor
from src.ml.data_filters import filter_rth_signals
from src.pipeline.stages.build_ohlcv import build_ohlcv


SEQ_FEATURE_COLUMNS = [
    "close",
    "return",
    "volatility",
    "dist_to_sma_90",
    "dist_to_ema_20",
    "sma_spread",
    "sma_90_slope_5bar",
    "ema_20_slope_5bar",
]


TARGET_COLUMNS = [
    "outcome",
    "future_price",
    "excursion_max",
    "excursion_min",
    "strength_signed",
    "strength_abs",
    "time_to_threshold_1",
    "time_to_threshold_2",
    "time_to_break_1",
    "time_to_break_2",
    "time_to_bounce_1",
    "time_to_bounce_2",
]


def _resolve_signals_path(signals_path: str | None) -> Path:
    if signals_path:
        return Path(signals_path)

    features_path = Path(__file__).resolve().parents[2] / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"features.json not found at {features_path}")

    with features_path.open("r") as f:
        features = json.load(f)

    output_path = features.get("output_path")
    if not output_path:
        raise ValueError("features.json missing output_path")

    return features_path.parent / output_path


def _load_signals(signals_path: Path, dates: List[str]) -> pd.DataFrame:
    if not signals_path.exists():
        raise FileNotFoundError(f"Signals parquet not found: {signals_path}")

    df = pd.read_parquet(signals_path)
    if df.empty:
        raise ValueError("Signals dataset is empty")

    if dates:
        df = df[df["date"].isin(dates)]

    df = filter_rth_signals(df)
    if df.empty:
        raise ValueError("No RTH signals found for requested dates")

    return df


def _get_warmup_dates(ingestor: DBNIngestor, date: str) -> List[str]:
    warmup_days = max(0, CONFIG.SMA_WARMUP_DAYS)
    if warmup_days == 0:
        return []

    available = ingestor.get_available_dates('trades')
    weekday_dates = [
        d for d in available
        if datetime.strptime(d, '%Y-%m-%d').weekday() < 5
    ]
    if date not in weekday_dates:
        return []

    idx = weekday_dates.index(date)
    start_idx = max(0, idx - warmup_days)
    return weekday_dates[start_idx:idx]


def _build_ohlcv_features(date: str) -> pd.DataFrame:
    ingestor = DBNIngestor()
    trades = list(ingestor.read_trades(date=date))
    if not trades:
        raise ValueError(f"No ES trades found for {date}")

    ohlcv_1m = build_ohlcv(trades, freq="1min")
    ohlcv_2m = build_ohlcv(trades, freq="2min")
    warmup_dates = _get_warmup_dates(ingestor, date)
    warmup_frames = []
    for warmup_date in warmup_dates:
        warmup_trades = list(ingestor.read_trades(date=warmup_date))
        if not warmup_trades:
            continue
        warmup_2m = build_ohlcv(warmup_trades, freq="2min")
        if not warmup_2m.empty:
            warmup_frames.append(warmup_2m)
    if warmup_frames:
        warmup_2m = pd.concat(warmup_frames, ignore_index=True).sort_values("timestamp")
        ohlcv_2m = pd.concat([warmup_2m, ohlcv_2m], ignore_index=True).sort_values("timestamp")

    if ohlcv_1m.empty:
        raise ValueError(f"Failed to build 1-minute OHLCV for {date}")

    ohlcv_1m = ohlcv_1m.sort_values("timestamp").reset_index(drop=True)
    ohlcv_2m = ohlcv_2m.sort_values("timestamp").reset_index(drop=True)

    ohlcv_1m["ts_ns"] = ohlcv_1m["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    ohlcv_1m["return"] = ohlcv_1m["close"].diff().fillna(0.0)
    ohlcv_1m["volatility"] = (
        ohlcv_1m["return"]
        .rolling(CONFIG.MEAN_REVERSION_VOL_WINDOW_MINUTES)
        .std()
        .fillna(0.0)
    )

    if not ohlcv_2m.empty:
        ohlcv_2m["sma_90"] = ohlcv_2m["close"].rolling(90).mean()
        ohlcv_2m["ema_20"] = ohlcv_2m["close"].ewm(span=20, adjust=False).mean()
        short_bars = max(1, CONFIG.SMA_SLOPE_SHORT_BARS)
        short_minutes = short_bars * 2
        ohlcv_2m["sma_90_slope_5bar"] = (
            ohlcv_2m["sma_90"] - ohlcv_2m["sma_90"].shift(short_bars)
        ) / short_minutes
        ohlcv_2m["ema_20_slope_5bar"] = (
            ohlcv_2m["ema_20"] - ohlcv_2m["ema_20"].shift(short_bars)
        ) / short_minutes

        sma_cols = [
            "timestamp",
            "sma_90",
            "ema_20",
            "sma_90_slope_5bar",
            "ema_20_slope_5bar",
        ]
        ohlcv_1m = pd.merge_asof(
            ohlcv_1m.sort_values("timestamp"),
            ohlcv_2m[sma_cols].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    ohlcv_1m["dist_to_sma_90"] = ohlcv_1m["close"] - ohlcv_1m.get("sma_90", np.nan)
    ohlcv_1m["dist_to_ema_20"] = ohlcv_1m["close"] - ohlcv_1m.get("ema_20", np.nan)
    ohlcv_1m["sma_spread"] = ohlcv_1m.get("sma_90", np.nan) - ohlcv_1m.get("ema_20", np.nan)

    ohlcv_1m[SEQ_FEATURE_COLUMNS] = ohlcv_1m[SEQ_FEATURE_COLUMNS].fillna(0.0)

    return ohlcv_1m


def _build_static_features(signals_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    if "direction" not in signals_df.columns:
        raise ValueError("Signals dataset missing direction column")

    static_df = signals_df.select_dtypes(include=["number", "bool"]).copy()
    direction_sign = np.where(signals_df["direction"].values == "UP", 1.0, -1.0)
    static_df["direction_sign"] = direction_sign

    static_df = static_df.drop(columns=[c for c in TARGET_COLUMNS if c in static_df.columns], errors="ignore")
    static_df = static_df.drop(columns=["ts_ns"], errors="ignore")

    return static_df.to_numpy(dtype=np.float32), static_df.columns.tolist()


def _encode_targets(signals_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if "outcome" not in signals_df.columns:
        raise ValueError("Signals dataset missing outcome column")
    if "strength_signed" not in signals_df.columns:
        raise ValueError("Signals dataset missing strength_signed column")

    outcome_map = {"BREAK": 1, "BOUNCE": 0, "CHOP": -1, "UNDEFINED": -1}
    y_break = signals_df["outcome"].map(outcome_map).fillna(-1).astype(np.int8).to_numpy()
    y_strength = signals_df["strength_signed"].fillna(0.0).astype(np.float32).to_numpy()

    return y_break, y_strength


def build_sequence_dataset(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    lookback_minutes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if signals_df.empty:
        raise ValueError("Signals dataset is empty")
    if ohlcv_df.empty:
        raise ValueError("OHLCV data is empty")

    seq_len = lookback_minutes
    feature_matrix = ohlcv_df[SEQ_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    bar_ts = ohlcv_df["ts_ns"].to_numpy(dtype=np.int64)

    n = len(signals_df)
    n_features = len(SEQ_FEATURE_COLUMNS)

    X = np.zeros((n, seq_len, n_features), dtype=np.float32)
    mask = np.zeros((n, seq_len), dtype=np.uint8)

    signal_ts = signals_df["ts_ns"].values.astype(np.int64)

    for i in range(n):
        ts = signal_ts[i]
        end_idx = np.searchsorted(bar_ts, ts, side="right") - 1
        if end_idx < 0:
            continue

        start_idx = end_idx - seq_len + 1
        if start_idx < 0:
            pad_len = -start_idx
            data = feature_matrix[: end_idx + 1]
            X[i, pad_len : pad_len + len(data)] = data
            mask[i, pad_len : pad_len + len(data)] = 1
        else:
            data = feature_matrix[start_idx : end_idx + 1]
            X[i] = data
            mask[i] = 1

    return X, mask, bar_ts, SEQ_FEATURE_COLUMNS


def _parse_dates(date: str | None, dates: str | None) -> List[str]:
    if date and dates:
        raise ValueError("Provide --date or --dates, not both")
    if date:
        return [date]
    if dates:
        return [d.strip() for d in dates.split(",") if d.strip()]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequence dataset builder for PatchTST")
    parser.add_argument("--signals-path", type=str, default=None, help="Path to signals parquet")
    parser.add_argument("--date", type=str, default=None, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated dates")
    parser.add_argument("--lookback-minutes", type=int, default=120, help="Lookback window length in minutes")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for .npz files")

    args = parser.parse_args()

    dates = _parse_dates(args.date, args.dates)
    signals_path = _resolve_signals_path(args.signals_path)
    output_dir = Path(args.output_dir) if args.output_dir else signals_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    signals_df = _load_signals(signals_path, dates)
    grouped = signals_df.groupby("date")

    for date, day_df in grouped:
        if dates and date not in dates:
            continue

        ohlcv_df = _build_ohlcv_features(date)
        X, mask, _, seq_feature_names = build_sequence_dataset(day_df, ohlcv_df, args.lookback_minutes)
        static_data, static_feature_names = _build_static_features(day_df)
        y_break, y_strength = _encode_targets(day_df)

        output_path = output_dir / f"sequence_dataset_{date}.npz"
        np.savez_compressed(
            output_path,
            X=X,
            mask=mask,
            y_break=y_break,
            y_strength=y_strength,
            static=static_data,
            event_id=day_df["event_id"].astype(str).to_numpy(),
            ts_ns=day_df["ts_ns"].to_numpy(dtype=np.int64),
            seq_feature_names=np.array(seq_feature_names),
            static_feature_names=np.array(static_feature_names),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
