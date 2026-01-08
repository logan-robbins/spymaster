"""
Phase 1: Pure Historical Baseline - Zero Features

For each level type (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW) and approach direction (from_below, from_above):
- Find all "touches" (price within TRIGGER_BAND of level)
- Track what happens in the next OUTCOME_WINDOW seconds
- Label outcome using simple thresholds:
  - BREAK: Price moves +TARGET_PTS in approach direction without retreating -STOP_PTS first
  - REJECT: Price retreats -STOP_PTS before reaching +TARGET_PTS
  - INDETERMINATE: Neither threshold hit within window

Output: Summary table with counts and percentages per level/direction.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class OutcomeParams:
    target_pts: float = 10.0
    stop_pts: float = 5.0
    window_seconds: int = 240
    trigger_band_pts: float = 1.0
    cooldown_seconds: int = 120


@dataclass
class Touch:
    dt: str
    symbol: str
    level_type: str
    level_price: float
    touch_ts: pd.Timestamp
    approach_direction: int
    outcome: str
    max_favorable: float
    max_adverse: float
    time_to_outcome_sec: float


def compute_mid_price(df: pd.DataFrame) -> np.ndarray:
    return ((df["bid_px_00"].values + df["ask_px_00"].values) / 2).astype(np.float64)


def compute_levels(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "PM_HIGH": float(df["pm_high"].iloc[0]),
        "PM_LOW": float(df["pm_low"].iloc[0]),
        "OR_HIGH": float(df["or_high"].iloc[0]),
        "OR_LOW": float(df["or_low"].iloc[0]),
    }


def get_approach_direction(mid_prices: np.ndarray, idx: int, level_price: float, lookback: int = 60) -> int:
    start = max(0, idx - lookback)
    if start == idx:
        return 0
    avg_price = mid_prices[start:idx].mean()
    if avg_price < level_price - 0.5:
        return 1
    elif avg_price > level_price + 0.5:
        return -1
    return 0


def compute_outcome(
    mid_prices: np.ndarray,
    timestamps: np.ndarray,
    trigger_idx: int,
    level_price: float,
    approach_direction: int,
    params: OutcomeParams,
) -> Tuple[str, float, float, float]:
    trigger_ts = timestamps[trigger_idx]
    window_end_ns = trigger_ts + params.window_seconds * 1_000_000_000

    max_favorable = 0.0
    max_adverse = 0.0

    for i in range(trigger_idx + 1, len(mid_prices)):
        if timestamps[i] > window_end_ns:
            break

        signed_move = (mid_prices[i] - level_price) * approach_direction

        if signed_move > max_favorable:
            max_favorable = signed_move
        if signed_move < max_adverse:
            max_adverse = signed_move

        if max_adverse <= -params.stop_pts:
            time_to_outcome = (timestamps[i] - trigger_ts) / 1_000_000_000
            return "REJECT", max_favorable, max_adverse, time_to_outcome

        if max_favorable >= params.target_pts:
            time_to_outcome = (timestamps[i] - trigger_ts) / 1_000_000_000
            return "BREAK", max_favorable, max_adverse, time_to_outcome

    time_to_outcome = min(
        (timestamps[min(len(timestamps) - 1, trigger_idx + 1)] - trigger_ts) / 1_000_000_000,
        params.window_seconds
    )
    return "INDETERMINATE", max_favorable, max_adverse, time_to_outcome


def find_touches_for_level(
    df: pd.DataFrame,
    mid_prices: np.ndarray,
    timestamps: np.ndarray,
    dt: str,
    symbol: str,
    level_type: str,
    level_price: float,
    valid_start_idx: int,
    valid_end_idx: int,
    params: OutcomeParams,
) -> List[Touch]:
    if np.isnan(level_price):
        return []

    touches = []
    cooldown_ns = params.cooldown_seconds * 1_000_000_000
    last_touch_ts = 0

    for i in range(valid_start_idx, valid_end_idx):
        if timestamps[i] < last_touch_ts + cooldown_ns:
            continue

        dist = abs(mid_prices[i] - level_price)
        if dist > params.trigger_band_pts:
            continue

        approach_dir = get_approach_direction(mid_prices, i, level_price)
        if approach_dir == 0:
            continue

        outcome, max_fav, max_adv, time_to = compute_outcome(
            mid_prices, timestamps, i, level_price, approach_dir, params
        )

        touches.append(Touch(
            dt=dt,
            symbol=symbol,
            level_type=level_type,
            level_price=level_price,
            touch_ts=pd.Timestamp(timestamps[i], unit="ns", tz="America/New_York"),
            approach_direction=approach_dir,
            outcome=outcome,
            max_favorable=max_fav,
            max_adverse=max_adv,
            time_to_outcome_sec=time_to,
        ))

        last_touch_ts = timestamps[i]

    return touches


def process_date(lake_root: Path, symbol: str, dt: str, params: OutcomeParams) -> List[Touch]:
    data_path = lake_root / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_with_levels/dt={dt}"

    if not data_path.exists():
        print(f"  Skipping {dt}: no data")
        return []

    df = pd.read_parquet(data_path)
    if len(df) == 0:
        return []

    ts_est = pd.to_datetime(df["ts_event_est"])
    date_only = pd.Timestamp(dt, tz="America/New_York").normalize()

    market_open = date_only + pd.Timedelta(hours=9, minutes=30)
    or_end = date_only + pd.Timedelta(hours=10)
    session_end = date_only + pd.Timedelta(hours=12, minutes=30)

    session_mask = (ts_est >= market_open) & (ts_est <= session_end)
    df_session = df[session_mask].copy()

    if len(df_session) == 0:
        return []

    df_session = df_session.sort_values("ts_event").reset_index(drop=True)

    mid_prices = compute_mid_price(df_session)
    timestamps = df_session["ts_event"].values.astype(np.int64)
    ts_est_session = pd.to_datetime(df_session["ts_event_est"])

    or_end_idx = (ts_est_session >= or_end).idxmax() if (ts_est_session >= or_end).any() else len(df_session)

    levels = compute_levels(df_session)

    all_touches = []

    for level_type, level_price in levels.items():
        if level_type in ["OR_HIGH", "OR_LOW"]:
            valid_start = or_end_idx
        else:
            valid_start = 0

        touches = find_touches_for_level(
            df_session,
            mid_prices,
            timestamps,
            dt,
            symbol,
            level_type,
            level_price,
            valid_start,
            len(df_session),
            params,
        )
        all_touches.extend(touches)

    return all_touches


def generate_summary(touches: List[Touch]) -> pd.DataFrame:
    if not touches:
        return pd.DataFrame()

    rows = []
    for t in touches:
        direction_str = "from_below" if t.approach_direction == 1 else "from_above"
        rows.append({
            "dt": t.dt,
            "symbol": t.symbol,
            "level_type": t.level_type,
            "level_price": t.level_price,
            "touch_ts": t.touch_ts,
            "direction": direction_str,
            "outcome": t.outcome,
            "max_favorable_pts": t.max_favorable,
            "max_adverse_pts": t.max_adverse,
            "time_to_outcome_sec": t.time_to_outcome_sec,
        })

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No touches found.")
        return

    print("\n" + "=" * 80)
    print("PHASE 1: PURE HISTORICAL BASELINE (Zero Features)")
    print("=" * 80)
    print(f"\nParams: Target={10}pts, Stop={5}pts, Window={240}sec")
    print(f"Total touches: {len(df)}")
    print()

    summary = df.groupby(["level_type", "direction", "outcome"]).size().unstack(fill_value=0)

    for (level_type, direction), row in summary.iterrows():
        total = row.sum()
        print(f"\n{level_type} | {direction}")
        print("-" * 40)
        for outcome in ["BREAK", "REJECT", "INDETERMINATE"]:
            count = row.get(outcome, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"  {outcome:15} {count:4d} ({pct:5.1f}%)")
        print(f"  {'TOTAL':15} {total:4d}")

    print("\n" + "=" * 80)
    print("BY LEVEL TYPE (all directions)")
    print("=" * 80)

    level_summary = df.groupby(["level_type", "outcome"]).size().unstack(fill_value=0)
    for level_type, row in level_summary.iterrows():
        total = row.sum()
        break_ct = row.get("BREAK", 0)
        reject_ct = row.get("REJECT", 0)
        indet_ct = row.get("INDETERMINATE", 0)
        break_pct = 100 * break_ct / total if total > 0 else 0
        reject_pct = 100 * reject_ct / total if total > 0 else 0
        print(f"{level_type:10} | BREAK: {break_ct:3d} ({break_pct:5.1f}%) | REJECT: {reject_ct:3d} ({reject_pct:5.1f}%) | INDET: {indet_ct:3d} | Total: {total}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Pure Historical Baseline")
    parser.add_argument("--symbols", type=str, default="ESU5,ESZ5", help="Comma-separated symbols")
    parser.add_argument("--dates", type=str, required=True, help="Date range: YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--target-pts", type=float, default=10.0, help="Target points for BREAK")
    parser.add_argument("--stop-pts", type=float, default=5.0, help="Stop points for REJECT")
    parser.add_argument("--window-sec", type=int, default=240, help="Outcome window in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    start_str, end_str = args.dates.split(":")
    dates = pd.date_range(start_str, end_str, freq="B").strftime("%Y-%m-%d").tolist()

    params = OutcomeParams(
        target_pts=args.target_pts,
        stop_pts=args.stop_pts,
        window_seconds=args.window_sec,
    )

    lake_root = Path(__file__).parent.parent.parent.parent.parent / "lake"

    all_touches = []

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        for dt in dates:
            touches = process_date(lake_root, symbol, dt, params)
            if touches:
                print(f"  {dt}: {len(touches)} touches")
                all_touches.extend(touches)

    df = generate_summary(all_touches)

    print_summary_table(df)

    if args.output and not df.empty:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
