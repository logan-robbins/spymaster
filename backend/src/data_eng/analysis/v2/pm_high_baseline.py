"""
PM_HIGH Touch → Break/Bounce Baseline (2-Minute Candles, Futures Only)

Implements the exact specification from PM_HIGH_TOUCH_OUTCOMES_PLAN.md:
- 2-minute OHLC candles from trade prints only
- Touch detection with OHLC-based debounce
- Outcome labeling with stop-loss aware categories
- Both approach directions: UP_FROM_BELOW, DOWN_FROM_ABOVE
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


class Permutation(str, Enum):
    UP_FROM_BELOW = "UP_FROM_BELOW"
    DOWN_FROM_ABOVE = "DOWN_FROM_ABOVE"


class Outcome(str, Enum):
    BREAK_UP = "BREAK_UP"
    BREAK_UP_STRONG = "BREAK_UP_STRONG"
    BOUNCE_DOWN = "BOUNCE_DOWN"
    BREAK_DOWN = "BREAK_DOWN"
    BREAK_DOWN_STRONG = "BREAK_DOWN_STRONG"
    BOUNCE_UP = "BOUNCE_UP"
    NO_EVENT = "NO_EVENT"


@dataclass
class Candle:
    start_ts: int
    end_ts: int
    open: float
    high: float
    low: float
    close: float
    trade_count: int
    volume: float


@dataclass
class Touch:
    dt: str
    symbol: str
    level_type: str
    level_price: float
    permutation: Permutation
    touch_candle_end_ts: int
    touch_candle_close: float
    touch_candle_open: float
    touch_candle_high: float
    touch_candle_low: float
    outcome: Outcome
    is_window_complete: bool
    window_high: Optional[float]
    window_low: Optional[float]
    max_favorable_pts: float
    max_adverse_pts: float


BAND_HALF = 5.0
STRONG_BREAK_PTS = 10.0
CANDLE_DURATION_NS = 2 * 60 * 1_000_000_000


def build_2min_candles(
    trades_df: pd.DataFrame,
    session_start_ns: int,
    session_end_ns: int,
) -> List[Candle]:
    if trades_df.empty:
        return []

    trades_df = trades_df.sort_values("ts_event").reset_index(drop=True)

    ts_event_ns = trades_df["ts_event"].values.astype(np.int64)
    prices = trades_df["price"].values.astype(np.float64)
    sizes = trades_df["size"].values.astype(np.float64)

    candles = []
    bin_start = session_start_ns
    bin_end = bin_start + CANDLE_DURATION_NS

    while bin_start < session_end_ns:
        mask = (ts_event_ns >= bin_start) & (ts_event_ns < bin_end)
        bin_prices = prices[mask]
        bin_sizes = sizes[mask]

        if len(bin_prices) > 0:
            candle = Candle(
                start_ts=bin_start,
                end_ts=bin_end,
                open=float(bin_prices[0]),
                high=float(bin_prices.max()),
                low=float(bin_prices.min()),
                close=float(bin_prices[-1]),
                trade_count=len(bin_prices),
                volume=float(bin_sizes.sum()),
            )
            candles.append(candle)

        bin_start = bin_end
        bin_end = bin_start + CANDLE_DURATION_NS

    return candles


def candle_intersects_band(candle: Candle, level: float) -> bool:
    band_low = level - BAND_HALF
    band_high = level + BAND_HALF
    return candle.low <= band_high and candle.high >= band_low


def detect_touches_with_debounce(
    candles: List[Candle],
    level: float,
) -> List[tuple[int, Permutation]]:
    if len(candles) < 2:
        return []

    touches = []
    in_zone = False

    band_low = level - BAND_HALF
    band_high = level + BAND_HALF

    for i in range(1, len(candles)):
        curr = candles[i]
        prev = candles[i - 1]

        curr_intersects = candle_intersects_band(curr, level)
        prev_close_in_band = band_low <= prev.close <= band_high
        curr_close_in_band = band_low <= curr.close <= band_high

        if not in_zone:
            if curr_close_in_band:
                if prev.close < band_low:
                    touches.append((i, Permutation.UP_FROM_BELOW))
                    in_zone = True
                elif prev.close > band_high:
                    touches.append((i, Permutation.DOWN_FROM_ABOVE))
                    in_zone = True

        if in_zone and not curr_intersects:
            in_zone = False

    return touches


def label_outcome_up_from_below(
    window_high: float,
    window_low: float,
    level: float,
) -> Outcome:
    if window_low <= level - BAND_HALF:
        return Outcome.BOUNCE_DOWN
    elif window_high >= level + STRONG_BREAK_PTS:
        return Outcome.BREAK_UP_STRONG
    elif window_high >= level + BAND_HALF:
        return Outcome.BREAK_UP
    else:
        return Outcome.NO_EVENT


def label_outcome_down_from_above(
    window_high: float,
    window_low: float,
    level: float,
) -> Outcome:
    if window_high >= level + BAND_HALF:
        return Outcome.BOUNCE_UP
    elif window_low <= level - STRONG_BREAK_PTS:
        return Outcome.BREAK_DOWN_STRONG
    elif window_low <= level - BAND_HALF:
        return Outcome.BREAK_DOWN
    else:
        return Outcome.NO_EVENT


def compute_excursions(
    window_high: float,
    window_low: float,
    level: float,
    permutation: Permutation,
) -> tuple[float, float]:
    if permutation == Permutation.UP_FROM_BELOW:
        max_favorable = window_high - level
        max_adverse = level - window_low
    else:
        max_favorable = level - window_low
        max_adverse = window_high - level
    return max_favorable, max_adverse


def process_date(
    lake_root: Path,
    symbol: str,
    dt: str,
) -> List[Touch]:
    data_path = lake_root / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_session_levels/dt={dt}"

    if not data_path.exists():
        return []

    df = pd.read_parquet(data_path)
    if len(df) == 0:
        return []

    pm_high = df["pm_high"].iloc[0]
    if pd.isna(pm_high):
        print(f"  {dt}: pm_high is NaN, skipping")
        return []

    level = float(pm_high)

    trades_df = df[df["action"] == "T"].copy()
    if trades_df.empty:
        print(f"  {dt}: no trades, skipping")
        return []

    sample_prices = trades_df["price"].head(5).tolist()
    if any(p > 1_000_000 for p in sample_prices):
        raise ValueError(
            f"Price unit mismatch detected on {dt}: prices appear scaled ({sample_prices}). "
            "Fix upstream before continuing."
        )

    date_pd = pd.Timestamp(dt, tz="America/New_York").normalize()
    session_start = date_pd + pd.Timedelta(hours=9, minutes=30)
    session_end = date_pd + pd.Timedelta(hours=12, minutes=30)

    session_start_ns = int(session_start.value)
    session_end_ns = int(session_end.value)

    ts_event_ns = trades_df["ts_event"].values.astype(np.int64)
    session_mask = (ts_event_ns >= session_start_ns) & (ts_event_ns < session_end_ns)
    trades_session = trades_df[session_mask].copy()

    if trades_session.empty:
        print(f"  {dt}: no trades in session window, skipping")
        return []

    candles = build_2min_candles(trades_session, session_start_ns, session_end_ns)

    if len(candles) < 3:
        print(f"  {dt}: too few candles ({len(candles)}), skipping")
        return []

    touches_raw = detect_touches_with_debounce(candles, level)

    touches = []
    for candle_idx, permutation in touches_raw:
        touch_candle = candles[candle_idx]

        window_candles_exist = candle_idx + 2 < len(candles)

        if window_candles_exist:
            c1 = candles[candle_idx + 1]
            c2 = candles[candle_idx + 2]
            window_high = max(c1.high, c2.high)
            window_low = min(c1.low, c2.low)
            is_complete = True

            if permutation == Permutation.UP_FROM_BELOW:
                outcome = label_outcome_up_from_below(window_high, window_low, level)
            else:
                outcome = label_outcome_down_from_above(window_high, window_low, level)

            max_fav, max_adv = compute_excursions(window_high, window_low, level, permutation)
        else:
            window_high = None
            window_low = None
            is_complete = False
            outcome = Outcome.NO_EVENT
            max_fav = 0.0
            max_adv = 0.0

        touch = Touch(
            dt=dt,
            symbol=symbol,
            level_type="PM_HIGH",
            level_price=level,
            permutation=permutation,
            touch_candle_end_ts=touch_candle.end_ts,
            touch_candle_close=touch_candle.close,
            touch_candle_open=touch_candle.open,
            touch_candle_high=touch_candle.high,
            touch_candle_low=touch_candle.low,
            outcome=outcome,
            is_window_complete=is_complete,
            window_high=window_high,
            window_low=window_low,
            max_favorable_pts=max_fav,
            max_adverse_pts=max_adv,
        )
        touches.append(touch)

    return touches


def touches_to_dataframe(touches: List[Touch]) -> pd.DataFrame:
    if not touches:
        return pd.DataFrame()

    rows = []
    for t in touches:
        rows.append({
            "dt": t.dt,
            "symbol": t.symbol,
            "level_type": t.level_type,
            "level_price": t.level_price,
            "permutation": t.permutation.value,
            "touch_candle_end_ts": t.touch_candle_end_ts,
            "touch_candle_close": t.touch_candle_close,
            "touch_candle_open": t.touch_candle_open,
            "touch_candle_high": t.touch_candle_high,
            "touch_candle_low": t.touch_candle_low,
            "outcome": t.outcome.value,
            "is_window_complete": t.is_window_complete,
            "window_high": t.window_high,
            "window_low": t.window_low,
            "max_favorable_pts": t.max_favorable_pts,
            "max_adverse_pts": t.max_adverse_pts,
        })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No touches found.")
        return

    complete_df = df[df["is_window_complete"]].copy()

    print("\n" + "=" * 80)
    print("PM_HIGH TOUCH BASELINE (2-Min Candles, Stop-Loss Aware Outcomes)")
    print("=" * 80)
    print(f"\nTotal touches: {len(df)}")
    print(f"Evaluable touches (window complete): {len(complete_df)}")
    print()

    for perm in ["UP_FROM_BELOW", "DOWN_FROM_ABOVE"]:
        perm_df = complete_df[complete_df["permutation"] == perm]
        if perm_df.empty:
            continue

        print(f"\n{perm}")
        print("-" * 60)

        outcome_counts = perm_df["outcome"].value_counts()
        total = len(perm_df)

        if perm == "UP_FROM_BELOW":
            outcomes_order = ["BREAK_UP_STRONG", "BREAK_UP", "BOUNCE_DOWN", "NO_EVENT"]
        else:
            outcomes_order = ["BREAK_DOWN_STRONG", "BREAK_DOWN", "BOUNCE_UP", "NO_EVENT"]

        for outcome in outcomes_order:
            count = outcome_counts.get(outcome, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"  {outcome:20} {count:4d}  ({pct:5.1f}%)")

        print(f"  {'TOTAL':20} {total:4d}")

    print("\n" + "=" * 80)
    print("BY SYMBOL")
    print("=" * 80)

    for symbol in complete_df["symbol"].unique():
        sym_df = complete_df[complete_df["symbol"] == symbol]
        print(f"\n{symbol}: {len(sym_df)} evaluable touches")

        for perm in ["UP_FROM_BELOW", "DOWN_FROM_ABOVE"]:
            perm_df = sym_df[sym_df["permutation"] == perm]
            if perm_df.empty:
                continue

            total = len(perm_df)
            break_outcomes = perm_df[perm_df["outcome"].str.contains("BREAK")]["outcome"].count()
            bounce_outcomes = perm_df[perm_df["outcome"].str.contains("BOUNCE")]["outcome"].count()
            no_event = perm_df[perm_df["outcome"] == "NO_EVENT"]["outcome"].count()

            break_pct = 100 * break_outcomes / total if total > 0 else 0
            bounce_pct = 100 * bounce_outcomes / total if total > 0 else 0

            print(f"  {perm:20} | Total: {total:3d} | Break: {break_outcomes:3d} ({break_pct:4.1f}%) | Bounce: {bounce_outcomes:3d} ({bounce_pct:4.1f}%) | NoEvent: {no_event:3d}")


def run_verification_checks(df: pd.DataFrame, candle_samples: List[dict]) -> None:
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    if candle_samples:
        print("\n1. CANDLE ALIGNMENT CHECK (sample from first date)")
        for sample in candle_samples[:5]:
            start_ts = pd.Timestamp(sample["start_ts"], unit="ns", tz="UTC").tz_convert("America/New_York")
            print(f"   {start_ts} | O:{sample['open']:.2f} H:{sample['high']:.2f} L:{sample['low']:.2f} C:{sample['close']:.2f} | trades:{sample['count']}")

    if not df.empty:
        print("\n2. TOUCH SANITY CHECK (debounce verification)")
        for symbol in df["symbol"].unique():
            sym_df = df[df["symbol"] == symbol].sort_values("touch_candle_end_ts")
            for dt in sym_df["dt"].unique():
                day_df = sym_df[sym_df["dt"] == dt]
                if len(day_df) > 1:
                    print(f"   {symbol} {dt}: {len(day_df)} touches detected")
                    for _, row in day_df.iterrows():
                        ts = pd.Timestamp(row["touch_candle_end_ts"], unit="ns", tz="UTC").tz_convert("America/New_York")
                        print(f"      {ts.strftime('%H:%M')} | {row['permutation']} | close={row['touch_candle_close']:.2f} | lvl={row['level_price']:.2f} | outcome={row['outcome']}")

        print("\n3. OUTCOME SANITY CHECK (sample touches)")
        sample_df = df[df["is_window_complete"]].head(10)
        for _, row in sample_df.iterrows():
            level = row["level_price"]
            band_low = level - BAND_HALF
            band_high = level + BAND_HALF

            w_high = row["window_high"]
            w_low = row["window_low"]
            outcome = row["outcome"]
            perm = row["permutation"]

            check_passed = "✓"

            if perm == "UP_FROM_BELOW":
                if w_low <= band_low and outcome != "BOUNCE_DOWN":
                    check_passed = "✗ (expected BOUNCE_DOWN)"
                elif w_high >= level + STRONG_BREAK_PTS and outcome not in ["BREAK_UP_STRONG", "BOUNCE_DOWN"]:
                    check_passed = "✗ (expected BREAK_UP_STRONG)"
            else:
                if w_high >= band_high and outcome != "BOUNCE_UP":
                    check_passed = "✗ (expected BOUNCE_UP)"
                elif w_low <= level - STRONG_BREAK_PTS and outcome not in ["BREAK_DOWN_STRONG", "BOUNCE_UP"]:
                    check_passed = "✗ (expected BREAK_DOWN_STRONG)"

            print(f"   {row['dt']} | {perm[:15]:15} | lvl={level:.2f} | wH={w_high:.2f} wL={w_low:.2f} | {outcome:20} | {check_passed}")


def main():
    parser = argparse.ArgumentParser(description="PM_HIGH Touch Baseline (2-Min Candles)")
    parser.add_argument("--symbols", type=str, default="ESU5,ESZ5", help="Comma-separated symbols")
    parser.add_argument("--dates", type=str, required=True, help="Date range: YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for touch-level table")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    start_str, end_str = args.dates.split(":")
    dates = pd.date_range(start_str, end_str, freq="B").strftime("%Y-%m-%d").tolist()

    lake_root = Path(__file__).parent.parent.parent.parent.parent / "lake"

    all_touches = []
    candle_samples = []

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        for dt in dates:
            touches = process_date(lake_root, symbol, dt)
            if touches:
                print(f"  {dt}: {len(touches)} touches")
                all_touches.extend(touches)

                if not candle_samples:
                    data_path = lake_root / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_session_levels/dt={dt}"
                    df_raw = pd.read_parquet(data_path)
                    trades_df = df_raw[df_raw["action"] == "T"].copy()
                    date_pd = pd.Timestamp(dt, tz="America/New_York").normalize()
                    session_start_ns = int((date_pd + pd.Timedelta(hours=9, minutes=30)).value)
                    session_end_ns = int((date_pd + pd.Timedelta(hours=12, minutes=30)).value)
                    ts_event_ns = trades_df["ts_event"].values.astype(np.int64)
                    session_mask = (ts_event_ns >= session_start_ns) & (ts_event_ns < session_end_ns)
                    trades_session = trades_df[session_mask].copy()
                    candles = build_2min_candles(trades_session, session_start_ns, session_end_ns)
                    for c in candles[:5]:
                        candle_samples.append({
                            "start_ts": c.start_ts,
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "count": c.trade_count,
                        })

    df = touches_to_dataframe(all_touches)

    print_summary(df)
    run_verification_checks(df, candle_samples)

    if args.output and not df.empty:
        df.to_csv(args.output, index=False)
        print(f"\nSaved touch-level table to {args.output}")


if __name__ == "__main__":
    main()
