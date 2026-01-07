from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import time

import numpy as np
import pandas as pd
from numba import njit

from src.data_eng.config import load_config
from src.data_eng.io import is_partition_complete, partition_ref, read_partition

LEVEL_TYPES = ["pm_high", "pm_low", "or_high", "or_low"]
ES_CONTRACTS = ["ESH5", "ESM5", "ESU5", "ESZ5", "ESH6"]
MONTH_CODES = {"H": 3, "M": 6, "U": 9, "Z": 12}
EPSILON = 1e-9


def get_front_month_contract(dt: str) -> str:
    from datetime import datetime
    date = datetime.strptime(dt, "%Y-%m-%d")
    for contract in ES_CONTRACTS:
        month_code = contract[2]
        year_digit = contract[3]
        month = MONTH_CODES[month_code]
        year = 2020 + int(year_digit)
        expiry = datetime(year, month, 1)
        if expiry >= date:
            return contract
    raise ValueError(f"No contract for {dt}")


@dataclass
class Episode:
    episode_id: str
    dt: str
    level_type: str
    level_price: float
    approach_direction: int
    outcome: str
    outcome_score: float
    trigger_bar: pd.Series
    lookback_bars: pd.DataFrame


@dataclass
class CompressionStrategy:
    name: str
    description: str
    compress_fn: Callable[[Episode], np.ndarray]
    dim: int


@dataclass
class StrategyResult:
    name: str
    dim: int
    accuracy: float
    top2_accuracy: float
    score_correlation: float
    n_test: int


def load_episodes(
    repo_root: Path,
    dates: List[str],
) -> List[Episode]:
    cfg = load_config(repo_root, repo_root / "src/data_eng/config/datasets.yaml")
    episodes = []

    for dt in dates:
        if pd.to_datetime(dt).weekday() >= 5:
            continue

        symbol = get_front_month_contract(dt)

        for level_type in LEVEL_TYPES:
            dataset_key = f"silver.future.market_by_price_10_{level_type}_approach"
            ref = partition_ref(cfg, dataset_key, symbol, dt)

            if not is_partition_complete(ref):
                continue

            df = read_partition(ref)
            if len(df) == 0:
                continue

            for episode_id in df["episode_id"].unique():
                ep_df = df[df["episode_id"] == episode_id].sort_values("bar_ts")

                trigger_mask = ep_df["is_trigger_bar"] == True
                if not trigger_mask.any():
                    continue

                trigger_idx = trigger_mask.idxmax()
                trigger_bar = ep_df.loc[trigger_idx]
                lookback_bars = ep_df[ep_df["is_pre_trigger"] == True]

                if len(lookback_bars) < 12:
                    continue

                episodes.append(Episode(
                    episode_id=episode_id,
                    dt=dt,
                    level_type=level_type.upper(),
                    level_price=float(trigger_bar.get("level_price", 0)),
                    approach_direction=int(trigger_bar.get("approach_direction", 0)),
                    outcome=str(trigger_bar.get("outcome", "UNKNOWN")),
                    outcome_score=float(trigger_bar.get("outcome_score", 0)),
                    trigger_bar=trigger_bar,
                    lookback_bars=lookback_bars,
                ))

    return episodes


APPROACH_FEATURES = [
    "bar5s_approach_dist_to_level_pts_eob",
    "bar5s_approach_side_of_level_eob",
]

OBI_FEATURES = [
    "bar5s_state_obi0_eob",
    "bar5s_state_obi10_eob",
]

CDI_FEATURES = [
    "bar5s_state_cdi_p0_1_eob",
    "bar5s_state_cdi_p1_2_eob",
    "bar5s_state_cdi_p2_3_eob",
]

DEPTH_FEATURES = [
    "bar5s_depth_bid10_qty_eob",
    "bar5s_depth_ask10_qty_eob",
    "bar5s_lvl_depth_above_qty_eob",
    "bar5s_lvl_depth_below_qty_eob",
    "bar5s_lvl_depth_imbal_eob",
]

WALL_FEATURES = [
    "bar5s_wall_bid_maxz_eob",
    "bar5s_wall_ask_maxz_eob",
    "bar5s_wall_bid_nearest_strong_dist_pts_eob",
    "bar5s_wall_ask_nearest_strong_dist_pts_eob",
]

FLOW_FEATURES = [
    "bar5s_lvl_flow_toward_net_sum",
    "bar5s_lvl_flow_away_net_sum",
    "bar5s_cumul_flow_net_bid",
    "bar5s_cumul_flow_net_ask",
    "bar5s_cumul_flow_imbal",
]

TRADE_FEATURES = [
    "bar5s_trade_signed_vol_sum",
    "bar5s_trade_vol_sum",
    "bar5s_cumul_signed_trade_vol",
]

VELOCITY_FEATURES = [
    "bar5s_deriv_dist_d1_w3",
    "bar5s_deriv_dist_d1_w12",
    "bar5s_deriv_dist_d1_w36",
]

ACCEL_FEATURES = [
    "bar5s_deriv_dist_d2_w3",
    "bar5s_deriv_dist_d2_w12",
    "bar5s_deriv_dist_d2_w36",
]

ALL_FEATURES = (
    APPROACH_FEATURES + OBI_FEATURES + CDI_FEATURES +
    DEPTH_FEATURES + WALL_FEATURES + FLOW_FEATURES +
    TRADE_FEATURES + VELOCITY_FEATURES + ACCEL_FEATURES
)


def safe_get(series: pd.Series, col: str, default: float = 0.0) -> float:
    val = series.get(col, default)
    if pd.isna(val):
        return default
    return float(val)


def safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df))
    return df[col].fillna(0).values


def agg_mean(vals: np.ndarray) -> float:
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def agg_std(vals: np.ndarray) -> float:
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def agg_delta(vals: np.ndarray) -> float:
    return float(vals[-1] - vals[0]) if len(vals) > 1 else 0.0


def agg_min(vals: np.ndarray) -> float:
    return float(np.min(vals)) if len(vals) > 0 else 0.0


def agg_max(vals: np.ndarray) -> float:
    return float(np.max(vals)) if len(vals) > 0 else 0.0


def agg_last(vals: np.ndarray) -> float:
    return float(vals[-1]) if len(vals) > 0 else 0.0


def compress_trigger_only(ep: Episode) -> np.ndarray:
    vec = []
    for feat in ALL_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))
    return np.array(vec, dtype=np.float64)


def compress_mean_only(ep: Episode) -> np.ndarray:
    vec = []
    for feat in ALL_FEATURES:
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_mean(vals))
    return np.array(vec, dtype=np.float64)


def compress_mean_std(ep: Episode) -> np.ndarray:
    vec = []
    for feat in ALL_FEATURES:
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_mean(vals))
        vec.append(agg_std(vals))
    return np.array(vec, dtype=np.float64)


def compress_mean_std_delta(ep: Episode) -> np.ndarray:
    vec = []
    for feat in ALL_FEATURES:
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_mean(vals))
        vec.append(agg_std(vals))
        vec.append(agg_delta(vals))
    return np.array(vec, dtype=np.float64)


def compress_trigger_plus_delta(ep: Episode) -> np.ndarray:
    vec = []
    for feat in ALL_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_delta(vals))
    return np.array(vec, dtype=np.float64)


def compress_trigger_plus_mean_delta(ep: Episode) -> np.ndarray:
    vec = []
    for feat in ALL_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_mean(vals))
        vec.append(agg_delta(vals))
    return np.array(vec, dtype=np.float64)


def compress_thirds(ep: Episode) -> np.ndarray:
    n = len(ep.lookback_bars)
    third = max(1, n // 3)
    early = ep.lookback_bars.iloc[:third]
    mid = ep.lookback_bars.iloc[third:2*third]
    late = ep.lookback_bars.iloc[2*third:]

    vec = []
    for feat in ALL_FEATURES:
        vec.append(agg_mean(safe_col(early, feat)))
        vec.append(agg_mean(safe_col(mid, feat)))
        vec.append(agg_mean(safe_col(late, feat)))
    return np.array(vec, dtype=np.float64)


def compress_thirds_with_trigger(ep: Episode) -> np.ndarray:
    n = len(ep.lookback_bars)
    third = max(1, n // 3)
    early = ep.lookback_bars.iloc[:third]
    mid = ep.lookback_bars.iloc[third:2*third]
    late = ep.lookback_bars.iloc[2*third:]

    vec = []
    for feat in ALL_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))
        vec.append(agg_mean(safe_col(early, feat)))
        vec.append(agg_mean(safe_col(mid, feat)))
        vec.append(agg_mean(safe_col(late, feat)))
    return np.array(vec, dtype=np.float64)


def compress_last_n_bars(n_bars: int):
    def _compress(ep: Episode) -> np.ndarray:
        recent = ep.lookback_bars.tail(n_bars)
        vec = []
        for feat in ALL_FEATURES:
            vals = safe_col(recent, feat)
            vec.append(agg_mean(vals))
            vec.append(agg_delta(vals))
        vec.append(safe_get(ep.trigger_bar, "bar5s_approach_dist_to_level_pts_eob"))
        return np.array(vec, dtype=np.float64)
    return _compress


def compress_multi_window(ep: Episode) -> np.ndarray:
    windows = [12, 36, 72, 180]
    vec = []

    for feat in ALL_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))

        for w in windows:
            recent = ep.lookback_bars.tail(w)
            vals = safe_col(recent, feat)
            vec.append(agg_mean(vals))

    return np.array(vec, dtype=np.float64)


def compress_family_focused(ep: Episode) -> np.ndarray:
    vec = []

    for feat in APPROACH_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_mean(vals))
        vec.append(agg_std(vals))
        vec.append(agg_delta(vals))

    for feat in OBI_FEATURES + CDI_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_mean(vals))
        vec.append(agg_delta(vals))
        recent = safe_col(ep.lookback_bars.tail(12), feat)
        vec.append(agg_mean(recent))

    for feat in WALL_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_max(vals))
        vec.append(agg_mean(vals))

    for feat in FLOW_FEATURES + TRADE_FEATURES:
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(agg_last(vals))
        n = len(ep.lookback_bars)
        third = max(1, n // 3)
        early = safe_col(ep.lookback_bars.iloc[:third], feat)
        late = safe_col(ep.lookback_bars.iloc[2*third:], feat)
        vec.append(agg_mean(late) - agg_mean(early))

    for feat in VELOCITY_FEATURES + ACCEL_FEATURES:
        vec.append(safe_get(ep.trigger_bar, feat))
        recent = safe_col(ep.lookback_bars.tail(12), feat)
        vec.append(agg_mean(recent))

    return np.array(vec, dtype=np.float64)


def compress_minimal_core(ep: Episode) -> np.ndarray:
    vec = []

    vec.append(safe_get(ep.trigger_bar, "bar5s_approach_dist_to_level_pts_eob"))
    vec.append(safe_get(ep.trigger_bar, "bar5s_deriv_dist_d1_w12"))
    vec.append(safe_get(ep.trigger_bar, "bar5s_deriv_dist_d2_w12"))

    vec.append(safe_get(ep.trigger_bar, "bar5s_state_obi0_eob"))
    vec.append(safe_get(ep.trigger_bar, "bar5s_state_obi10_eob"))
    obi_vals = safe_col(ep.lookback_bars, "bar5s_state_obi0_eob")
    vec.append(agg_delta(obi_vals))

    vec.append(safe_get(ep.trigger_bar, "bar5s_wall_bid_maxz_eob"))
    vec.append(safe_get(ep.trigger_bar, "bar5s_wall_ask_maxz_eob"))

    vec.append(safe_get(ep.trigger_bar, "bar5s_lvl_depth_imbal_eob"))
    depth_vals = safe_col(ep.lookback_bars, "bar5s_lvl_depth_imbal_eob")
    vec.append(agg_delta(depth_vals))

    flow_toward = safe_col(ep.lookback_bars, "bar5s_lvl_flow_toward_net_sum")
    flow_away = safe_col(ep.lookback_bars, "bar5s_lvl_flow_away_net_sum")
    vec.append(np.sum(flow_toward))
    vec.append(np.sum(flow_away))

    signed_vol = safe_col(ep.lookback_bars, "bar5s_trade_signed_vol_sum")
    vec.append(np.sum(signed_vol))

    n = len(ep.lookback_bars)
    third = max(1, n // 3)
    early_vol = np.sum(safe_col(ep.lookback_bars.iloc[:third], "bar5s_trade_vol_sum"))
    late_vol = np.sum(safe_col(ep.lookback_bars.iloc[2*third:], "bar5s_trade_vol_sum"))
    vec.append(late_vol - early_vol)

    return np.array(vec, dtype=np.float64)


def build_strategies() -> List[CompressionStrategy]:
    strategies = [
        CompressionStrategy(
            name="trigger_only",
            description="Just the trigger bar snapshot",
            compress_fn=compress_trigger_only,
            dim=len(ALL_FEATURES),
        ),
        CompressionStrategy(
            name="mean_only",
            description="Mean of each feature over lookback",
            compress_fn=compress_mean_only,
            dim=len(ALL_FEATURES),
        ),
        CompressionStrategy(
            name="mean_std",
            description="Mean + std of each feature",
            compress_fn=compress_mean_std,
            dim=len(ALL_FEATURES) * 2,
        ),
        CompressionStrategy(
            name="mean_std_delta",
            description="Mean + std + delta of each feature",
            compress_fn=compress_mean_std_delta,
            dim=len(ALL_FEATURES) * 3,
        ),
        CompressionStrategy(
            name="trigger_plus_delta",
            description="Trigger value + delta over lookback",
            compress_fn=compress_trigger_plus_delta,
            dim=len(ALL_FEATURES) * 2,
        ),
        CompressionStrategy(
            name="trigger_plus_mean_delta",
            description="Trigger + mean + delta",
            compress_fn=compress_trigger_plus_mean_delta,
            dim=len(ALL_FEATURES) * 3,
        ),
        CompressionStrategy(
            name="thirds",
            description="Mean of early/mid/late thirds",
            compress_fn=compress_thirds,
            dim=len(ALL_FEATURES) * 3,
        ),
        CompressionStrategy(
            name="thirds_with_trigger",
            description="Trigger + early/mid/late means",
            compress_fn=compress_thirds_with_trigger,
            dim=len(ALL_FEATURES) * 4,
        ),
        CompressionStrategy(
            name="last_12_bars",
            description="Last 1 minute: mean + delta",
            compress_fn=compress_last_n_bars(12),
            dim=len(ALL_FEATURES) * 2 + 1,
        ),
        CompressionStrategy(
            name="last_36_bars",
            description="Last 3 minutes: mean + delta",
            compress_fn=compress_last_n_bars(36),
            dim=len(ALL_FEATURES) * 2 + 1,
        ),
        CompressionStrategy(
            name="last_72_bars",
            description="Last 6 minutes: mean + delta",
            compress_fn=compress_last_n_bars(72),
            dim=len(ALL_FEATURES) * 2 + 1,
        ),
        CompressionStrategy(
            name="multi_window",
            description="Trigger + means at 1/3/6/15 min windows",
            compress_fn=compress_multi_window,
            dim=len(ALL_FEATURES) * 5,
        ),
        CompressionStrategy(
            name="family_focused",
            description="Feature-family-aware aggregations",
            compress_fn=compress_family_focused,
            dim=2*4 + 5*4 + 4*3 + 8*2 + 6*2,
        ),
        CompressionStrategy(
            name="minimal_core",
            description="15 hand-picked core dimensions",
            compress_fn=compress_minimal_core,
            dim=15,
        ),
    ]
    return strategies


def normalize_vectors(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(vectors, axis=0)
    std = np.nanstd(vectors, axis=0)
    std = np.where(std < EPSILON, 1.0, std)

    normalized = (vectors - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(normalized, -10, 10)

    return normalized, mean, std


@njit
def knn_accuracy(
    train_vecs: np.ndarray,
    train_outcomes: np.ndarray,
    test_vecs: np.ndarray,
    test_outcomes: np.ndarray,
    k: int,
    n_outcomes: int,
) -> Tuple[float, float]:
    n_test = len(test_vecs)
    n_train = len(train_vecs)

    train_norms = np.sqrt(np.sum(train_vecs * train_vecs, axis=1)) + EPSILON
    test_norms = np.sqrt(np.sum(test_vecs * test_vecs, axis=1)) + EPSILON

    correct = 0
    top2_correct = 0

    for i in range(n_test):
        sims = np.zeros(n_train)
        for j in range(n_train):
            dot = 0.0
            for d in range(train_vecs.shape[1]):
                dot += test_vecs[i, d] * train_vecs[j, d]
            sims[j] = dot / (test_norms[i] * train_norms[j])

        idx = np.argsort(-sims)[:k]

        outcome_counts = np.zeros(n_outcomes)
        for j in idx:
            outcome_counts[train_outcomes[j]] += 1

        pred = np.argmax(outcome_counts)
        if pred == test_outcomes[i]:
            correct += 1

        sorted_outcomes = np.argsort(-outcome_counts)
        if test_outcomes[i] == sorted_outcomes[0] or test_outcomes[i] == sorted_outcomes[1]:
            top2_correct += 1

    return correct / n_test, top2_correct / n_test


def evaluate_strategy(
    strategy: CompressionStrategy,
    train_episodes: List[Episode],
    test_episodes: List[Episode],
    k: int = 10,
) -> StrategyResult:
    train_vectors = []
    train_outcomes = []
    train_scores = []

    for ep in train_episodes:
        try:
            vec = strategy.compress_fn(ep)
            train_vectors.append(vec)
            train_outcomes.append(ep.outcome)
            train_scores.append(ep.outcome_score)
        except Exception:
            continue

    test_vectors = []
    test_outcomes = []
    test_scores = []

    for ep in test_episodes:
        try:
            vec = strategy.compress_fn(ep)
            test_vectors.append(vec)
            test_outcomes.append(ep.outcome)
            test_scores.append(ep.outcome_score)
        except Exception:
            continue

    if len(train_vectors) < 10 or len(test_vectors) < 5:
        return StrategyResult(
            name=strategy.name,
            dim=strategy.dim,
            accuracy=0.0,
            top2_accuracy=0.0,
            score_correlation=0.0,
            n_test=len(test_vectors),
        )

    train_vecs = np.array(train_vectors)
    test_vecs = np.array(test_vectors)

    all_vecs = np.vstack([train_vecs, test_vecs])
    all_norm, mean, std = normalize_vectors(all_vecs)
    train_norm = all_norm[:len(train_vecs)]
    test_norm = all_norm[len(train_vecs):]

    all_outcomes = train_outcomes + test_outcomes
    outcome_labels = sorted(set(all_outcomes))
    outcome_map = {o: i for i, o in enumerate(outcome_labels)}

    train_outcome_idx = np.array([outcome_map[o] for o in train_outcomes], dtype=np.int32)
    test_outcome_idx = np.array([outcome_map[o] for o in test_outcomes], dtype=np.int32)

    accuracy, top2_accuracy = knn_accuracy(
        train_norm.astype(np.float64),
        train_outcome_idx,
        test_norm.astype(np.float64),
        test_outcome_idx,
        k,
        len(outcome_labels),
    )

    train_norms = np.linalg.norm(train_norm, axis=1, keepdims=True) + EPSILON
    test_norms = np.linalg.norm(test_norm, axis=1, keepdims=True) + EPSILON
    train_unit = train_norm / train_norms
    test_unit = test_norm / test_norms

    sim = test_unit @ train_unit.T
    top_k_idx = np.argsort(-sim, axis=1)[:, :k]

    predicted_scores = []
    for i in range(len(test_vectors)):
        neighbor_scores = [train_scores[j] for j in top_k_idx[i]]
        predicted_scores.append(np.mean(neighbor_scores))

    if len(predicted_scores) > 1 and np.std(predicted_scores) > EPSILON:
        score_corr = np.corrcoef(predicted_scores, test_scores)[0, 1]
        if np.isnan(score_corr):
            score_corr = 0.0
    else:
        score_corr = 0.0

    return StrategyResult(
        name=strategy.name,
        dim=strategy.dim,
        accuracy=float(accuracy),
        top2_accuracy=float(top2_accuracy),
        score_correlation=float(score_corr),
        n_test=len(test_vectors),
    )


def run_analysis(
    repo_root: Path,
    dates: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    t0 = time.time()

    print("Loading episodes...", flush=True)
    episodes = load_episodes(repo_root, dates)
    print(f"Loaded {len(episodes)} episodes in {time.time() - t0:.1f}s", flush=True)

    if len(episodes) < 20:
        raise ValueError(f"Not enough episodes: {len(episodes)}")

    episodes_by_date = {}
    for ep in episodes:
        episodes_by_date.setdefault(ep.dt, []).append(ep)

    sorted_dates = sorted(episodes_by_date.keys())
    split_idx = int(len(sorted_dates) * 0.8)
    train_dates = set(sorted_dates[:split_idx])
    test_dates = set(sorted_dates[split_idx:])

    train_episodes = [ep for ep in episodes if ep.dt in train_dates]
    test_episodes = [ep for ep in episodes if ep.dt in test_dates]

    print(f"Train: {len(train_episodes)} episodes ({len(train_dates)} days)", flush=True)
    print(f"Test: {len(test_episodes)} episodes ({len(test_dates)} days)", flush=True)

    strategies = build_strategies()
    results = []

    for strategy in strategies:
        print(f"Evaluating: {strategy.name}...", flush=True)
        result = evaluate_strategy(strategy, train_episodes, test_episodes)
        results.append(result)
        print(f"  -> accuracy={result.accuracy:.1%}, top2={result.top2_accuracy:.1%}, dim={result.dim}", flush=True)

    results.sort(key=lambda r: r.accuracy, reverse=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = [
        {
            "rank": i + 1,
            "name": r.name,
            "dim": r.dim,
            "accuracy": r.accuracy,
            "top2_accuracy": r.top2_accuracy,
            "score_correlation": r.score_correlation,
            "n_test": r.n_test,
        }
        for i, r in enumerate(results)
    ]

    (output_dir / "strategy_ranking.json").write_text(
        json.dumps(results_data, indent=2)
    )

    summary = {
        "n_episodes": len(episodes),
        "n_train": len(train_episodes),
        "n_test": len(test_episodes),
        "train_dates": sorted(train_dates),
        "test_dates": sorted(test_dates),
        "n_strategies_tested": len(strategies),
        "best_strategy": results[0].name if results else None,
        "best_accuracy": results[0].accuracy if results else 0.0,
        "feature_families": {
            "approach": APPROACH_FEATURES,
            "obi": OBI_FEATURES,
            "cdi": CDI_FEATURES,
            "depth": DEPTH_FEATURES,
            "wall": WALL_FEATURES,
            "flow": FLOW_FEATURES,
            "trade": TRADE_FEATURES,
            "velocity": VELOCITY_FEATURES,
            "accel": ACCEL_FEATURES,
        },
        "total_features_per_bar": len(ALL_FEATURES),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nAnalysis complete in {time.time() - t0:.1f}s", flush=True)
    print(f"Results saved to {output_dir}", flush=True)

    return {"summary": summary, "results": results_data}


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode compression strategy analysis")
    parser.add_argument("--dates", required=True, help="Date range (YYYY-MM-DD:YYYY-MM-DD)")
    args = parser.parse_args()

    from src.data_eng.utils import expand_date_range
    dates = expand_date_range(dates=args.dates)

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "ablation_results" / "episode_compression_analysis"

    result = run_analysis(repo_root, dates, output_dir)

    print("\n=== Strategy Ranking ===")
    for r in result["results"]:
        print(f"{r['rank']:2d}. {r['name']:25s} acc={r['accuracy']:.1%} top2={r['top2_accuracy']:.1%} dim={r['dim']:3d}")


if __name__ == "__main__":
    main()
