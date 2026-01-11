from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .query import SimilarVector, TriggerVectorRetriever

H_FIRE = 1
K = 200
K_RAW = 2000
W_POW = 8

STOP_TICKS = 6.0
MIN_RESOLVE_RATE = 0.60
MAX_WHIPSAW_RATE = 0.25

P_MIN = 0.70
MARGIN_MIN = 0.20
P_CHOP_MAX = 0.35

COOLDOWN_WINDOWS = 6
MIN_GAP_WINDOWS = 3

LONG_CLASSES = {"BREAK_UP", "REJECT_UP"}
SHORT_CLASSES = {"BREAK_DOWN", "REJECT_DOWN"}


@dataclass(frozen=True)
class TriggerMetrics:
    p_break: float
    p_reject: float
    p_chop: float
    p_top1: float
    margin: float
    risk_q80_ticks: float
    resolve_rate: float
    whipsaw_rate: float
    c_top1: str
    neighbors_ok: bool


@dataclass(frozen=True)
class TriggerDecision:
    fire_flag: int
    signal: str
    metrics: TriggerMetrics


@dataclass(frozen=True)
class TriggerThresholds:
    p_min: float
    margin_min: float
    p_chop_max: float
    stop_ticks: float
    min_resolve_rate: float
    max_whipsaw_rate: float


class TriggerEngine:
    def __init__(
        self,
        retriever: TriggerVectorRetriever,
        k: int = K,
        k_raw: int = K_RAW,
        w_pow: int = W_POW,
    ) -> None:
        self.retriever = retriever
        self.k = int(k)
        self.k_raw = int(k_raw)
        self.w_pow = int(w_pow)

    def score_vector(
        self,
        level_id: str,
        approach_dir: str,
        vector: np.ndarray,
        session_date: str | None = None,
        exclude_session_date: bool = False,
    ) -> TriggerMetrics:
        neighbors = self.retriever.find_similar(
            level_id=level_id,
            approach_dir=approach_dir,
            vector=vector,
            k=self.k_raw,
        )
        if exclude_session_date and session_date is not None:
            neighbors = [
                n for n in neighbors if n.metadata.get("session_date") != session_date
            ]
        if len(neighbors) < self.k:
            return _empty_metrics()
        return compute_metrics(neighbors[: self.k], approach_dir, self.w_pow)

    def decide(
        self,
        metrics: TriggerMetrics,
        thresholds: TriggerThresholds,
    ) -> TriggerDecision:
        fire_flag, signal = apply_fire_rule(metrics, thresholds)
        return TriggerDecision(fire_flag=fire_flag, signal=signal, metrics=metrics)


class EpisodeGate:
    def __init__(
        self,
        cooldown_windows: int = COOLDOWN_WINDOWS,
        min_gap_windows: int = MIN_GAP_WINDOWS,
    ) -> None:
        self.cooldown_windows = int(cooldown_windows)
        self.min_gap_windows = int(min_gap_windows)
        self.episode_id = 0
        self.prev_dir: str | None = None
        self.none_streak = 0
        self.last_flip_idx: int | None = None
        self.cooldown_left = 0
        self.fired_episodes: set[int] = set()

    def step(self, idx: int, approach_dir: str) -> tuple[int, bool]:
        cooldown_block = False
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            cooldown_block = True

        if self.prev_dir is None or approach_dir != self.prev_dir:
            self.episode_id += 1
            self.prev_dir = approach_dir
            if approach_dir == "approach_none":
                self.none_streak = 1
            else:
                self.none_streak = 0
            if approach_dir in ("approach_up", "approach_down"):
                self.last_flip_idx = idx
            else:
                self.last_flip_idx = None
        else:
            if approach_dir == "approach_none":
                self.none_streak += 1
                if self.none_streak == 3:
                    self.episode_id += 1
            else:
                self.none_streak = 0

        gap_block = False
        if approach_dir in ("approach_up", "approach_down") and self.last_flip_idx is not None:
            if idx - self.last_flip_idx < self.min_gap_windows:
                gap_block = True

        episode_block = self.episode_id in self.fired_episodes
        return self.episode_id, (cooldown_block or gap_block or episode_block)

    def register_fire(self, episode_id: int) -> None:
        self.fired_episodes.add(int(episode_id))
        self.cooldown_left = self.cooldown_windows


def compute_metrics(
    neighbors: List[SimilarVector],
    approach_dir: str,
    w_pow: int,
) -> TriggerMetrics:
    if len(neighbors) == 0:
        return _empty_metrics()

    class_map = _class_map(approach_dir)
    sims = np.array([max(n.similarity, 0.0) for n in neighbors], dtype=np.float64)
    weights = np.power(sims, float(w_pow))
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        return _empty_metrics()

    label_key = "true_outcome_h1"
    counts = {name: 0.0 for name in class_map.values()}
    for n, w in zip(neighbors, weights):
        label = str(n.metadata.get(label_key, "CHOP"))
        if label == "WHIPSAW":
            label = "CHOP"
        if label not in counts:
            label = "CHOP"
        counts[label] += float(w)

    p_break = counts[class_map["break"]] / w_sum
    p_reject = counts[class_map["reject"]] / w_sum
    p_chop = counts["CHOP"] / w_sum

    probs = {
        class_map["break"]: p_break,
        class_map["reject"]: p_reject,
        "CHOP": p_chop,
    }
    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    c_top1, p_top1 = sorted_probs[0]
    p_top2 = sorted_probs[1][1]
    margin = float(p_top1 - p_top2)

    risk_vals = []
    resolve_vals = []
    whipsaw_vals = []
    for n in neighbors:
        risk_vals.append(_risk_ticks(n.metadata, approach_dir, c_top1))
        resolve_vals.append(_resolve_flag(n.metadata))
        whipsaw_vals.append(_whipsaw_flag(n.metadata))

    risk_arr = np.array(risk_vals, dtype=np.float64)
    resolve_arr = np.array(resolve_vals, dtype=np.float64)
    whipsaw_arr = np.array(whipsaw_vals, dtype=np.float64)

    risk_q80 = weighted_quantile(risk_arr, weights, 0.80)
    resolve_rate = float(np.sum(weights * resolve_arr) / w_sum)
    whipsaw_rate = float(np.sum(weights * whipsaw_arr) / w_sum)

    return TriggerMetrics(
        p_break=float(p_break),
        p_reject=float(p_reject),
        p_chop=float(p_chop),
        p_top1=float(p_top1),
        margin=margin,
        risk_q80_ticks=risk_q80,
        resolve_rate=resolve_rate,
        whipsaw_rate=whipsaw_rate,
        c_top1=str(c_top1),
        neighbors_ok=True,
    )


def apply_fire_rule(
    metrics: TriggerMetrics,
    thresholds: TriggerThresholds,
) -> tuple[int, str]:
    if not metrics.neighbors_ok:
        return 0, "NONE"
    if metrics.p_top1 < thresholds.p_min:
        return 0, "NONE"
    if metrics.margin < thresholds.margin_min:
        return 0, "NONE"
    if metrics.p_chop > thresholds.p_chop_max:
        return 0, "NONE"
    if metrics.risk_q80_ticks > thresholds.stop_ticks:
        return 0, "NONE"
    if metrics.resolve_rate < thresholds.min_resolve_rate:
        return 0, "NONE"
    if metrics.whipsaw_rate > thresholds.max_whipsaw_rate:
        return 0, "NONE"
    if metrics.c_top1 == "CHOP":
        return 0, "NONE"
    if metrics.c_top1 in LONG_CLASSES:
        return 1, "LONG"
    if metrics.c_top1 in SHORT_CLASSES:
        return 1, "SHORT"
    return 0, "NONE"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or weights.size == 0:
        return 0.0
    if float(np.sum(weights)) <= 0.0:
        return 0.0
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    cum = np.cumsum(w_sorted)
    target = q * float(cum[-1])
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), v_sorted.shape[0] - 1)
    return float(v_sorted[idx])


def _empty_metrics() -> TriggerMetrics:
    return TriggerMetrics(
        p_break=0.0,
        p_reject=0.0,
        p_chop=0.0,
        p_top1=0.0,
        margin=0.0,
        risk_q80_ticks=0.0,
        resolve_rate=0.0,
        whipsaw_rate=0.0,
        c_top1="CHOP",
        neighbors_ok=False,
    )


def _class_map(approach_dir: str) -> Dict[str, str]:
    if approach_dir == "approach_up":
        return {"break": "BREAK_UP", "reject": "REJECT_DOWN", "chop": "CHOP"}
    if approach_dir == "approach_down":
        return {"break": "BREAK_DOWN", "reject": "REJECT_UP", "chop": "CHOP"}
    raise ValueError(f"Unexpected approach_dir: {approach_dir}")


def _risk_ticks(meta: Dict[str, object], approach_dir: str, c_top1: str) -> float:
    upper = _to_float(meta.get("mae_before_upper_ticks"))
    lower = _to_float(meta.get("mae_before_lower_ticks"))
    if approach_dir == "approach_up":
        if c_top1 == "BREAK_UP":
            return upper
        if c_top1 == "REJECT_DOWN":
            return lower
        return max(upper, lower)
    if approach_dir == "approach_down":
        if c_top1 == "BREAK_DOWN":
            return lower
        if c_top1 == "REJECT_UP":
            return upper
        return max(upper, lower)
    return max(upper, lower)


def _resolve_flag(meta: Dict[str, object]) -> float:
    offset = meta.get("first_hit_bar_offset")
    if offset is None:
        return 0.0
    if isinstance(offset, str):
        return 0.0
    if isinstance(offset, float) and np.isnan(offset):
        return 0.0
    return 1.0 if int(offset) <= 1 else 0.0


def _whipsaw_flag(meta: Dict[str, object]) -> float:
    flag = meta.get("whipsaw_flag")
    if flag is None:
        return 0.0
    if isinstance(flag, float) and np.isnan(flag):
        return 0.0
    return 1.0 if int(flag) == 1 else 0.0


def _to_float(val: object) -> float:
    if val is None:
        return 0.0
    if isinstance(val, str):
        return 0.0
    try:
        out = float(val)
        if np.isnan(out):
            return 0.0
        return out
    except (TypeError, ValueError):
        return 0.0
