from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import uuid

from .level_universe import Level
from src.common.config import CONFIG


@dataclass
class TouchEpisode:
    episode_id: str
    date: str
    level_kind: str
    level_price: float
    direction: str
    t0_ns: int
    t1_ns: int
    attempt_index: int
    cluster_id: int


@dataclass
class AttemptTrends:
    barrier_replenishment_trend: float = 0.0
    barrier_delta_liq_trend: float = 0.0
    tape_velocity_trend: float = 0.0
    tape_imbalance_trend: float = 0.0


class EpisodeDetector:
    """
    Touch detection + attempt clustering for repeated tests at a level.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self._clusters: Dict[Tuple[str, str, str], Dict[str, float | int]] = {}
        self._first_metrics: Dict[Tuple[str, int], Dict[str, float]] = {}

        self._cluster_time_ns = int(self.config.TOUCH_CLUSTER_TIME_MINUTES * 60 * 1e9)
        self._cluster_price_band = self.config.TOUCH_CLUSTER_PRICE_BAND
        self._confirmation_ns = int(self.config.CONFIRMATION_WINDOW_SECONDS * 1e9)

    def register_touch(
        self,
        level: Level,
        direction: str,
        ts_ns: int
    ) -> TouchEpisode:
        date_str = self._date_from_ts(ts_ns)
        key = (date_str, level.kind.value, direction)

        cluster = self._clusters.get(key)
        if cluster is None:
            cluster = {
                "cluster_id": 0,
                "last_ts_ns": ts_ns,
                "last_price": level.price,
                "attempt_index": 0
            }

        new_cluster = False
        if ts_ns - cluster["last_ts_ns"] > self._cluster_time_ns:
            new_cluster = True
        if abs(level.price - cluster["last_price"]) > self._cluster_price_band:
            new_cluster = True

        if new_cluster:
            cluster["cluster_id"] += 1
            cluster["attempt_index"] = 0

        cluster["attempt_index"] += 1
        cluster["last_ts_ns"] = ts_ns
        cluster["last_price"] = level.price
        self._clusters[key] = cluster

        episode = TouchEpisode(
            episode_id=str(uuid.uuid4()),
            date=date_str,
            level_kind=level.kind.value,
            level_price=level.price,
            direction=direction,
            t0_ns=ts_ns,
            t1_ns=ts_ns + self._confirmation_ns,
            attempt_index=cluster["attempt_index"],
            cluster_id=cluster["cluster_id"]
        )
        return episode

    def compute_attempt_trends(
        self,
        episode: TouchEpisode,
        barrier_replenishment_ratio: Optional[float],
        barrier_delta_liq: Optional[float],
        tape_velocity: Optional[float],
        tape_imbalance: Optional[float]
    ) -> AttemptTrends:
        key = (episode.level_kind, episode.cluster_id)
        if key not in self._first_metrics:
            self._first_metrics[key] = {
                "barrier_replenishment_ratio": barrier_replenishment_ratio or 0.0,
                "barrier_delta_liq": barrier_delta_liq or 0.0,
                "tape_velocity": tape_velocity or 0.0,
                "tape_imbalance": tape_imbalance or 0.0
            }

        first = self._first_metrics[key]
        return AttemptTrends(
            barrier_replenishment_trend=(barrier_replenishment_ratio or 0.0) - first["barrier_replenishment_ratio"],
            barrier_delta_liq_trend=(barrier_delta_liq or 0.0) - first["barrier_delta_liq"],
            tape_velocity_trend=(tape_velocity or 0.0) - first["tape_velocity"],
            tape_imbalance_trend=(tape_imbalance or 0.0) - first["tape_imbalance"]
        )

    @staticmethod
    def _date_from_ts(ts_ns: int) -> str:
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        )
        return dt.strftime("%Y-%m-%d")
