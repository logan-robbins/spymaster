from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import math

from .level_universe import Level, LevelKind
from .market_state import MarketState
from .fuel_engine import FuelEngine
from src.common.config import CONFIG


class ViewportState(str, Enum):
    FAR = "FAR"
    APPROACHING = "APPROACHING"
    IN_MONITOR_BAND = "IN_MONITOR_BAND"
    TOUCH = "TOUCH"
    CONFIRMATION = "CONFIRMATION"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True)
class ViewportPin:
    kind: LevelKind
    price: Optional[float] = None


@dataclass
class ViewportTarget:
    level: Level
    state: ViewportState
    pinned: bool
    relevance: float
    touch_ts_ns: Optional[int] = None
    confirm_ts_ns: Optional[int] = None


class ViewportManager:
    """
    Maintain a ranked viewport of active target levels with persistence and migration.
    """

    def __init__(
        self,
        fuel_engine: Optional[FuelEngine] = None,
        config=None,
        trading_date: Optional[str] = None
    ):
        self.config = config or CONFIG
        self.fuel_engine = fuel_engine
        self._pins: List[ViewportPin] = []
        self._targets: Dict[str, ViewportTarget] = {}
        self._trading_date = trading_date

        self.scan_radius = self.config.VIEWPORT_SCAN_RADIUS
        self.monitor_band = self.config.MONITOR_BAND
        self.touch_band = self.config.TOUCH_BAND
        self.confirmation_ns = int(self.config.CONFIRMATION_WINDOW_SECONDS * 1e9)
        self.max_targets = self.config.VIEWPORT_MAX_TARGETS

        self.w_distance = self.config.VIEWPORT_W_DISTANCE
        self.w_velocity = self.config.VIEWPORT_W_VELOCITY
        self.w_confluence = self.config.VIEWPORT_W_CONFLUENCE
        self.w_gamma = self.config.VIEWPORT_W_GAMMA

    def pin_target(self, kind: LevelKind | str, price: Optional[float] = None) -> None:
        if isinstance(kind, str):
            kind = LevelKind(kind)
        pin = ViewportPin(kind=kind, price=price)
        if pin not in self._pins:
            self._pins.append(pin)

    def unpin_target(self, kind: LevelKind | str, price: Optional[float] = None) -> None:
        if isinstance(kind, str):
            kind = LevelKind(kind)
        pin = ViewportPin(kind=kind, price=price)
        self._pins = [p for p in self._pins if p != pin]

    def clear_pins(self) -> None:
        self._pins = []

    def update(
        self,
        universe: List[Level],
        market_state: MarketState,
        ts_ns: Optional[int] = None
    ) -> List[ViewportTarget]:
        spot = market_state.get_spot()
        if spot is None:
            return []
        ts_ns = ts_ns if ts_ns is not None else market_state.get_current_ts_ns()

        pins = self._resolve_pins(universe)
        scored_levels = []
        for level in universe:
            distance = abs(spot - level.price)
            approach_velocity = self._compute_approach_velocity(level, spot, market_state)
            confluence_score = self._compute_confluence_score(level, universe)
            gamma_score = self._compute_gamma_score(level, market_state)
            relevance = self._compute_relevance(distance, approach_velocity, confluence_score, gamma_score)
            scored_levels.append((level, distance, approach_velocity, relevance))

        # Auto-focus within scan radius.
        auto_candidates = [
            (level, relevance)
            for level, distance, _, relevance in scored_levels
            if distance <= self.scan_radius
        ]
        auto_candidates.sort(key=lambda item: item[1], reverse=True)

        auto_levels = [level for level, _ in auto_candidates[: self.max_targets]]
        active_ids = {level.id for level in auto_levels}
        active_ids.update(level.id for level in pins)

        updated_targets: Dict[str, ViewportTarget] = {}
        for level, distance, approach_velocity, relevance in scored_levels:
            if level.id not in active_ids:
                continue
            prior = self._targets.get(level.id)
            state, touch_ts_ns, confirm_ts_ns = self._next_state(
                prior, level, distance, approach_velocity, ts_ns
            )
            pinned = any(self._is_pin_match(pin, level) for pin in self._pins)
            updated_targets[level.id] = ViewportTarget(
                level=level,
                state=state,
                pinned=pinned,
                relevance=relevance,
                touch_ts_ns=touch_ts_ns,
                confirm_ts_ns=confirm_ts_ns
            )

        # Persist resolved pinned targets if they remain near.
        for level_id, target in self._targets.items():
            if level_id in updated_targets:
                continue
            if target.pinned or abs(spot - target.level.price) <= self.scan_radius:
                updated_targets[level_id] = target

        self._targets = updated_targets
        return sorted(
            self._targets.values(),
            key=lambda t: (t.pinned is False, -t.relevance)
        )

    def _compute_relevance(
        self,
        distance: float,
        approach_velocity: float,
        confluence_score: float,
        gamma_score: float
    ) -> float:
        if distance >= self.scan_radius:
            distance_score = 0.0
        else:
            distance_score = 1.0 - (distance / max(self.scan_radius, 1e-6))
        velocity_score = math.tanh(approach_velocity / self.config.APPROACH_VELOCITY_NORM)
        return (
            self.w_distance * distance_score
            + self.w_velocity * velocity_score
            + self.w_confluence * confluence_score
            + self.w_gamma * gamma_score
        )

    def _compute_approach_velocity(
        self,
        level: Level,
        spot: float,
        market_state: MarketState
    ) -> float:
        closes = market_state.get_recent_minute_closes(self.config.LOOKBACK_MINUTES)
        if len(closes) < 2:
            return 0.0
        price_change = closes[-1] - closes[0]
        minutes = len(closes)
        direction = "UP" if spot < level.price else "DOWN"
        if direction == "UP":
            return price_change / minutes
        return -price_change / minutes

    def _compute_confluence_score(self, level: Level, universe: List[Level]) -> float:
        key_weights = {
            LevelKind.PM_HIGH: 1.0,
            LevelKind.PM_LOW: 1.0,
            LevelKind.OR_HIGH: 0.9,
            LevelKind.OR_LOW: 0.9,
            LevelKind.SMA_90: 0.8,
            LevelKind.EMA_20: 0.8,
            LevelKind.VWAP: 0.7,
            LevelKind.SESSION_HIGH: 0.6,
            LevelKind.SESSION_LOW: 0.6,
            LevelKind.CALL_WALL: 1.0,
            LevelKind.PUT_WALL: 1.0
        }
        band = self.config.CONFLUENCE_BAND
        weighted = 0.0
        total_weight = 0.0
        for other in universe:
            if other.id == level.id:
                continue
            weight = key_weights.get(other.kind)
            if weight is None:
                continue
            distance = abs(other.price - level.price)
            if distance > band:
                continue
            decay = max(0.0, 1.0 - (distance / band))
            weighted += weight * decay
            total_weight += weight
        if total_weight <= 0:
            return 0.0
        return weighted / total_weight

    def _compute_gamma_score(self, level: Level, market_state: MarketState) -> float:
        if self.fuel_engine is None:
            return 0.0
        try:
            fuel = self.fuel_engine.compute_fuel_state(
                level_price=level.price,
                market_state=market_state,
                exp_date_filter=self._trading_date
            )
            return math.tanh(-fuel.net_dealer_gamma / self.config.GAMMA_EXPOSURE_NORM)
        except Exception:
            return 0.0

    def _next_state(
        self,
        prior: Optional[ViewportTarget],
        level: Level,
        distance: float,
        approach_velocity: float,
        ts_ns: int
    ) -> tuple[ViewportState, Optional[int], Optional[int]]:
        state = prior.state if prior else ViewportState.FAR
        touch_ts_ns = prior.touch_ts_ns if prior else None
        confirm_ts_ns = prior.confirm_ts_ns if prior else None

        if distance <= self.touch_band:
            if state != ViewportState.TOUCH:
                touch_ts_ns = ts_ns
                confirm_ts_ns = ts_ns + self.confirmation_ns
            state = ViewportState.TOUCH
        elif distance <= self.monitor_band:
            state = ViewportState.IN_MONITOR_BAND
        elif distance <= self.scan_radius and approach_velocity > 0:
            state = ViewportState.APPROACHING
        else:
            state = ViewportState.FAR

        if touch_ts_ns is not None and ts_ns >= touch_ts_ns + self.confirmation_ns:
            state = ViewportState.CONFIRMATION
        if state == ViewportState.CONFIRMATION and distance > self.monitor_band:
            state = ViewportState.RESOLVED

        return state, touch_ts_ns, confirm_ts_ns

    def _resolve_pins(self, universe: List[Level]) -> List[Level]:
        pinned_levels = []
        for level in universe:
            if any(self._is_pin_match(pin, level) for pin in self._pins):
                pinned_levels.append(level)
        return pinned_levels

    @staticmethod
    def _is_pin_match(pin: ViewportPin, level: Level) -> bool:
        if level.kind != pin.kind:
            return False
        if pin.price is None:
            return True
        return abs(level.price - pin.price) < 1e-6
