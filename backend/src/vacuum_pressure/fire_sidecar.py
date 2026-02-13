"""Core FIRE-entry outcome tracking for the vacuum-pressure sidecar.

Tracks whether each transition into FIRE reaches a target move in the
event direction within a configurable horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NS_PER_SECOND = 1_000_000_000
STATE_FIRE = "FIRE"
DIR_UP = "UP"
DIR_DOWN = "DOWN"


@dataclass(frozen=True)
class FireEvent:
    """A single transition into FIRE."""

    event_id: int
    fire_ts_ns: int
    fire_price: float
    direction: int
    tick_target: float
    tick_size: float
    target_price: float
    deadline_ts_ns: int


@dataclass(frozen=True)
class FireResolution:
    """Resolved outcome for a previously opened FIRE event."""

    event_id: int
    fire_ts_ns: int
    resolved_ts_ns: int
    fire_price: float
    resolved_price: float
    direction: int
    target_price: float
    status: str
    time_to_outcome_s: float


class FireOutcomeTracker:
    """Track FIRE entry outcomes and maintain rolling metrics."""

    def __init__(
        self,
        *,
        tick_size: float,
        tick_target: float = 8.0,
        max_horizon_s: float = 10.0,
    ) -> None:
        if tick_size <= 0:
            raise ValueError(f"tick_size must be > 0, got {tick_size}")
        if tick_target <= 0:
            raise ValueError(f"tick_target must be > 0, got {tick_target}")
        if max_horizon_s <= 0:
            raise ValueError(f"max_horizon_s must be > 0, got {max_horizon_s}")

        self._tick_size = float(tick_size)
        self._tick_target = float(tick_target)
        self._max_horizon_ns = int(max_horizon_s * NS_PER_SECOND)

        self._pending: list[FireEvent] = []
        self._prev_event_state: str | None = None
        self._next_event_id: int = 1

        self.fires: int = 0
        self.resolved: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self._sum_time_to_hit_s: float = 0.0

    @staticmethod
    def _parse_direction(event_direction: str) -> int:
        direction = event_direction.strip().upper()
        if direction == DIR_UP:
            return 1
        if direction == DIR_DOWN:
            return -1
        raise ValueError(
            f"Unsupported event_direction for FIRE transition: {event_direction!r} "
            f"(expected '{DIR_UP}' or '{DIR_DOWN}')"
        )

    @staticmethod
    def _normalize_state(event_state: str) -> str:
        return event_state.strip().upper()

    def _resolve_pending(
        self,
        *,
        window_end_ts_ns: int,
        mid_price: float,
    ) -> list[FireResolution]:
        outcomes: list[FireResolution] = []
        still_pending: list[FireEvent] = []

        for pending in self._pending:
            elapsed_ns = window_end_ts_ns - pending.fire_ts_ns
            hit = (
                mid_price >= pending.target_price
                if pending.direction > 0
                else mid_price <= pending.target_price
            )
            expired = elapsed_ns >= self._max_horizon_ns

            if hit:
                time_to_outcome_s = max(0.0, elapsed_ns / NS_PER_SECOND)
                outcomes.append(
                    FireResolution(
                        event_id=pending.event_id,
                        fire_ts_ns=pending.fire_ts_ns,
                        resolved_ts_ns=window_end_ts_ns,
                        fire_price=pending.fire_price,
                        resolved_price=mid_price,
                        direction=pending.direction,
                        target_price=pending.target_price,
                        status="hit",
                        time_to_outcome_s=time_to_outcome_s,
                    )
                )
                self.resolved += 1
                self.hits += 1
                self._sum_time_to_hit_s += time_to_outcome_s
            elif expired:
                time_to_outcome_s = max(0.0, elapsed_ns / NS_PER_SECOND)
                outcomes.append(
                    FireResolution(
                        event_id=pending.event_id,
                        fire_ts_ns=pending.fire_ts_ns,
                        resolved_ts_ns=window_end_ts_ns,
                        fire_price=pending.fire_price,
                        resolved_price=mid_price,
                        direction=pending.direction,
                        target_price=pending.target_price,
                        status="miss",
                        time_to_outcome_s=time_to_outcome_s,
                    )
                )
                self.resolved += 1
                self.misses += 1
            else:
                still_pending.append(pending)

        self._pending = still_pending
        return outcomes

    def update(
        self,
        *,
        window_end_ts_ns: int,
        mid_price: float,
        event_state: str,
        event_direction: str,
    ) -> tuple[FireEvent | None, list[FireResolution]]:
        """Process one signals window and return entry/resolution updates."""
        if window_end_ts_ns <= 0:
            raise ValueError(f"window_end_ts_ns must be > 0, got {window_end_ts_ns}")
        if mid_price <= 0:
            raise ValueError(f"mid_price must be > 0, got {mid_price}")

        outcomes = self._resolve_pending(
            window_end_ts_ns=window_end_ts_ns,
            mid_price=mid_price,
        )

        state = self._normalize_state(event_state)
        fire_event: FireEvent | None = None
        is_fire_entry = state == STATE_FIRE and self._prev_event_state != STATE_FIRE

        if is_fire_entry:
            direction = self._parse_direction(event_direction)
            target_delta = self._tick_target * self._tick_size
            target_price = (
                mid_price + target_delta
                if direction > 0
                else mid_price - target_delta
            )
            fire_event = FireEvent(
                event_id=self._next_event_id,
                fire_ts_ns=window_end_ts_ns,
                fire_price=mid_price,
                direction=direction,
                tick_target=self._tick_target,
                tick_size=self._tick_size,
                target_price=target_price,
                deadline_ts_ns=window_end_ts_ns + self._max_horizon_ns,
            )
            self._next_event_id += 1
            self._pending.append(fire_event)
            self.fires += 1

        self._prev_event_state = state
        return fire_event, outcomes

    def metrics(self) -> dict[str, Any]:
        """Return a rolling metrics snapshot."""
        hit_rate = (self.hits / self.resolved) if self.resolved > 0 else None
        avg_time_to_hit = (
            self._sum_time_to_hit_s / self.hits
            if self.hits > 0
            else None
        )
        return {
            "fires": self.fires,
            "resolved": self.resolved,
            "hits": self.hits,
            "misses": self.misses,
            "pending": len(self._pending),
            "hit_rate": hit_rate,
            "avg_time_to_hit": avg_time_to_hit,
        }
