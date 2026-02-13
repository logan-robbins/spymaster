"""Deterministic replay evaluator for vacuum-pressure directional signals.

This module evaluates fixed-threshold directional gating with no model fitting.
It is designed for short-horizon quality checks on 1-second signal outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SECONDS_TO_NS = 1_000_000_000
DEFAULT_HORIZONS_SECONDS: tuple[int, ...] = (2, 5, 10)
DEFAULT_FIRE_HORIZONS_SECONDS: tuple[int, ...] = (20, 30, 45)
DEFAULT_FIRE_TARGET_TICKS: tuple[float, ...] = (8.0,)
REGIME_LIFT = "LIFT"
REGIME_DRAG = "DRAG"
REGIME_THRESHOLD = 0.5


@dataclass(frozen=True)
class ThresholdGate:
    """Deterministic threshold gate for directional alerts.

    Args:
        min_abs_net_lift: Minimum absolute net-lift magnitude to trigger.
        min_cross_confidence: Minimum cross-timescale confidence.
        min_abs_d1_15s: Minimum absolute medium-scale slope.
        require_regime_alignment: If True, long alerts require ``LIFT`` and
            short alerts require ``DRAG``.
    """

    min_abs_net_lift: float
    min_cross_confidence: float
    min_abs_d1_15s: float
    require_regime_alignment: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "min_abs_net_lift": self.min_abs_net_lift,
            "min_cross_confidence": self.min_cross_confidence,
            "min_abs_d1_15s": self.min_abs_d1_15s,
            "require_regime_alignment": self.require_regime_alignment,
        }


def prepare_signal_frame(
    df_signals: pd.DataFrame,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Normalize and validate signal inputs for deterministic evaluation."""
    required = {"window_end_ts_ns", "mid_price"}
    missing = required.difference(df_signals.columns)
    if missing:
        raise ValueError(f"Signals frame missing required columns: {sorted(missing)}")

    df = df_signals.sort_values("window_end_ts_ns").copy()
    df = df.drop_duplicates(subset=["window_end_ts_ns"], keep="last")

    if "net_lift" not in df.columns:
        if "composite" in df.columns:
            df["net_lift"] = df["composite"].astype(float)
        else:
            raise ValueError("Signals frame must contain 'net_lift' or 'composite'.")
    if "cross_confidence" not in df.columns:
        if "confidence" in df.columns:
            df["cross_confidence"] = df["confidence"].astype(float)
        else:
            df["cross_confidence"] = 0.0
    if "d1_15s" not in df.columns:
        if "d1_smooth" in df.columns:
            df["d1_15s"] = df["d1_smooth"].astype(float)
        elif "d1_composite" in df.columns:
            df["d1_15s"] = df["d1_composite"].astype(float)
        else:
            df["d1_15s"] = 0.0
    if "regime" not in df.columns:
        df["regime"] = "UNKNOWN"
    if "book_valid" not in df.columns:
        df["book_valid"] = True

    out_columns = [
        "window_end_ts_ns",
        "mid_price",
        "net_lift",
        "cross_confidence",
        "d1_15s",
        "regime",
        "book_valid",
    ]
    for col in extra_columns:
        if col not in df.columns:
            raise ValueError(f"Signals frame missing requested extra column: {col!r}")
        out_columns.append(col)
    return df[out_columns].reset_index(drop=True)


def ensure_event_columns(df_signals: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``event_state`` and ``event_direction`` exist deterministically.

    Replay-path signals can be missing event-state outputs. In that case, this
    reconstructs event labels from available directional fields using the same
    deterministic state machine used by the incremental live engine.
    """
    if "window_end_ts_ns" not in df_signals.columns:
        raise ValueError("Signals frame missing required column: 'window_end_ts_ns'")

    df = df_signals.sort_values("window_end_ts_ns").copy()
    has_state = "event_state" in df.columns
    has_direction = "event_direction" in df.columns
    if has_state and has_direction:
        return df

    from .incremental import DirectionalEventStateMachine

    machine = DirectionalEventStateMachine()
    event_state: list[str] = []
    event_direction: list[str] = []
    event_strength: list[float] = []
    event_confidence: list[float] = []

    net_lift = (
        df["net_lift"].astype(float).to_numpy(copy=False)
        if "net_lift" in df.columns
        else df.get("composite", pd.Series(0.0, index=df.index))
        .astype(float)
        .to_numpy(copy=False)
    )
    d1_15s = (
        df["d1_15s"].astype(float).to_numpy(copy=False)
        if "d1_15s" in df.columns
        else df.get("d1_smooth", df.get("d1_composite", pd.Series(0.0, index=df.index)))
        .astype(float)
        .to_numpy(copy=False)
    )
    d2_15s = (
        df["d2_15s"].astype(float).to_numpy(copy=False)
        if "d2_15s" in df.columns
        else df.get("d2_smooth", df.get("d2_composite", pd.Series(0.0, index=df.index)))
        .astype(float)
        .to_numpy(copy=False)
    )
    cross_conf = (
        df["cross_confidence"].astype(float).to_numpy(copy=False)
        if "cross_confidence" in df.columns
        else df.get("confidence", df.get("wtd_deriv_conf", pd.Series(0.0, index=df.index)))
        .astype(float)
        .to_numpy(copy=False)
    )
    projection_coh = (
        df["projection_coherence"].astype(float).to_numpy(copy=False)
        if "projection_coherence" in df.columns
        else df.get("wtd_deriv_conf", pd.Series(0.0, index=df.index))
        .astype(float)
        .to_numpy(copy=False)
    )
    projection_value = (
        df["proj_15s"].astype(float).to_numpy(copy=False)
        if "proj_15s" in df.columns
        else df.get("wtd_projection", pd.Series(0.0, index=df.index))
        .astype(float)
        .to_numpy(copy=False)
    )

    if "regime" not in df.columns:
        regime = np.full(len(df), "NEUTRAL", dtype=object)
        regime[net_lift > REGIME_THRESHOLD] = REGIME_LIFT
        regime[net_lift < -REGIME_THRESHOLD] = REGIME_DRAG
        df["regime"] = regime

    for nl, d1, d2, conf, pcoh, proj_val in zip(
        net_lift, d1_15s, d2_15s, cross_conf, projection_coh, projection_value
    ):
        out = machine.update(
            net_lift=float(nl) if np.isfinite(nl) else 0.0,
            d1_15s=float(d1) if np.isfinite(d1) else 0.0,
            d2_15s=float(d2) if np.isfinite(d2) else 0.0,
            cross_confidence=float(conf) if np.isfinite(conf) else 0.0,
            projection_coherence=float(pcoh) if np.isfinite(pcoh) else 0.0,
            projection_direction=int(np.sign(proj_val)) if np.isfinite(proj_val) else 0,
        )
        event_state.append(out["event_state"])
        event_direction.append(out["event_direction"])
        event_strength.append(float(out["event_strength"]))
        event_confidence.append(float(out["event_confidence"]))

    df["event_state"] = event_state
    df["event_direction"] = event_direction
    if "event_strength" not in df.columns:
        df["event_strength"] = event_strength
    if "event_confidence" not in df.columns:
        df["event_confidence"] = event_confidence
    return df


def _resolve_event_direction(event_direction: Sequence[Any]) -> np.ndarray:
    """Map event_direction labels to signed direction (+1/-1/0)."""
    labels = pd.Series(event_direction, copy=False).astype(str).str.upper().to_numpy()
    direction = np.zeros(labels.shape[0], dtype=np.int8)
    direction[labels == "UP"] = 1
    direction[labels == "DOWN"] = -1
    return direction


def _first_touch_step(mask: np.ndarray) -> np.ndarray:
    """Return first 1-indexed step where mask is true, 0 when never true."""
    touched = np.any(mask, axis=0)
    first = np.where(touched, np.argmax(mask, axis=0) + 1, 0)
    return first.astype(np.int64, copy=False)


def _summarize_fire_outcomes(
    hit_mask: np.ndarray,
    false_mask: np.ndarray,
    unresolved_mask: np.ndarray,
    unevaluable_mask: np.ndarray,
    first_hit_step: np.ndarray,
) -> dict[str, Any]:
    """Summarize FIRE outcome classification for a cohort."""
    evaluable_count = int((hit_mask | false_mask | unresolved_mask).sum())
    unevaluable_count = int(unevaluable_mask.sum())
    total_count = evaluable_count + unevaluable_count

    hit_count = int(hit_mask.sum())
    false_count = int(false_mask.sum())
    unresolved_count = int(unresolved_mask.sum())

    if hit_count > 0:
        hit_steps = first_hit_step[hit_mask].astype(np.float64, copy=False)
        mean_hit_s: float | None = float(np.mean(hit_steps))
        median_hit_s: float | None = float(np.median(hit_steps))
    else:
        mean_hit_s = None
        median_hit_s = None

    if evaluable_count > 0:
        hit_rate: float | None = float(hit_count / evaluable_count)
        false_rate: float | None = float(false_count / evaluable_count)
        unresolved_rate: float | None = float(unresolved_count / evaluable_count)
    else:
        hit_rate = None
        false_rate = None
        unresolved_rate = None

    return {
        "total_fire_events": total_count,
        "evaluable_fire_events": evaluable_count,
        "unevaluable_fire_events": unevaluable_count,
        "hit_events": hit_count,
        "false_fire_events": false_count,
        "unresolved_events": unresolved_count,
        "hit_rate": hit_rate,
        "false_fire_rate": false_rate,
        "unresolved_rate": unresolved_rate,
        "mean_time_to_hit_s": mean_hit_s,
        "median_time_to_hit_s": median_hit_s,
    }


def evaluate_fire_events(
    frame: pd.DataFrame,
    horizon_s: int,
    tick_size: float,
    target_ticks: float = 8.0,
    fire_state: str = "FIRE",
) -> dict[str, Any]:
    """Evaluate deterministic FIRE-event quality using first-touch barriers.

    Outcome for each FIRE event is determined by the first barrier reached
    within the horizon:
      - HIT: signed move reaches +target_ticks first
      - FALSE: signed move reaches -target_ticks first
      - UNRESOLVED: neither barrier reached within horizon
    """
    if horizon_s < 1:
        raise ValueError(f"horizon_s must be >= 1, got {horizon_s}")
    if target_ticks <= 0.0:
        raise ValueError(f"target_ticks must be > 0, got {target_ticks}")
    for required in ("event_state", "event_direction", "regime"):
        if required not in frame.columns:
            raise ValueError(f"frame missing required column: {required!r}")

    moves_by_step = _forward_move_ticks_by_step(frame, horizon_s, tick_size)
    event_state = frame["event_state"].astype(str).str.upper().to_numpy(copy=False)
    direction = _resolve_event_direction(frame["event_direction"])
    regime = frame["regime"].astype(str).to_numpy(copy=False)
    book_valid = frame["book_valid"].fillna(False).to_numpy(dtype=bool, copy=False)

    fire_mask = (event_state == fire_state.upper()) & (direction != 0) & book_valid
    fire_idx = np.flatnonzero(fire_mask)
    if fire_idx.size == 0:
        empty = _summarize_fire_outcomes(
            hit_mask=np.zeros(0, dtype=bool),
            false_mask=np.zeros(0, dtype=bool),
            unresolved_mask=np.zeros(0, dtype=bool),
            unevaluable_mask=np.zeros(0, dtype=bool),
            first_hit_step=np.zeros(0, dtype=np.int64),
        )
        return {
            "horizon_s": int(horizon_s),
            "target_ticks": float(target_ticks),
            "overall": empty,
            "by_event_direction": {},
            "by_regime": {},
            "by_regime_and_direction": {},
        }

    fire_direction = direction[fire_idx]
    signed_steps = np.vstack(
        [
            fire_direction * moves_by_step[step][fire_idx]
            for step in range(1, horizon_s + 1)
        ]
    )
    finite_steps = np.isfinite(signed_steps)
    hit_steps = signed_steps >= target_ticks
    false_steps = signed_steps <= -target_ticks

    first_hit_step = _first_touch_step(hit_steps)
    first_false_step = _first_touch_step(false_steps)

    # Horizon evaluation requires an observed price at the horizon boundary.
    # Partial tails near session end are treated as unevaluable.
    evaluable = finite_steps[-1]
    hit_mask = evaluable & (first_hit_step > 0) & (
        (first_false_step == 0) | (first_hit_step < first_false_step)
    )
    false_mask = evaluable & (first_false_step > 0) & (
        (first_hit_step == 0) | (first_false_step < first_hit_step)
    )
    unresolved_mask = evaluable & ~(hit_mask | false_mask)
    unevaluable_mask = ~evaluable

    outcome = {
        "horizon_s": int(horizon_s),
        "target_ticks": float(target_ticks),
        "overall": _summarize_fire_outcomes(
            hit_mask=hit_mask,
            false_mask=false_mask,
            unresolved_mask=unresolved_mask,
            unevaluable_mask=unevaluable_mask,
            first_hit_step=first_hit_step,
        ),
    }

    fire_direction_labels = np.where(fire_direction > 0, "UP", "DOWN")
    fire_regime = regime[fire_idx]

    by_direction: dict[str, Any] = {}
    for direction_label in ("UP", "DOWN"):
        mask = fire_direction_labels == direction_label
        if not np.any(mask):
            continue
        by_direction[direction_label] = _summarize_fire_outcomes(
            hit_mask=hit_mask[mask],
            false_mask=false_mask[mask],
            unresolved_mask=unresolved_mask[mask],
            unevaluable_mask=unevaluable_mask[mask],
            first_hit_step=first_hit_step[mask],
        )

    by_regime: dict[str, Any] = {}
    for reg in sorted(pd.unique(fire_regime)):
        mask = fire_regime == reg
        by_regime[reg] = _summarize_fire_outcomes(
            hit_mask=hit_mask[mask],
            false_mask=false_mask[mask],
            unresolved_mask=unresolved_mask[mask],
            unevaluable_mask=unevaluable_mask[mask],
            first_hit_step=first_hit_step[mask],
        )

    by_regime_and_direction: dict[str, Any] = {}
    for reg in sorted(pd.unique(fire_regime)):
        for direction_label in ("UP", "DOWN"):
            mask = (fire_regime == reg) & (fire_direction_labels == direction_label)
            if not np.any(mask):
                continue
            key = f"{reg}|{direction_label}"
            by_regime_and_direction[key] = _summarize_fire_outcomes(
                hit_mask=hit_mask[mask],
                false_mask=false_mask[mask],
                unresolved_mask=unresolved_mask[mask],
                unevaluable_mask=unevaluable_mask[mask],
                first_hit_step=first_hit_step[mask],
            )

    outcome["by_event_direction"] = by_direction
    outcome["by_regime"] = by_regime
    outcome["by_regime_and_direction"] = by_regime_and_direction
    return outcome


def _fire_candidate_objective(
    candidate: Mapping[str, Any],
    min_evaluable_fires: int,
) -> float:
    """Score FIRE settings deterministically for recommendation ranking."""
    overall = candidate["overall"]
    evaluable = int(overall["evaluable_fire_events"])
    if evaluable < min_evaluable_fires:
        return float("-inf")

    hit_rate = float(overall["hit_rate"] or 0.0)
    false_rate = float(overall["false_fire_rate"] or 0.0)
    unresolved_rate = float(overall["unresolved_rate"] or 0.0)
    mean_hit_s = float(overall["mean_time_to_hit_s"] or 0.0)
    return float(
        hit_rate
        - 1.25 * false_rate
        - 0.75 * unresolved_rate
        - 0.05 * mean_hit_s
    )


def sweep_fire_operating_grid(
    frame: pd.DataFrame,
    horizons_s: Sequence[int],
    target_ticks_values: Sequence[float],
    tick_size: float,
    min_evaluable_fires: int = 25,
    top_k: int = 10,
) -> dict[str, Any]:
    """Sweep FIRE horizon/target settings and recommend operating defaults."""
    if not horizons_s:
        raise ValueError("horizons_s must not be empty")
    if not target_ticks_values:
        raise ValueError("target_ticks_values must not be empty")

    candidates: list[dict[str, Any]] = []
    for horizon_s, target_ticks in product(horizons_s, target_ticks_values):
        evaluated = evaluate_fire_events(
            frame=frame,
            horizon_s=int(horizon_s),
            tick_size=tick_size,
            target_ticks=float(target_ticks),
        )
        evaluated["objective_score"] = _fire_candidate_objective(
            candidate=evaluated,
            min_evaluable_fires=min_evaluable_fires,
        )
        candidates.append(evaluated)

    ranked = sorted(
        candidates,
        key=lambda x: (
            float(x["objective_score"]),
            float(x["overall"]["hit_rate"] or 0.0),
            -float(x["overall"]["false_fire_rate"] or 0.0),
            -float(x["overall"]["unresolved_rate"] or 0.0),
            -float(x["overall"]["mean_time_to_hit_s"] or x["horizon_s"]),
        ),
        reverse=True,
    )
    recommended = ranked[0] if ranked else None
    if recommended is not None and not np.isfinite(float(recommended["objective_score"])):
        recommended = None
    return {
        "recommended": recommended,
        "top_candidates": ranked[: max(1, int(top_k))],
        "search_space_size": len(candidates),
    }


def _forward_move_ticks_by_step(
    frame: pd.DataFrame,
    max_horizon_s: int,
    tick_size: float,
) -> dict[int, np.ndarray]:
    """Compute signed future move in ticks for each forward step."""
    if tick_size <= 0:
        raise ValueError(f"tick_size must be > 0, got {tick_size}")

    ts = frame["window_end_ts_ns"]
    mid = frame["mid_price"].to_numpy(dtype=np.float64, copy=False)
    mid_by_ts = pd.Series(mid, index=ts.to_numpy(dtype=np.int64, copy=False))

    move_by_step: dict[int, np.ndarray] = {}
    for step in range(1, max_horizon_s + 1):
        target_ts = ts + step * SECONDS_TO_NS
        future_mid = target_ts.map(mid_by_ts).to_numpy(dtype=np.float64, copy=False)
        move_by_step[step] = (future_mid - mid) / tick_size
    return move_by_step


def _mean_lead_time_seconds(
    direction: np.ndarray,
    step_moves_ticks: Mapping[int, np.ndarray],
    horizon_s: int,
    min_move_ticks: float,
    hit_mask: np.ndarray,
) -> float | None:
    """Compute mean lead time for hit alerts within horizon."""
    signed_steps = np.vstack(
        [direction * step_moves_ticks[step] for step in range(1, horizon_s + 1)]
    )
    hit_steps = signed_steps >= min_move_ticks
    any_hit = np.any(hit_steps, axis=0)
    first_hit = np.argmax(hit_steps, axis=0) + 1
    lead_time = np.where(any_hit, first_hit.astype(np.float64), np.nan)
    hit_lead = lead_time[hit_mask]
    if hit_lead.size == 0:
        return None
    if np.isnan(hit_lead).all():
        return None
    return float(np.nanmean(hit_lead))


def evaluate_threshold_gate(
    frame: pd.DataFrame,
    gate: ThresholdGate,
    horizons_s: Sequence[int],
    tick_size: float,
    min_move_ticks: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one deterministic threshold gate across horizons."""
    if not horizons_s:
        raise ValueError("horizons_s must not be empty")

    horizons = sorted(set(int(h) for h in horizons_s))
    max_h = max(horizons)
    moves_by_step = _forward_move_ticks_by_step(frame, max_h, tick_size)

    net_lift = frame["net_lift"].to_numpy(dtype=np.float64, copy=False)
    confidence = frame["cross_confidence"].to_numpy(dtype=np.float64, copy=False)
    d1_15s = frame["d1_15s"].to_numpy(dtype=np.float64, copy=False)
    regime = frame["regime"].astype(str).to_numpy(copy=False)
    book_valid = frame["book_valid"].fillna(False).to_numpy(dtype=bool, copy=False)
    direction = np.sign(net_lift)

    finite_signal = np.isfinite(net_lift) & np.isfinite(confidence) & np.isfinite(d1_15s)
    alert_mask = (
        book_valid
        & finite_signal
        & (direction != 0.0)
        & (np.abs(net_lift) >= gate.min_abs_net_lift)
        & (confidence >= gate.min_cross_confidence)
        & (np.abs(d1_15s) >= gate.min_abs_d1_15s)
    )
    if gate.require_regime_alignment:
        align = ((direction > 0.0) & (regime == REGIME_LIFT)) | (
            (direction < 0.0) & (regime == REGIME_DRAG)
        )
        alert_mask &= align

    horizon_metrics: dict[str, Any] = {}
    for horizon_s in horizons:
        key = f"{horizon_s}s"
        realized_ticks = direction * moves_by_step[horizon_s]
        evaluable = np.isfinite(realized_ticks)
        candidate = alert_mask & evaluable

        alerts = int(candidate.sum())
        hits_mask = candidate & (realized_ticks >= min_move_ticks)
        false_mask = candidate & (realized_ticks <= -min_move_ticks)
        stale_mask = candidate & ~(hits_mask | false_mask)

        hits = int(hits_mask.sum())
        false_alerts = int(false_mask.sum())
        stale_alerts = int(stale_mask.sum())
        eval_windows = int(evaluable.sum())
        eval_minutes = eval_windows / 60.0 if eval_windows > 0 else np.nan

        lead_time_s = _mean_lead_time_seconds(
            direction=direction,
            step_moves_ticks=moves_by_step,
            horizon_s=horizon_s,
            min_move_ticks=min_move_ticks,
            hit_mask=hits_mask,
        )

        regime_perf: dict[str, Any] = {}
        for reg in sorted(pd.unique(regime)):
            reg_evaluable = evaluable & (regime == reg)
            reg_candidate = candidate & (regime == reg)
            reg_alerts = int(reg_candidate.sum())
            if reg_alerts == 0:
                continue
            reg_hits = int((hits_mask & (regime == reg)).sum())
            reg_false = int((false_mask & (regime == reg)).sum())
            reg_eval_count = int(reg_evaluable.sum())
            reg_eval_minutes = reg_eval_count / 60.0 if reg_eval_count > 0 else np.nan
            regime_perf[reg] = {
                "alerts": reg_alerts,
                "hits": reg_hits,
                "false_alerts": reg_false,
                "hit_rate": float(reg_hits / reg_alerts),
                "false_alert_density_per_minute": (
                    float(reg_false / reg_eval_minutes)
                    if np.isfinite(reg_eval_minutes) and reg_eval_minutes > 0
                    else None
                ),
            }

        horizon_metrics[key] = {
            "alerts": alerts,
            "hits": hits,
            "false_alerts": false_alerts,
            "stale_alerts": stale_alerts,
            "hit_rate": float(hits / alerts) if alerts > 0 else None,
            "alert_rate": float(alerts / eval_windows) if eval_windows > 0 else None,
            "false_alert_density_per_minute": (
                float(false_alerts / eval_minutes)
                if np.isfinite(eval_minutes) and eval_minutes > 0
                else None
            ),
            "mean_lead_time_s": lead_time_s,
            "regime_stratified": regime_perf,
            "evaluable_windows": eval_windows,
        }

    return {
        "thresholds": gate.to_dict(),
        "horizons": horizon_metrics,
        "sample_windows": int(len(frame)),
    }


def _candidate_objective(
    candidate: Mapping[str, Any],
    primary_horizon_s: int,
    min_alerts: int,
    target_alert_rate: float,
) -> float:
    """Score candidate gate for recommendation ranking."""
    key = f"{primary_horizon_s}s"
    primary = candidate["horizons"][key]
    alerts = int(primary["alerts"])
    if alerts < min_alerts:
        return float("-inf")

    hit_rate = float(primary["hit_rate"] or 0.0)
    false_density = float(primary["false_alert_density_per_minute"] or 0.0)
    lead_time = float(primary["mean_lead_time_s"] or primary_horizon_s)
    alert_rate = float(primary["alert_rate"] or 0.0)

    coverage_credit = min(alert_rate, target_alert_rate) / max(target_alert_rate, 1e-9)
    score = (
        hit_rate
        - 0.05 * false_density
        - 0.10 * (lead_time / max(primary_horizon_s, 1))
        + 0.20 * coverage_credit
    )
    return float(score)


def sweep_threshold_grid(
    frame: pd.DataFrame,
    horizons_s: Sequence[int],
    tick_size: float,
    min_move_ticks: float,
    net_lift_thresholds: Sequence[float],
    confidence_thresholds: Sequence[float],
    d1_15s_thresholds: Sequence[float],
    require_regime_alignment_values: Sequence[bool],
    primary_horizon_s: int = 5,
    min_alerts: int = 100,
    target_alert_rate: float = 0.01,
    top_k: int = 10,
) -> dict[str, Any]:
    """Grid-search deterministic thresholds and recommend a fixed set."""
    candidates: list[dict[str, Any]] = []

    for min_abs_net_lift, min_cross_confidence, min_abs_d1_15s, align in product(
        net_lift_thresholds,
        confidence_thresholds,
        d1_15s_thresholds,
        require_regime_alignment_values,
    ):
        gate = ThresholdGate(
            min_abs_net_lift=float(min_abs_net_lift),
            min_cross_confidence=float(min_cross_confidence),
            min_abs_d1_15s=float(min_abs_d1_15s),
            require_regime_alignment=bool(align),
        )
        result = evaluate_threshold_gate(
            frame=frame,
            gate=gate,
            horizons_s=horizons_s,
            tick_size=tick_size,
            min_move_ticks=min_move_ticks,
        )
        result["objective_score"] = _candidate_objective(
            candidate=result,
            primary_horizon_s=primary_horizon_s,
            min_alerts=min_alerts,
            target_alert_rate=target_alert_rate,
        )
        candidates.append(result)

    ranked = sorted(
        candidates,
        key=lambda x: (
            float(x["objective_score"]),
            float(x["horizons"][f"{primary_horizon_s}s"]["hit_rate"] or 0.0),
            -float(x["horizons"][f"{primary_horizon_s}s"]["false_alert_density_per_minute"] or 0.0),
            -int(x["horizons"][f"{primary_horizon_s}s"]["alerts"]),
        ),
        reverse=True,
    )

    recommended = ranked[0] if ranked else None
    return {
        "recommended": recommended,
        "top_candidates": ranked[: max(1, int(top_k))],
        "search_space_size": len(candidates),
    }


def parse_csv_floats(raw: str) -> list[float]:
    """Parse comma-separated float list preserving input order."""
    vals = [v.strip() for v in raw.split(",")]
    parsed = [float(v) for v in vals if v]
    if not parsed:
        raise ValueError(f"No float values parsed from: {raw!r}")
    return parsed


def parse_csv_bools(raw: str) -> list[bool]:
    """Parse comma-separated bool list."""
    out: list[bool] = []
    for token in (v.strip().lower() for v in raw.split(",")):
        if token in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif token in {"0", "false", "f", "no", "n"}:
            out.append(False)
        elif token:
            raise ValueError(f"Invalid bool token in list: {token!r}")
    if not out:
        raise ValueError(f"No boolean values parsed from: {raw!r}")
    return out


def parse_csv_ints(raw: str) -> list[int]:
    """Parse comma-separated integer list preserving input order."""
    vals = [v.strip() for v in raw.split(",")]
    parsed = [int(v) for v in vals if v]
    if not parsed:
        raise ValueError(f"No integer values parsed from: {raw!r}")
    return parsed

