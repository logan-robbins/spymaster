from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

TICK_INT = 250_000_000
MAX_TICKS = 200
FULL_TICKS = np.arange(-MAX_TICKS, MAX_TICKS + 1, dtype=float)
CENTER_INDEX = MAX_TICKS

FORCE_RADIUS = 64
FORCE_SIGMA = 12.0
DAMP_RADIUS = 16
DAMP_SIGMA = 6.0
SPREAD_RADIUS = 24
SPREAD_SIGMA = 8.0

_FORCE_K = np.arange(1, FORCE_RADIUS + 1, dtype=float)
_FORCE_W = np.exp(-(_FORCE_K**2) / (2.0 * FORCE_SIGMA**2))
_FORCE_W = _FORCE_W / _FORCE_W.sum()
_FORCE_POS = np.concatenate([np.arange(-FORCE_RADIUS, 0), np.arange(1, FORCE_RADIUS + 1)])
_FORCE_W_POS = _FORCE_W[np.abs(_FORCE_POS) - 1]

_DAMP_K = np.arange(1, DAMP_RADIUS + 1, dtype=float)
_DAMP_W = np.exp(-(_DAMP_K**2) / (2.0 * DAMP_SIGMA**2))
_DAMP_W = _DAMP_W / _DAMP_W.sum()
_DAMP_POS = np.concatenate([np.arange(-DAMP_RADIUS, 0), np.arange(1, DAMP_RADIUS + 1)])
_DAMP_W_POS = _DAMP_W[np.abs(_DAMP_POS) - 1]

_CORRIDOR_K = np.arange(1, FORCE_RADIUS + 1, dtype=float)
_CORRIDOR_W = np.exp(-_CORRIDOR_K / 16.0)


@dataclass(frozen=True)
class WindowFields:
    ticks: np.ndarray
    g_dir: np.ndarray
    u_near: np.ndarray
    u_p_slow: np.ndarray
    omega_total: np.ndarray
    omega_prom: np.ndarray
    nu: np.ndarray
    kappa: np.ndarray


@dataclass(frozen=True)
class RunDiagnostics:
    d_up: int
    d_down: int
    run_score_up: float
    run_score_down: float


def tick_index_from_price(price_int: float) -> float:
    return float(price_int) / float(TICK_INT)


def gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    axis = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(axis**2) / (2.0 * sigma**2))
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        return kernel
    return kernel / kernel_sum


def gaussian_smooth(field: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    kernel = gaussian_kernel(sigma, radius)
    return np.convolve(field, kernel, mode="same")


def _side_array(
    df: pd.DataFrame,
    column: str,
    side: str,
    ticks: np.ndarray = FULL_TICKS,
) -> np.ndarray:
    if df.empty or column not in df.columns:
        return np.zeros_like(ticks, dtype=float)
    subset = df.loc[df["side"] == side, ["rel_ticks", column]]
    if subset.empty:
        return np.zeros_like(ticks, dtype=float)
    series = subset.groupby("rel_ticks")[column].mean()
    return series.reindex(ticks.astype(int), fill_value=0.0).to_numpy(dtype=float)


def _directional_array(
    df: pd.DataFrame,
    column: str,
    ticks: np.ndarray = FULL_TICKS,
) -> np.ndarray:
    bid = _side_array(df, column, "B", ticks)
    ask = _side_array(df, column, "A", ticks)
    out = np.zeros_like(ticks, dtype=float)
    pos_mask = ticks > 0
    neg_mask = ticks < 0
    out[pos_mask] = ask[pos_mask]
    out[neg_mask] = bid[neg_mask]
    return out


def _options_omega_base(
    df_opt: pd.DataFrame,
    ticks: np.ndarray = FULL_TICKS,
) -> np.ndarray:
    if df_opt.empty or "Omega_opt" not in df_opt.columns:
        return np.zeros_like(ticks, dtype=float)
    series = df_opt.groupby("rel_ticks")["Omega_opt"].sum()
    return series.reindex(ticks.astype(int), fill_value=0.0).to_numpy(dtype=float)


def build_window_fields(
    df_fut: pd.DataFrame,
    df_opt: pd.DataFrame,
    ticks: np.ndarray = FULL_TICKS,
) -> WindowFields:
    g_dir = _directional_array(df_fut, "pressure_grad", ticks)
    u_near = _directional_array(df_fut, "u_near", ticks)
    u_p_slow = _directional_array(df_fut, "u_p_slow", ticks)
    omega_fut = _directional_array(df_fut, "Omega", ticks)

    omega_opt_base = _options_omega_base(df_opt, ticks)
    omega_opt_spread = gaussian_smooth(omega_opt_base, SPREAD_SIGMA, SPREAD_RADIUS)

    omega_total = omega_fut + 0.70 * omega_opt_spread
    omega_near = gaussian_smooth(omega_total, sigma=6.0, radius=16)
    omega_far = gaussian_smooth(omega_total, sigma=24.0, radius=64)
    omega_prom = omega_near - omega_far

    nu = 1.0 + omega_near + 2.0 * np.maximum(0.0, omega_prom)
    kappa = 1.0 / nu

    return WindowFields(
        ticks=ticks,
        g_dir=g_dir,
        u_near=u_near,
        u_p_slow=u_p_slow,
        omega_total=omega_total,
        omega_prom=omega_prom,
        nu=nu,
        kappa=kappa,
    )


def force_and_damping(fields: WindowFields, delta: float) -> Tuple[float, float]:
    positions = _FORCE_POS - delta
    g_vals = np.interp(
        positions,
        fields.ticks,
        fields.g_dir,
        left=fields.g_dir[0],
        right=fields.g_dir[-1],
    )
    k_vals = np.interp(
        positions,
        fields.ticks,
        fields.kappa,
        left=fields.kappa[0],
        right=fields.kappa[-1],
    )
    force = float(np.sum(_FORCE_W_POS * k_vals * g_vals))

    damp_positions = _DAMP_POS - delta
    nu_vals = np.interp(
        damp_positions,
        fields.ticks,
        fields.nu,
        left=fields.nu[0],
        right=fields.nu[-1],
    )
    nu_local = float(np.sum(_DAMP_W_POS * nu_vals))
    return force, nu_local


def _weighted_sum(values: Iterable[float], weights: np.ndarray) -> float:
    values_arr = np.asarray(list(values), dtype=float)
    if values_arr.size == 0:
        return 0.0
    w = weights[: values_arr.size].astype(float)
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0
    return float(np.sum(values_arr * (w / w_sum)))


def compute_run_diagnostics(fields: WindowFields) -> RunDiagnostics:
    omega_prom = fields.omega_prom
    center = CENTER_INDEX

    up_slice = omega_prom[center + 1 : center + 1 + FORCE_RADIUS]
    w_up = float(np.max(up_slice)) if up_slice.size > 0 else 0.0
    t_up = 0.60 * w_up
    up_mask = up_slice >= t_up
    d_up = int(np.argmax(up_mask) + 1) if np.any(up_mask) else FORCE_RADIUS

    down_slice = omega_prom[center - FORCE_RADIUS : center]
    w_down = float(np.max(down_slice)) if down_slice.size > 0 else 0.0
    t_down = 0.60 * w_down
    down_mask = down_slice[::-1] >= t_down
    d_down = int(np.argmax(down_mask) + 1) if np.any(down_mask) else FORCE_RADIUS

    r = np.maximum(0.0, -fields.u_near)

    vacuum_up_vals = [r[center + k] for k in range(1, d_up)]
    vacuum_down_vals = [r[center - k] for k in range(1, d_down)]
    vacuum_up = _weighted_sum(vacuum_up_vals, _CORRIDOR_W)
    vacuum_down = _weighted_sum(vacuum_down_vals, _CORRIDOR_W)

    reinforce_up = float(np.maximum(0.0, fields.u_p_slow[center + d_up]))
    reinforce_down = float(np.maximum(0.0, fields.u_p_slow[center - d_down]))

    run_score_up = float(vacuum_up - reinforce_up)
    run_score_down = float(vacuum_down - reinforce_down)

    return RunDiagnostics(
        d_up=d_up,
        d_down=d_down,
        run_score_up=run_score_up,
        run_score_down=run_score_down,
    )


def run_forecast(
    fields: WindowFields,
    spot_ticks: float,
    v0: float,
    beta: float,
    gamma: float,
    horizon_s: int = 30,
) -> Tuple[list[dict], RunDiagnostics]:
    diagnostics = compute_run_diagnostics(fields)
    force_0, nu_0 = force_and_damping(fields, delta=0.0)
    c0 = float(np.tanh(abs(force_0) / (1.0 + nu_0)))

    v = float(np.clip(v0, -8.0, 8.0))
    delta = 0.0
    results: list[dict] = []

    for h in range(1, horizon_s + 1):
        force_h, nu_h = force_and_damping(fields, delta=delta)
        accel = beta * force_h / (1.0 + nu_h)
        v = float(np.clip((1.0 - gamma) * v + accel, -8.0, 8.0))
        delta = float(np.clip(delta + v, -80.0, 80.0))

        predicted_tick_delta = int(np.rint(delta))
        predicted_spot_tick = int(np.rint(spot_ticks + predicted_tick_delta))

        if predicted_tick_delta > 0:
            gate = float(np.tanh(max(0.0, diagnostics.run_score_up)))
        elif predicted_tick_delta < 0:
            gate = float(np.tanh(max(0.0, diagnostics.run_score_down)))
        else:
            gate = 0.0

        confidence = c0 * gate

        results.append(
            {
                "horizon_s": h,
                "predicted_tick_delta": predicted_tick_delta,
                "predicted_spot_tick": predicted_spot_tick,
                "confidence": confidence,
            }
        )

    return results, diagnostics
