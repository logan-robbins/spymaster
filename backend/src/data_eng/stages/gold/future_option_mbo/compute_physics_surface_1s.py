from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....filters.gold_strict_filters import apply_gold_strict_filters
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)

logger = logging.getLogger(__name__)

EPS_QTY = 1.0
PRICE_SCALE = 1e-9
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
STRIKE_STEP_POINTS = 5.0
STRIKE_STEP_INT = int(round(STRIKE_STEP_POINTS / PRICE_SCALE))
STRIKE_TICKS = 20  # $5 strike buckets at $0.25 tick size
MAX_STRIKE_TICKS = 200  # ±$50 from spot -> ±200 ticks


class GoldComputeOptionPhysicsSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_compute_option_physics_surface_1s",
            io=StageIO(
                inputs=["silver.future_option_mbo.depth_and_flow_1s"],
                output="gold.future_option_mbo.physics_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        flow_key = "silver.future_option_mbo.depth_and_flow_1s"
        flow_ref = partition_ref(cfg, flow_key, symbol, dt)
        if not is_partition_complete(flow_ref):
            raise FileNotFoundError(f"Input not ready: {flow_ref.dataset_key} dt={dt}")

        flow_contract = load_avro_contract(repo_root / cfg.dataset(flow_key).contract)
        df_flow = enforce_contract(read_partition(flow_ref), flow_contract)

        # Apply gold strict filters for institutional-grade data
        original_flow_count = len(df_flow)
        df_flow, flow_stats = apply_gold_strict_filters(df_flow, product_type="futures_options", return_stats=True)
        if flow_stats.get("total_filtered", 0) > 0:
            logger.info(
                f"Gold filters for {dt}: flow={flow_stats.get('total_filtered', 0)}/{original_flow_count}"
            )

        df_out = self.transform(df_flow)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": flow_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(flow_ref)},
        ]

        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_out,
            contract_path=out_contract_path,
            inputs=lineage,
            stage=self.name,
        )

    def transform(self, df_flow: pd.DataFrame) -> pd.DataFrame:
        if df_flow.empty:
            return pd.DataFrame(
                columns=[
                    "window_end_ts_ns",
                    "event_ts_ns",
                    "spot_ref_price_int",
                    "strike_price_int",
                    "strike_points",
                    "rel_ticks",
                    "right",
                    "side",
                    "add_intensity",
                    "fill_intensity",
                    "pull_intensity",
                    "liquidity_velocity",
                    "rho_opt",
                    "phi_rest_opt",
                    "u_opt_ema_2",
                    "u_opt_ema_8",
                    "u_opt_ema_32",
                    "u_opt_ema_128",
                    "u_opt_band_fast",
                    "u_opt_band_mid",
                    "u_opt_band_slow",
                    "u_opt_wave_energy",
                    "du_opt_dt",
                    "d2u_opt_dt2",
                    "u_opt_p",
                    "u_opt_p_slow",
                    "u_opt_near",
                    "u_opt_far",
                    "u_opt_prom",
                    "du_opt_dx",
                    "d2u_opt_dx2",
                    "Omega_opt",
                    "Omega_opt_near",
                    "Omega_opt_far",
                    "Omega_opt_prom",
                    "nu_opt",
                    "kappa_opt",
                    "pressure_grad_opt",
                ]
            )

        df = df_flow.copy()
        df["spot_ref_price_int"] = df["spot_ref_price_int"].astype("int64")
        df["strike_price_int"] = df["strike_price_int"].astype("int64")
        df["strike_ref_price_int"] = _round_to_nearest_strike_int(df["spot_ref_price_int"])

        strike_delta = df["strike_price_int"] - df["strike_ref_price_int"]
        if (strike_delta % STRIKE_STEP_INT != 0).any():
            raise ValueError("Option strikes not aligned to $5 grid")
        df["rel_strike"] = (strike_delta // STRIKE_STEP_INT).astype(int)
        df["spot_offset_ticks"] = (
            (df["strike_ref_price_int"] - df["spot_ref_price_int"]) // TICK_INT
        ).astype(int)

        depth_start = df["depth_qty_start"].astype(float).to_numpy()
        depth_end = df["depth_qty_end"].astype(float).to_numpy() # Need end for rho
        depth_rest = df["depth_qty_rest"].astype(float).to_numpy()
        add_qty = df["add_qty"].astype(float).to_numpy()
        fill_qty = df["fill_qty"].astype(float).to_numpy()
        pull_qty = df["pull_qty"].astype(float).to_numpy()

        denom = depth_start + EPS_QTY

        df["add_intensity"] = add_qty / denom
        df["fill_intensity"] = fill_qty / denom
        df["pull_intensity"] = pull_qty / denom
        df["liquidity_velocity"] = df["add_intensity"] - df["pull_intensity"] - df["fill_intensity"]

        # -------------------------------------------------------------------------
        # Options physics fields on strike lattice (bucketed at $5 / 20 ticks)
        # -------------------------------------------------------------------------
        # rho_opt
        df["rho_opt"] = np.log(1.0 + depth_end)

        # phi_rest_opt
        df["phi_rest_opt"] = depth_rest / (depth_end + 1.0)

        # Temporal EMAs (per strike/right/side)
        df = df.sort_values(["strike_price_int", "right", "side", "window_end_ts_ns"])

        alpha_2 = 1.0 - np.exp(-1.0 / 2.0)
        alpha_8 = 1.0 - np.exp(-1.0 / 8.0)
        alpha_32 = 1.0 - np.exp(-1.0 / 32.0)
        alpha_128 = 1.0 - np.exp(-1.0 / 128.0)

        g = df.groupby(["strike_price_int", "right", "side"])["liquidity_velocity"]
        df["u_opt_ema_2"] = g.ewm(alpha=alpha_2, adjust=False).mean().reset_index(level=[0, 1, 2], drop=True)
        df["u_opt_ema_8"] = g.ewm(alpha=alpha_8, adjust=False).mean().reset_index(level=[0, 1, 2], drop=True)
        df["u_opt_ema_32"] = g.ewm(alpha=alpha_32, adjust=False).mean().reset_index(level=[0, 1, 2], drop=True)
        df["u_opt_ema_128"] = g.ewm(alpha=alpha_128, adjust=False).mean().reset_index(level=[0, 1, 2], drop=True)

        # Bands + wave energy
        df["u_opt_band_fast"] = df["u_opt_ema_2"] - df["u_opt_ema_8"]
        df["u_opt_band_mid"] = df["u_opt_ema_8"] - df["u_opt_ema_32"]
        df["u_opt_band_slow"] = df["u_opt_ema_32"] - df["u_opt_ema_128"]
        df["u_opt_wave_energy"] = np.sqrt(
            df["u_opt_band_fast"] ** 2 + df["u_opt_band_mid"] ** 2 + df["u_opt_band_slow"] ** 2
        )

        # Temporal derivatives (per strike/right/side)
        u2_prev = df.groupby(["strike_price_int", "right", "side"])["u_opt_ema_2"].shift(1)
        df["du_opt_dt"] = (df["u_opt_ema_2"] - u2_prev).fillna(0.0)
        du_prev = df.groupby(["strike_price_int", "right", "side"])["du_opt_dt"].shift(1)
        df["d2u_opt_dt2"] = (df["du_opt_dt"] - du_prev).fillna(0.0)

        # Persistence-weighted velocities
        df["u_opt_p"] = df["phi_rest_opt"] * df["u_opt_ema_8"]
        df["u_opt_p_slow"] = df["phi_rest_opt"] * df["u_opt_ema_32"]

        # Omega_opt
        term_rest = 0.5 + 0.5 * df["phi_rest_opt"]
        term_reinforce = 1.0 + np.maximum(0.0, df["u_opt_p_slow"])
        df["Omega_opt"] = df["rho_opt"] * term_rest * term_reinforce

        # Spatial smoothing on strike lattice (strike spacing = 20 ticks)
        max_strike_steps = int(round(MAX_STRIKE_TICKS / STRIKE_TICKS))

        def apply_strike_smoothing(
            df_in: pd.DataFrame,
            target_col: str,
            sigma_ticks: float,
            n_ticks: int,
            out_col_name: str,
        ) -> pd.DataFrame:
            """Gaussian smoothing on the $5 strike lattice (strike units)."""
            df_work = df_in.copy()
            if "rel_strike" not in df_work.columns or "spot_offset_ticks" not in df_work.columns:
                raise ValueError("Missing strike lattice metadata for smoothing")

            pivoted = (
                df_work.groupby(["window_end_ts_ns", "right", "side", "rel_strike"])[target_col]
                .mean()
                .unstack("rel_strike")
            )

            full_strikes = np.arange(-max_strike_steps, max_strike_steps + 1)
            pivoted = pivoted.reindex(columns=full_strikes, fill_value=0.0)

            n_strikes = max(1, int(round(n_ticks / STRIKE_TICKS)))
            sigma_strikes = max(1.0, sigma_ticks / STRIKE_TICKS)
            win_size = int(2 * n_strikes + 1)

            # Gaussian weighted-mean smoothing across strikes (axis=1).
            # Uses scipy.signal.windows.gaussian + np.convolve to avoid the
            # pandas rolling(..., axis=1) parameter, which is deprecated in
            # pandas 2.x and removed in pandas 3.x.
            from scipy.signal.windows import gaussian as _gaussian_win

            _kernel = _gaussian_win(win_size, std=sigma_strikes)
            _kernel = _kernel / _kernel.sum()
            _half = win_size // 2
            _vals = pivoted.values.copy().astype(float)
            _result = np.full_like(_vals, np.nan)
            for _row_i in range(_vals.shape[0]):
                _conv = np.convolve(_vals[_row_i], _kernel, mode="same")
                _result[_row_i, _half : _vals.shape[1] - _half] = (
                    _conv[_half : _vals.shape[1] - _half]
                )
            smoothed = pd.DataFrame(_result, index=pivoted.index, columns=pivoted.columns)

            smoothed = smoothed.reset_index()
            smoothed = smoothed.melt(
                id_vars=["window_end_ts_ns", "right", "side"],
                var_name="rel_strike",
                value_name=out_col_name,
            )
            spot_offsets = df_work[["window_end_ts_ns", "spot_offset_ticks"]].drop_duplicates()
            smoothed = smoothed.merge(spot_offsets, on="window_end_ts_ns", how="left")
            smoothed["rel_ticks"] = (
                smoothed["rel_strike"].astype(int) * STRIKE_TICKS
                + smoothed["spot_offset_ticks"].astype(int)
            )
            return smoothed[["window_end_ts_ns", "right", "side", "rel_ticks", out_col_name]]

        u_near_df = apply_strike_smoothing(df, "u_opt_ema_8", 6.0, 16, "u_opt_near")
        u_far_df = apply_strike_smoothing(df, "u_opt_ema_32", 24.0, 64, "u_opt_far")
        omega_near_df = apply_strike_smoothing(df, "Omega_opt", 6.0, 16, "Omega_opt_near")
        omega_far_df = apply_strike_smoothing(df, "Omega_opt", 24.0, 64, "Omega_opt_far")

        df = df.merge(u_near_df, on=["window_end_ts_ns", "right", "side", "rel_ticks"], how="left")
        df = df.merge(u_far_df, on=["window_end_ts_ns", "right", "side", "rel_ticks"], how="left")
        df = df.merge(omega_near_df, on=["window_end_ts_ns", "right", "side", "rel_ticks"], how="left")
        df = df.merge(omega_far_df, on=["window_end_ts_ns", "right", "side", "rel_ticks"], how="left")

        df["u_opt_near"] = df["u_opt_near"].fillna(0.0)
        df["u_opt_far"] = df["u_opt_far"].fillna(0.0)
        df["Omega_opt_near"] = df["Omega_opt_near"].fillna(0.0)
        df["Omega_opt_far"] = df["Omega_opt_far"].fillna(0.0)

        df["u_opt_prom"] = df["u_opt_near"] - df["u_opt_far"]
        df["Omega_opt_prom"] = df["Omega_opt_near"] - df["Omega_opt_far"]

        # Spatial derivatives on strike lattice (scaled to tick units)
        df = df.sort_values(["window_end_ts_ns", "right", "side", "rel_ticks"])
        g_space = df.groupby(["window_end_ts_ns", "right", "side"])
        u_near = df["u_opt_near"]
        df["du_opt_dx"] = (
            g_space["u_opt_near"].shift(-1) - g_space["u_opt_near"].shift(1)
        ) / (2.0 * STRIKE_TICKS)
        df["d2u_opt_dx2"] = (
            g_space["u_opt_near"].shift(-1) - 2.0 * u_near + g_space["u_opt_near"].shift(1)
        ) / float(STRIKE_TICKS ** 2)
        df["du_opt_dx"] = df["du_opt_dx"].fillna(0.0)
        df["d2u_opt_dx2"] = df["d2u_opt_dx2"].fillna(0.0)

        # Viscosity + permeability (options lattice)
        df["nu_opt"] = 1.0 + df["Omega_opt_near"] + 2.0 * np.maximum(0.0, df["Omega_opt_prom"])
        df["kappa_opt"] = 1.0 / df["nu_opt"]

        # Pressure gradient (directional)
        df["pressure_grad_opt"] = np.where(df["side"] == "B", df["u_opt_p"], -df["u_opt_p"])

        df["event_ts_ns"] = df["window_end_ts_ns"]

        return df[
            [
                "window_end_ts_ns",
                "event_ts_ns",
                "spot_ref_price_int",
                "strike_price_int",
                "strike_points",
                "rel_ticks",
                "right",
                "side",
                "add_intensity",
                "fill_intensity",
                "pull_intensity",
                "liquidity_velocity",
                "rho_opt",
                "phi_rest_opt",
                "u_opt_ema_2",
                "u_opt_ema_8",
                "u_opt_ema_32",
                "u_opt_ema_128",
                "u_opt_band_fast",
                "u_opt_band_mid",
                "u_opt_band_slow",
                "u_opt_wave_energy",
                "du_opt_dt",
                "d2u_opt_dt2",
                "u_opt_p",
                "u_opt_p_slow",
                "u_opt_near",
                "u_opt_far",
                "u_opt_prom",
                "du_opt_dx",
                "d2u_opt_dx2",
                "Omega_opt",
                "Omega_opt_near",
                "Omega_opt_far",
                "Omega_opt_prom",
                "nu_opt",
                "kappa_opt",
                "pressure_grad_opt",
            ]
        ]


def _round_to_nearest_strike_int(price_int: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(price_int, dtype="int64")
    return ((values + STRIKE_STEP_INT // 2) // STRIKE_STEP_INT) * STRIKE_STEP_INT
