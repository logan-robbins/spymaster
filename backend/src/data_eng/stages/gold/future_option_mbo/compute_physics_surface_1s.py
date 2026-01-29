from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)

EPS_QTY = 1.0


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
                ]
            )

        df = df_flow.copy()

        depth_start = df["depth_qty_start"].astype(float).to_numpy()
        depth_end = df["depth_qty_end"].astype(float).to_numpy() # Need end for rho
        depth_rest = df["depth_qty_rest"].astype(float).to_numpy()
        add_qty = df["add_qty"].astype(float).to_numpy()
        fill_qty = df["fill_qty"].astype(float).to_numpy()
        pull_qty = df["pull_qty_total"].astype(float).to_numpy()

        denom = depth_start + EPS_QTY

        df["add_intensity"] = add_qty / denom
        df["fill_intensity"] = fill_qty / denom
        df["pull_intensity"] = pull_qty / denom
        df["liquidity_velocity"] = df["add_intensity"] - df["pull_intensity"] - df["fill_intensity"]

        # -------------------------------------------------------------------------
        # 8.2 Options obstacle field on strike lattice
        # -------------------------------------------------------------------------
        import numpy as np
        
        # rho_opt
        df["rho_opt"] = np.log(1.0 + depth_end)
        
        # phi_rest_opt
        df["phi_rest_opt"] = depth_rest / (depth_end + 1.0)
        
        # EMAs
        # Sort to ensure time ordering for EWM
        df = df.sort_values(["strike_price_int", "right", "side", "window_end_ts_ns"])
        
        # Define alphas
        # alpha = 1 - exp(-dt/tau), dt=1s
        alpha_8 = 1.0 - np.exp(-1.0 / 8.0)
        alpha_32 = 1.0 - np.exp(-1.0 / 32.0)
        
        # Group by buckets (strike/right/side)
        # Note: 'strike_price_int' identifies the absolute level.
        g = df.groupby(["strike_price_int", "right", "side"])["liquidity_velocity"]
        
        # Compute EWM and reset MultiIndex to align with df index
        u_opt_ema_8 = g.ewm(alpha=alpha_8, adjust=False).mean()
        u_opt_ema_32 = g.ewm(alpha=alpha_32, adjust=False).mean()
        
        # Drop the groupby index levels to get a Series with the original df index
        df["u_opt_ema_8"] = u_opt_ema_8.reset_index(level=[0, 1, 2], drop=True)
        df["u_opt_ema_32"] = u_opt_ema_32.reset_index(level=[0, 1, 2], drop=True)
        
        # u_opt_p_slow
        df["u_opt_p_slow"] = df["phi_rest_opt"] * df["u_opt_ema_32"]
        
        # Omega_opt
        # Omega_opt = rho_opt * (0.5 + 0.5*phi_rest_opt) * (1 + max(0, u_opt_p_slow))
        term_rest = 0.5 + 0.5 * df["phi_rest_opt"]
        term_reinforce = 1.0 + np.maximum(0.0, df["u_opt_p_slow"])
        df["Omega_opt"] = df["rho_opt"] * term_rest * term_reinforce

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
                "u_opt_ema_8",
                "u_opt_ema_32",
                "u_opt_p_slow",
                "Omega_opt",
            ]
        ]
