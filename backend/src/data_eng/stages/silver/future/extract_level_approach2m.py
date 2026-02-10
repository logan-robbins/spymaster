"""Silver stage: extract 2-minute candle level approach datasets.

Takes bar5s data and session levels, produces approach2m datasets for each level type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig, ProductConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)
from .level2m import compute_level_approach2m

LEVEL_TYPES = ["pm_high", "pm_low", "or_high", "or_low"]


class SilverExtractLevelApproach2m(Stage):
    """Silver stage: extract 2-minute candle level approach datasets.

    For each level type (pm_high, pm_low, or_high, or_low):
    - Reads bar5s data and session levels
    - Detects trigger candles (touched + closed in zone)
    - Computes microstructure signatures
    - Produces approach2m dataset

    Input: bar5s + first4h (for level prices)
    Output: one dataset per level type
    """

    def __init__(self) -> None:
        super().__init__(
            name="silver_extract_level_approach2m",
            io=StageIO(
                inputs=[
                    "silver.future.market_by_price_10_bar5s",
                    "silver.future.market_by_price_10_first4h",
                ],
                output="silver.future.market_by_price_10_pm_high_approach2m",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str, product: ProductConfig | None = None) -> None:
        output_keys = [
            f"silver.future.market_by_price_10_{lt}_approach2m"
            for lt in LEVEL_TYPES
        ]

        all_complete = all(
            is_partition_complete(partition_ref(cfg, k, symbol, dt))
            for k in output_keys
        )
        if all_complete:
            return

        bar5s_key = "silver.future.market_by_price_10_bar5s"
        first4h_key = "silver.future.market_by_price_10_first4h"

        bar5s_ref = partition_ref(cfg, bar5s_key, symbol, dt)
        first4h_ref = partition_ref(cfg, first4h_key, symbol, dt)

        if not is_partition_complete(bar5s_ref):
            raise FileNotFoundError(f"Input not ready: {bar5s_key} dt={dt}")
        if not is_partition_complete(first4h_ref):
            raise FileNotFoundError(f"Input not ready: {first4h_key} dt={dt}")

        bar5s_contract_path = repo_root / cfg.dataset(bar5s_key).contract
        bar5s_contract = load_avro_contract(bar5s_contract_path)
        df_bar5s = read_partition(bar5s_ref)
        df_bar5s = enforce_contract(df_bar5s, bar5s_contract)

        df_first4h = read_partition(first4h_ref)
        if len(df_first4h) == 0:
            levels = {}
        else:
            levels = {
                "pm_high": float(df_first4h["pm_high"].iloc[0]),
                "pm_low": float(df_first4h["pm_low"].iloc[0]),
                "or_high": float(df_first4h["or_high"].iloc[0]),
                "or_low": float(df_first4h["or_low"].iloc[0]),
            }

        for level_type in LEVEL_TYPES:
            output_key = f"silver.future.market_by_price_10_{level_type}_approach2m"
            out_ref = partition_ref(cfg, output_key, symbol, dt)

            if is_partition_complete(out_ref):
                continue

            level_price = levels.get(level_type, float("nan"))

            if len(df_bar5s) == 0 or pd.isna(level_price):
                df_out = pd.DataFrame()
            else:
                df_out = compute_level_approach2m(
                    df_bar5s, level_price, level_type, dt, symbol
                )

            out_contract_path = repo_root / cfg.dataset(output_key).contract
            out_contract = load_avro_contract(out_contract_path)

            if len(df_out) > 0:
                df_out = enforce_contract(df_out, out_contract)

            lineage: List[Dict[str, Any]] = [
                {
                    "dataset": bar5s_ref.dataset_key,
                    "dt": dt,
                    "manifest_sha256": read_manifest_hash(bar5s_ref),
                },
                {
                    "dataset": first4h_ref.dataset_key,
                    "dt": dt,
                    "manifest_sha256": read_manifest_hash(first4h_ref),
                },
            ]

            write_partition(
                cfg=cfg,
                dataset_key=output_key,
                symbol=symbol,
                dt=dt,
                df=df_out,
                contract_path=out_contract_path,
                inputs=lineage,
                stage=self.name,
            )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        raise NotImplementedError("Use run() directly")
