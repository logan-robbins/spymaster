from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import databento as db
import pandas as pd
from databento.common.enums import PriceType

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import is_partition_complete, partition_ref, read_partition, write_partition

class BronzeIngestFutureOptionStatistics(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="bronze_ingest_future_option_statistics",
            io=StageIO(
                inputs=[],
                output="bronze.future_option.statistics",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        checkpoint_key = "bronze_cache.future_option.statistics_0dte"
        checkpoint_ref = partition_ref(cfg, checkpoint_key, symbol, dt)

        if is_partition_complete(checkpoint_ref):
            checkpoint_contract = load_avro_contract(repo_root / cfg.dataset(checkpoint_key).contract)
            df_out = read_partition(checkpoint_ref)
            df_out = enforce_contract(df_out, checkpoint_contract)
        else:
            date_compact = dt.replace("-", "")
            raw_path = (
                cfg.lake_root
                / "raw"
                / "source=databento"
                / "product_type=future_option_mbo"
                / f"symbol={symbol}"
                / "table=statistics"
            )

            dbn_files = list(raw_path.glob(f"*{date_compact}*.dbn"))
            if not dbn_files:
                raise FileNotFoundError(f"No statistics DBN files found for date {dt} in {raw_path}/")

            def_files = _definition_files(cfg.lake_root, date_compact)
            if not def_files:
                raise FileNotFoundError(f"No instrument definition files found for {date_compact}")

            meta_map = _load_definitions(def_files, dt)

            all_dfs: List[pd.DataFrame] = []
            for dbn_file in dbn_files:
                store = db.DBNStore.from_file(str(dbn_file))
                df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
                df = df.reset_index()
                if df.empty:
                    continue
                all_dfs.append(df)

            if not all_dfs:
                raise ValueError(f"No statistics records found for {dt}")

            df_all = pd.concat(all_dfs, ignore_index=True)

            df_all["ts_event"] = df_all["ts_event"].astype("int64")
            df_all["ts_recv"] = df_all["ts_recv"].fillna(0).astype("int64")
            df_all["instrument_id"] = df_all["instrument_id"].astype("int64")
            df_all["quantity"] = df_all["quantity"].fillna(0).astype("int64")
            df_all["stat_type"] = df_all["stat_type"].astype("int64")

            df_all = df_all.loc[df_all["stat_type"] == 1].copy()
            if df_all.empty:
                raise ValueError(f"No open interest statistics for {dt}")

            meta_df = pd.DataFrame.from_dict(meta_map, orient="index")
            meta_df.index.name = "instrument_id"
            meta_df = meta_df.reset_index()

            df_all = df_all.merge(meta_df, on="instrument_id", how="left")
            missing = (
                df_all["underlying"].isna()
                | df_all["right"].isna()
                | df_all["strike"].isna()
                | df_all["expiration"].isna()
            )
            if missing.any():
                raise ValueError("Missing instrument definitions for statistics rows")

            df_out = pd.DataFrame(
                {
                    "ts_event_ns": df_all["ts_event"].astype("int64"),
                    "ts_recv_ns": df_all["ts_recv"].astype("int64"),
                    "source": "DATABENTO",
                    "underlying": df_all["underlying"].astype(str),
                    "option_symbol": df_all["symbol"].astype(str),
                    "exp_date": pd.to_datetime(df_all["expiration"].astype("int64"), utc=True).dt.date.astype(str),
                    "strike": df_all["strike"].astype("int64") * 1e-9,
                    "right": df_all["right"].astype(str),
                    "open_interest": df_all["quantity"].astype("int64").astype(float),
                }
            )

            contract_path = repo_root / cfg.dataset(self.io.output).contract
            contract = load_avro_contract(contract_path)
            df_out = enforce_contract(df_out, contract)

            write_partition(
                cfg=cfg,
                dataset_key=checkpoint_key,
                symbol=symbol,
                dt=dt,
                df=df_out.copy(),
                contract_path=repo_root / cfg.dataset(checkpoint_key).contract,
                inputs=[],
                stage=self.name,
            )

        contract_path = repo_root / cfg.dataset(self.io.output).contract
        contract = load_avro_contract(contract_path)

        for underlying in sorted(df_out["underlying"].unique()):
            out_ref = partition_ref(cfg, self.io.output, underlying, dt)
            if is_partition_complete(out_ref):
                continue
            df_curr = df_out.loc[df_out["underlying"] == underlying].copy()
            if df_curr.empty:
                continue
            df_curr = enforce_contract(df_curr, contract)
            write_partition(
                cfg=cfg,
                dataset_key=self.io.output,
                symbol=underlying,
                dt=dt,
                df=df_curr,
                contract_path=contract_path,
                inputs=[],
                stage=self.name,
            )


def _definition_files(lake_root: Path, date_compact: str) -> List[Path]:
    base = lake_root / "raw" / "source=databento" / "dataset=definition"
    if not base.exists():
        return []
    return sorted(base.glob(f"*{date_compact}*.dbn*"))


def _load_definitions(files: List[Path], session_date: str) -> Dict[int, Dict[str, object]]:
    dfs = []
    for path in files:
        store = db.DBNStore.from_file(str(path))
        df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
        if df.empty:
            continue
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("Instrument definitions empty")
    df_all = pd.concat(dfs, ignore_index=True)
    required = {"instrument_id", "instrument_class", "underlying", "strike_price", "expiration"}
    missing = required.difference(df_all.columns)
    if missing:
        raise ValueError(f"Missing definition columns: {sorted(missing)}")
    df_all = df_all.sort_values("ts_event").groupby("instrument_id", as_index=False).last()
    df_all = df_all.loc[df_all["instrument_class"].isin({"C", "P"})].copy()
    exp_dates = (
        pd.to_datetime(df_all["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("Etc/GMT+5")
        .dt.date.astype(str)
    )
    df_all = df_all.loc[exp_dates == session_date].copy()
    meta = {}
    for row in df_all.itertuples(index=False):
        meta[int(row.instrument_id)] = {
            "underlying": str(row.underlying),
            "right": str(row.instrument_class),
            "strike": int(row.strike_price),
            "expiration": int(row.expiration),
        }
    return meta
