from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import databento as db
import numpy as np
import pandas as pd
from databento.common.enums import PriceType

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import is_partition_complete, partition_ref, read_partition, write_partition
from ....utils import session_window_ns

# CMBP-1 schema uses rtype 177 (CBBO - Consolidated BBO)
RTYPE_CMBP = 177
NULL_PRICE = np.iinfo("int64").max

RIGHT_CLASSES = {"C", "P"}


class BronzeIngestEquityOptionMbo(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="bronze_ingest_equity_option_mbo",
            io=StageIO(
                inputs=[],
                output="bronze.equity_option_mbo.cmbp_1",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        checkpoint_key = "bronze_cache.equity_option_mbo.cmbp_1_0dte"
        date_compact = dt.replace("-", "")
        raw_path = (
            cfg.lake_root
            / "raw"
            / "source=databento"
            / "product_type=equity_option_cmbp_1"
            / f"symbol={symbol}"
            / "table=cmbp_1"
        )
        checkpoint_ref = partition_ref(cfg, checkpoint_key, symbol, dt)
        if is_partition_complete(checkpoint_ref):
            checkpoint_contract = load_avro_contract(repo_root / cfg.dataset(checkpoint_key).contract)
            df_all = read_partition(checkpoint_ref)
            df_all = enforce_contract(df_all, checkpoint_contract)
        else:
            dbn_files = list(raw_path.glob(f"*{date_compact}*.dbn*"))
            if not dbn_files:
                raise FileNotFoundError(f"No DBN files found for date {dt} in {raw_path}/")

            def_files = _definition_files(cfg.lake_root, date_compact)
            if not def_files:
                raise FileNotFoundError(f"No instrument definition files found for {date_compact}")

            meta_map = _load_definitions(def_files, dt, symbol)

            all_dfs: List[pd.DataFrame] = []
            for dbn_file in dbn_files:
                store = db.DBNStore.from_file(str(dbn_file))
                df_raw = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
                df_raw = df_raw.reset_index()
                if df_raw.empty:
                    continue

                df = df_raw.copy()
                if "rtype" in df.columns:
                    df = df.loc[df["rtype"] == RTYPE_CMBP].copy()
                if df.empty:
                    continue

                df = df.loc[df["instrument_id"].isin(meta_map.keys())].copy()
                if df.empty:
                    continue

                df = _apply_definition_meta(df, meta_map)
                all_dfs.append(df)

            if not all_dfs:
                raise ValueError(f"No option CMBP-1 records found for {dt}")

            df_all = pd.concat(all_dfs, ignore_index=True, copy=False)
            df_all["ts_event"] = df_all["ts_event"].astype("int64")
            if "ts_recv" not in df_all.columns:
                df_all["ts_recv"] = df_all["ts_event"]
            df_all["ts_recv"] = df_all["ts_recv"].astype("int64")

            session_start_ns, session_end_ns = session_window_ns(dt)
            df_all = df_all.loc[
                (df_all["ts_event"] >= session_start_ns) & (df_all["ts_event"] < session_end_ns)
            ].copy()
            if df_all.empty:
                raise ValueError(f"No option CMBP-1 records in session window for {dt}")

            for col in ("price", "bid_px_00", "ask_px_00"):
                if col in df_all.columns:
                    null_price = df_all[col].astype("int64") == NULL_PRICE
                    if null_price.any():
                        df_all.loc[null_price, col] = 0

            df_all = df_all.sort_values(["ts_event", "ts_recv"], ascending=[True, True])

            required_cols = {
                "ts_event",
                "ts_recv",
                "publisher_id",
                "instrument_id",
                "bid_px_00",
                "ask_px_00",
                "bid_sz_00",
                "ask_sz_00",
                "underlying",
                "right",
                "strike",
                "expiration",
            }
            missing_cols = required_cols.difference(df_all.columns)
            if missing_cols:
                raise ValueError(f"Missing required CMBP-1 columns: {sorted(missing_cols)}")

            df_all["publisher_id"] = df_all["publisher_id"].astype("int64")
            df_all["instrument_id"] = df_all["instrument_id"].astype("int64")
            if "flags" in df_all.columns:
                df_all["flags"] = df_all["flags"].astype("int64")
            if "ts_in_delta" in df_all.columns:
                df_all["ts_in_delta"] = df_all["ts_in_delta"].astype("int64")
            if "rtype" in df_all.columns:
                df_all["rtype"] = df_all["rtype"].astype("int64")
            if "bid_px_00" in df_all.columns:
                df_all["bid_px_00"] = df_all["bid_px_00"].astype("int64")
            if "ask_px_00" in df_all.columns:
                df_all["ask_px_00"] = df_all["ask_px_00"].astype("int64")
            if "bid_sz_00" in df_all.columns:
                df_all["bid_sz_00"] = df_all["bid_sz_00"].astype("int64")
            if "ask_sz_00" in df_all.columns:
                df_all["ask_sz_00"] = df_all["ask_sz_00"].astype("int64")
            if "bid_pb_00" in df_all.columns:
                df_all["bid_pb_00"] = df_all["bid_pb_00"].astype("int64")
            if "ask_pb_00" in df_all.columns:
                df_all["ask_pb_00"] = df_all["ask_pb_00"].astype("int64")
            if "price" in df_all.columns:
                df_all["price"] = df_all["price"].astype("int64")
            if "size" in df_all.columns:
                df_all["size"] = df_all["size"].astype("int64")

        contract_path = repo_root / cfg.dataset(self.io.output).contract
        contract = load_avro_contract(contract_path)

        df_all = df_all.loc[:, contract.fields].copy()

        if not is_partition_complete(checkpoint_ref):
            df_checkpoint = enforce_contract(df_all.copy(), contract)
            write_partition(
                cfg=cfg,
                dataset_key=checkpoint_key,
                symbol=symbol,
                dt=dt,
                df=df_checkpoint,
                contract_path=repo_root / cfg.dataset(checkpoint_key).contract,
                inputs=[],
                stage=self.name,
            )

        underlying_symbols = [str(s) for s in df_all["underlying"].unique() if pd.notna(s)]
        for underlying in underlying_symbols:
            out_ref = partition_ref(cfg, self.io.output, underlying, dt)
            if is_partition_complete(out_ref):
                continue

            df_underlying = df_all.loc[df_all["underlying"] == underlying].copy()
            if df_underlying.empty:
                continue

            df_underlying = enforce_contract(df_underlying, contract)
            write_partition(
                cfg=cfg,
                dataset_key=self.io.output,
                symbol=underlying,
                dt=dt,
                df=df_underlying,
                contract_path=contract_path,
                inputs=[],
                stage=self.name,
            )


def _definition_files(lake_root: Path, date_compact: str) -> List[Path]:
    base = lake_root / "raw" / "source=databento" / "dataset=definition" / "venue=opra"
    if not base.exists():
        return []
    return sorted(base.glob(f"*{date_compact}*.dbn*"))


def _load_definitions(files: List[Path], session_date: str, symbol: str) -> Dict[int, Dict[str, object]]:
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
    df_all = df_all.loc[df_all["instrument_class"].isin(RIGHT_CLASSES)].copy()
    df_all = df_all.loc[df_all["underlying"].astype(str).str.upper() == symbol.upper()].copy()
    exp_dates = (
        pd.to_datetime(df_all["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("America/New_York")
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


def _apply_definition_meta(df: pd.DataFrame, meta_map: Dict[int, Dict[str, object]]) -> pd.DataFrame:
    meta_df = pd.DataFrame.from_dict(meta_map, orient="index")
    meta_df.index.name = "instrument_id"
    meta_df = meta_df.reset_index()
    df = df.merge(meta_df, on="instrument_id", how="left")
    missing = df["underlying"].isna() | df["right"].isna() | df["strike"].isna() | df["expiration"].isna()
    if missing.any():
        raise ValueError("Missing instrument definitions for option CMBP-1 rows")
    return df
