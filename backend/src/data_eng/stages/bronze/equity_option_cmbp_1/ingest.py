from __future__ import annotations

from pathlib import Path
from typing import List

import databento as db
import pandas as pd
from databento.common.enums import PriceType

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import is_partition_complete, partition_ref, write_partition
from ....utils import session_window_ns


class BronzeIngestEquityOptionCmbp1(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="bronze_ingest_equity_option_cmbp_1",
            io=StageIO(
                inputs=[],
                output="bronze.equity_option_cmbp_1.cmbp_1",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        date_compact = dt.replace("-", "")
        raw_path = (
            cfg.lake_root
            / "raw"
            / "source=databento"
            / "product_type=equity_option_cmbp_1"
            / f"symbol={symbol}"
            / "table=cmbp_1_dbn"
        )
        dbn_files = list(raw_path.glob(f"*{date_compact}*.dbn"))
        if not dbn_files:
            raise FileNotFoundError(f"No DBN files found for date {dt} in {raw_path}/")

        all_dfs: List[pd.DataFrame] = []
        for dbn_file in dbn_files:
            store = db.DBNStore.from_file(str(dbn_file))
            df_raw = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
            df_raw = df_raw.reset_index()
            if df_raw.empty:
                continue

            df_raw = df_raw.loc[df_raw["symbol"].notna()].copy()
            if df_raw.empty:
                continue

            rtypes = df_raw["rtype"].dropna().unique()
            if len(rtypes) > 1:
                raise ValueError(f"Multiple rtype values found in CMBP-1 data: {sorted(rtypes)}")

            all_dfs.append(df_raw)

        if not all_dfs:
            raise ValueError(f"No CMBP-1 records found for {dt}")

        df_all = pd.concat(all_dfs, ignore_index=True, copy=False)
        df_all["ts_event"] = df_all["ts_event"].astype("int64")
        df_all["ts_recv"] = df_all["ts_recv"].astype("int64")

        session_start_ns, session_end_ns = session_window_ns(dt)
        df_all = df_all.loc[
            (df_all["ts_event"] >= session_start_ns) & (df_all["ts_event"] < session_end_ns)
        ].copy()
        if df_all.empty:
            raise ValueError(f"No CMBP-1 records in session window for {dt}")

        int_cols = [
            "ts_recv",
            "flags",
            "ts_event",
            "ts_in_delta",
            "rtype",
            "bid_px_00",
            "publisher_id",
            "ask_px_00",
            "instrument_id",
            "bid_sz_00",
            "ask_sz_00",
            "bid_pb_00",
            "price",
            "ask_pb_00",
            "size",
        ]
        for col in int_cols:
            if col in df_all.columns:
                df_all[col] = df_all[col].astype("int64")

        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        contract_path = repo_root / cfg.dataset(self.io.output).contract
        contract = load_avro_contract(contract_path)
        df_all = enforce_contract(df_all, contract)
        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_all,
            contract_path=contract_path,
            inputs=[],
            stage=self.name,
        )
