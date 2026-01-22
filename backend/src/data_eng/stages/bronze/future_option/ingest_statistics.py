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
        date_compact = dt.replace("-", "")
        # Path convention for statistics
        # raw/source=databento/product_type=future_option/symbol=ES/table=statistics
        raw_path = (
            cfg.lake_root
            / "raw"
            / "source=databento"
            / "product_type=future_option"
            / f"symbol={symbol}"
            / "table=statistics"
        )
        
        dbn_files = list(raw_path.glob(f"*{date_compact}*.dbn"))
        if not dbn_files:
             # Try without table=statistics if missing?
             # ingest_preview looks for table=market_by_order_dbn
             # I'll strict to the table=statistics as standard.
             return

        all_dfs: List[pd.DataFrame] = []
        for dbn_file in dbn_files:
            try:
                store = db.DBNStore.from_file(str(dbn_file))
                # Statistics
                df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
                if df.empty: continue
                all_dfs.append(df)
            except Exception:
                continue

        if not all_dfs:
            return

        df_all = pd.concat(all_dfs, ignore_index=True)
        
        # Casts
        if "ts_event" in df_all.columns: df_all["ts_event"] = df_all["ts_event"].astype("int64")
        if "ts_recv" in df_all.columns: df_all["ts_recv"] = df_all["ts_recv"].fillna(0).astype("int64")
        if "instrument_id" in df_all.columns: df_all["instrument_id"] = df_all["instrument_id"].astype("int64")
        if "price" in df_all.columns: df_all["price"] = df_all["price"].fillna(2**63 - 1).astype("int64")
        if "size" in df_all.columns: df_all["size"] = df_all["size"].fillna(0).astype("int64")
        if "stat_type" in df_all.columns: df_all["stat_type"] = df_all["stat_type"].astype("int64")
        if "update_action" in df_all.columns: df_all["update_action"] = df_all["update_action"].astype("int64")
        if "flags" in df_all.columns: df_all["flags"] = df_all["flags"].astype("int64")

        contract_path = repo_root / cfg.dataset(self.io.output).contract
        contract = load_avro_contract(contract_path)
        
        # Write partition
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if not is_partition_complete(out_ref):
            df_curr = enforce_contract(df_all, contract)
            write_partition(
                cfg=cfg,
                dataset_key=self.io.output,
                symbol=symbol,
                dt=dt,
                df=df_curr,
                contract_path=contract_path,
                inputs=[],
                stage=self.name
            )
