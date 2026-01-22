from __future__ import annotations

from pathlib import Path
from typing import List

import databento as db
import numpy as np
import pandas as pd
from databento.common.enums import PriceType

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import is_partition_complete, partition_ref, write_partition

class BronzeIngestInstrumentDefinitions(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="bronze_ingest_instrument_definitions",
            io=StageIO(
                inputs=[],
                output="bronze.shared.instrument_definitions",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        date_compact = dt.replace("-", "")
        search_paths = [
            cfg.lake_root / "raw" / "source=databento" / "dataset=definition",
            cfg.lake_root / "raw" / "source=databento" / "product_type=future" / f"symbol={symbol}" / "table=definition",
        ]
        
        dbn_files = []
        for p in search_paths:
            if p.exists():
                dbn_files.extend(list(p.glob(f"*{date_compact}*.dbn")))
        
        if not dbn_files:
             return

        all_dfs: List[pd.DataFrame] = []
        for dbn_file in dbn_files:
            try:
                store = db.DBNStore.from_file(str(dbn_file))
                df = store.to_df(price_type=PriceType.FIXED, pretty_ts=False, map_symbols=True)
                if df.empty: continue
                all_dfs.append(df)
            except Exception:
                continue

        if not all_dfs:
            return

        df_all = pd.concat(all_dfs, ignore_index=True)
        
        if "ts_event" in df_all.columns: df_all["ts_event"] = df_all["ts_event"].fillna(0).astype("int64")
        if "instrument_id" in df_all.columns: df_all["instrument_id"] = df_all["instrument_id"].fillna(0).astype("int64") 
        if "expiration" in df_all.columns: df_all["expiration"] = df_all["expiration"].fillna(np.iinfo(np.int64).max).astype("int64")
        if "strike_price" in df_all.columns: df_all["strike_price"] = df_all["strike_price"].fillna(0).astype("int64")
        if "underlying_id" in df_all.columns: df_all["underlying_id"] = df_all["underlying_id"].fillna(0).astype("int64")
        
        contract_path = repo_root / cfg.dataset(self.io.output).contract
        contract = load_avro_contract(contract_path)
        
        if "symbol" in df_all.columns:
            syms = df_all["symbol"].unique()
            for s in syms:
                if not isinstance(s, str): continue
                
                out_ref = partition_ref(cfg, self.io.output, s, dt)
                if is_partition_complete(out_ref): continue
                
                df_s = df_all[df_all["symbol"] == s].copy()
                df_s = enforce_contract(df_s, contract)
                
                write_partition(
                    cfg=cfg,
                    dataset_key=self.io.output,
                    symbol=s,
                    dt=dt,
                    df=df_s,
                    contract_path=contract_path,
                    inputs=[],
                    stage=self.name
                )
