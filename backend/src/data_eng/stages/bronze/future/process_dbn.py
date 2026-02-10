from __future__ import annotations

from pathlib import Path
from typing import List

import databento as db
import numpy as np
import pandas as pd

from src.data_eng.config import AppConfig, ProductConfig
from src.data_eng.io import is_partition_complete, partition_ref, write_partition
from src.data_eng.contracts import load_avro_contract, enforce_contract

from ...base import Stage, StageIO


def extract_flags_vectorized(flags: np.ndarray) -> pd.DataFrame:
    """Extract flag bits into boolean columns."""
    return pd.DataFrame({
        'is_last': (flags & 0x80).astype(bool),
        'is_snapshot': (flags & 0x20).astype(bool),
        'is_bad_ts': (flags & 0x08).astype(bool),
        'is_bad_book': (flags & 0x04).astype(bool),
    })


def add_est_timestamp(df: pd.DataFrame) -> pd.Series:
    """
    Convert ts_event (ns, UTC) to ISO-8601 America/New_York string with offset.
    """
    ts = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    ts_ny = ts.dt.tz_convert("America/New_York")
    est_str = ts_ny.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    return est_str.str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)


class BronzeProcessDBN(Stage):
    """Bronze stage: DBN → Parquet with minimal transformations.
    
    Transformations:
    - Filter to MBP-10 records only (rtype=10)
    - Filter out spreads (symbol contains '-')
    - Filter out crossed/locked books (bid_px_00 >= ask_px_00)
    - Drop ts_in_delta column
    - Extract flag bits into boolean columns
    
    Input: Raw DBN files in lake/raw/source=databento/product_type=future/symbol={symbol}/table=market_by_price_10_dbn/
    Output: silver.future.market_by_price_10 (one partition per contract)
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="bronze_process_dbn",
            io=StageIO(
                inputs=[],
                output="silver.future.market_by_price_10",
            ),
        )
    
    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str, product: ProductConfig | None = None, dbn_root: Path | None = None) -> None:
        """Process DBN files for all contracts matching symbol prefix on date dt."""
        
        if dbn_root is None:
            dbn_root = cfg.lake_root / "raw"
        
        date_compact = dt.replace("-", "")
        dbn_pattern = f"*{date_compact}*.dbn"
        
        raw_path = dbn_root / "source=databento" / "product_type=future" / f"symbol={symbol}" / "table=market_by_price_10_dbn"
        dbn_files = list(raw_path.glob(dbn_pattern))
        
        if not dbn_files:
            raise FileNotFoundError(f"No DBN files found for date {dt} in {raw_path}/")
        
        print(f"  Processing {len(dbn_files)} DBN file(s) for {dt}")
        
        all_dfs: List[pd.DataFrame] = []
        
        for dbn_file in dbn_files:
            print(f"    Reading {dbn_file.name}...")
            
            store = db.DBNStore.from_file(str(dbn_file))
            df_raw = store.to_df().reset_index()
            
            if df_raw.empty:
                continue
            
            is_mbp10 = df_raw['rtype'].to_numpy() == 10
            df = df_raw[is_mbp10].copy()
            
            if df.empty:
                continue
            
            is_spread = df['symbol'].str.contains('-', regex=False).to_numpy()
            df = df[~is_spread].copy()
            
            if df.empty:
                continue
            
            is_crossed = df['bid_px_00'].to_numpy() >= df['ask_px_00'].to_numpy()
            n_crossed = is_crossed.sum()
            if n_crossed > 0:
                print(f"      Filtered {n_crossed} crossed/locked book rows")
            df = df[~is_crossed].copy()
            
            if df.empty:
                continue
            
            if 'ts_in_delta' in df.columns:
                df.drop(columns=['ts_in_delta'], inplace=True)
            
            flag_df = extract_flags_vectorized(df['flags'].to_numpy())
            df = pd.concat([df, flag_df], axis=1)
            
            all_dfs.append(df)
            print(f"      Loaded {len(df):,} MBP-10 records")
        
        if not all_dfs:
            raise ValueError(f"No MBP-10 records found for {dt}")
        
        print(f"    Concatenating records...")
        df_all = pd.concat(all_dfs, ignore_index=True, copy=False)
        df_all.sort_values('ts_event', inplace=True)
        
        all_contracts = [str(c) for c in df_all['symbol'].unique() if pd.notna(c)]
        print(f"    Found {len(all_contracts)} contracts: {sorted(all_contracts)}")

        contract_path = repo_root / cfg.dataset(self.io.output).contract
        contract = load_avro_contract(contract_path)

        for contract_symbol in all_contracts:
            out_ref = partition_ref(cfg, self.io.output, contract_symbol, dt)
            if is_partition_complete(out_ref):
                print(f"    Skipping {contract_symbol} (already complete)")
                continue

            symbol_mask = df_all['symbol'].to_numpy() == contract_symbol
            df_contract = df_all[symbol_mask].copy()
            if df_contract.empty:
                continue

            df_contract["ts_event_est"] = add_est_timestamp(df_contract)
            df_contract = enforce_contract(df_contract, contract)

            print(f"    Writing {contract_symbol}: {len(df_contract):,} records")
            write_partition(
                cfg=cfg,
                dataset_key=self.io.output,
                symbol=contract_symbol,
                dt=dt,
                df=df_contract,
                contract_path=contract_path,
                inputs=[],
                stage=self.name,
            )

        print(f"  ✓ Bronze complete for {len(all_contracts)} contract(s)")
    
    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        """Not used for Bronze stage (overrides run() instead)."""
        raise NotImplementedError("Bronze stage uses custom run() method")



