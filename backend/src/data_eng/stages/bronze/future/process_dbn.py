from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import databento as db
import numpy as np
import pandas as pd

from src.data_eng.config import AppConfig
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


def get_front_month_contract(contracts: List[str], dt: str) -> str:
    """
    Determine front month contract for given date.
    
    ES contract months: H (Mar), M (Jun), U (Sep), Z (Dec)
    Front month is the nearest unexpired quarterly contract.
    """
    month_codes = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}
    date = datetime.strptime(dt, '%Y-%m-%d')
    
    contract_dates = []
    for contract in contracts:
        if len(contract) < 4:
            continue
        month_code = contract[2]
        year_digit = contract[3]
        
        if month_code not in month_codes:
            continue
        
        month = month_codes[month_code]
        year = 2020 + int(year_digit)
        
        expiry = datetime(year, month, 1)
        
        if expiry >= date:
            contract_dates.append((contract, expiry))
    
    if not contract_dates:
        raise ValueError(f"No valid front month contract found for {dt} in {contracts}")
    
    contract_dates.sort(key=lambda x: x[1])
    return contract_dates[0][0]


class BronzeProcessDBN(Stage):
    """Bronze stage: DBN → Parquet with minimal transformations.
    
    Transformations:
    - Filter to MBP-10 records only (rtype=10)
    - Filter out spreads (symbol contains '-')
    - Drop ts_in_delta column
    - Extract flag bits into boolean columns
    
    Input: Raw DBN files in lake/raw/source=databento/product_type=future/symbol={symbol}/table=market_by_price_10/
    Output: bronze.future.market_by_price_10
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="bronze_process_dbn",
            io=StageIO(
                inputs=[],
                output="bronze.future.market_by_price_10",
            ),
        )
    
    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str, dbn_root: Path | None = None) -> None:
        """Process DBN files for all contracts matching symbol prefix on date dt."""
        
        if dbn_root is None:
            dbn_root = cfg.lake_root / "raw"
        
        date_compact = dt.replace("-", "")
        dbn_pattern = f"*{date_compact}*.dbn"
        
        raw_path = dbn_root / "source=databento" / "product_type=future" / f"symbol={symbol}" / "table=market_by_price_10"
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
        
        front_month = get_front_month_contract(all_contracts, dt)
        print(f"    Front month: {front_month}")
        
        out_ref = partition_ref(cfg, self.io.output, front_month, dt)
        
        if is_partition_complete(out_ref):
            print(f"    Skipping (already complete)")
            return
        
        symbol_mask = df_all['symbol'].to_numpy() == front_month
        df_contract = df_all[symbol_mask].copy()
        
        contract_path = repo_root / cfg.dataset(self.io.output).contract
        contract = load_avro_contract(contract_path)
        df_contract = enforce_contract(df_contract, contract)
        
        print(f"    Writing {front_month}: {len(df_contract):,} records")
        
        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=front_month,
            dt=dt,
            df=df_contract,
            contract_path=contract_path,
            inputs=[],
            stage=self.name,
        )
        
        print(f"  ✓ Bronze complete: {front_month}")
    
    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        """Not used for Bronze stage (overrides run() instead)."""
        raise NotImplementedError("Bronze stage uses custom run() method")



