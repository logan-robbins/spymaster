from __future__ import annotations

from pathlib import Path
from typing import List

import databento as db
import numpy as np
import pandas as pd
from databento_dbn import MBP10Msg

from src.data_eng.config import AppConfig
from src.data_eng.io import write_partition


def extract_flags_vectorized(flags: np.ndarray) -> pd.DataFrame:
    """Vectorized flag extraction using bitwise operations."""
    return pd.DataFrame({
        'is_last': (flags & 0x80).astype(bool),
        'is_snapshot': (flags & 0x20).astype(bool),
        'is_bad_ts': (flags & 0x08).astype(bool),
        'is_bad_book': (flags & 0x04).astype(bool),
    })


def process_mbp10_to_bronze(
    cfg: AppConfig,
    repo_root: Path,
    symbol: str,
    dt: str,
    dbn_root: Path | None = None,
    chunk_size: int = 5_000_000,
) -> None:
    """
    Process MBP-10 DBN files into Bronze layer with minimal transformations.
    
    Transformations:
    - Drop ts_in_delta
    - Extract flag bits into boolean columns
    - Nothing else (research-quality minimal processing)
    
    Args:
        cfg: AppConfig
        repo_root: Repository root path
        symbol: Symbol prefix (e.g., 'ES')
        dt: Date in YYYY-MM-DD format
        dbn_root: Root directory for DBN files (defaults to repo_root/data/raw)
        chunk_size: Number of records to accumulate before writing
    """
    
    if dbn_root is None:
        dbn_root = cfg.lake_root / "raw"
    
    date_compact = dt.replace("-", "")
    dbn_pattern = f"*{date_compact}*.dbn"
    
    raw_path = dbn_root / "source=databento" / f"product_type=future" / f"symbol={symbol}" / "table=market_by_price_10"
    dbn_files = list(raw_path.glob(dbn_pattern))
    
    if not dbn_files:
        raise FileNotFoundError(
            f"No DBN files found for date {dt} in {raw_path}/"
        )
    
    print(f"  Processing {len(dbn_files)} DBN file(s) for {dt}")
    
    all_dfs: List[pd.DataFrame] = []
    total_records = 0
    
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
        total_records += len(df)
        
        print(f"      Loaded {len(df):,} MBP-10 records (spreads filtered)")
    
    if not all_dfs:
        raise ValueError(f"No MBP-10 records found for {dt}")
    
    print(f"    Concatenating {total_records:,} total records...")
    df_all = pd.concat(all_dfs, ignore_index=True, copy=False)
    
    df_all.sort_values('ts_event', inplace=True)
    
    contracts = df_all['symbol'].unique()
    print(f"    Found {len(contracts)} contracts: {sorted(contracts)}")
    
    contract_path = repo_root / cfg.dataset("bronze.future.market_by_price_10").contract
    
    for contract_symbol in contracts:
        symbol_mask = df_all['symbol'].to_numpy() == contract_symbol
        df_contract = df_all[symbol_mask].copy()
        
        print(f"    Writing {contract_symbol} to Bronze: {len(df_contract):,} records...")
        
        write_partition(
            cfg=cfg,
            dataset_key="bronze.future.market_by_price_10",
            symbol=contract_symbol,
            dt=dt,
            df=df_contract,
            contract_path=contract_path,
            inputs=[],
            stage="bronze_process_dbn",
        )
    
    print(f"  ✓ Bronze MBP-10 complete for {dt}: {len(df_all):,} records across {len(contracts)} contracts")



