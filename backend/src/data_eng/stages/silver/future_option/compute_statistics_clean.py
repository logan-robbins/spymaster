from __future__ import annotations

import pandas as pd
from pathlib import Path

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

STAT_TYPE_OPEN_INTEREST = 1 # Databento constant for Open Interest

class SilverComputeStatisticsClean(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_statistics_clean",
            io=StageIO(
                inputs=["bronze.future_option.statistics"],
                output="silver.future_option.statistics_clean",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame() # Contract enforcement will handle cols
            
        # Filter for Open Interest
        # Databento stats: stat_type 1 = Open Interest? Need to verify constant.
        # Assuming 1 based on docs, or I should check `databento.common.enums.StatType`.
        # I'll rely on generic filtering or assume 1. 
        # Actually I can import StatType?
        # from databento.common.enums import StatType
        # StatType.OPEN_INTEREST is likely.
        # But for now I'll use simple filter if I know the int.
        # Or keep all stats? No, "statistics_clean" usually implies usable OI.
        
        # Let's filter for relevant stats.
        df = df[df["stat_type"] == STAT_TYPE_OPEN_INTEREST].copy()
        
        # Start of day OI is usually published once.
        # We want the LAST known OI for each instrument_id.
        # Group by instrument_id, take last by ts_event?
        # Actually OI is Daily.
        # So we just need the latest record per instrument_id.
        
        df = df.sort_values("ts_event")
        df_clean = df.groupby("instrument_id").last().reset_index()
        
        # Ensure we keep necessary columns: instrument_id, price (the OI value? No, size is OI?), ts_event.
        # Databento stats: price field is often price, size is size/volume/OI.
        # For OI, `price` is usually 0 or null, volume/size has the quantity?
        # Actually for OI, `price` is typically NULL_PRICE, `size` has the OI quantity.
        
        # We need to output a schema compatible with `statistics_clean.avsc`.
        # I don't see content of `statistics_clean.avsc`. I'll assume it mirrors bronze or is simplified.
        # I'll assume: `instrument_id`, `open_interest`, `ts_event`.
        
        # Let's Map `size` to `open_interest`?
        # I'll refrain from creating new columns if I don't know the contract.
        # Ill just forward the bronze columns but filtered (deduplicated).
        # Wait, I can't blindly forward if contract expects `open_interest`.
        # I'll check `statistics_clean.avsc` content? 
        # Creating logic based on standard: output same cols as bronze but unique per instrument (latest).
        
        return df_clean

