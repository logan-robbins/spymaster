from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import List, Tuple

import databento as db
import numpy as np
import pandas as pd
from databento.common.enums import PriceType

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import is_partition_complete, partition_ref, write_partition
from ....utils import session_window_ns

RTYPE_MBO = 160
NULL_PRICE = np.iinfo("int64").max

OPTION_SYMBOL_PATTERN = re.compile(r"^(ES[HMUZ]\d{1,2})\s+([CP])(\d+)$")

MONTH_CODE_TO_MONTH = {"H": 3, "M": 6, "U": 9, "Z": 12}


def parse_option_symbol(symbol: str) -> Tuple[str, str, int] | None:
    match = OPTION_SYMBOL_PATTERN.match(symbol)
    if not match:
        return None
    underlying = match.group(1)
    right = match.group(2)
    strike = int(match.group(3))
    return underlying, right, strike


def compute_expiration_date(underlying: str) -> date:
    month_code = underlying[2]
    year_digit = underlying[3:]
    year = 2020 + int(year_digit)
    month = MONTH_CODE_TO_MONTH[month_code]

    first_day = date(year, month, 1)
    first_friday_offset = (4 - first_day.weekday()) % 7
    first_friday = first_day.day + first_friday_offset
    third_friday = first_friday + 14
    return date(year, month, third_friday)


class BronzeIngestFutureOptionMbo(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="bronze_ingest_future_option_mbo",
            io=StageIO(
                inputs=[],
                output="bronze.future_option_mbo.mbo",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        date_compact = dt.replace("-", "")
        raw_path = (
            cfg.lake_root
            / "raw"
            / "source=databento"
            / "product_type=future_option"
            / f"symbol={symbol}"
            / "table=market_by_order_dbn"
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

            df = df_raw.loc[df_raw["rtype"] == RTYPE_MBO].copy()
            if df.empty:
                continue

            df = df.loc[df["symbol"].notna()].copy()
            if df.empty:
                continue

            is_spread = df["symbol"].str.startswith("UD:", na=False).to_numpy()
            df = df.loc[~is_spread].copy()
            if df.empty:
                continue

            is_calendar = df["symbol"].str.contains("-", regex=False, na=False).to_numpy()
            df = df.loc[~is_calendar].copy()
            if df.empty:
                continue

            parsed = df["symbol"].apply(parse_option_symbol)
            valid_mask = parsed.notna()
            df = df.loc[valid_mask].copy()
            if df.empty:
                continue

            parsed_valid = parsed.loc[valid_mask]
            df["underlying"] = parsed_valid.apply(lambda x: x[0])
            df["right"] = parsed_valid.apply(lambda x: x[1])
            df["strike"] = parsed_valid.apply(lambda x: x[2])
            df["expiration"] = df["underlying"].apply(compute_expiration_date).astype(str)

            all_dfs.append(df)

        if not all_dfs:
            raise ValueError(f"No option MBO records found for {dt}")

        df_all = pd.concat(all_dfs, ignore_index=True, copy=False)
        df_all["ts_event"] = df_all["ts_event"].astype("int64")
        df_all["ts_recv"] = df_all["ts_recv"].astype("int64")

        session_start_ns, session_end_ns = session_window_ns(dt)
        df_all = df_all.loc[
            (df_all["ts_event"] >= session_start_ns) & (df_all["ts_event"] < session_end_ns)
        ].copy()
        if df_all.empty:
            raise ValueError(f"No option MBO records in session window for {dt}")

        null_price = df_all["price"].astype("int64") == NULL_PRICE
        if null_price.any():
            needs_price = df_all["action"].isin({"A", "M"}).to_numpy()
            if (null_price.to_numpy() & needs_price).any():
                raise ValueError("Missing price for add/modify actions in DBN MBO data")
            df_all.loc[null_price, "price"] = 0

        df_all = df_all.sort_values(["ts_event", "sequence"], ascending=[True, True])
        df_all["order_id"] = df_all["order_id"].astype("int64")
        df_all["size"] = df_all["size"].astype("int64")
        df_all["sequence"] = df_all["sequence"].astype("int64")
        df_all["price"] = df_all["price"].astype("int64")
        df_all["channel_id"] = df_all["channel_id"].astype("int64")
        df_all["rtype"] = df_all["rtype"].astype("int64")
        df_all["publisher_id"] = df_all["publisher_id"].astype("int64")
        df_all["flags"] = df_all["flags"].astype("int64")
        df_all["instrument_id"] = df_all["instrument_id"].astype("int64")
        df_all["ts_in_delta"] = df_all["ts_in_delta"].astype("int64")
        df_all["strike"] = df_all["strike"].astype("int64")

        contract_path = repo_root / cfg.dataset(self.io.output).contract
        contract = load_avro_contract(contract_path)

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
