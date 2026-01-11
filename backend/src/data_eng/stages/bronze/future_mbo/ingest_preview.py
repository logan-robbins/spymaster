from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import is_partition_complete, partition_ref, write_partition
from ....utils import session_window_ns

PRICE_SCALE_INT = Decimal("1000000000")


class BronzeIngestMboPreview(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="bronze_ingest_mbo_preview",
            io=StageIO(
                inputs=[],
                output="bronze.future_mbo.mbo",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        source_path = repo_root.parent / "mbo_preview.json"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing mbo_preview.json at {source_path}")

        raw = json.loads(source_path.read_text())
        rows = []
        for entry in raw:
            symbol_val = entry.get("symbol")
            if symbol_val != symbol:
                continue

            hd = entry.get("hd", {})
            ts_event = hd.get("ts_event")
            ts_recv = entry.get("ts_recv")
            if ts_event is None or ts_recv is None:
                raise ValueError("Missing ts_event or ts_recv in mbo_preview.json")

            ts_event_ns = pd.to_datetime(ts_event, utc=True).value
            ts_recv_ns = pd.to_datetime(ts_recv, utc=True).value

            action = entry.get("action")
            side = entry.get("side")
            if action is None or side is None:
                raise ValueError("Missing action or side in mbo_preview.json")
            price_raw = entry.get("price")
            price_int = _parse_price(price_raw)
            if price_int is None and action in {"A", "M"}:
                raise ValueError(f"Missing price for action {action}")
            if price_int is None:
                price_int = 0

            row = {
                "ts_recv": int(ts_recv_ns),
                "size": int(entry.get("size", 0)),
                "ts_event": int(ts_event_ns),
                "channel_id": int(entry.get("channel_id", 0)),
                "rtype": int(hd.get("rtype", 0)),
                "order_id": _parse_int(entry.get("order_id")),
                "publisher_id": int(hd.get("publisher_id", 0)),
                "flags": int(entry.get("flags", 0)),
                "instrument_id": int(hd.get("instrument_id", 0)),
                "ts_in_delta": int(entry.get("ts_in_delta", 0)),
                "action": action,
                "sequence": int(entry.get("sequence", 0)),
                "side": side,
                "symbol": symbol_val,
                "price": int(price_int),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) == 0:
            out_contract_path = repo_root / cfg.dataset(self.io.output).contract
            out_contract = load_avro_contract(out_contract_path)
            df_out = pd.DataFrame(columns=out_contract.fields)
            write_partition(
                cfg=cfg,
                dataset_key=self.io.output,
                symbol=symbol,
                dt=dt,
                df=df_out,
                contract_path=out_contract_path,
                inputs=[],
                stage=self.name,
            )
            return

        session_start_ns, session_end_ns = session_window_ns(dt)
        df = df.loc[(df["ts_event"] >= session_start_ns) & (df["ts_event"] < session_end_ns)].copy()
        if len(df) == 0:
            out_contract_path = repo_root / cfg.dataset(self.io.output).contract
            out_contract = load_avro_contract(out_contract_path)
            df_out = pd.DataFrame(columns=out_contract.fields)
            write_partition(
                cfg=cfg,
                dataset_key=self.io.output,
                symbol=symbol,
                dt=dt,
                df=df_out,
                contract_path=out_contract_path,
                inputs=[],
                stage=self.name,
            )
            return

        df = df.sort_values(["ts_event", "sequence"], ascending=[True, True])
        df["order_id"] = df["order_id"].astype("int64")
        df["size"] = df["size"].astype("int64")
        df["ts_event"] = df["ts_event"].astype("int64")
        df["ts_recv"] = df["ts_recv"].astype("int64")
        df["sequence"] = df["sequence"].astype("int64")
        df["price"] = df["price"].astype("int64")
        df["channel_id"] = df["channel_id"].astype("int64")
        df["rtype"] = df["rtype"].astype("int64")
        df["publisher_id"] = df["publisher_id"].astype("int64")
        df["flags"] = df["flags"].astype("int64")
        df["instrument_id"] = df["instrument_id"].astype("int64")
        df["ts_in_delta"] = df["ts_in_delta"].astype("int64")

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df = enforce_contract(df, out_contract)

        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df,
            contract_path=out_contract_path,
            inputs=[],
            stage=self.name,
        )


def _parse_price(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        if abs(value) < 1_000_000_000:
            return int(Decimal(value) * PRICE_SCALE_INT)
        return int(value)
    return int(Decimal(str(value)) * PRICE_SCALE_INT)


def _parse_int(value: object) -> int:
    if value is None:
        raise ValueError("Missing order_id")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid order_id: {value}") from exc
