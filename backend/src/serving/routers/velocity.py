"""Minimal WebSocket router for velocity streaming to frontend2."""
from __future__ import annotations

import asyncio
import json

import pandas as pd
import pyarrow as pa
from fastapi import APIRouter, WebSocket

from ..velocity_streaming import VelocityStreamService

router = APIRouter()
service = VelocityStreamService()

SNAP_SCHEMA = pa.schema([
    ("window_end_ts_ns", pa.int64()),
    ("mid_price", pa.float64()),
    ("spot_ref_price_int", pa.int64()),
    ("book_valid", pa.bool_()),
])

VELOCITY_SCHEMA = pa.schema([
    ("window_end_ts_ns", pa.int64()),
    ("spot_ref_price_int", pa.int64()),
    ("rel_ticks", pa.int32()),
    ("side", pa.string()),
    ("liquidity_velocity", pa.float64()),
])


def _df_to_arrow_bytes(df: pd.DataFrame, schema: pa.Schema) -> bytes:
    if df.empty:
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema})
    else:
        # Ensure correct dtypes
        df = df.copy()
        for field in schema:
            if field.name in df.columns:
                if pa.types.is_int64(field.type):
                    df[field.name] = df[field.name].astype("int64")
                elif pa.types.is_int32(field.type):
                    df[field.name] = df[field.name].astype("int32")
                elif pa.types.is_float64(field.type):
                    df[field.name] = df[field.name].astype("float64")
                elif pa.types.is_boolean(field.type):
                    df[field.name] = df[field.name].astype("bool")
        table = pa.Table.from_pandas(df[schema.names], schema=schema)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


@router.websocket("/v1/velocity/stream")
async def velocity_stream(
    websocket: WebSocket,
    symbol: str = "ESH6",
    dt: str = "2026-01-06",
    speed: float = 1.0,
    skip_minutes: int = 5,
):
    """WebSocket endpoint for velocity streaming.

    Protocol:
    1. JSON: {"type": "batch_start", "window_end_ts_ns": "123456789", "surfaces": ["snap", "velocity"]}
    2. JSON: {"type": "surface_header", "surface": "snap"}
    3. Binary: Arrow IPC bytes for snap
    4. JSON: {"type": "surface_header", "surface": "velocity"}
    5. Binary: Arrow IPC bytes for velocity
    ... repeat for each window
    """
    await websocket.accept()
    print(f"Velocity stream connected: symbol={symbol}, dt={dt}, speed={speed}, skip_minutes={skip_minutes}")

    try:
        last_window_ts = None
        skip_count = skip_minutes * 60  # Skip N minutes worth of 1s windows
        skipped = 0
        for window_id, batch in service.iter_batches(symbol, dt):
            # Skip initial windows
            if skipped < skip_count:
                skipped += 1
                continue
            # Simulate real-time delay
            if last_window_ts is not None:
                delta_ns = window_id - last_window_ts
                delta_sec = delta_ns / 1_000_000_000.0
                to_sleep = delta_sec / speed
                if to_sleep > 0:
                    await asyncio.sleep(to_sleep)
            last_window_ts = window_id

            # Send batch_start
            await websocket.send_text(json.dumps({
                "type": "batch_start",
                "window_end_ts_ns": str(window_id),
                "surfaces": ["snap", "velocity"],
            }))

            # Send snap
            await websocket.send_text(json.dumps({"type": "surface_header", "surface": "snap"}))
            snap_bytes = _df_to_arrow_bytes(batch["snap"], SNAP_SCHEMA)
            await websocket.send_bytes(snap_bytes)

            # Send velocity
            await websocket.send_text(json.dumps({"type": "surface_header", "surface": "velocity"}))
            velocity_bytes = _df_to_arrow_bytes(batch["velocity"], VELOCITY_SCHEMA)
            await websocket.send_bytes(velocity_bytes)

    except Exception as e:
        print(f"Velocity stream error: {e}")
    finally:
        print("Velocity stream disconnected")
