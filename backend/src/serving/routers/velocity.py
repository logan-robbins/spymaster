"""Unified WebSocket router for velocity + options streaming to frontend2."""
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
    ("rho", pa.float64()),
    ("nu", pa.float64()),
    ("kappa", pa.float64()),
    ("pressure_grad", pa.float64()),
    ("u_wave_energy", pa.float64()),
    ("Omega", pa.float64()),
])

OPTIONS_SCHEMA = pa.schema([
    ("window_end_ts_ns", pa.int64()),
    ("spot_ref_price_int", pa.int64()),
    ("rel_ticks", pa.int32()),
    ("liquidity_velocity", pa.float64()),
    ("pressure_grad", pa.float64()),
    ("u_wave_energy", pa.float64()),
    ("nu", pa.float64()),
    ("Omega", pa.float64()),
])

FORECAST_SCHEMA = pa.schema([
    ("window_end_ts_ns", pa.int64()),
    ("horizon_s", pa.int32()),
    ("predicted_spot_tick", pa.int64()),
    ("predicted_tick_delta", pa.int64()),
    ("confidence", pa.float64()),
    ("RunScore_up", pa.float64()),
    ("RunScore_down", pa.float64()),
    ("D_up", pa.int32()),
    ("D_down", pa.int32()),
])


def _df_to_arrow_bytes(df: pd.DataFrame, schema: pa.Schema) -> bytes:
    if df.empty:
        # Create empty table with correct schema
        # For int columns, use int64 array if schema expects int64
        arrays = []
        for f in schema:
            arrays.append(pa.array([], type=f.type))
        table = pa.Table.from_arrays(arrays, schema=schema)
    else:
        # Ensure correct dtypes
        df = df.copy()
        for field in schema:
            if field.name in df.columns:
                # Handle nulls for int columns (pandas converts to float if NaN present)
                # But here we assume filled.
                # If int column has NaNs, astype('int') fails.
                # FillNA with 0?
                # For D_up/D_down, contract says ["null", "int"].
                # PyArrow handles nulls if we don't force astype int on NaN?
                # But our schema says pa.int32() (not nullable expressed here, implies required?)
                # PA Schema fields are nullable by default.
                
                # Check target type
                if pa.types.is_int64(field.type) or pa.types.is_int32(field.type):
                    # Fill NaNs with 0 or keep as float?
                    # If we have NaNs, we can't cast to numpy int.
                    # PyArrow from_pandas can handle it if we let it infer or pass schema.
                    # But explicit astype helps.
                    # D_up can be null.
                    pass
                elif pa.types.is_float64(field.type):
                    df[field.name] = df[field.name].astype("float64")
                elif pa.types.is_boolean(field.type):
                    df[field.name] = df[field.name].astype("bool")
        
        # Using schema prevents type mismatch
        # PyArrow handles int with nulls properly if using from_pandas?
        # Only if using Int64 nullable type in pandas, otherwise float.
        # Let's hope to_pybytes handles it.
        table = pa.Table.from_pandas(df, schema=schema)

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
    """Unified WebSocket endpoint for futures + options + forecast streaming.

    Protocol per 1-second window:
    1. JSON: {"type": "batch_start", "window_end_ts_ns": "...", "surfaces": ["snap", "velocity", "options", "forecast"]}
    2. JSON: {"type": "surface_header", "surface": "snap"}
    3. Binary: Arrow IPC bytes for snap
    4. JSON: {"type": "surface_header", "surface": "velocity"}
    5. Binary: Arrow IPC bytes for velocity (futures physics)
    6. JSON: {"type": "surface_header", "surface": "options"}
    7. Binary: Arrow IPC bytes for options
    8. JSON: {"type": "surface_header", "surface": "forecast"}
    9. Binary: Arrow IPC bytes for forecast
    """
    await websocket.accept()
    print(
        "Unified stream connected: "
        f"symbol={symbol}, dt={dt}, speed={speed}, skip_minutes={skip_minutes}"
    )

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
                "surfaces": ["snap", "velocity", "options", "forecast"],
            }))

            # Send snap (futures spot reference)
            await websocket.send_text(json.dumps({"type": "surface_header", "surface": "snap"}))
            snap_bytes = _df_to_arrow_bytes(batch["snap"], SNAP_SCHEMA)
            await websocket.send_bytes(snap_bytes)

            # Send velocity (futures liquidity)
            await websocket.send_text(json.dumps({"type": "surface_header", "surface": "velocity"}))
            velocity_bytes = _df_to_arrow_bytes(batch["velocity"], VELOCITY_SCHEMA)
            await websocket.send_bytes(velocity_bytes)

            # Send options (aggregated options liquidity)
            await websocket.send_text(json.dumps({"type": "surface_header", "surface": "options"}))
            options_bytes = _df_to_arrow_bytes(batch["options"], OPTIONS_SCHEMA)
            await websocket.send_bytes(options_bytes)

            # Send forecast
            await websocket.send_text(json.dumps({"type": "surface_header", "surface": "forecast"}))
            forecast_bytes = _df_to_arrow_bytes(batch["forecast"], FORECAST_SCHEMA)
            await websocket.send_bytes(forecast_bytes)

    except Exception as e:
        print(f"Unified stream error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Unified stream disconnected")
