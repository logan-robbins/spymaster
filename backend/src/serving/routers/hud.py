"""
HUD Router - Bootstrap and Streaming endpoints for the SpyMaster HUD.

Implements:
- GET /v1/hud/bootstrap - Returns 30-min window of HUD data as Arrow IPC
- WS  /v1/hud/stream   - Streams 1s updates as Arrow IPC
"""
from pathlib import Path
from typing import Optional
import io

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response

from ..config import settings

router = APIRouter(prefix="/v1/hud", tags=["hud"])

# Constants from IMPLEMENT.md
HUD_MAX_TICKS = 600
WINDOW_NS = 1_000_000_000
HUD_HISTORY_WINDOWS = 1800  # 30 minutes at 1s cadence

def _get_lake_path(dataset_key: str, symbol: str) -> Path:
    """Convert dataset key to lake path."""
    paths = {
        "silver.future_mbo.book_snapshot_1s": f"silver/product_type=future_mbo/symbol={symbol}/table=book_snapshot_1s",
        "silver.future_mbo.wall_surface_1s": f"silver/product_type=future_mbo/symbol={symbol}/table=wall_surface_1s",
        "silver.future_mbo.vacuum_surface_1s": f"silver/product_type=future_mbo/symbol={symbol}/table=vacuum_surface_1s",
        "silver.future_mbo.radar_vacuum_1s": f"silver/product_type=future_mbo/symbol={symbol}/table=radar_vacuum_1s",
        "silver.future_mbo.physics_bands_1s": f"silver/product_type=future_mbo/symbol={symbol}/table=physics_bands_1s",
        "silver.future_option_mbo.gex_surface_1s": f"silver/product_type=future_option_mbo/symbol={symbol}/table=gex_surface_1s",
    }
    return settings.lake_root / paths.get(dataset_key, "")


def _query_parquet(dataset_key: str, symbol: str, dt: str, 
                   start_ts_ns: Optional[int] = None, 
                   end_ts_ns: Optional[int] = None) -> pd.DataFrame:
    """Query parquet data using DuckDB with predicate pushdown."""
    path = _get_lake_path(dataset_key, symbol) / f"dt={dt}"
    
    if not path.exists():
        return pd.DataFrame()
    
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()
    
    # Build query with predicates
    file_path = str(parquet_files[0])
    
    query = f"SELECT * FROM read_parquet('{file_path}')"
    
    conditions = []
    if start_ts_ns is not None:
        conditions.append(f"window_end_ts_ns >= {start_ts_ns}")
    if end_ts_ns is not None:
        conditions.append(f"window_end_ts_ns <= {end_ts_ns}")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY window_end_ts_ns"
    
    con = duckdb.connect(":memory:")
    return con.execute(query).df()


def _df_to_arrow_ipc(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Arrow IPC format (uncompressed for browser compatibility)."""
    if df.empty:
        # Return empty but valid Arrow IPC
        table = pa.Table.from_pandas(pd.DataFrame())
    else:
        table = pa.Table.from_pandas(df)
    
    sink = io.BytesIO()
    # No compression for browser compatibility (apache-arrow JS lacks zstd codec)
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    
    return sink.getvalue()


@router.get("/bootstrap")
async def bootstrap(
    symbol: str = Query(..., description="Symbol (e.g., ESH6)"),
    dt: str = Query(..., description="Session date (YYYY-MM-DD)"),
    end_ts_ns: Optional[int] = Query(None, description="End timestamp (ns). Defaults to latest available.")
):
    """
    Bootstrap endpoint: returns last 30 minutes of HUD data as Arrow IPC.
    
    Returns a single Arrow IPC batch containing:
    - book_snapshot_1s
    - wall_surface_1s (quantized to HUD texture)
    - vacuum_surface_1s (quantized to HUD texture)
    - gex_surface_1s (quantized to HUD texture)
    """
    # Query book snapshot for time range detection
    df_snap = _query_parquet("silver.future_mbo.book_snapshot_1s", symbol, dt)
    
    if df_snap.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol} on {dt}")
    
    # Determine time range
    if end_ts_ns is None:
        end_ts_ns = int(df_snap["window_end_ts_ns"].max())
    
    start_ts_ns = end_ts_ns - (HUD_HISTORY_WINDOWS * WINDOW_NS)
    
    # Query all surfaces
    df_snap = _query_parquet("silver.future_mbo.book_snapshot_1s", symbol, dt, start_ts_ns, end_ts_ns)
    df_wall = _query_parquet("silver.future_mbo.wall_surface_1s", symbol, dt, start_ts_ns, end_ts_ns)
    df_vacuum = _query_parquet("silver.future_mbo.vacuum_surface_1s", symbol, dt, start_ts_ns, end_ts_ns)
    df_gex = _query_parquet("silver.future_option_mbo.gex_surface_1s", symbol, dt, start_ts_ns, end_ts_ns)
    
    # Build response payload
    response_data = {
        "book_snapshot": _df_to_arrow_ipc(df_snap),
        "wall_surface": _df_to_arrow_ipc(df_wall),
        "vacuum_surface": _df_to_arrow_ipc(df_vacuum),
        "gex_surface": _df_to_arrow_ipc(df_gex),
    }
    
    # For now, return book_snapshot as primary payload
    # Frontend will need to handle multiple Arrow batches
    return Response(
        content=response_data["book_snapshot"],
        media_type="application/vnd.apache.arrow.stream",
        headers={
            "X-HUD-Symbol": symbol,
            "X-HUD-Dt": dt,
            "X-HUD-Start-Ts-Ns": str(start_ts_ns),
            "X-HUD-End-Ts-Ns": str(end_ts_ns),
            "X-HUD-Windows": str(len(df_snap)),
        }
    )


@router.get("/surfaces")
async def get_surfaces(
    symbol: str = Query(..., description="Symbol (e.g., ESH6)"),
    dt: str = Query(..., description="Session date (YYYY-MM-DD)"),
    surface: str = Query(..., description="Surface type: book_snapshot, wall, vacuum, gex"),
    start_ts_ns: Optional[int] = Query(None, description="Start timestamp (ns)"),
    end_ts_ns: Optional[int] = Query(None, description="End timestamp (ns)")
):
    """
    Get a specific surface as Arrow IPC.
    """
    surface_map = {
        "book_snapshot": "silver.future_mbo.book_snapshot_1s",
        "wall": "silver.future_mbo.wall_surface_1s",
        "vacuum": "silver.future_mbo.vacuum_surface_1s",
        "gex": "silver.future_option_mbo.gex_surface_1s",
        "radar": "silver.future_mbo.radar_vacuum_1s",
        "physics_bands": "silver.future_mbo.physics_bands_1s",
    }
    
    dataset_key = surface_map.get(surface)
    if not dataset_key:
        raise HTTPException(status_code=400, detail=f"Unknown surface: {surface}")
    
    df = _query_parquet(dataset_key, symbol, dt, start_ts_ns, end_ts_ns)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No {surface} data found for {symbol} on {dt}")
    
    return Response(
        content=_df_to_arrow_ipc(df),
        media_type="application/vnd.apache.arrow.stream",
        headers={
            "X-HUD-Symbol": symbol,
            "X-HUD-Surface": surface,
            "X-HUD-Rows": str(len(df)),
        }
    )


@router.websocket("/stream")
async def stream(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for streaming 1s HUD updates.
    
    Streams one Arrow IPC record batch every second containing:
    - window_end_ts_ns
    - spot_ref_price_int
    - Quantized texture columns
    """
    await websocket.accept()
    
    try:
        # For development: simulate streaming from historical data
        # In production: connect to real-time data source
        await websocket.send_json({
            "type": "connected",
            "symbol": symbol,
            "message": "Stream connected. Real-time data not yet implemented."
        })
        
        # Keep connection alive and wait for messages
        while True:
            data = await websocket.receive_text()
            # Handle control messages (ping, subscribe, etc.)
            if data == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        pass
