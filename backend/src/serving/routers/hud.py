from __future__ import annotations

import asyncio
import io
from typing import Dict

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response

from ..hud_streaming import HudStreamService

router = APIRouter(prefix="/v1/hud", tags=["hud"])

WINDOW_NS = 1_000_000_000
HUD_HISTORY_WINDOWS = 1800

SERVICE = HudStreamService()


def _df_to_arrow_ipc(df: pd.DataFrame) -> bytes:
    if df.empty:
        table = pa.Table.from_pandas(pd.DataFrame())
    else:
        table = pa.Table.from_pandas(df)
    sink = io.BytesIO()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _filter_range(df: pd.DataFrame, start_ts_ns: int | None, end_ts_ns: int | None) -> pd.DataFrame:
    if df.empty:
        return df
    if start_ts_ns is None and end_ts_ns is None:
        return df
    mask = pd.Series(True, index=df.index)
    if start_ts_ns is not None:
        mask &= df["window_end_ts_ns"] >= start_ts_ns
    if end_ts_ns is not None:
        mask &= df["window_end_ts_ns"] <= end_ts_ns
    return df.loc[mask]


@router.get("/bootstrap")
async def bootstrap(
    symbol: str = Query(..., description="Symbol (e.g., ESH6)"),
    dt: str = Query(..., description="Session date (YYYY-MM-DD)"),
    end_ts_ns: int | None = Query(None, description="End timestamp (ns). Defaults to latest available."),
):
    frames = SERVICE.bootstrap_frames(symbol, dt, end_ts_ns=end_ts_ns)
    df_snap = frames["snap"]
    if df_snap.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol} on {dt}")

    end_ts = int(df_snap["window_end_ts_ns"].max())
    start_ts = int(df_snap["window_end_ts_ns"].min())

    return Response(
        content=_df_to_arrow_ipc(df_snap),
        media_type="application/vnd.apache.arrow.stream",
        headers={
            "X-HUD-Symbol": symbol,
            "X-HUD-Dt": dt,
            "X-HUD-Start-Ts-Ns": str(start_ts),
            "X-HUD-End-Ts-Ns": str(end_ts),
            "X-HUD-Windows": str(len(df_snap)),
        },
    )


@router.get("/surfaces")
async def get_surfaces(
    symbol: str = Query(..., description="Symbol (e.g., ESH6)"),
    dt: str = Query(..., description="Session date (YYYY-MM-DD)"),
    surface: str = Query(..., description="Surface type: book_snapshot, wall, vacuum, radar, physics_bands, gex"),
    start_ts_ns: int | None = Query(None, description="Start timestamp (ns)"),
    end_ts_ns: int | None = Query(None, description="End timestamp (ns)"),
):
    cache = SERVICE.load_cache(symbol, dt)
    surface_map: Dict[str, pd.DataFrame] = {
        "book_snapshot": cache.snap,
        "wall": cache.wall,
        "vacuum": cache.vacuum,
        "radar": cache.radar,
        "physics_bands": cache.physics,
        "gex": cache.gex,
    }
    df = surface_map.get(surface)
    if df is None:
        raise HTTPException(status_code=400, detail=f"Unknown surface: {surface}")
    df = _filter_range(df, start_ts_ns, end_ts_ns)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No {surface} data found for {symbol} on {dt}")
    return Response(
        content=_df_to_arrow_ipc(df),
        media_type="application/vnd.apache.arrow.stream",
        headers={
            "X-HUD-Symbol": symbol,
            "X-HUD-Surface": surface,
            "X-HUD-Rows": str(len(df)),
        },
    )


@router.websocket("/stream")
async def stream(
    websocket: WebSocket,
    symbol: str,
    dt: str,
    start_ts_ns: int | None = None,
    speed: float = 1.0,
):
    await websocket.accept()
    try:
        iterator = SERVICE.iter_batches(symbol, dt, start_ts_ns=start_ts_ns)
        last_window_ts = None
        
        for window_id, batch in iterator:
            if last_window_ts is not None:
                delta_ns = window_id - last_window_ts
                delta_sec = delta_ns / 1_000_000_000.0
                to_sleep = delta_sec / speed
                if to_sleep > 0:
                    await asyncio.sleep(to_sleep)
            
            last_window_ts = window_id

            # Send batch metadata first
            await websocket.send_json(
                {
                    "type": "batch_start",
                    "window_end_ts_ns": str(window_id),
                    "surfaces": list(batch.keys()),
                }
            )

            # Send each surface as a separate binary message
            for surface, df in batch.items():
                # We send a small header frame or just rely on order?
                # Better: Send metadata FRAME, then BINARY frame.
                # Client needs to know which binary corresponds to which surface.
                # Let's send a JSON header for EACH surface immediately followed by its binary.
                await websocket.send_json(
                    {
                        "type": "surface_header",
                        "surface": surface,
                        "window_end_ts_ns": str(window_id),
                    }
                )
                await websocket.send_bytes(_df_to_arrow_ipc(df))
            
            # Send batch end marker? Not strictly necessary if "batch_start" implies new group.
    except WebSocketDisconnect:
        return
