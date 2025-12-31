"""
Pentaview Stream Publisher - Publishes historical stream data to NATS.

This service:
- Loads Pentaview stream bars from gold/streams/pentaview/
- Publishes to NATS subject `pentaview.streams` 
- Formats data for demo frontend (candles + streams + projections)
- Syncs with replay engine timing
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from src.common.bus import NATSBus
from src.common.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PentaviewStreamer:
    """Publishes Pentaview stream data to NATS for Gateway relay."""
    
    def __init__(
        self,
        bus: NATSBus,
        date: str,
        data_root: Path = Path("data"),
        publish_interval: float = 30.0
    ):
        """
        Initialize streamer.
        
        Args:
            bus: NATS bus instance
            date: Date to stream (YYYY-MM-DD format)
            data_root: Root data directory
            publish_interval: Publish cadence in seconds (default 30s)
        """
        self.bus = bus
        self.date = date
        self.data_root = data_root
        self.publish_interval = publish_interval
        self.running = False
        
        # Load stream data
        self.stream_df = self._load_stream_data()
        logger.info(f"📊 Loaded {len(self.stream_df):,} stream bars for {date}")
    
    def _load_stream_data(self) -> pd.DataFrame:
        """Load Pentaview stream bars from gold layer."""
        # Path: data/gold/streams/pentaview/version=3.1.0/date=YYYY-MM-DD_00:00:00/stream_bars.parquet
        date_partition = f"date={self.date}_00:00:00"
        stream_path = (
            self.data_root / "gold" / "streams" / "pentaview" / 
            "version=3.1.0" / date_partition / "stream_bars.parquet"
        )
        
        if not stream_path.exists():
            raise FileNotFoundError(f"Stream data not found: {stream_path}")
        
        df = pd.read_parquet(stream_path)
        
        # Ensure timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def _build_payload(self, row: pd.Series) -> Dict[str, Any]:
        """
        Build WebSocket payload in demo-expected format.
        
        Expected format:
        {
            "candles": [{"time": 1702900800, "open": 4500.0, ...}],
            "streams": {
                "sigma_p": [{"time": 1702900800, "value": 0.45}],
                ...
            },
            "projections": {
                "q10": [{"time": 1702900830, "value": 0.40}, ...],
                ...
            }
        }
        """
        # Convert timestamp to unix epoch (seconds)
        ts = int(row['timestamp'].timestamp())
        
        # Build candle (if OHLCV columns exist)
        candles = []
        if 'open' in row and 'high' in row:
            candles = [{
                "time": ts,
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
                "volume": float(row.get('volume', 0))
            }]
        
        # Build streams
        streams = {}
        stream_names = ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']
        for name in stream_names:
            if name in row:
                streams[name] = [{
                    "time": ts,
                    "value": float(row[name]) if pd.notna(row[name]) else 0.0
                }]
        
        # Build projections (if columns exist)
        projections = {}
        quantile_names = ['q10', 'q50', 'q90']
        
        for q_name in quantile_names:
            # Check for coefficient columns: {q}_a1, {q}_a2, {q}_a3
            a1_col = f"{q_name}_a1"
            a2_col = f"{q_name}_a2"
            a3_col = f"{q_name}_a3"
            
            if a1_col in row and pd.notna(row[a1_col]):
                # Generate 10 forecast points (5 minutes @ 30s cadence)
                a1 = float(row[a1_col])
                a2 = float(row[a2_col]) if a2_col in row and pd.notna(row[a2_col]) else 0.0
                a3 = float(row[a3_col]) if a3_col in row and pd.notna(row[a3_col]) else 0.0
                current_value = float(row.get('sigma_p', 0))  # Use sigma_p as baseline
                
                forecast_points = []
                for i in range(1, 11):  # 10 bars ahead
                    t = i * 30  # 30 seconds per bar
                    # Polynomial: v(t) = v0 + a1*t + a2*t^2 + a3*t^3
                    value = current_value + a1 * t + a2 * (t ** 2) + a3 * (t ** 3)
                    forecast_points.append({
                        "time": ts + t,
                        "value": float(np.clip(value, -1.0, 1.0))  # Clip to [-1, 1]
                    })
                
                projections[q_name] = forecast_points
        
        return {
            "candles": candles,
            "streams": streams,
            "projections": projections,
            "metadata": {
                "date": self.date,
                "timestamp": ts,
                "version": "v30s_20251115_20251215"
            }
        }
    
    async def start(self):
        """Start publishing stream data."""
        self.running = True
        logger.info(f"🚀 Starting Pentaview streamer (cadence={self.publish_interval}s)")
        
        # Iterate through stream bars and publish
        for idx, row in self.stream_df.iterrows():
            if not self.running:
                break
            
            try:
                payload = self._build_payload(row)
                
                # Publish to NATS
                await self.bus.publish(
                    subject="pentaview.streams",
                    payload=payload
                )
                
                # Log summary
                n_streams = len(payload['streams'])
                n_projections = len(payload['projections'])
                ts_str = row['timestamp'].strftime('%H:%M:%S')
                logger.info(f"📡 Published Pentaview data @ {ts_str} (streams={n_streams}, projections={n_projections})")
                
                # Wait for next interval
                await asyncio.sleep(self.publish_interval)
                
            except Exception as e:
                logger.error(f"❌ Error publishing stream bar: {e}")
                import traceback
                traceback.print_exc()
    
    async def stop(self):
        """Stop streaming."""
        self.running = False
        logger.info("🛑 Pentaview streamer stopped")


async def main():
    """Main entry point for standalone mode."""
    import os
    import signal
    
    # Configuration
    date = os.environ.get("REPLAY_DATE", "2025-12-18")
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    data_root = Path(os.environ.get("DATA_ROOT", "data"))
    
    logger.info("=" * 60)
    logger.info("📊 PENTAVIEW STREAM PUBLISHER")
    logger.info("=" * 60)
    logger.info(f"Date: {date}")
    logger.info(f"NATS: {nats_url}")
    logger.info(f"Data root: {data_root}")
    
    # Initialize NATS
    bus = NATSBus(servers=[nats_url])
    await bus.connect()
    
    # Initialize streamer
    streamer = PentaviewStreamer(
        bus=bus,
        date=date,
        data_root=data_root,
        publish_interval=5.0  # Faster for demo (5s instead of 30s)
    )
    
    # Handle shutdown
    stop_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutdown signal received")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start streaming
    streaming_task = asyncio.create_task(streamer.start())
    
    # Wait for shutdown
    await stop_event.wait()
    
    # Stop streamer
    await streamer.stop()
    streaming_task.cancel()
    
    try:
        await streaming_task
    except asyncio.CancelledError:
        pass
    
    # Close NATS
    await bus.close()
    logger.info("👋 Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

