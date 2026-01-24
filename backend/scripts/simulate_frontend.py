import asyncio
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path.cwd()))
from src.serving.hud_streaming import HudStreamService

def main():
    service = HudStreamService()
    symbol = "ESH6"
    dt = "2026-01-06"
    
    print(f"Starting simulation for {symbol} on {dt}...")
    print("Initializing service (loading Silver cache)...")
    
    start_load = time.time()
    # Pre-load cache to separate loading time from streaming time
    service.load_cache(symbol, dt)
    print(f"Cache loaded in {time.time() - start_load:.2f}s")
    
    print("-" * 50)
    print("STREAMING START (1s Cadence Verification)")
    print("-" * 50)
    
    last_recv_time = time.time()
    frame_count = 0
    
    # Start simulating from 09:30:00 EST
    # We need to know the NS timestamp for that.
    # 2026-01-06 09:30:00 EST is... well, we can just let it start from beginning of file
    # since we filtered data to 09:30-10:30 anyway.
    
    for window_id, batch in service.simulate_stream(symbol, dt, speed=1.0):
        now = time.time()
        delta = now - last_recv_time
        last_recv_time = now
        frame_count += 1
        
        if frame_count == 1:
            print(f"DEBUG: Batch Keys: {list(batch.keys())}")
            if "vacuum" in batch:
                print(f"DEBUG: Vacuum Sample: {batch['vacuum'].head(1)}")
        
        vacuum_rows = len(batch.get("vacuum", []))
        radar_rows = len(batch.get("radar", []))
        
        print(f"[{frame_count:04d}] Window {window_id} @ {time.strftime('%H:%M:%S')} | Delta: {delta:.3f}s | Vacuum: {vacuum_rows} | Radar: {radar_rows}")
        
        if frame_count >= 10:
            print("Stopping after 10 frames.")
            break

if __name__ == "__main__":
    main()
