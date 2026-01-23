from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import asyncio

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.serving.hud_streaming import HudStreamService
from src.serving.routers.hud import _df_to_arrow_ipc

def verify_service():
    print("Initializing HudStreamService...")
    service = HudStreamService()
    
    symbol = "ESH6"
    dt = "2026-01-06"
    
    print(f"Loading cache for {symbol} {dt}...")
    cache = service.load_cache(symbol, dt)
    
    print(f"Cache loaded. Window count: {len(cache.window_ids)}")
    if not cache.window_ids:
        print("FAIL: No windows found in cache")
        sys.exit(1)
        
    # Check dataframes
    for key in ["snap", "wall", "vacuum", "radar", "physics", "gex"]:
        df = getattr(cache, key)
        print(f"Surface {key}: {len(df)} rows")
        if df.empty:
            print(f"FAIL: Surface {key} is empty")
            sys.exit(1)
            
    # Check ring buffer bootstrap
    print("Testing bootstrap_frames...")
    frames = service.bootstrap_frames(symbol, dt)
    for key, df in frames.items():
        print(f"Bootstrap {key}: {len(df)} rows")
        if df.empty:
            print(f"FAIL: Bootstrap {key} empty")
            sys.exit(1)
            
    # Test batch iteration (first 5 batches)
    print("Testing iter_batches...")
    count = 0
    for wid, batch in service.iter_batches(symbol, dt):
        count += 1
        if count > 5:
            break
        print(f"Batch {wid}: Keys {list(batch.keys())}")
        # Verify Arrow serialization
        for k, df in batch.items():
            arrow_bytes = _df_to_arrow_ipc(df)
            if len(arrow_bytes) == 0:
                print(f"FAIL: Arrow serialization empty for {k}")
                sys.exit(1)

    print("PASS: Service verification successful.")

if __name__ == "__main__":
    verify_service()
