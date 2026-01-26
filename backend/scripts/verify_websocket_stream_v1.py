import asyncio
import websockets
import pyarrow as pa
import struct
import sys
import time
import io

async def verify_websocket():
    uri = "ws://localhost:8000/v1/hud/stream?symbol=ESH6&dt=2026-01-06"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Read first few messages
            messages_received = 0
            start_time = time.time()
            
            # State to track partial batches
            current_window_ts = None
            surfaces_seen = set()
            
            while messages_received < 20 and (time.time() - start_time < 15):
                message = await websocket.recv()
                
                # Check for control frame (JSON) vs Binary
                if isinstance(message, str) or (isinstance(message, bytes) and message.startswith(b'{')):
                    # Likely a JSON control frame (though the schema says format is Arrow IPC, 
                    # usually arrow is binary, but control frames might be JSON if text)
                    # frontend_data.json says: "control_frames": "type": "batch_start"...
                    # Let's inspect raw message
                    try:
                        import json
                        data = json.loads(message)
                        msg_type = data.get("type")
                        print(f"Control Frame: {msg_type}")
                        
                        if msg_type == "batch_start":
                            current_window_ts = data.get("window_end_ts_ns")
                            print(f"  New Window: {current_window_ts}")
                            surfaces_seen.clear()
                        elif msg_type == "surface_header":
                            print(f"  Surface Header: {data.get('surface')}")
                        
                    except Exception as e:
                        print(f"Unknown text message: {message[:100]}... Error: {e}")
                        
                else:
                    # Binary message - Arrow IPC
                    print(f"Binary Message: {len(message)} bytes")
                    try:
                        reader = pa.ipc.open_stream(message)
                        table = reader.read_all()
                        print(f"  Arrow Table: {table.num_rows} rows, {table.num_columns} columns")
                        print(f"  Schema: {table.schema.names}")
                        messages_received += 1
                    except Exception as e:
                        print(f"  Failed to parse Arrow: {e}")
                
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(verify_websocket())
    except KeyboardInterrupt:
        pass
