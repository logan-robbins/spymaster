import asyncio
import websockets
import json
import pyarrow as pa
import time
import sys
import traceback

from .protocol import (
    SurfaceId, PacketHeader, 
    pack_snap, pack_wall_entry, pack_vacuum_entry, pack_physics_entry, pack_gex_entry
)
from .transform import (
    compute_wall_intensity, compute_wall_erosion, normalize_vacuum, normalize_physics,
    W_LOG_MAX
)
from .emitter import UdpEmitter

MAX_PAYLOAD_BYTES = 1200

class BridgeClient:
    def __init__(self, uri: str, udp_ip: str = "127.0.0.1", udp_port: int = 7777):
        self.uri = uri
        self.emitter = UdpEmitter(udp_ip, udp_port)
        self.running = False
        
        # Current window state
        self.current_window_ts = 0
        self.spot_ref_price_int = 0
        
    async def run(self):
        self.running = True
        while self.running:
            try:
                print(f"Connecting to {self.uri}...")
                async with websockets.connect(self.uri) as websocket:
                    print("Bridge Connected.")
                    self.current_surface_name = None
                    
                    async for message in websocket:
                        await self.handle_message(message)
                        
            except Exception as e:
                print(f"Bridge Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(1) # Reconnect delay

    async def handle_message(self, message):
        # Control Frame (JSON)
        if isinstance(message, str) or (isinstance(message, bytes) and message.startswith(b'{')):
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "batch_start":
                    self.current_window_ts = data.get("window_end_ts_ns", 0)
                    # print(f"Window: {self.current_window_ts}")
                    
                elif msg_type == "surface_header":
                    self.current_surface_name = data.get("surface")
                    
            except Exception as e:
                print(f"JSON Parse Error: {e}")
                
        # Binary Frame (Arrow)
        else:
            if not self.current_surface_name:
                return # Should not happen
            
            try:
                reader = pa.ipc.open_stream(message)
                table = reader.read_all()
                self.process_surface(self.current_surface_name, table)
            except Exception as e:
                print(f"Arrow Error ({self.current_surface_name}): {e}")
                traceback.print_exc()

    def process_surface(self, name: str, table: pa.Table):
        # Convert to python dicts/lists for easy iteration (perf is fine for V1 < 1000 rows)
        # Or iterate pyarrow columns directly.
        
        # We need 'rel_ticks' for almost all surfaces.
        # Check if empty
        if table.num_rows == 0:
            return

        # V1: Update SpotRef from SNAP
        if name == "snap":
            self.process_snap(table)
        elif name == "wall":
            self.process_wall(table)
        elif name == "vacuum":
            self.process_vacuum(table)
        elif name == "physics":
            self.process_physics(table)
        elif name == "gex":
            self.process_gex(table)
            
    def process_snap(self, table):
        # Expect 1 row
        c_spot = table.column("spot_ref_price_int")
        c_mid = table.column("mid_price")
        c_valid = table.column("book_valid")
        
        for i in range(table.num_rows):
            self.spot_ref_price_int = c_spot[i].as_py()
            mid = c_mid[i].as_py()
            valid = c_valid[i].as_py()
            
            payload = pack_snap(mid, valid)
            self.emit_packet(SurfaceId.SNAP, payload, 1)
            
    def process_wall(self, table):
        # Columns: rel_ticks, side, depth_qty_rest, d1_depth_qty (optional)
        # Need to handle missing optional columns safely
        
        cols = table.column_names
        rel_ticks = table.column("rel_ticks").to_pylist()
        sides = table.column("side").to_pylist()
        depths = table.column("depth_qty_rest").to_pylist()
        
        d1s = [0.0] * table.num_rows
        if "d1_depth_qty" in cols:
            d1s = table.column("d1_depth_qty").to_pylist()
            
        entries = []
        for i in range(table.num_rows):
            # Compute Intensity
            d_rest = depths[i]
            d1 = d1s[i]
            side_str = sides[i]
            side_int = 1 if side_str == 'A' else 0
            
            intensity = compute_wall_intensity(d_rest)
            erosion = compute_wall_erosion(d1, d_rest) # Using d_rest as start approx or passed
            
            entries.append(pack_wall_entry(rel_ticks[i], side_int, intensity, erosion))
            
        self.emit_batched(SurfaceId.WALL, entries)

    def process_vacuum(self, table):
        # Columns: rel_ticks, vacuum_score, d2_pull_add_log (optional)
        cols = table.column_names
        rel_ticks = table.column("rel_ticks").to_pylist()
        scores = table.column("vacuum_score").to_pylist()
        
        turbs = [0.0] * table.num_rows
        if "d2_pull_add_log" in cols:
            turbs = table.column("d2_pull_add_log").to_pylist()
            
        entries = []
        for i in range(table.num_rows):
            entries.append(pack_vacuum_entry(rel_ticks[i], scores[i], turbs[i]))
            
        self.emit_batched(SurfaceId.VACUUM, entries)

    def process_physics(self, table):
        # Columns: rel_ticks, physics_score_signed
        rel_ticks = table.column("rel_ticks").to_pylist()
        scores = table.column("physics_score_signed").to_pylist()
        
        entries = []
        for i in range(table.num_rows):
            entries.append(pack_physics_entry(rel_ticks[i], scores[i]))
            
        self.emit_batched(SurfaceId.PHYSICS, entries)

    def process_gex(self, table):
        # Columns: rel_ticks, gex_abs, gex_imbalance_ratio
        rel_ticks = table.column("rel_ticks").to_pylist()
        gex_abs = table.column("gex_abs").to_pylist()
        ratios = table.column("gex_imbalance_ratio").to_pylist()
        
        entries = []
        for i in range(table.num_rows):
            entries.append(pack_gex_entry(rel_ticks[i], gex_abs[i], ratios[i]))
            
        self.emit_batched(SurfaceId.GEX, entries)

    def emit_batched(self, surface_id: int, entries: list):
        if not entries:
            return
            
        # Chunk entries
        # Approx size check.
        # Just hardcode a safe chunk size.
        # Wall(11) -> 100 is 1100 bytes. Safe.
        CHUNK_SIZE = 100
        
        for i in range(0, len(entries), CHUNK_SIZE):
            chunk = entries[i:i+CHUNK_SIZE]
            payload = b''.join(chunk)
            self.emit_packet(surface_id, payload, len(chunk))

    def emit_packet(self, surface_id: int, payload: bytes, count: int):
        try:
            # Ensure proper types
            sid = int(surface_id)
            ts = int(self.current_window_ts)
            spot = int(self.spot_ref_price_int)
            cnt = int(count)
            flg = 0
            
            header = PacketHeader(
                surface_id=sid,
                window_end_ts_ns=ts,
                spot_ref_price_int=spot,
                count=cnt,
                flags=flg,
                prediction_horizon_ns=0
            )
            data = header.pack() + payload
            self.emitter.send(data)
        except Exception as e:
            print(f"Packet Error: {e} | TS={self.current_window_ts} Spot={self.spot_ref_price_int}")
            traceback.print_exc()

    def stop(self):
        self.running = False
        self.emitter.close()
