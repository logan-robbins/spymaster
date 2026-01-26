import struct
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Union

class SurfaceId(IntEnum):
    SNAP = 1
    WALL = 2
    VACUUM = 3
    PHYSICS = 4
    GEX = 5

@dataclass
class PacketHeader:
    magic: bytes = b'MWT1'
    version: int = 1
    surface_id: int = 0
    window_end_ts_ns: int = 0
    spot_ref_price_int: int = 0
    count: int = 0
    flags: int = 0
    prediction_horizon_ns: int = 0 # Reserved for Section 13

    def pack(self) -> bytes:
        # <4s H H q q I I q
        # 32 + 8 = 40 bytes
        return struct.pack('<4sHHqqIIq', 
                           self.magic, 
                           self.version, 
                           self.surface_id, 
                           self.window_end_ts_ns, 
                           self.spot_ref_price_int, 
                           self.count, 
                           self.flags,
                           self.prediction_horizon_ns)

# Payload encoders
# We'll use struct.pack for efficiency

def pack_snap(mid_price: float, book_valid: bool) -> bytes:
    # d ? 7x
    # 8 + 1 + 7 = 16 bytes
    return struct.pack('<d?7x', mid_price, book_valid)

def pack_wall_entry(rel_ticks: int, side: int, intensity: float, erosion: float) -> bytes:
    # h B f f
    # 2 + 1 + 4 + 4 = 11 bytes. 
    # NOTE: UE struct alignment might be an issue if we don't pack.
    # Let's assume on UE side we read byte-by-byte or use #pragma pack(1).
    return struct.pack('<hBff', rel_ticks, side, intensity, erosion)

def pack_vacuum_entry(rel_ticks: int, vacuum_score: float, turbulence: float) -> bytes:
    # h f f
    # 2 + 4 + 4 = 10 bytes
    return struct.pack('<hff', rel_ticks, vacuum_score, turbulence)

def pack_physics_entry(rel_ticks: int, score_signed: float) -> bytes:
    # h f
    # 2 + 4 = 6 bytes
    return struct.pack('<hf', rel_ticks, score_signed)

def pack_gex_entry(rel_ticks: int, gex_abs: float, balance_ratio: float) -> bytes:
    # h f f
    # 2 + 4 + 4 = 10 bytes
    return struct.pack('<hff', rel_ticks, gex_abs, balance_ratio)
