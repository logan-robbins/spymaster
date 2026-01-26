import socket
import struct
import time
import sys

# Header Struct: <4sHHqqIIq (40 bytes)
HEADER_FMT = '<4sHHqqIIq'
HEADER_SIZE = 40

def main():
    ip = "127.0.0.1"
    port = 7777
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    
    print(f"Listening on UDP {ip}:{port}...")
    
    start_time = time.time()
    packet_counts = {}
    
    while time.time() - start_time < 15:
        try:
            sock.settimeout(1.0)
            data, addr = sock.recvfrom(65535)
            
            if len(data) < HEADER_SIZE:
                print(f"Short packet: {len(data)}")
                continue
                
            header_bytes = data[:HEADER_SIZE]
            magic, ver, sid, ts, spot, count, flags, pred_ns = struct.unpack(HEADER_FMT, header_bytes)
            
            if magic != b'MWT1':
                print(f"Invalid Magic: {magic}")
                continue
                
            if sid not in packet_counts:
                packet_counts[sid] = 0
            packet_counts[sid] += 1
            
            # Print periodic stats (every 100 packets total)
            total = sum(packet_counts.values())
            if total % 50 == 0:
                print(f"Total: {total} | Breakdown: {packet_counts} | Last TS: {ts}")
                
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break
            
    print("Test Complete.")
    print(f"Final Counts: {packet_counts}")
    if sum(packet_counts.values()) > 0:
        print("PASS: Received UDP packets")
    else:
        print("FAIL: No packets received")
        sys.exit(1)

if __name__ == "__main__":
    main()
