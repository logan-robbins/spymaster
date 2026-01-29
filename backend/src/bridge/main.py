import argparse
import asyncio
from .client import BridgeClient

def main():
    parser = argparse.ArgumentParser(description="Market Wind Tunnel Bridge (WebSocket -> UDP)")
    parser.add_argument("--symbol", type=str, default="ESH6", help="Symbol to stream")
    parser.add_argument("--dt", type=str, default="2026-01-06", help="Date to stream")
    parser.add_argument("--ws-url", type=str, default="ws://localhost:8001/v1/velocity/stream", help="WebSocket Base URL")
    parser.add_argument("--udp-ip", type=str, default="127.0.0.1", help="Target UDP IP")
    parser.add_argument("--udp-port", type=int, default=7777, help="Target UDP Port")
    
    args = parser.parse_args()
    
    full_uri = f"{args.ws_url}?symbol={args.symbol}&dt={args.dt}"
    
    print(f"Starting Bridge...")
    print(f"Source: {full_uri}")
    print(f"Target: {args.udp_ip}:{args.udp_port}")
    
    client = BridgeClient(full_uri, args.udp_ip, args.udp_port)
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nStopping Bridge.")
        client.stop()

if __name__ == "__main__":
    main()
