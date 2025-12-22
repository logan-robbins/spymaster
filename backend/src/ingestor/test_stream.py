import os
import asyncio
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")

if not API_KEY:
    print("❌ Critical Error: POLYGON_API_KEY not found in .env")
    exit(1)

# Hardcoded ticker for testing
TICKER = "O:SPY251220C00600000" # Example ticker, might need updating to a valid future one if this is old
# Actually, let's use a dynamic one or just a known active one. 
# It's better to pick something relevant. 
# But for a simple test, we just want to see IF we connect.
# Let's use * for all trades just to see flow? No, that's too much.
# Let's try to subscribe to all SPY trades if possible, or just one specific active option.
# The prompt says "hardcode 1 active ticker". 
# I will use a placeholder and comment that it might need updating.
# Or better, I'll use a stock ticker 'T.SPY' first to verify Auth, then Option 'O:...'
# APP.md says: `Listen for T (Trade) events.`
# It implies options trades, but verifying auth with stock trade is safer as it's always active.
# Actually, APP.md 2.3 says `market='options'`. So I must use options.
# I will guess a strike that is likely to exist. SPY is always active.
# I'll just use a generic format or maybe I can list trades for SPY...
# I will try to subscribe to "O:SPY*" if Polygon allows wildcard? 
# Polygon docs say: A symbol... 
# Let's just pick one that is likely active: The ATM call for the next expiration.
# But I don't know the date.
# I will use a dummy ticker and see if I get a "success" subscription message at least.

async def handle_msg(msgs: list[WebSocketMessage]):
    for m in msgs:
        print(f"📩 Received: {m}")

def main():
    print(f"🔑 Using API Key: {API_KEY[:4]}...{API_KEY[-4:]}")
    print("🔌 Connecting to Polygon options stream...")
    
    # Using the client in synchronous mode wrapper or async? 
    # polygon-api-client has a WebSocketClient.
    
    client = WebSocketClient(
        api_key=API_KEY,
        market='options',
        subscriptions=[f"T.{TICKER}"], # T = Trades
        verbose=True
    )
    
    print(f"Run loop for {TICKER}")
    client.run(handle_msg=handle_msg)

if __name__ == "__main__":
    main()
