"""
Pentaview Stream Viewer - Real-time visualization of Pentaview projections

Connects to Gateway WebSocket to display:
- 2-minute OHLCV candles (left Y-axis)
- Stream overlays: sigma_p, sigma_m, sigma_f, sigma_b, sigma_r (right Y-axis, -1 to +1)
- Projection bands: q10, q50, q90 extending 5 minutes ahead
- Updates every 30 seconds
"""
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_sock import Sock
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Gateway WebSocket URL (backend)
GATEWAY_WS_URL = "ws://localhost:8000/ws/stream"

@app.route('/')
def index():
    """Serve Pentaview visualization UI"""
    return render_template('index.html')

@app.route('/config')
def config():
    """Configuration endpoint for frontend"""
    return jsonify({
        'gateway_ws_url': GATEWAY_WS_URL,
        'streams': ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r'],
        'projection_horizon': 10,  # 10 bars @ 30s = 5 minutes
        'model_version': 'v30s_20251115_20251215'
    })

@sock.route('/ws/pentaview')
def pentaview_websocket(ws):
    """
    WebSocket proxy: Forwards Gateway stream data to frontend
    
    This allows the frontend to connect to a local WebSocket
    while we handle the Gateway connection server-side.
    """
    import asyncio
    import websockets
    import threading
    
    stop_event = threading.Event()
    
    async def forward_messages():
        try:
            async with websockets.connect(GATEWAY_WS_URL) as gateway_ws:
                logger.info(f"Connected to Gateway: {GATEWAY_WS_URL}")
                
                while not stop_event.is_set():
                    try:
                        # Receive with timeout to check stop_event periodically
                        message = await asyncio.wait_for(gateway_ws.recv(), timeout=1.0)
                        # Forward raw message to frontend
                        ws.send(message)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error receiving message: {e}")
                        break
                    
        except Exception as e:
            logger.error(f"Gateway connection error: {e}")
            try:
                ws.send(json.dumps({'error': str(e)}))
            except:
                pass
    
    # Run async forwarding in a thread
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(forward_messages())
    
    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()
    
    # Keep connection alive and handle disconnects
    try:
        while True:
            # Receive from frontend (mainly to detect disconnects)
            data = ws.receive()
            if data is None:
                break
    except Exception as e:
        logger.info(f"Frontend disconnected: {e}")
    finally:
        stop_event.set()
        thread.join(timeout=2)

if __name__ == '__main__':
    logger.info("Starting Pentaview Stream Viewer...")
    logger.info("Open http://localhost:5000 in your browser")
    logger.info(f"Connecting to Gateway: {GATEWAY_WS_URL}")
    app.run(debug=True, host='0.0.0.0', port=5000)
