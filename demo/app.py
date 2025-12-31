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
    
    async def forward_messages():
        try:
            async with websockets.connect(GATEWAY_WS_URL) as gateway_ws:
                logger.info(f"Connected to Gateway: {GATEWAY_WS_URL}")
                
                async for message in gateway_ws:
                    # Forward raw message to frontend
                    ws.send(message)
                    
        except Exception as e:
            logger.error(f"Gateway connection error: {e}")
            ws.send(json.dumps({'error': str(e)}))
    
    # Run async loop
    asyncio.run(forward_messages())

if __name__ == '__main__':
    logger.info("Starting Pentaview Stream Viewer...")
    logger.info("Open http://localhost:5000 in your browser")
    logger.info(f"Connecting to Gateway: {GATEWAY_WS_URL}")
    app.run(debug=True, host='0.0.0.0', port=5000)
