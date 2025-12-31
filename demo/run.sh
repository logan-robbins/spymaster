#!/bin/bash
# TradingView Chart Demo Startup Script

cd "$(dirname "$0")"

echo "Starting TradingView Forward Projection Chart Demo..."
echo ""
echo "📊 Chart will be available at: http://localhost:5000"
echo ""
echo "Features:"
echo "  • 2-minute candlestick chart"
echo "  • Purple dotted projection line (5 bars ahead)"
echo "  • Based on average slope of last 10 bars"
echo "  • Interactive controls to adjust parameters"
echo ""

source .venv/bin/activate
python app.py

