#!/bin/bash
# Pentaview Stream Viewer - Launch Script

echo "🚀 Starting Pentaview Stream Viewer..."
echo ""
echo "Prerequisites:"
echo "  ✓ Gateway running on ws://localhost:8000/ws/stream"
echo "  ✓ Replay engine publishing stream data"
echo ""
echo "Starting Flask server on http://localhost:5000"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the app
python app.py
