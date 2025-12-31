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

# Ensure we're using Python 3.12 with uv
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create/sync virtual environment with Python 3.12
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    uv venv --python 3.12
    echo "Installing dependencies..."
    uv pip sync
fi

# Run the app with uv
echo "Launching viewer..."
uv run python app.py
