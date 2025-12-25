#!/bin/bash
# Spymaster Replay Pipeline
# Usage: ./scripts/run_replay.sh [DATE] [SPEED]
#
# Examples:
#   ./scripts/run_replay.sh                    # Replay Nov 2 at max speed
#   ./scripts/run_replay.sh 2025-11-03 1.0     # Replay Nov 3 at realtime
#   ./scripts/run_replay.sh 2025-12-16 0       # Replay Dec 16 at max speed

set -e

DATE=${1:-2025-12-16}
SPEED=${2:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$BACKEND_DIR")"

echo "=============================================="
echo "  SPYMASTER REPLAY PIPELINE"
echo "=============================================="
echo "  Date:  $DATE"
echo "  Speed: ${SPEED}x (0 = max speed)"
echo "=============================================="

# Ensure Docker services are running
echo ""
echo ">>> Ensuring NATS, Core, Gateway are running..."
cd "$PROJECT_DIR"
docker-compose up -d nats core gateway lake

# Wait for services to be healthy
echo ""
echo ">>> Waiting for services to be ready..."
sleep 3

# Check Gateway health
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo ">>> Gateway is healthy"
else
    echo ">>> WARNING: Gateway may not be ready"
fi

# Run replay
echo ""
echo ">>> Starting replay..."
cd "$BACKEND_DIR"
REPLAY_SPEED=$SPEED REPLAY_DATE=$DATE uv run python -m src.ingestor.replay_publisher

echo ""
echo "=============================================="
echo "  REPLAY COMPLETE"
echo "=============================================="
echo ""
echo "To connect frontend:"
echo "  cd frontend && npm run start"
echo "  Open http://localhost:4200"
echo ""
echo "To test WebSocket:"
echo "  curl http://localhost:8000/health"
echo ""
