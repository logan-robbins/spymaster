#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BACKEND_DIR"

SYMBOL="${1:-ESU5}"
DATES="${2:-2025-06-05,2025-06-06,2025-06-09,2025-06-10}"
INDEX_TYPE="${3:-flat}"
OUTPUT_DIR="${4:-databases/indices}"

echo "Building FAISS indices and metadata store..."
echo "  Symbol: $SYMBOL"
echo "  Dates: $DATES"
echo "  Index Type: $INDEX_TYPE"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

uv run python -m src.data_eng.stages.gold.future.build_faiss_index \
    --symbol "$SYMBOL" \
    --dates "$DATES" \
    --output-dir "$OUTPUT_DIR" \
    --index-type "$INDEX_TYPE"

echo ""
echo "Done! Indices written to $OUTPUT_DIR"
