#!/bin/bash
#
# Quick-start script: Download ES Futures (June 1 - Nov 1, 2025)
# This fills the gap between your existing data (Nov 2 - Dec 19).
#
# Usage:
#   cd backend
#   bash scripts/download_es_june_to_nov.sh
#

set -e

echo "========================================================================"
echo "ES Futures Batch Download: June 1 - Nov 1, 2025"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Submit batch jobs to Databento (trades + MBP-10)"
echo "  2. Wait for server-side processing (~5-15 min)"
echo "  3. Download ~109 days of data in parallel (8 connections)"
echo "  4. Skip any dates you already have"
echo ""
echo "Expected download:"
echo "  - Trades: ~5 GB compressed"
echo "  - MBP-10: ~1 TB compressed"
echo "  - Time: ~30-60 min (depends on network)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting batch download..."
echo ""

# Run batch download
uv run python scripts/download_es_futures_batch.py \
    --start 2025-06-01 \
    --end 2025-11-01 \
    --workers 8

echo ""
echo "========================================================================"
echo "✅ Download complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Process to Bronze layer:"
echo "     uv run python -m scripts.backfill_bronze_futures --all"
echo ""
echo "  2. Run pipeline:"
echo "     uv run python -m scripts.run_pipeline --start 2025-06-01 --end 2025-11-01"
echo ""
echo "  3. Build ML indices:"
echo "     See IMPLEMENTATION_READY.md Section 13 for gold layer processing"
echo ""

