#!/bin/bash
# Download Polygon flat files for December dates

cd /Users/loganrobbins/research/qmachina/spymaster/backend

echo "Starting Polygon flat file downloads for December..."
echo "Using S3 flat files (50-100x faster than API)"
echo ""

dates=(
  "2025-12-01"
  "2025-12-02"
  "2025-12-03"
  "2025-12-04"
  "2025-12-05"
  "2025-12-08"
  "2025-12-09"
  "2025-12-10"
  "2025-12-11"
  "2025-12-12"
)

for date in "${dates[@]}"; do
  echo "============================================================"
  echo "Downloading $date..."
  echo "============================================================"
  uv run python -u -m src.ingestor.polygon_flatfiles --date "$date"
  if [ $? -eq 0 ]; then
    echo "✅ $date completed"
  else
    echo "❌ $date failed"
  fi
  echo ""
done

echo "============================================================"
echo "All downloads complete!"
echo "============================================================"

