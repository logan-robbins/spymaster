#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BACKEND_DIR"

usage() {
    echo "Build FAISS indices and metadata store from setup vectors"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --symbol         Symbol to process (default: ESU5)"
    echo "  --dates          Date range: 2025-06-05:2025-06-10 or comma-separated"
    echo "  --start-date     Start date (YYYY-MM-DD)"
    echo "  --end-date       End date (YYYY-MM-DD)"
    echo "  --index-type     Index type: flat, ivf_flat, ivf_pq (default: flat)"
    echo "  --output-dir     Output directory (default: databases/indices)"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --dates 2025-06-05:2025-06-10"
    echo "  $0 --start-date 2025-06-05 --end-date 2025-06-10 --symbol ESU5"
    echo "  $0 --dates 2025-06-05,2025-06-06,2025-06-09"
    exit 0
}

SYMBOL="ESU5"
DATES=""
START_DATE=""
END_DATE=""
INDEX_TYPE="flat"
OUTPUT_DIR="databases/indices"

while [[ $# -gt 0 ]]; do
    case $1 in
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --dates)
            DATES="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --index-type)
            INDEX_TYPE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$DATES" && (-z "$START_DATE" || -z "$END_DATE") ]]; then
    echo "Error: Must provide --dates or both --start-date and --end-date"
    echo ""
    usage
fi

echo "Building FAISS indices and metadata store..."
echo "  Symbol: $SYMBOL"

if [[ -n "$DATES" ]]; then
    echo "  Dates: $DATES"
else
    echo "  Range: $START_DATE to $END_DATE"
fi

echo "  Index Type: $INDEX_TYPE"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

CMD="uv run python -m src.data_eng.stages.gold.future.build_faiss_index --symbol $SYMBOL --index-type $INDEX_TYPE --output-dir $OUTPUT_DIR"

if [[ -n "$DATES" ]]; then
    CMD="$CMD --dates $DATES"
else
    CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
fi

eval $CMD

echo ""
echo "Done! Indices written to $OUTPUT_DIR"
