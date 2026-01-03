#!/bin/bash
# Regenerate Bronze futures data with proper contract symbols
# 
# This script:
# 1. Backs up options bronze data (no source files available)
# 2. Cleans futures bronze data for target date range
# 3. Regenerates from DBN files using backfill script
# 4. Verifies the new data has correct symbol values
#
# Usage:
#   cd backend/
#   ./scripts/regenerate_bronze_futures.sh
#
# NOTE: The read-time price filter already handles mixed contract data.
#       This regeneration is optional but provides cleaner bronze data.

set -e

# Configuration
START_DATE="2025-06-02"
END_DATE="2025-09-30"
WORKERS=4
BRONZE_ROOT="data/bronze"
OPTIONS_BACKUP="data/bronze_options_backup_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "BRONZE FUTURES REGENERATION"
echo "========================================"
echo "Date range: $START_DATE to $END_DATE"
echo "Workers: $WORKERS"
echo ""

# Step 1: Backup options (cannot regenerate - no source files)
echo "Step 1: Backing up options bronze data..."
if [ -d "$BRONZE_ROOT/options" ]; then
    cp -r "$BRONZE_ROOT/options" "$OPTIONS_BACKUP"
    echo "  ✅ Options backed up to: $OPTIONS_BACKUP"
else
    echo "  ⚠️  No options data found to backup"
fi
echo ""

# Step 2: Clean futures bronze data
echo "Step 2: Cleaning futures bronze data..."
FUTURES_PATH="$BRONZE_ROOT/futures/mbp10/symbol=ES"

# Count existing dates
EXISTING_COUNT=$(find "$FUTURES_PATH" -maxdepth 1 -type d -name "date=2025-0[6-9]*" 2>/dev/null | wc -l | tr -d ' ')
echo "  Found $EXISTING_COUNT existing date directories"

if [ "$EXISTING_COUNT" -gt 0 ]; then
    read -p "  Delete existing futures data for Jun-Sep 2025? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$FUTURES_PATH"/date=2025-06-*
        rm -rf "$FUTURES_PATH"/date=2025-07-*
        rm -rf "$FUTURES_PATH"/date=2025-08-*
        rm -rf "$FUTURES_PATH"/date=2025-09-*
        echo "  ✅ Deleted existing futures data"
    else
        echo "  ⏭️  Skipped deletion"
    fi
fi
echo ""

# Step 3: Regenerate from DBN files
echo "Step 3: Regenerating from DBN files..."
echo "  This may take a while..."
echo ""

uv run python -m scripts.backfill_bronze_futures \
    --all \
    --workers "$WORKERS" \
    --force \
    --batch-size 15000000

echo ""

# Step 4: Verify new data
echo "Step 4: Verifying new bronze data..."
uv run python << 'VERIFY_EOF'
import duckdb
from pathlib import Path

print("\n📊 VERIFICATION:")
print("-" * 60)

base = Path("data/bronze/futures/mbp10/symbol=ES")
dates = sorted([d.name.replace("date=", "") for d in base.iterdir() if d.name.startswith("date=2025-0")])

if not dates:
    print("  ❌ No bronze data found!")
    exit(1)

print(f"  Dates regenerated: {len(dates)}")
print(f"  Range: {dates[0]} to {dates[-1]}")

# Check symbol values in a sample
sample_date = dates[len(dates)//2]  # Middle date
sample_path = base / f"date={sample_date}" / "**" / "*.parquet"

conn = duckdb.connect()
query = f"""
SELECT DISTINCT symbol, COUNT(*) as cnt
FROM read_parquet('{sample_path}', hive_partitioning=true)
GROUP BY symbol
ORDER BY cnt DESC
LIMIT 10
"""
result = conn.execute(query).fetchdf()
conn.close()

print(f"\n  Symbol values in {sample_date}:")
for _, row in result.iterrows():
    print(f"    {row['symbol']}: {row['cnt']:,} records")

# Check for proper contract codes (ESM5, ESU5, etc.)
has_contract_codes = any(
    s.startswith('ES') and len(s) == 4 and s[2].isalpha() and s[3].isdigit()
    for s in result['symbol']
)

if has_contract_codes:
    print("\n  ✅ Contract codes found (e.g., ESM5, ESU5)")
else:
    print("\n  ⚠️  No contract codes found - symbology may need update")
    print("      Read-time price filter will still work correctly")

VERIFY_EOF

echo ""
echo "========================================"
echo "REGENERATION COMPLETE"
echo "========================================"
echo "Options backup: $OPTIONS_BACKUP"
echo ""
echo "Next steps:"
echo "  1. Run pipeline to verify: uv run python -m scripts.run_pipeline --pipeline bronze_to_silver --start 2025-06-11 --end 2025-06-11 --level PM_HIGH --write-outputs"
echo "  2. Check silver output for correct level prices"
echo ""

