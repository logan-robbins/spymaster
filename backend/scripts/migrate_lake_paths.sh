#!/usr/bin/env bash
# migrate_lake_paths.sh -- Rename lake directories from old VP names to new qMachina names.
#
# Before:
#   lake/research/vp_immutable/     -> lake/research/datasets/
#   lake/research/vp_harness/       -> lake/research/harness/
#   lake/research/vp_experiments/   -> lake/research/experiments/
#
# This script is idempotent: if the new directory already exists and the old does not,
# the rename is silently skipped.
#
# Usage:
#   bash scripts/migrate_lake_paths.sh [--lake-root backend/lake]

set -euo pipefail

LAKE_ROOT="${1:-$(dirname "$0")/../lake}"
LAKE_ROOT="$(cd "$LAKE_ROOT" && pwd)"
RESEARCH="$LAKE_ROOT/research"

echo "Lake root:    $LAKE_ROOT"
echo "Research dir: $RESEARCH"
echo

rename_dir() {
    local old_name="$1"
    local new_name="$2"
    local old_path="$RESEARCH/$old_name"
    local new_path="$RESEARCH/$new_name"

    if [ -d "$old_path" ] && [ ! -d "$new_path" ]; then
        echo "  RENAME $old_name -> $new_name"
        mv "$old_path" "$new_path"
    elif [ -d "$old_path" ] && [ -d "$new_path" ]; then
        echo "  CONFLICT: both $old_name and $new_name exist. Manual merge required."
        exit 1
    elif [ ! -d "$old_path" ] && [ -d "$new_path" ]; then
        echo "  SKIP $old_name -> $new_name (already migrated)"
    else
        echo "  SKIP $old_name -> $new_name (source does not exist)"
    fi
}

rename_dir "vp_immutable" "datasets"
rename_dir "vp_harness" "harness"
rename_dir "vp_experiments" "experiments"

echo
echo "Migration complete."
