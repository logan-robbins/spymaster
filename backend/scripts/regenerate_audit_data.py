
"""
Regenerate Silver Data for Audit
Re-runs bronze_to_silver pipeline for the 3x3 day audit periods
to fix missing Market Tide and OFI features.
"""
import sys
from pathlib import Path
import logging
import argparse

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.pipeline.pipelines.bronze_to_silver import build_bronze_to_silver_pipeline
# from src.pipeline.core.runner import PipelineRunner hiding broken import
from src.common.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="4.0.0")
    args = parser.parse_args()

    # Define Audit Periods (Same as audit_silver_stats.py)
    audit_dates = [
        # June (Early)
        "2025-06-05", "2025-06-06", "2025-06-09",
        # August (Mid)
        "2025-08-20", "2025-08-21", "2025-08-22",
        # Oct (Late)
        "2025-10-29", "2025-10-30", "2025-09-30"
    ]
    
    print(f"Regenerating Silver data for {len(audit_dates)} days...")
    print(f"Version: {args.version}")
    
    # We must rebuild pipeline for each run or ensure it's stateless enough.
    # Pipeline object handles sequential stage execution.
    pipeline = build_bronze_to_silver_pipeline()
    
    for date in audit_dates:
        print(f"\n>>> Processing {date} <<<")
        try:
            # Overwrite enabled by default in run() via overwrite_partitions=True
            pipeline.run(date=date, write_outputs=True)
            print(f"SUCCESS: {date}")
        except Exception as e:
            print(f"FAILED: {date} - {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
