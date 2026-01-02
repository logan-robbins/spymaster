import sys
from pathlib import Path
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.normalization import ComputeNormalizationStage

logging.basicConfig(level=logging.INFO)

def main():
    backend_dir = Path(__file__).parent.parent
    data_root = backend_dir / "data"
    
    # Input: State Table (Silver)
    state_table_dir = data_root / "silver" / "state" / "es_level_state" / "version=4.0.0"
    
    # Output: Gold Normalization
    output_dir = data_root / "gold" / "normalization"
    
    print(f"Computing normalization stats...")
    print(f"  Input: {state_table_dir}")
    print(f"  Output: {output_dir}")
    
    stage = ComputeNormalizationStage(
        state_table_dir=state_table_dir,
        output_dir=output_dir,
        lookback_days=120
    )
    stage.set_end_date('2025-10-31')
    
    result = stage.execute()
    print(f"Success! Stats saved to: {result['output_file']}")

if __name__ == "__main__":
    main()
