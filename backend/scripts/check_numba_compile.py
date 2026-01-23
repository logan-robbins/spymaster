
import os
import sys
from pathlib import Path

# Add backend to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "src"))

from data_eng.stages.silver.future_option_mbo.compute_gex_surface_1s import SilverComputeGexSurface1s
from data_eng.config import AppConfig

def main():
    stage = SilverComputeGexSurface1s()
    # Dummy run or load context?
    # Actually simpler to just run the full pipeline script if arguments allow.
    # But for invalidation, we might want to just run this file.
    pass

if __name__ == "__main__":
    # We will just use the verify script to import and compile,
    # catching errors early.
    try:
        from numba import njit
        @njit
        def test_compile():
            pass
        test_compile()
        print("Numba basic compile ok")
        
        # Import the stage module which triggers the JIT compilation on use?
        # Numba is lazy. We need to actually call the mocked function to check compilation.
        from data_eng.stages.silver.future_option_mbo.compute_gex_surface_1s import _numba_mbo_to_mids
        print("Imported _numba_mbo_to_mids")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
