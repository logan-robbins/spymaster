"""
Config override context manager for hyperparameter optimization.

Allows temporary CONFIG modifications without polluting global state.
"""

from typing import Any, Dict


class ConfigOverride:
    """Context manager for temporary config changes.
    
    Usage:
        with ConfigOverride(MONITOR_BAND=10.0, OUTCOME_THRESHOLD=50.0):
            signals = pipeline.run(date)
        # CONFIG restored to original values
    """
    
    def __init__(self, **overrides: Dict[str, Any]):
        """
        Initialize config override.
        
        Args:
            **overrides: CONFIG attribute overrides
        """
        self.overrides = overrides
        self.original: Dict[str, Any] = {}
    
    def __enter__(self):
        """Apply config overrides."""
        from src.common.config import CONFIG
        
        for key, value in self.overrides.items():
            if not hasattr(CONFIG, key):
                raise AttributeError(f"CONFIG has no attribute '{key}'")
            
            self.original[key] = getattr(CONFIG, key)
            setattr(CONFIG, key, value)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original config values."""
        from src.common.config import CONFIG
        
        for key, value in self.original.items():
            setattr(CONFIG, key, value)
        
        return False  # Don't suppress exceptions

