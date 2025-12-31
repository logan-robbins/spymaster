

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_context_metrics():
    """
    Run a unit test to validate 'Glass Box' context metrics.
    Checks:
    1. Are metrics populated?
    2. Does 'base_rate_reject' correlate with actual REJECT outcomes?
    3. Does 'historical_avg_volatility' correlate with actual max excursion?
    """
    logger.info("Testing outcome_aggregation logic directly with mock data...")
    from src.ml.outcome_aggregation import aggregate_query_results
    
    # Create mock retrieved metadata
    # Scenario: 4 neighbors. 
    # 3 REJECTs (Small volatility), 1 BREAK (Huge volatility)
    # This simulates a "High Probability Rejection" but "High Risk if Wrong" scenario.
    mock_retrieved = pd.DataFrame({
        'similarity': [0.99, 0.95, 0.90, 0.85], # High similarity
        'outcome_4min': ['REJECT', 'REJECT', 'BREAK', 'REJECT'],
        'excursion_favorable': [1.0, 2.0, 20.0, 1.0], # The BREAK goes 20 pts
        'excursion_adverse': [5.0, 4.0, 2.0, 5.0],    # The BREAK has small adverse
        'strength_abs': [50, 40, 100, 45],
        'date': [pd.Timestamp('2025-11-01')] * 4
    })
    
    logger.info(f"Mock Data:\n{mock_retrieved}")
    
    agg_result = aggregate_query_results(mock_retrieved, query_date=pd.Timestamp('2025-12-01'))
    
    metrics = agg_result.get('context_metrics', {})
    logger.info(f"Computed Metrics: {metrics}")
    
    # 1. Check Base Rate Calculation
    # Weights will be roughly equal (slightly higher for 0.99)
    # base_rate_reject should be ~0.75
    # base_rate_break should be ~0.25
    assert metrics['base_rate_reject'] > 0.6, "Reject rate should be dominant"
    assert metrics['base_rate_break'] > 0.1, "Break rate should be non-zero"
    
    # 2. Check Volatility
    # Max excursion is 20.0 from the break. Weighted avg should be pulled up significantly.
    # If it was simple average: (5+5+20+5)/4 = 8.75
    # Weighted might be slightly different.
    assert metrics['historical_max_excursion'] == 20.0, "Max excursion failed"
    assert metrics['historical_avg_volatility'] > 6.0, "Volatility should reflect the 20pt move"
    
    print("\n--- Validation Successful ---")
    print(f"Base Rate Reject: {metrics['base_rate_reject']:.4f}")
    print(f"Base Rate Break: {metrics['base_rate_break']:.4f}")
    print(f"Avg Volatility: {metrics['historical_avg_volatility']:.4f}")
    print(f"Max Excursion: {metrics['historical_max_excursion']:.4f}")

if __name__ == "__main__":
    validate_context_metrics()

