"""
Experiment Runner - Agent C Implementation

Runs statistical analysis and backtests on level signals.

Provides methods for:
- Simple backtest by level kind (win rate analysis)
- Physics correlation analysis (wall_ratio vs outcomes)
- Generate formatted reports for research
"""

from typing import List, Dict, Optional
from collections import defaultdict
import statistics

from src.common.schemas.levels_signals import LevelSignal, OutcomeLabel, LevelKind


class ExperimentRunner:
    """
    Agent C: The Research Scientist
    
    Runs statistical experiments on labeled level signals to validate hypotheses:
    - Which level kinds have highest bounce rates?
    - Does wall_ratio correlate with outcome?
    - How does first 15m performance differ?
    """
    
    def __init__(self, signals: Optional[List[LevelSignal]] = None):
        """
        Initialize Experiment Runner.
        
        Args:
            signals: Optional list of signals to analyze. Can also be passed to methods.
        """
        self.signals = signals or []
    
    def run_simple_backtest(
        self, 
        signals: Optional[List[LevelSignal]] = None,
        print_report: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Run simple backtest grouped by level kind.
        
        Calculates:
        - Total count per level kind
        - Bounce rate (% of outcomes that are BOUNCE)
        - Break rate (% of outcomes that are BREAK)
        - Chop rate (% of outcomes that are CHOP)
        
        Args:
            signals: List of signals to analyze (uses self.signals if None)
            print_report: Whether to print formatted markdown report
        
        Returns:
            Dictionary with statistics by level kind
            
        Example Output:
            {
                "PM_HIGH": {
                    "count": 45,
                    "bounce_rate": 0.67,
                    "break_rate": 0.22,
                    "chop_rate": 0.11
                },
                ...
            }
        """
        signals_to_analyze = signals if signals is not None else self.signals
        
        if not signals_to_analyze:
            print("⚠️  No signals to analyze")
            return {}
        
        # Group signals by level kind
        grouped: Dict[LevelKind, List[LevelSignal]] = defaultdict(list)
        for signal in signals_to_analyze:
            grouped[signal.level_kind].append(signal)
        
        # Calculate statistics
        results = {}
        for level_kind, level_signals in grouped.items():
            total = len(level_signals)
            bounce_count = sum(1 for s in level_signals if s.outcome == OutcomeLabel.BOUNCE)
            break_count = sum(1 for s in level_signals if s.outcome == OutcomeLabel.BREAK)
            chop_count = sum(1 for s in level_signals if s.outcome == OutcomeLabel.CHOP)
            
            results[level_kind.value] = {
                "count": total,
                "bounce_rate": bounce_count / total if total > 0 else 0.0,
                "break_rate": break_count / total if total > 0 else 0.0,
                "chop_rate": chop_count / total if total > 0 else 0.0,
            }
        
        # Print formatted report
        if print_report:
            self._print_backtest_report(results)
        
        return results
    
    def run_physics_correlation(
        self,
        signals: Optional[List[LevelSignal]] = None,
        print_report: bool = True
    ) -> Dict[str, float]:
        """
        Calculate correlation between physics metrics and outcomes.
        
        Analyzes:
        - wall_ratio vs outcome (Bounce=1, Break=0, Chop excluded)
        - gamma_exposure vs outcome
        - tape_velocity vs outcome
        
        Args:
            signals: List of signals to analyze (uses self.signals if None)
            print_report: Whether to print formatted report
        
        Returns:
            Dictionary of correlation coefficients
            
        Example Output:
            {
                "wall_ratio_correlation": 0.34,
                "gamma_exposure_correlation": -0.12,
                "tape_velocity_correlation": 0.08
            }
        """
        signals_to_analyze = signals if signals is not None else self.signals
        
        if not signals_to_analyze:
            print("⚠️  No signals to analyze")
            return {}
        
        # Filter to only BOUNCE and BREAK (exclude CHOP and UNDEFINED)
        resolved_signals = [
            s for s in signals_to_analyze 
            if s.outcome in [OutcomeLabel.BOUNCE, OutcomeLabel.BREAK]
        ]
        
        if len(resolved_signals) < 2:
            print("⚠️  Need at least 2 resolved signals (BOUNCE/BREAK) for correlation")
            return {}
        
        # Encode outcomes: BOUNCE=1, BREAK=0
        outcomes = [1.0 if s.outcome == OutcomeLabel.BOUNCE else 0.0 for s in resolved_signals]
        
        # Extract physics metrics
        wall_ratios = [s.wall_ratio for s in resolved_signals]
        gamma_exposures = [s.gamma_exposure for s in resolved_signals]
        tape_velocities = [s.tape_velocity for s in resolved_signals]
        
        # Calculate correlations
        results = {
            "wall_ratio_correlation": self._pearson_correlation(wall_ratios, outcomes),
            "gamma_exposure_correlation": self._pearson_correlation(gamma_exposures, outcomes),
            "tape_velocity_correlation": self._pearson_correlation(tape_velocities, outcomes),
            "sample_size": len(resolved_signals),
            "bounce_count": sum(outcomes),
            "break_count": len(outcomes) - sum(outcomes),
        }
        
        if print_report:
            self._print_correlation_report(results)
        
        return results
    
    def run_time_based_analysis(
        self,
        signals: Optional[List[LevelSignal]] = None,
        print_report: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance differences by time of day.
        
        Compares:
        - First 15 minutes (09:30-09:45 ET) vs rest of day
        - Bounce/break rates for each period
        
        Args:
            signals: List of signals to analyze (uses self.signals if None)
            print_report: Whether to print formatted report
        
        Returns:
            Dictionary with statistics by time period
        """
        signals_to_analyze = signals if signals is not None else self.signals
        
        if not signals_to_analyze:
            print("⚠️  No signals to analyze")
            return {}
        
        # Split by time period
        first_15m_signals = [s for s in signals_to_analyze if s.is_first_15m]
        rest_of_day_signals = [s for s in signals_to_analyze if not s.is_first_15m]
        
        results = {
            "first_15m": self._calculate_outcome_stats(first_15m_signals),
            "rest_of_day": self._calculate_outcome_stats(rest_of_day_signals),
        }
        
        if print_report:
            self._print_time_analysis_report(results)
        
        return results
    
    # --- Helper Methods ---
    
    def _calculate_outcome_stats(self, signals: List[LevelSignalV1]) -> Dict[str, float]:
        """Calculate outcome statistics for a list of signals."""
        if not signals:
            return {"count": 0, "bounce_rate": 0.0, "break_rate": 0.0, "chop_rate": 0.0}
        
        total = len(signals)
        bounce_count = sum(1 for s in signals if s.outcome == OutcomeLabel.BOUNCE)
        break_count = sum(1 for s in signals if s.outcome == OutcomeLabel.BREAK)
        chop_count = sum(1 for s in signals if s.outcome == OutcomeLabel.CHOP)
        
        return {
            "count": total,
            "bounce_rate": bounce_count / total,
            "break_rate": break_count / total,
            "chop_rate": chop_count / total,
        }
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Returns correlation between -1 and 1, or 0.0 if calculation fails.
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            n = len(x)
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            
            sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
            sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
            
            denominator = (sum_sq_x * sum_sq_y) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        except Exception:
            return 0.0
    
    def _print_backtest_report(self, results: Dict[str, Dict[str, float]]):
        """Print formatted markdown backtest report."""
        print("\n" + "="*60)
        print("📊 SIMPLE BACKTEST REPORT - BY LEVEL KIND")
        print("="*60)
        print()
        print("| Level Kind      | Count | Bounce % | Break % | Chop % |")
        print("|-----------------|-------|----------|---------|--------|")
        
        # Sort by bounce rate descending
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["bounce_rate"],
            reverse=True
        )
        
        for level_kind, stats in sorted_results:
            print(
                f"| {level_kind:15s} | "
                f"{stats['count']:5d} | "
                f"{stats['bounce_rate']*100:7.1f}% | "
                f"{stats['break_rate']*100:6.1f}% | "
                f"{stats['chop_rate']*100:5.1f}% |"
            )
        
        print()
        print("✅ Higher Bounce % = Level holds more often (rejects price)")
        print()
    
    def _print_correlation_report(self, results: Dict[str, float]):
        """Print formatted correlation analysis report."""
        print("\n" + "="*60)
        print("🔬 PHYSICS CORRELATION ANALYSIS")
        print("="*60)
        print()
        print(f"Sample Size: {results['sample_size']} signals")
        print(f"  - Bounces: {int(results['bounce_count'])}")
        print(f"  - Breaks:  {int(results['break_count'])}")
        print()
        print("Correlation with BOUNCE (1=Bounce, 0=Break):")
        print(f"  wall_ratio:      {results['wall_ratio_correlation']:+.3f}")
        print(f"  gamma_exposure:  {results['gamma_exposure_correlation']:+.3f}")
        print(f"  tape_velocity:   {results['tape_velocity_correlation']:+.3f}")
        print()
        print("Interpretation:")
        print("  +1.0 = Perfect positive correlation (higher value → more bounces)")
        print("  -1.0 = Perfect negative correlation (higher value → more breaks)")
        print("   0.0 = No correlation")
        print()
    
    def _print_time_analysis_report(self, results: Dict[str, Dict[str, float]]):
        """Print formatted time-based analysis report."""
        print("\n" + "="*60)
        print("⏰ TIME-BASED ANALYSIS")
        print("="*60)
        print()
        
        for period, stats in results.items():
            period_label = "First 15 Minutes (09:30-09:45 ET)" if period == "first_15m" else "Rest of Day"
            print(f"{period_label}:")
            print(f"  Count:       {stats['count']}")
            print(f"  Bounce Rate: {stats['bounce_rate']*100:.1f}%")
            print(f"  Break Rate:  {stats['break_rate']*100:.1f}%")
            print(f"  Chop Rate:   {stats['chop_rate']*100:.1f}%")
            print()

