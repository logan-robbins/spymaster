"""
Pentaview Trading Edge Analysis

Answers day-trading relevant questions:
1. When do tradeable moves occur? (Breakout probability)
2. What predicts direction? (Confluence analysis)
3. Which stream combinations work best? (Multi-signal edge)
4. What timing windows are predictable? (Entry timing)

Usage:
    uv run python scripts/analyze_pentaview_edge.py --date 2025-12-18
    uv run python scripts/analyze_pentaview_edge.py --start 2025-11-17 --end 2025-12-18
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BreakoutEvent:
    """Represents a tradeable move in a stream."""
    timestamp: pd.Timestamp
    stream: str
    direction: str  # 'UP' or 'DOWN'
    magnitude: float  # Size of move
    bars_to_peak: int  # How many bars until peak
    prior_state: Dict[str, float]  # All 5 stream values before breakout


@dataclass
class EdgeMetrics:
    """Trading-relevant edge metrics."""
    # Breakout prediction
    breakout_rate: float  # % of bars that lead to breakouts
    breakout_precision: float  # When we predict breakout, how often correct?
    
    # Directional edge
    directional_accuracy_on_moves: float  # Accuracy when moves happen (not flat)
    up_bias_accuracy: float  # Accuracy when streams positive
    down_bias_accuracy: float  # Accuracy when streams negative
    
    # Confluence edge
    all_aligned_accuracy: float  # When all 5 streams same sign
    majority_aligned_accuracy: float  # When 3+ streams same sign
    divergent_accuracy: float  # When streams mixed
    
    # Timing edge
    immediate_accuracy: float  # 0-2 minute (bars 1-4)
    near_term_accuracy: float  # 2-5 minute (bars 5-10)
    
    # Setup quality
    strong_setup_accuracy: float  # High magnitude + aligned
    weak_setup_accuracy: float  # Low magnitude or divergent
    
    # Sample sizes
    n_total: int
    n_breakouts: int
    n_aligned: int


class PentaviewEdgeAnalyzer:
    """Analyze trading edge in Pentaview streams."""
    
    def __init__(
        self,
        data_root: Path = Path("data"),
        canonical_version: str = "v4.0.0",
        breakout_threshold: float = 0.10  # Minimum 5min change to be "tradeable"
    ):
        self.data_root = data_root
        self.canonical_version = canonical_version
        self.breakout_threshold = breakout_threshold
        
        self.stream_names = ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']
    
    def load_stream_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load stream bars for a date."""
        date_partition = f"date={date}_00:00:00"
        stream_path = (
            self.data_root / "gold" / "streams" / "pentaview" /
            f"version={self.canonical_version}" / date_partition / "stream_bars.parquet"
        )
        
        if not stream_path.exists():
            logger.warning(f"Stream data not found: {stream_path}")
            return None
        
        return pd.read_parquet(stream_path)
    
    def identify_breakouts(self, df: pd.DataFrame, stream: str) -> List[BreakoutEvent]:
        """Find tradeable breakout events in a stream."""
        breakouts = []
        
        for i in range(len(df) - 10):  # Need 10 bars future
            current_row = df.iloc[i]
            future_vals = df[stream].iloc[i+1:i+11].values
            current_val = current_row[stream]
            
            # Check for significant move in next 10 bars
            max_future = np.max(future_vals)
            min_future = np.min(future_vals)
            
            up_move = max_future - current_val
            down_move = current_val - min_future
            
            # Determine if breakout occurred
            if up_move >= self.breakout_threshold:
                # Upside breakout
                bars_to_peak = np.argmax(future_vals) + 1
                
                # Capture prior stream state
                prior_state = {s: float(current_row[s]) for s in self.stream_names if s in current_row}
                
                breakouts.append(BreakoutEvent(
                    timestamp=current_row['timestamp'],
                    stream=stream,
                    direction='UP',
                    magnitude=float(up_move),
                    bars_to_peak=int(bars_to_peak),
                    prior_state=prior_state
                ))
                
            elif down_move >= self.breakout_threshold:
                # Downside breakout
                bars_to_peak = np.argmin(future_vals) + 1
                
                prior_state = {s: float(current_row[s]) for s in self.stream_names if s in current_row}
                
                breakouts.append(BreakoutEvent(
                    timestamp=current_row['timestamp'],
                    stream=stream,
                    direction='DOWN',
                    magnitude=float(down_move),
                    bars_to_peak=int(bars_to_peak),
                    prior_state=prior_state
                ))
        
        return breakouts
    
    def analyze_stream_confluence(self, breakouts: List[BreakoutEvent]) -> Dict[str, Any]:
        """Analyze if stream alignment predicts breakout direction."""
        
        results = {
            'all_aligned': {'correct': 0, 'total': 0},
            'majority_aligned': {'correct': 0, 'total': 0},
            'divergent': {'correct': 0, 'total': 0},
            'positive_bias': {'up': 0, 'down': 0, 'total': 0},
            'negative_bias': {'up': 0, 'down': 0, 'total': 0}
        }
        
        for event in breakouts:
            state = event.prior_state
            target_stream = event.stream
            target_val = state.get(target_stream, 0.0)
            
            # Get other stream values
            other_vals = [state[s] for s in self.stream_names if s != target_stream and s in state]
            
            if len(other_vals) < 4:
                continue
            
            # Count alignment
            n_positive = sum(1 for v in other_vals if v > 0.05)
            n_negative = sum(1 for v in other_vals if v < -0.05)
            
            # Classify confluence
            if n_positive == 4:  # All others positive
                results['all_aligned']['total'] += 1
                if (event.direction == 'UP' and target_val > 0) or (event.direction == 'DOWN' and target_val < 0):
                    results['all_aligned']['correct'] += 1
            
            elif n_negative == 4:  # All others negative
                results['all_aligned']['total'] += 1
                if (event.direction == 'DOWN' and target_val < 0) or (event.direction == 'UP' and target_val > 0):
                    results['all_aligned']['correct'] += 1
            
            elif n_positive >= 3 or n_negative >= 3:  # Majority aligned
                results['majority_aligned']['total'] += 1
                expected_dir = 'UP' if n_positive >= 3 else 'DOWN'
                if event.direction == expected_dir:
                    results['majority_aligned']['correct'] += 1
            
            else:  # Divergent
                results['divergent']['total'] += 1
            
            # Bias tracking
            if target_val > 0.05:
                results['positive_bias']['total'] += 1
                if event.direction == 'UP':
                    results['positive_bias']['up'] += 1
                else:
                    results['positive_bias']['down'] += 1
            
            elif target_val < -0.05:
                results['negative_bias']['total'] += 1
                if event.direction == 'DOWN':
                    results['negative_bias']['down'] += 1
                else:
                    results['negative_bias']['up'] += 1
        
        # Compute accuracies
        summary = {}
        
        if results['all_aligned']['total'] > 0:
            summary['all_aligned_accuracy'] = results['all_aligned']['correct'] / results['all_aligned']['total']
            summary['all_aligned_n'] = results['all_aligned']['total']
        
        if results['majority_aligned']['total'] > 0:
            summary['majority_aligned_accuracy'] = results['majority_aligned']['correct'] / results['majority_aligned']['total']
            summary['majority_aligned_n'] = results['majority_aligned']['total']
        
        if results['divergent']['total'] > 0:
            summary['divergent_n'] = results['divergent']['total']
        
        # Bias accuracy
        if results['positive_bias']['total'] > 0:
            summary['positive_bias_up_rate'] = results['positive_bias']['up'] / results['positive_bias']['total']
            summary['positive_bias_n'] = results['positive_bias']['total']
        
        if results['negative_bias']['total'] > 0:
            summary['negative_bias_down_rate'] = results['negative_bias']['down'] / results['negative_bias']['total']
            summary['negative_bias_n'] = results['negative_bias']['total']
        
        return summary
    
    def analyze_single_date(self, date: str) -> Dict[str, Any]:
        """Run full analysis for one date."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Analyzing {date}")
        logger.info(f"{'='*70}")
        
        df = self.load_stream_data(date)
        if df is None or df.empty:
            return {}
        
        logger.info(f"Loaded {len(df):,} stream bars")
        
        # Find all breakouts across all streams
        all_breakouts = []
        for stream in self.stream_names:
            if stream not in df.columns:
                continue
            
            breakouts = self.identify_breakouts(df, stream)
            all_breakouts.extend(breakouts)
            
            logger.info(f"  {stream}: {len(breakouts)} breakouts (>{self.breakout_threshold} move)")
        
        logger.info(f"\nTotal breakouts: {len(all_breakouts)}")
        logger.info(f"Breakout rate: {len(all_breakouts)/(len(df)*5):.1%} of bar-stream pairs")
        
        # Analyze confluence
        logger.info(f"\n{'─'*70}")
        logger.info("CONFLUENCE ANALYSIS")
        logger.info(f"{'─'*70}")
        
        confluence = self.analyze_stream_confluence(all_breakouts)
        
        if 'all_aligned_accuracy' in confluence:
            logger.info(f"All Streams Aligned (4/4 same sign):")
            logger.info(f"  Directional Accuracy: {confluence['all_aligned_accuracy']:.1%}")
            logger.info(f"  Sample Size: {confluence['all_aligned_n']}")
        
        if 'majority_aligned_accuracy' in confluence:
            logger.info(f"\nMajority Aligned (3+/5 same sign):")
            logger.info(f"  Directional Accuracy: {confluence['majority_aligned_accuracy']:.1%}")
            logger.info(f"  Sample Size: {confluence['majority_aligned_n']}")
        
        if 'positive_bias_up_rate' in confluence:
            logger.info(f"\nPositive Stream Value:")
            logger.info(f"  Up-Move Rate: {confluence['positive_bias_up_rate']:.1%}")
            logger.info(f"  Sample Size: {confluence['positive_bias_n']}")
        
        if 'negative_bias_down_rate' in confluence:
            logger.info(f"\nNegative Stream Value:")
            logger.info(f"  Down-Move Rate: {confluence['negative_bias_down_rate']:.1%}")
            logger.info(f"  Sample Size: {confluence['negative_bias_n']}")
        
        # Timing analysis
        logger.info(f"\n{'─'*70}")
        logger.info("TIMING ANALYSIS")
        logger.info(f"{'─'*70}")
        
        timing_bins = {
            'immediate': [b for b in all_breakouts if b.bars_to_peak <= 4],
            'near_term': [b for b in all_breakouts if 5 <= b.bars_to_peak <= 10]
        }
        
        for window, events in timing_bins.items():
            if events:
                logger.info(f"{window.upper()} (bars {1 if window=='immediate' else 5}-{4 if window=='immediate' else 10}):")
                logger.info(f"  Breakouts: {len(events)} ({len(events)/len(all_breakouts):.1%} of total)")
                
                # Average magnitude
                avg_mag = np.mean([e.magnitude for e in events])
                logger.info(f"  Avg Magnitude: {avg_mag:.4f}")
        
        # Stream-specific analysis
        logger.info(f"\n{'─'*70}")
        logger.info("STREAM-SPECIFIC PATTERNS")
        logger.info(f"{'─'*70}")
        
        for stream in self.stream_names:
            stream_breakouts = [b for b in all_breakouts if b.stream == stream]
            if not stream_breakouts:
                continue
            
            up_breakouts = [b for b in stream_breakouts if b.direction == 'UP']
            down_breakouts = [b for b in stream_breakouts if b.direction == 'DOWN']
            
            # Prior stream value analysis
            if up_breakouts:
                avg_prior_up = np.mean([b.prior_state.get(stream, 0) for b in up_breakouts])
            else:
                avg_prior_up = np.nan
            
            if down_breakouts:
                avg_prior_down = np.mean([b.prior_state.get(stream, 0) for b in down_breakouts])
            else:
                avg_prior_down = np.nan
            
            logger.info(f"\n{stream}:")
            logger.info(f"  Up-breakouts: {len(up_breakouts)} (avg prior value: {avg_prior_up:+.3f})")
            logger.info(f"  Down-breakouts: {len(down_breakouts)} (avg prior value: {avg_prior_down:+.3f})")
            
            # Momentum vs mean-reversion
            if up_breakouts and not np.isnan(avg_prior_up):
                pct_momentum = sum(1 for b in up_breakouts if b.prior_state.get(stream, 0) > 0.05) / len(up_breakouts)
                logger.info(f"  Up-move from positive: {pct_momentum:.1%} (momentum trading)")
        
        return {
            'date': date,
            'total_bars': len(df),
            'total_breakouts': len(all_breakouts),
            'breakout_rate': len(all_breakouts) / (len(df) * 5),
            'confluence_metrics': confluence,
            'timing_distribution': {
                'immediate': len(timing_bins['immediate']),
                'near_term': len(timing_bins['near_term'])
            }
        }
    
    def run_multi_date_analysis(self, dates: List[str]) -> Dict[str, Any]:
        """Aggregate analysis across multiple dates."""
        logger.info("="*70)
        logger.info("PENTAVIEW TRADING EDGE ANALYSIS")
        logger.info("="*70)
        logger.info(f"Dates: {len(dates)} days ({dates[0]} to {dates[-1]})")
        logger.info(f"Breakout threshold: {self.breakout_threshold}")
        logger.info("="*70)
        
        all_results = []
        
        for date in dates:
            result = self.analyze_single_date(date)
            if result:
                all_results.append(result)
        
        # Aggregate summary
        logger.info(f"\n{'='*70}")
        logger.info("AGGREGATE SUMMARY")
        logger.info(f"{'='*70}")
        
        total_breakouts = sum(r['total_breakouts'] for r in all_results)
        total_bars = sum(r['total_bars'] for r in all_results)
        
        logger.info(f"Total bars analyzed: {total_bars:,}")
        logger.info(f"Total breakouts: {total_breakouts}")
        logger.info(f"Overall breakout rate: {total_breakouts/(total_bars*5):.2%}")
        
        # Aggregate confluence metrics
        all_aligned_correct = 0
        all_aligned_total = 0
        majority_aligned_correct = 0
        majority_aligned_total = 0
        positive_up = 0
        positive_total = 0
        negative_down = 0
        negative_total = 0
        
        for r in all_results:
            conf = r.get('confluence_metrics', {})
            
            if 'all_aligned_accuracy' in conf and 'all_aligned_n' in conf:
                all_aligned_correct += conf['all_aligned_accuracy'] * conf['all_aligned_n']
                all_aligned_total += conf['all_aligned_n']
            
            if 'majority_aligned_accuracy' in conf and 'majority_aligned_n' in conf:
                majority_aligned_correct += conf['majority_aligned_accuracy'] * conf['majority_aligned_n']
                majority_aligned_total += conf['majority_aligned_n']
            
            if 'positive_bias_up_rate' in conf and 'positive_bias_n' in conf:
                positive_up += conf['positive_bias_up_rate'] * conf['positive_bias_n']
                positive_total += conf['positive_bias_n']
            
            if 'negative_bias_down_rate' in conf and 'negative_bias_n' in conf:
                negative_down += conf['negative_bias_down_rate'] * conf['negative_bias_n']
                negative_total += conf['negative_bias_n']
        
        logger.info(f"\nCONFLUENCE EDGE:")
        if all_aligned_total > 0:
            logger.info(f"  All 4 Streams Aligned: {all_aligned_correct/all_aligned_total:.1%} accuracy (n={all_aligned_total})")
        
        if majority_aligned_total > 0:
            logger.info(f"  Majority Aligned (3+): {majority_aligned_correct/majority_aligned_total:.1%} accuracy (n={majority_aligned_total})")
        
        logger.info(f"\nDIRECTIONAL BIAS EDGE:")
        if positive_total > 0:
            logger.info(f"  When Stream > 0.05: {positive_up/positive_total:.1%} move UP (n={positive_total})")
        
        if negative_total > 0:
            logger.info(f"  When Stream < -0.05: {negative_down/negative_total:.1%} move DOWN (n={negative_total})")
        
        # Key Insight Summary
        logger.info(f"\n{'='*70}")
        logger.info("KEY TRADING INSIGHTS")
        logger.info(f"{'='*70}")
        
        if positive_total > 0 and negative_total > 0:
            pos_edge = (positive_up/positive_total) - 0.5  # Edge over coin flip
            neg_edge = (negative_down/negative_total) - 0.5
            
            logger.info(f"✓ Directional Bias Edge:")
            logger.info(f"    Positive streams → UP: {pos_edge:+.1%} edge")
            logger.info(f"    Negative streams → DOWN: {neg_edge:+.1%} edge")
        
        if all_aligned_total > 10:
            aligned_acc = all_aligned_correct/all_aligned_total
            aligned_edge = aligned_acc - 0.5
            logger.info(f"\n✓ Confluence Edge:")
            logger.info(f"    All aligned → Correct direction: {aligned_edge:+.1%} edge")
        
        # Overall assessment
        logger.info(f"\n{'='*70}")
        if positive_total > 0:
            pos_rate = positive_up/positive_total
            if pos_rate > 0.60:
                logger.info("🎯 ACTIONABLE EDGE FOUND:")
                logger.info(f"   When stream strongly positive → {pos_rate:.0%} probability UP-move")
                logger.info("   Trade Setup: Wait for positive alignment + positive bias")
            elif pos_rate > 0.52:
                logger.info("⚠️  WEAK EDGE:")
                logger.info(f"   Slight directional bias ({pos_rate:.1%}) but may not overcome costs")
            else:
                logger.info("❌ NO ACTIONABLE EDGE:")
                logger.info(f"   Direction essentially random ({pos_rate:.1%})")
        
        return {
            'dates_analyzed': len(dates),
            'total_breakouts': total_breakouts,
            'breakout_rate': total_breakouts/(total_bars*5),
            'aggregate_metrics': {
                'all_aligned_accuracy': all_aligned_correct/all_aligned_total if all_aligned_total > 0 else None,
                'majority_aligned_accuracy': majority_aligned_correct/majority_aligned_total if majority_aligned_total > 0 else None,
                'positive_bias_up_rate': positive_up/positive_total if positive_total > 0 else None,
                'negative_bias_down_rate': negative_down/negative_total if negative_total > 0 else None
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Pentaview Trading Edge Analysis")
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--data-root', type=str, default='data', help='Data root directory')
    parser.add_argument('--breakout-threshold', type=float, default=0.10, 
                       help='Minimum 5min move to qualify as breakout')
    parser.add_argument('--canonical-version', type=str, default='v4.0.0', help='Data version')
    
    args = parser.parse_args()
    
    # Determine dates
    if args.date:
        dates = [args.date]
    elif args.start and args.end:
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        dates = pd.date_range(start, end, freq='D').strftime('%Y-%m-%d').tolist()
    else:
        # Default: Dec 18
        dates = ['2025-12-18']
    
    # Run analysis
    analyzer = PentaviewEdgeAnalyzer(
        data_root=Path(args.data_root),
        canonical_version=args.canonical_version,
        breakout_threshold=args.breakout_threshold
    )
    
    results = analyzer.run_multi_date_analysis(dates)
    
    # Save results
    output_dir = Path(args.data_root) / 'ml' / 'ablation_pentaview'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"edge_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

