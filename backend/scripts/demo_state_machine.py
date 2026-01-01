"""
Demo script for stream state machine.

Demonstrates:
1. Alert detection on synthetic stream bars
2. Position-aware exit scoring
3. State machine with hysteresis

Usage:
    uv run python -m scripts.demo_state_machine
"""
import argparse
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

from src.ml.stream_state_machine import (
    detect_alerts, compute_exit_score, StreamStateMachine,
    AlertType, AlertSeverity
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_scenarios() -> Dict[str, pd.Series]:
    """Create synthetic test scenarios for different alert patterns."""
    scenarios = {}
    
    # ========== EXHAUSTION UP ==========
    scenarios['exhaustion_up'] = pd.Series({
        'timestamp': pd.Timestamp('2025-12-16 09:45:00'),
        'level_kind': 'OR_HIGH',
        'direction': 'UP',
        'sigma_p': 0.45,  # High pressure
        'sigma_p_slope': -0.08,  # But slowing (negative slope)
        'sigma_p_curvature': -0.02,
        'sigma_p_jerk': -0.025,  # Accelerating slowdown
        'sigma_m': 0.40,
        'sigma_f': 0.38,
        'sigma_b': 0.25,
        'sigma_d': 0.15,
        'sigma_s': 0.35,
        'sigma_m_slope': -0.05,
        'sigma_f_slope': -0.06,
        'sigma_b_slope': -0.03,
        'alignment_adj': 0.35
    })
    
    # ========== CONTINUATION DOWN ==========
    scenarios['continuation_down'] = pd.Series({
        'timestamp': pd.Timestamp('2025-12-16 10:00:00'),
        'level_kind': 'PM_LOW',
        'direction': 'DOWN',
        'sigma_p': -0.50,  # Strong selling
        'sigma_p_slope': -0.12,  # And accelerating (more negative)
        'sigma_p_curvature': -0.01,
        'sigma_p_jerk': 0.00,
        'sigma_m': -0.48,
        'sigma_f': -0.52,
        'sigma_b': -0.30,
        'sigma_d': 0.20,  # Fuel regime (amplifies)
        'sigma_s': 0.40,  # Good setup
        'sigma_m_slope': -0.10,
        'sigma_f_slope': -0.13,
        'sigma_b_slope': -0.08,
        'alignment_adj': -0.48
    })
    
    # ========== FLOW DIVERGENCE ==========
    scenarios['flow_divergence'] = pd.Series({
        'timestamp': pd.Timestamp('2025-12-16 10:15:00'),
        'level_kind': 'SMA_90',
        'direction': 'UP',
        'sigma_p': 0.10,
        'sigma_p_slope': 0.02,
        'sigma_p_curvature': 0.00,
        'sigma_p_jerk': 0.00,
        'sigma_m': 0.40,  # Price drifting up
        'sigma_f': -0.45,  # But sell aggression dominates!
        'sigma_b': -0.10,
        'sigma_d': -0.10,
        'sigma_s': 0.15,
        'sigma_m_slope': 0.03,
        'sigma_f_slope': -0.08,
        'sigma_b_slope': 0.00,
        'alignment_adj': 0.05
    })
    
    # ========== BARRIER OPPOSES PRESSURE ==========
    scenarios['barrier_opposition'] = pd.Series({
        'timestamp': pd.Timestamp('2025-12-16 10:30:00'),
        'level_kind': 'OR_LOW',
        'direction': 'DOWN',
        'sigma_p': -0.42,  # Selling pressure
        'sigma_p_slope': -0.05,
        'sigma_p_curvature': 0.00,
        'sigma_p_jerk': 0.00,
        'sigma_m': -0.38,
        'sigma_f': -0.40,
        'sigma_b': 0.35,  # But barrier favors upside!
        'sigma_d': 0.00,
        'sigma_s': 0.30,
        'sigma_m_slope': -0.04,
        'sigma_f_slope': -0.05,
        'sigma_b_slope': 0.02,
        'alignment_adj': -0.15
    })
    
    # ========== LOW QUALITY SETUP ==========
    scenarios['low_quality'] = pd.Series({
        'timestamp': pd.Timestamp('2025-12-16 10:45:00'),
        'level_kind': 'EMA_20',
        'direction': 'UP',
        'sigma_p': 0.25,
        'sigma_p_slope': 0.03,
        'sigma_p_curvature': 0.00,
        'sigma_p_jerk': 0.00,
        'sigma_m': 0.22,
        'sigma_f': 0.28,
        'sigma_b': 0.15,
        'sigma_d': -0.35,  # Pin regime
        'sigma_s': -0.45,  # Very poor setup quality!
        'sigma_m_slope': 0.02,
        'sigma_f_slope': 0.03,
        'sigma_b_slope': 0.01,
        'alignment_adj': 0.10
    })
    
    return scenarios


def demo_alert_detection():
    """Demonstrate alert detection on test scenarios."""
    logger.info("="*70)
    logger.info("STREAM STATE MACHINE DEMO: Alert Detection")
    logger.info("="*70)
    
    scenarios = create_test_scenarios()
    
    for scenario_name, bar in scenarios.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Scenario: {scenario_name.upper().replace('_', ' ')}")
        logger.info(f"{'='*70}")
        logger.info(f"Timestamp: {bar['timestamp']}")
        logger.info(f"Level: {bar['level_kind']} ({bar['direction']})")
        logger.info(f"Stream values:")
        logger.info(f"  Pressure (P): {bar['sigma_p']:.3f}")
        logger.info(f"  Momentum (M): {bar['sigma_m']:.3f}")
        logger.info(f"  Flow (F): {bar['sigma_f']:.3f}")
        logger.info(f"  Barrier (B): {bar['sigma_b']:.3f}")
        logger.info(f"  Dealer (D): {bar['sigma_d']:.3f}")
        logger.info(f"  Setup (S): {bar['sigma_s']:.3f}")
        
        # Detect alerts
        alerts = detect_alerts(bar)
        
        if alerts:
            logger.info(f"\n📢 DETECTED ALERTS ({len(alerts)}):")
            for alert in alerts:
                severity_icon = {
                    'CRITICAL': '🔴',
                    'WARNING': '⚠️ ',
                    'INFO': 'ℹ️ '
                }.get(alert.severity, '•')
                
                logger.info(f"  {severity_icon} [{alert.severity}] {alert.alert_type}")
                logger.info(f"      {alert.message}")
                logger.info(f"      Confidence: {alert.confidence:.2f}")
        else:
            logger.info("\n  No alerts triggered")


def demo_exit_scoring():
    """Demonstrate position-aware exit scoring."""
    logger.info("\n\n" + "="*70)
    logger.info("STREAM STATE MACHINE DEMO: Position-Aware Exit Scoring")
    logger.info("="*70)
    
    scenarios = create_test_scenarios()
    
    # Test LONG position on various scenarios
    logger.info("\n--- LONG POSITION ---")
    for scenario_name, bar in scenarios.items():
        exit_result = compute_exit_score(bar, position_sign=+1.0)
        
        zone_colors = {
            'HOLD_ADD': '🟢',
            'HOLD_TRAIL': '🟡',
            'REDUCE': '🟠',
            'EXIT': '🔴'
        }
        icon = zone_colors.get(exit_result['zone'], '•')
        
        logger.info(f"\n{scenario_name.replace('_', ' ').title()}:")
        logger.info(f"  {icon} Exit Score: {exit_result['exit_score']:+.3f}")
        logger.info(f"  Zone: {exit_result['zone']}")
        logger.info(f"  Recommendation: {exit_result['recommendation']}")
        if exit_result['jerk_penalty'] > 0:
            logger.info(f"  ⚡ Jerk penalty applied: {exit_result['jerk_penalty']:.3f}")
    
    # Test SHORT position
    logger.info("\n\n--- SHORT POSITION ---")
    # Use continuation_down scenario (good for shorts)
    bar = scenarios['continuation_down']
    exit_result = compute_exit_score(bar, position_sign=-1.0)
    
    icon = zone_colors.get(exit_result['zone'], '•')
    logger.info(f"Continuation Down (SHORT):")
    logger.info(f"  {icon} Exit Score: {exit_result['exit_score']:+.3f}")
    logger.info(f"  Zone: {exit_result['zone']}")
    logger.info(f"  Recommendation: {exit_result['recommendation']}")


def demo_state_machine_hysteresis():
    """Demonstrate state machine with hysteresis."""
    logger.info("\n\n" + "="*70)
    logger.info("STREAM STATE MACHINE DEMO: Hysteresis (Sustained Alerts)")
    logger.info("="*70)
    
    # Create time series with evolving exhaustion pattern
    base_time = pd.Timestamp('2025-12-16 09:30:00')
    state_machine = StreamStateMachine(hold_time_seconds=3.0)
    
    logger.info("\nSimulating 8-second evolution (1 bar/second):")
    logger.info("Pressure starts high, slope gradually goes negative...")
    
    for i in range(8):
        ts = base_time + pd.Timedelta(seconds=i)
        ts_ns = int(ts.value)
        
        # Evolving bar: pressure high but slope becoming more negative
        bar = pd.Series({
            'timestamp': ts,
            'sigma_p': 0.45,
            'sigma_p_slope': -0.02 * (i / 2),  # Gradually more negative
            'sigma_p_curvature': -0.01,
            'sigma_p_jerk': -0.025,
            'sigma_m': 0.40,
            'sigma_f': 0.38,
            'sigma_b': 0.25,
            'sigma_d': 0.15,
            'sigma_s': 0.35,
            'sigma_m_slope': -0.03,
            'sigma_f_slope': -0.04,
            'sigma_b_slope': -0.02,
            'alignment_adj': 0.32
        })
        
        sustained_alerts = state_machine.update(bar, ts_ns)
        instant_alerts = detect_alerts(bar)
        
        logger.info(f"\n  t={i}s: slope={bar['sigma_p_slope']:+.3f}")
        logger.info(f"    Instant alerts: {len(instant_alerts)}")
        logger.info(f"    Sustained alerts (>3s): {len(sustained_alerts)}")
        
        if sustained_alerts:
            for alert in sustained_alerts:
                logger.info(f"      ✓ {alert.alert_type} SUSTAINED")


def main() -> int:
    parser = argparse.ArgumentParser(description='Demo stream state machine')
    parser.add_argument('--section', type=str, default='all',
                       choices=['all', 'alerts', 'exit', 'hysteresis'],
                       help='Which demo to run')
    
    args = parser.parse_args()
    
    if args.section in ['all', 'alerts']:
        demo_alert_detection()
    
    if args.section in ['all', 'exit']:
        demo_exit_scoring()
    
    if args.section in ['all', 'hysteresis']:
        demo_state_machine_hysteresis()
    
    logger.info("\n\n" + "="*70)
    logger.info("✓ Demo complete!")
    logger.info("="*70)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

