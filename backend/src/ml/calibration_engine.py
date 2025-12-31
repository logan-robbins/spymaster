"""
Calibration Engine for Spymaster.
Measures "Mirror Quality": deviations between predicted probabilities and actual outcome frequencies.
Essential for Phase 4: Vector Optimization.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class CalibrationResult:
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    prob_true: np.ndarray  # y-axis of reliability diagram
    prob_pred: np.ndarray  # x-axis of reliability diagram (bin centers)
    bin_counts: np.ndarray # Number of samples per bin
    brier_score: float     # Mean Squared Error of probs

class CalibrationEngine:
    """
    Computes calibration metrics for "Glass Box" retrieval probabilities.
    """
    
    def __init__(self, n_bins: int = 10, strategy: str = 'uniform'):
        """
        Args:
            n_bins: Number of bins for reliability diagram.
            strategy: 'uniform' (equal width bins) or 'quantile' (equal count bins).
        """
        self.n_bins = n_bins
        self.strategy = strategy

    def compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationResult:
        """
        Compute ECE, MCE, and Reliability Curve data.
        
        Args:
            y_true: Binary ground truth (0 or 1).
            y_prob: Predicted probability of positive class (0.0 to 1.0).
            
        Returns:
            CalibrationResult object.
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        if len(y_true) != len(y_prob):
            raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_prob)}")
            
        if self.strategy == 'quantile':
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            bins = np.percentile(y_prob, quantiles * 100)
            bins[-1] = 1.0 + 1e-8 # Ensure inclusion
        else:
            bins = np.linspace(0.0, 1.0, self.n_bins + 1)
            
        binids = np.digitize(y_prob, bins) - 1
        
        bin_sums = np.bincount(binids, weights=y_prob, minlength=self.n_bins)
        bin_true = np.bincount(binids, weights=y_true, minlength=self.n_bins)
        bin_total = np.bincount(binids, minlength=self.n_bins)
        
        nonzero = bin_total > 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        counts = bin_total[nonzero]
        
        # Expected Calibration Error (weighted average of absolute difference)
        ece = np.sum(counts * np.abs(prob_true - prob_pred)) / np.sum(counts)
        
        # Maximum Calibration Error (max absolute difference)
        mce = np.max(np.abs(prob_true - prob_pred))
        
        # Brier Score
        brier = np.mean((y_prob - y_true) ** 2)
        
        return CalibrationResult(
            ece=ece,
            mce=mce,
            prob_true=prob_true,
            prob_pred=prob_pred,
            bin_counts=counts,
            brier_score=brier
        )

    def plot_reliability_curve(self, result: CalibrationResult, title: str = "Reliability Diagram", save_path: Optional[str] = None):
        """
        Plot the reliability diagram.
        """
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        
        # Plot calibration curve
        plt.plot(result.prob_pred, result.prob_true, "s-", label=f"Model (ECE={result.ece:.3f})")
        
        # Plot histogram of predictions (small subplot or twinx)
        # For simplicity, just the curve for now.
        
        plt.ylabel("Fraction of Positives (Actual)")
        plt.xlabel("Mean Predicted Value (Probability)")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def compute_calibration_report(df: pd.DataFrame, prob_col: str, outcome_col: str) -> Dict[str, Any]:
    """
    Helper to compute calibration from a dataframe.
    """
    # Convert outcome to binary (assuming outcome_col is string like "BREAK")
    # This logic assumes we are predicting "BREAK".
    # Need to verify if the model predicts BREAK probability.
    
    # If outcome_col contains 'BREAK', 'BOUNCE', etc.
    y_true = (df[outcome_col] == 'BREAK').astype(int)
    y_prob = df[prob_col]
    
    engine = CalibrationEngine()
    result = engine.compute_metrics(y_true, y_prob)
    
    return {
        "ece": result.ece,
        "mce": result.mce,
        "brier": result.brier_score,
        "prob_true": result.prob_true.tolist(),
        "prob_pred": result.prob_pred.tolist(),
        "bin_counts": result.bin_counts.tolist()
    }
