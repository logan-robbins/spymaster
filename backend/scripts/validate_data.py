"""
Data Validation Script for Spymaster Signal Data

Validates the generated signals parquet file for:
- Missing values (NaN/None)
- Zero distributions per feature
- Categorical value validity
- Numeric range sanity
- Label distribution balance
- Feature correlations with outcomes

Usage:
    cd backend/
    uv run python -m scripts.validate_data
    uv run python -m scripts.validate_data --verbose
    uv run python -m scripts.validate_data --input path/to/signals.parquet
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    severity: str  # "ERROR", "WARNING", "INFO"
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    file_path: str
    total_signals: int
    results: List[ValidationResult]

    @property
    def errors(self) -> List[ValidationResult]:
        return [r for r in self.results if r.severity == "ERROR"]

    @property
    def warnings(self) -> List[ValidationResult]:
        return [r for r in self.results if r.severity == "WARNING"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


class DataValidator:
    """Validates signal data quality."""

    # Expected columns and their properties
    REQUIRED_COLUMNS = [
        'ts_ns', 'level_price', 'level_kind', 'level_kind_name',
        'direction', 'distance', 'spot', 'date', 'symbol', 'outcome'
    ]

    NUMERIC_COLUMNS = {
        'spot': {'min': 400.0, 'max': 1000.0, 'allow_zero': False},
        'level_price': {'min': 400.0, 'max': 1000.0, 'allow_zero': False},
        'distance': {'min': 0.0, 'max': 50.0, 'allow_zero': True},
        'gamma_exposure': {'min': -1e9, 'max': 1e9, 'allow_zero': True},
        'tape_imbalance': {'min': -1.0, 'max': 1.0, 'allow_zero': True},
        'tape_velocity': {'min': -10000.0, 'max': 10000.0, 'allow_zero': True},
        'wall_ratio': {'min': 0.0, 'max': 100.0, 'allow_zero': True},
        'barrier_delta_liq': {'min': -1e6, 'max': 1e6, 'allow_zero': True},
        'future_price_5min': {'min': 400.0, 'max': 1000.0, 'allow_zero': False},
        'excursion_max': {'min': 0.0, 'max': 50.0, 'allow_zero': True},
        'excursion_min': {'min': 0.0, 'max': 50.0, 'allow_zero': True},
    }

    CATEGORICAL_COLUMNS = {
        'direction': ['UP', 'DOWN'],
        'outcome': ['BREAK', 'BOUNCE', 'CHOP', 'UNDEFINED'],
        'fuel_effect': ['AMPLIFY', 'DAMPEN', 'NEUTRAL'],
        'barrier_state': ['VACUUM', 'WALL', 'ABSORPTION', 'CONSUMED', 'WEAK', 'NEUTRAL'],
        'level_kind_name': [
            'PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SESSION_HIGH', 'SESSION_LOW',
            'SMA_200', 'SMA_400', 'VWAP', 'ROUND', 'STRIKE', 'CALL_WALL', 'PUT_WALL'
        ],
        'symbol': ['SPY'],
    }

    # Features that should have non-zero values for meaningful analysis
    SHOULD_HAVE_NONZERO = {
        'gamma_exposure': 0.80,  # At least 80% should be non-zero
        'tape_imbalance': 0.10,  # At least 10% should be non-zero
        'tape_velocity': 0.10,
        'wall_ratio': 0.03,
    }

    ALLOWED_MISSING_COLUMNS = {
        'sma_200',
        'sma_400',
        'dist_to_sma_200',
        'dist_to_sma_400',
        'sma_200_slope',
        'sma_400_slope',
        'sma_200_slope_5bar',
        'sma_400_slope_5bar',
        'sma_spread',
        'mean_reversion_pressure_200',
        'mean_reversion_pressure_400',
        'mean_reversion_velocity_200',
        'mean_reversion_velocity_400',
        'confluence_min_distance',
        'time_to_threshold_1',
        'time_to_threshold_2',
        'future_price_5min',
        'excursion_max',
        'excursion_min',
        'strength_signed',
        'strength_abs',
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def validate(self, df: pd.DataFrame, file_path: str) -> ValidationReport:
        """Run all validation checks on the dataframe."""
        self.results = []

        # Basic checks
        self._check_row_count(df)
        self._check_required_columns(df)
        self._check_missing_values(df)

        # Data type checks
        self._check_numeric_ranges(df)
        self._check_categorical_values(df)

        # Distribution checks
        self._check_zero_distributions(df)
        self._check_label_balance(df)
        self._check_date_distribution(df)

        # Sanity checks
        self._check_spot_level_consistency(df)
        self._check_timestamp_ordering(df)
        self._check_nan_inf_values(df)

        return ValidationReport(
            file_path=file_path,
            total_signals=len(df),
            results=self.results
        )

    def _add_result(self, check_name: str, passed: bool, severity: str,
                   message: str, details: Optional[Dict] = None):
        """Add a validation result."""
        self.results.append(ValidationResult(
            check_name=check_name,
            passed=passed,
            severity=severity,
            message=message,
            details=details
        ))

    def _check_row_count(self, df: pd.DataFrame):
        """Check that we have a reasonable number of signals."""
        n = len(df)

        if n == 0:
            self._add_result("row_count", False, "ERROR", "DataFrame is empty")
        elif n < 100:
            self._add_result("row_count", False, "WARNING",
                           f"Only {n} signals - may be insufficient for training")
        else:
            self._add_result("row_count", True, "INFO",
                           f"{n:,} signals found")

    def _check_required_columns(self, df: pd.DataFrame):
        """Check all required columns exist."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]

        if missing:
            self._add_result("required_columns", False, "ERROR",
                           f"Missing required columns: {missing}")
        else:
            self._add_result("required_columns", True, "INFO",
                           f"All {len(self.REQUIRED_COLUMNS)} required columns present")

    def _check_missing_values(self, df: pd.DataFrame):
        """Check for missing values in each column."""
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if len(cols_with_missing) > 0:
            details = {col: int(count) for col, count in cols_with_missing.items()}
            pct_missing = {col: f"{count/len(df)*100:.1f}%"
                          for col, count in cols_with_missing.items()}

            # Critical columns should have no missing values
            critical_missing = [col for col in ['outcome', 'spot', 'level_price']
                              if col in cols_with_missing.index]

            allowed_missing = [col for col in cols_with_missing.index if col in self.ALLOWED_MISSING_COLUMNS]
            unexpected_missing = [col for col in cols_with_missing.index if col not in self.ALLOWED_MISSING_COLUMNS]

            if critical_missing:
                self._add_result("missing_values", False, "ERROR",
                               f"Critical columns have missing values: {critical_missing}",
                               details=details)
            elif unexpected_missing:
                self._add_result("missing_values", False, "WARNING",
                               f"Columns with missing values: {unexpected_missing}",
                               details={col: pct_missing[col] for col in unexpected_missing})
            else:
                self._add_result("missing_values", True, "INFO",
                               "Only expected columns have missing values")

            if allowed_missing:
                self._add_result("missing_values_expected", True, "INFO",
                               f"Expected missing values in: {allowed_missing}",
                               details={col: pct_missing[col] for col in allowed_missing})
        else:
            self._add_result("missing_values", True, "INFO",
                           "No missing values found")

    def _check_numeric_ranges(self, df: pd.DataFrame):
        """Check numeric columns are within expected ranges."""
        violations = []

        for col, constraints in self.NUMERIC_COLUMNS.items():
            if col not in df.columns:
                continue

            series = df[col]
            valid = series.notna()

            if not valid.any():
                continue

            min_val = series[valid].min()
            max_val = series[valid].max()

            if min_val < constraints['min']:
                violations.append(f"{col}: min={min_val:.4f} < expected {constraints['min']}")
            if max_val > constraints['max']:
                violations.append(f"{col}: max={max_val:.4f} > expected {constraints['max']}")

        if violations:
            self._add_result("numeric_ranges", False, "WARNING",
                           f"Range violations found",
                           details={'violations': violations})
        else:
            self._add_result("numeric_ranges", True, "INFO",
                           "All numeric columns within expected ranges")

    def _check_categorical_values(self, df: pd.DataFrame):
        """Check categorical columns have valid values."""
        violations = []

        for col, valid_values in self.CATEGORICAL_COLUMNS.items():
            if col not in df.columns:
                continue

            unique_vals = df[col].dropna().unique()
            invalid = [v for v in unique_vals if v not in valid_values]

            if invalid:
                violations.append(f"{col}: unexpected values {invalid}")

        if violations:
            self._add_result("categorical_values", False, "WARNING",
                           f"Invalid categorical values found",
                           details={'violations': violations})
        else:
            self._add_result("categorical_values", True, "INFO",
                           "All categorical columns have valid values")

    def _check_zero_distributions(self, df: pd.DataFrame):
        """Check that features aren't all zeros."""
        zero_columns = []
        low_nonzero = []

        for col, min_nonzero in self.SHOULD_HAVE_NONZERO.items():
            if col not in df.columns:
                continue

            series = df[col]
            nonzero_pct = (series != 0).sum() / len(series)

            if nonzero_pct == 0:
                zero_columns.append(col)
            elif nonzero_pct < min_nonzero:
                low_nonzero.append(f"{col}: {nonzero_pct*100:.1f}% non-zero (expected >{min_nonzero*100:.0f}%)")

        if zero_columns:
            self._add_result("zero_distribution", False, "ERROR",
                           f"Columns with ALL zeros (data quality issue): {zero_columns}")
        elif low_nonzero:
            self._add_result("zero_distribution", False, "WARNING",
                           f"Columns with low non-zero rates",
                           details={'low_nonzero': low_nonzero})
        else:
            self._add_result("zero_distribution", True, "INFO",
                           "Feature distributions look healthy")

    def _check_label_balance(self, df: pd.DataFrame):
        """Check outcome label distribution is not severely imbalanced."""
        if 'outcome' not in df.columns:
            return

        dist = df['outcome'].value_counts(normalize=True)

        break_bounce = dist.reindex(['BREAK', 'BOUNCE']).dropna()
        if len(break_bounce) < 2:
            self._add_result("label_balance", False, "ERROR",
                           f"Only one of BREAK/BOUNCE labels found: {list(break_bounce.index)}")
            return

        min_pct = break_bounce.min()
        if min_pct < 0.05:
            self._add_result("label_balance", False, "WARNING",
                           f"Severe BREAK/BOUNCE imbalance: minority at {min_pct*100:.1f}%",
                           details=dist.to_dict())
        elif min_pct < 0.20:
            self._add_result("label_balance", False, "WARNING",
                           f"Moderate BREAK/BOUNCE imbalance: {break_bounce.to_dict()}",
                           details=dist.to_dict())
        else:
            self._add_result("label_balance", True, "INFO",
                           f"BREAK/BOUNCE distribution: {break_bounce.to_dict()}",
                           details=dist.to_dict())

    def _check_date_distribution(self, df: pd.DataFrame):
        """Check signals are distributed across dates."""
        if 'date' not in df.columns:
            return

        date_counts = df['date'].value_counts().sort_index()
        n_dates = len(date_counts)

        if n_dates < 2:
            self._add_result("date_distribution", False, "WARNING",
                           f"Only {n_dates} date(s) in data - may limit generalization")
        else:
            # Check for highly uneven distribution
            cv = date_counts.std() / date_counts.mean()  # Coefficient of variation
            if cv > 0.5:
                self._add_result("date_distribution", False, "WARNING",
                               f"Uneven date distribution (CV={cv:.2f})",
                               details=date_counts.to_dict())
            else:
                self._add_result("date_distribution", True, "INFO",
                               f"{n_dates} dates: {list(date_counts.index)}")

    def _check_spot_level_consistency(self, df: pd.DataFrame):
        """Check that spot and level_price are close to each other."""
        if 'spot' not in df.columns or 'level_price' not in df.columns:
            return

        # Distance should generally be less than $20
        distance = (df['spot'] - df['level_price']).abs()
        large_distance = (distance > 20).sum()

        if large_distance > 0:
            pct = large_distance / len(df) * 100
            if pct > 10:
                self._add_result("spot_level_consistency", False, "WARNING",
                               f"{pct:.1f}% of signals have spot-level distance > $20")
            else:
                self._add_result("spot_level_consistency", True, "INFO",
                               f"Spot-level consistency OK (only {pct:.1f}% with distance > $20)")
        else:
            self._add_result("spot_level_consistency", True, "INFO",
                           "All signals have reasonable spot-level distance")

    def _check_timestamp_ordering(self, df: pd.DataFrame):
        """Check that timestamps are properly ordered within each date."""
        if 'ts_ns' not in df.columns or 'date' not in df.columns:
            return

        issues = []
        for date in df['date'].unique():
            date_df = df[df['date'] == date]
            ts_values = date_df['ts_ns'].values
            if not np.all(ts_values[:-1] <= ts_values[1:]):
                issues.append(date)

        if issues:
            self._add_result("timestamp_ordering", False, "WARNING",
                           f"Timestamps not monotonically increasing for dates: {issues}")
        else:
            self._add_result("timestamp_ordering", True, "INFO",
                           "Timestamps properly ordered within each date")

    def _check_nan_inf_values(self, df: pd.DataFrame):
        """Check for NaN and Inf values in numeric columns."""
        inf_cols = []

        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)

        if inf_cols:
            self._add_result("inf_values", False, "ERROR",
                           f"Columns with Inf values: {inf_cols}")
        else:
            self._add_result("inf_values", True, "INFO",
                           "No Inf values found in numeric columns")


def print_report(report: ValidationReport, verbose: bool = False):
    """Print validation report to console."""
    print("\n" + "=" * 70)
    print("DATA VALIDATION REPORT")
    print("=" * 70)
    print(f"File: {report.file_path}")
    print(f"Total signals: {report.total_signals:,}")
    print()

    # Summary
    n_errors = len(report.errors)
    n_warnings = len(report.warnings)
    n_passed = sum(1 for r in report.results if r.passed)

    status = "PASSED" if report.passed else "FAILED"
    status_color = "\033[92m" if report.passed else "\033[91m"
    reset_color = "\033[0m"

    print(f"Status: {status_color}{status}{reset_color}")
    print(f"Checks: {n_passed} passed, {n_errors} errors, {n_warnings} warnings")
    print()

    # Errors
    if report.errors:
        print("\033[91mERRORS:\033[0m")
        for r in report.errors:
            print(f"  [{r.check_name}] {r.message}")
            if verbose and r.details:
                for k, v in r.details.items():
                    print(f"    {k}: {v}")
        print()

    # Warnings
    if report.warnings:
        print("\033[93mWARNINGS:\033[0m")
        for r in report.warnings:
            print(f"  [{r.check_name}] {r.message}")
            if verbose and r.details:
                for k, v in r.details.items():
                    print(f"    {k}: {v}")
        print()

    # Info (only in verbose mode)
    if verbose:
        info_results = [r for r in report.results if r.severity == "INFO" and r.passed]
        if info_results:
            print("\033[94mINFO:\033[0m")
            for r in info_results:
                print(f"  [{r.check_name}] {r.message}")
            print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Validate Spymaster signal data quality'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/lake/gold/research/signals_vectorized.parquet',
        help='Path to signals parquet file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    file_path = Path(args.input)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return 1

    # Load data
    print(f"Loading {file_path}...")
    df = pq.read_table(str(file_path)).to_pandas()

    # Validate
    validator = DataValidator(verbose=args.verbose)
    report = validator.validate(df, str(file_path))

    # Output
    if args.json:
        output = {
            'file_path': report.file_path,
            'total_signals': report.total_signals,
            'passed': report.passed,
            'results': [
                {
                    'check': r.check_name,
                    'passed': r.passed,
                    'severity': r.severity,
                    'message': r.message,
                    'details': r.details
                }
                for r in report.results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
