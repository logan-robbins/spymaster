"""
Bronze data quality assurance utilities.

QA Gates for Bronze data validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .contract_selector import ContractSelector, ContractSelection


@dataclass
class BronzeQAReport:
    """Quality assurance report for Bronze ES futures data."""
    
    date: str
    contract_selection: Optional[ContractSelection]
    
    # Gate results
    front_month_purity_pass: bool
    dominance_ratio: float
    roll_contaminated: bool
    
    # Diagnostics
    trades_count: int
    mbp10_count: int
    price_min: float
    price_max: float
    spy_equiv_min: float
    spy_equiv_max: float
    
    # Warnings
    warnings: List[str]
    
    def __str__(self) -> str:
        """Human-readable report."""
        lines = [
            f"Bronze QA Report: {self.date}",
            "=" * 60,
            f"Front-Month Selection:",
            f"  Symbol: {self.contract_selection.front_month_symbol if self.contract_selection else 'N/A'}",
            f"  Dominance: {self.dominance_ratio:.1%}",
            f"  Roll Contaminated: {self.roll_contaminated}",
            f"  Gate: {'PASS' if self.front_month_purity_pass else 'FAIL'}",
            "",
            f"Data Counts:",
            f"  Trades: {self.trades_count:,}",
            f"  MBP-10 Snapshots: {self.mbp10_count:,}",
            "",
            f"Price Range:",
            f"  ES: ${self.price_min:.2f} - ${self.price_max:.2f}",
            f"  SPY Equiv: ${self.spy_equiv_min:.2f} - ${self.spy_equiv_max:.2f}",
        ]
        
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)


class BronzeQA:
    """
    Bronze data quality assurance checker.
    
    Implements QA gates for Bronze data validation.
    """
    
    def __init__(self, bronze_root: str):
        """
        Initialize QA checker.
        
        Args:
            bronze_root: Path to Bronze layer root
        """
        self.bronze_root = bronze_root
        self.selector = ContractSelector(bronze_root)
    
    def check_date(
        self,
        date: str,
        trades_df: Optional[pd.DataFrame] = None,
        mbp10_df: Optional[pd.DataFrame] = None
    ) -> BronzeQAReport:
        """
        Run QA checks for a date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            trades_df: Optional pre-loaded trades DataFrame
            mbp10_df: Optional pre-loaded MBP-10 DataFrame
        
        Returns:
            BronzeQAReport with gate results and diagnostics
        """
        warnings = []
        
        # Gate 1: Front-month purity
        try:
            selection = self.selector.select_front_month(date)
            front_month_purity_pass = not selection.roll_contaminated
            dominance_ratio = selection.dominance_ratio
            roll_contaminated = selection.roll_contaminated
            
            if roll_contaminated:
                warnings.append(
                    f"Roll contamination detected: {selection.front_month_symbol} "
                    f"dominance {dominance_ratio:.1%} < {self.selector.dominance_threshold:.1%}"
                )
                if selection.runner_up_symbol:
                    warnings.append(
                        f"Runner-up contract {selection.runner_up_symbol} "
                        f"has {selection.runner_up_ratio:.1%} of volume"
                    )
        except Exception as e:
            selection = None
            front_month_purity_pass = False
            dominance_ratio = 0.0
            roll_contaminated = True
            warnings.append(f"Front-month selection failed: {e}")
        
        # Load data if not provided
        if trades_df is None:
            from src.lake.bronze_writer import BronzeReader
            reader = BronzeReader(data_root=self.bronze_root.replace('/bronze', ''))
            trades_df = reader.read_futures_trades(
                symbol='ES',
                date=date,
                front_month_only=True
            )
        
        if mbp10_df is None:
            from src.lake.bronze_writer import BronzeReader
            reader = BronzeReader(data_root=self.bronze_root.replace('/bronze', ''))
            mbp10_df = reader.read_futures_mbp10(
                symbol='ES',
                date=date,
                front_month_only=True
            )
        
        # Data counts
        trades_count = len(trades_df)
        mbp10_count = len(mbp10_df)
        
        if trades_count == 0:
            warnings.append("No trades data found")
        if mbp10_count == 0:
            warnings.append("No MBP-10 data found")
        
        # Price range
        if not trades_df.empty and 'price' in trades_df.columns:
            price_min = trades_df['price'].min()
            price_max = trades_df['price'].max()
            spy_equiv_min = price_min / 10.0
            spy_equiv_max = price_max / 10.0
            
            # Sanity checks
            if price_min < 3000 or price_max > 10000:
                warnings.append(
                    f"ES price range ${price_min:.2f}-${price_max:.2f} "
                    f"outside expected bounds [3000, 10000]"
                )
        else:
            price_min = 0.0
            price_max = 0.0
            spy_equiv_min = 0.0
            spy_equiv_max = 0.0
        
        return BronzeQAReport(
            date=date,
            contract_selection=selection,
            front_month_purity_pass=front_month_purity_pass,
            dominance_ratio=dominance_ratio,
            roll_contaminated=roll_contaminated,
            trades_count=trades_count,
            mbp10_count=mbp10_count,
            price_min=price_min,
            price_max=price_max,
            spy_equiv_min=spy_equiv_min,
            spy_equiv_max=spy_equiv_max,
            warnings=warnings
        )
    
    def check_batch(self, dates: List[str]) -> Dict[str, BronzeQAReport]:
        """
        Run QA checks for multiple dates.
        
        Args:
            dates: List of date strings (YYYY-MM-DD)
        
        Returns:
            Dict mapping date -> BronzeQAReport
        """
        reports = {}
        for date in dates:
            try:
                reports[date] = self.check_date(date)
            except Exception as e:
                print(f"ERROR: QA check failed for {date}: {e}")
        return reports

