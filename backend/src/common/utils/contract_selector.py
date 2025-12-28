"""
Front-month contract selector for ES futures.

Implements volume-dominant selection to ensure data quality by preventing
roll-period contamination where multiple ES contracts trade simultaneously.

Per the Final Call v1 spec:
- Use volume-dominant selection (no schedule-based assumptions)
- Compute dominance ratio as quality gate
- Apply same chosen symbol to BOTH trades and MBP-10
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import duckdb


@dataclass
class ContractSelection:
    """Result of front-month contract selection for a date."""
    
    date: str  # YYYY-MM-DD
    front_month_symbol: str  # e.g., 'ESZ5'
    dominance_ratio: float  # 0.0-1.0, fraction of volume/trades
    roll_contaminated: bool  # True if dominance < threshold
    
    # Diagnostics
    total_contracts: int  # Number of ES contracts found
    runner_up_symbol: Optional[str] = None  # Second-place contract
    runner_up_ratio: Optional[float] = None


class ContractSelector:
    """
    Select front-month ES contract per date using volume dominance.
    
    This is the authoritative Bronze data quality gate. Mixing ES contracts
    creates phantom liquidity ("ghost walls") and corrupts physics features.
    
    Usage:
        selector = ContractSelector(bronze_root="/path/to/data/lake/bronze")
        selection = selector.select_front_month(date="2025-12-16")
        if selection.roll_contaminated:
            logger.warning(f"{date}: Roll contamination detected")
    """
    
    def __init__(
        self,
        bronze_root: str,
        dominance_threshold: float = 0.60,
        rth_only: bool = True
    ):
        """
        Initialize contract selector.
        
        Args:
            bronze_root: Path to Bronze layer root (e.g., backend/data/lake/bronze)
            dominance_threshold: Minimum volume fraction to pass quality gate (default 0.60)
            rth_only: Count only RTH trades (09:30-13:30 ET) for selection (default True)
        """
        self.bronze_root = Path(bronze_root)
        self.dominance_threshold = dominance_threshold
        self.rth_only = rth_only
        self.duckdb = duckdb.connect(":memory:")
    
    def select_front_month(
        self,
        date: str,
        metric: str = "trade_count"
    ) -> ContractSelection:
        """
        Select front-month ES contract for a date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            metric: Selection metric - "trade_count" (default) or "volume"
        
        Returns:
            ContractSelection with chosen symbol and quality metrics
        
        Raises:
            FileNotFoundError: If no Bronze trades data exists for date
            ValueError: If no valid ES contracts found
        """
        base_dir = self.bronze_root / "futures" / "trades" / "symbol=ES" / f"date={date}"
        
        if not base_dir.exists():
            raise FileNotFoundError(f"No Bronze trades data for {date}: {base_dir}")
        
        parquet_files = list(base_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No trades Parquet files for {date}: {base_dir}")
        
        # Build time filter for RTH if requested
        time_filter = ""
        if self.rth_only:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            # Skip weekend dates (futures don't trade RTH on weekends)
            if date_obj.weekday() < 5:  # Monday=0, Friday=4
                # 09:30-13:30 ET = 14:30-18:30 UTC (approximate, ignoring DST edge cases)
                start_ns = int(
                    datetime(
                        date_obj.year, date_obj.month, date_obj.day,
                        14, 30, tzinfo=timezone.utc
                    ).timestamp() * 1e9
                )
                end_ns = int(
                    datetime(
                        date_obj.year, date_obj.month, date_obj.day,
                        18, 30, tzinfo=timezone.utc
                    ).timestamp() * 1e9
                )
                time_filter = f"WHERE ts_event_ns >= {start_ns} AND ts_event_ns <= {end_ns}"
        
        # Query per-symbol metrics
        pattern = str(base_dir / "**" / "*.parquet")
        
        if metric == "volume":
            metric_col = "SUM(size)"
        else:  # trade_count
            metric_col = "COUNT(*)"
        
        query = f"""
            SELECT
                symbol,
                {metric_col} AS metric_value
            FROM read_parquet('{pattern}', hive_partitioning=false)
            {time_filter}
            GROUP BY symbol
            ORDER BY metric_value DESC
        """
        
        df = self.duckdb.execute(query).fetchdf()
        
        if df.empty:
            raise ValueError(f"No ES contract data found for {date}")
        
        # Extract results
        total_metric = df["metric_value"].sum()
        front_month = df.iloc[0]["symbol"]
        front_month_metric = df.iloc[0]["metric_value"]
        dominance = front_month_metric / total_metric if total_metric > 0 else 0.0
        
        runner_up_symbol = None
        runner_up_ratio = None
        if len(df) > 1:
            runner_up_symbol = df.iloc[1]["symbol"]
            runner_up_ratio = df.iloc[1]["metric_value"] / total_metric
        
        # Quality gate
        roll_contaminated = dominance < self.dominance_threshold
        
        return ContractSelection(
            date=date,
            front_month_symbol=front_month,
            dominance_ratio=dominance,
            roll_contaminated=roll_contaminated,
            total_contracts=len(df),
            runner_up_symbol=runner_up_symbol,
            runner_up_ratio=runner_up_ratio
        )
    
    def select_batch(
        self,
        dates: list[str],
        metric: str = "trade_count"
    ) -> Dict[str, ContractSelection]:
        """
        Select front-month contracts for multiple dates.
        
        Args:
            dates: List of date strings (YYYY-MM-DD)
            metric: Selection metric
        
        Returns:
            Dict mapping date -> ContractSelection
        """
        results = {}
        for date in dates:
            try:
                results[date] = self.select_front_month(date, metric)
            except (FileNotFoundError, ValueError) as e:
                # Log but continue batch
                print(f"WARNING: {date}: {e}")
        return results

