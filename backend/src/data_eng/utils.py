from __future__ import annotations

from typing import List, Optional
import pandas as pd


def expand_date_range(
    dates: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[str]:
    """
    Expand date arguments into a sorted list of unique date strings (YYYY-MM-DD).

    Supports:
    - dates="2025-06-01,2025-06-02"
    - dates="2025-06-01:2025-06-05" (inclusive range)
    - start_date="2025-06-01", end_date="2025-06-05"
    - Single date via dates="2025-06-01"

    If explicit start/end dates are provided, they take precedence over 'dates'.
    """
    if start_date and end_date:
        # Generate range inclusive
        dt_range = pd.date_range(start=start_date, end=end_date)
        return sorted([d.strftime("%Y-%m-%d") for d in dt_range])

    if not dates:
        return []

    # Check for range syntax "start:end"
    if ":" in dates:
        parts = dates.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {dates}. Expected start:end")
        
        start, end = parts[0].strip(), parts[1].strip()
        dt_range = pd.date_range(start=start, end=end)
        return sorted([d.strftime("%Y-%m-%d") for d in dt_range])

    # Handle comma-separated list
    expanded = set()
    for d in dates.split(","):
        clean_d = d.strip()
        if clean_d:
            expanded.add(clean_d)
            
    return sorted(list(expanded))
