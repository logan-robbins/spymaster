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


# Product type categories for session window selection
_EQUITY_PRODUCT_TYPES = frozenset({"equity_mbo", "equity_option_cmbp_1"})
_FUTURES_PRODUCT_TYPES = frozenset({"future_mbo", "future_option_mbo"})


def session_window_ns(
    session_date: str, product_type: str = "equity_mbo"
) -> tuple[int, int]:
    """Return (start_ns, end_ns) for bronze ingestion window.

    Windows are product-type-aware to capture venue-specific session starts:

    **Equities** (``equity_mbo``, ``equity_option_cmbp_1``):
        02:00–16:00 ET (EST, UTC-5).  XNAS session begins ~03:05 ET with a
        Clear record (``action=R``) followed by Add events for every resting
        order.  The 02:00 ET start provides a safe buffer before the earliest
        observed session start.  End at 16:00 ET covers full Regular Trading
        Hours close.

    **Futures** (``future_mbo``, ``future_option_mbo``):
        00:00 UTC – next-day 00:00 UTC (exclusive).  CME Globex runs nearly
        24 hours; Databento provides a synthetic MBO snapshot at 00:00 UTC
        with ``F_SNAPSHOT=32`` flag.  Capturing from midnight ensures the
        daily snapshot is included.

    Args:
        session_date: ISO-8601 date string (``YYYY-MM-DD``).
        product_type: One of the four canonical product types.  Defaults to
            ``"equity_mbo"`` for backward compatibility.

    Returns:
        Tuple of ``(start_ns, end_ns)`` in UTC nanoseconds since epoch.
        The end boundary is exclusive (callers use ``ts_event < end_ns``).
    """
    if product_type in _FUTURES_PRODUCT_TYPES:
        # Full UTC day: 00:00 UTC to next-day 00:00 UTC (exclusive)
        start_utc = pd.Timestamp(f"{session_date} 00:00:00", tz="UTC")
        end_utc = start_utc + pd.Timedelta(days=1)
        return int(start_utc.value), int(end_utc.value)

    if product_type not in _EQUITY_PRODUCT_TYPES:
        raise ValueError(
            f"Unknown product_type={product_type!r}. "
            f"Expected one of {sorted(_EQUITY_PRODUCT_TYPES | _FUTURES_PRODUCT_TYPES)}"
        )

    # Equities: 02:00 ET – 16:00 ET (America/New_York handles EST/EDT correctly)
    start_local = pd.Timestamp(f"{session_date} 02:00:00", tz="America/New_York")
    end_local = pd.Timestamp(f"{session_date} 16:00:00", tz="America/New_York")
    return int(start_local.tz_convert("UTC").value), int(end_local.tz_convert("UTC").value)
