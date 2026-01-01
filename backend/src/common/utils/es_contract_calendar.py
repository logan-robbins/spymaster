"""
ES futures contract calendar for front-month determination.

ES contracts expire on the 3rd Friday of March, June, September, December.
Roll typically happens 8 days before expiration.
"""

from datetime import datetime, timedelta
from typing import Tuple


# ES contract month codes
MONTH_CODES = {
    3: 'H',   # March
    6: 'M',   # June
    9: 'U',   # September
    12: 'Z'   # December
}


def get_third_friday(year: int, month: int) -> datetime:
    """
    Get third Friday of a month.
    
    Args:
        year: Year
        month: Month (1-12)
    
    Returns:
        datetime of third Friday
    """
    # First day of month
    first = datetime(year, month, 1)
    
    # Find first Friday (weekday 4 = Friday)
    days_until_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_until_friday)
    
    # Third Friday is 14 days later
    third_friday = first_friday + timedelta(days=14)
    
    return third_friday


def get_front_month_contract_code(date: str) -> str:
    """
    Determine front-month ES contract code for a date.
    
    Uses CME roll schedule: 8 days before expiration, roll to next contract.
    
    Args:
        date: Date string (YYYY-MM-DD)
    
    Returns:
        Contract code (e.g., 'ESZ5' for Dec 2025)
    
    Example:
        >>> get_front_month_contract_code('2025-11-03')
        'ESZ5'  # December 2025 is front month
    """
    dt = datetime.strptime(date, '%Y-%m-%d')
    year = dt.year
    
    # ES contract months: March, June, September, December
    contract_months = [3, 6, 9, 12]
    
    # Find next contract expiration
    expirations = []
    for month in contract_months:
        exp_date = get_third_friday(year, month)
        if exp_date >= dt:
            expirations.append((month, year, exp_date))
    
    # If no expiration this year, check next year
    if not expirations:
        for month in contract_months:
            exp_date = get_third_friday(year + 1, month)
            expirations.append((month, year + 1, exp_date))
    
    # Sort by expiration date
    expirations.sort(key=lambda x: x[2])
    
    # Get nearest expiration
    next_month, next_year, next_exp = expirations[0]
    
    # Check if we should roll (8 days before expiration)
    roll_date = next_exp - timedelta(days=8)
    
    if dt >= roll_date:
        # Roll to next contract
        if len(expirations) > 1:
            next_month, next_year, _ = expirations[1]
        else:
            # Need to compute next quarter
            next_idx = (contract_months.index(next_month) + 1) % 4
            next_month = contract_months[next_idx]
            if next_idx == 0:  # Rolled to March of next year
                next_year += 1
    
    # Build contract code: ESH5 (H = March, 5 = 2025)
    month_code = MONTH_CODES[next_month]
    year_code = str(next_year)[-1]  # Last digit of year
    
    contract_code = f'ES{month_code}{year_code}'
    
    return contract_code


def get_active_contracts(date: str) -> list[str]:
    """
    Get all active ES contracts for a date (handles rollover overlap).
    
    During the 8-day rollover period, returns BOTH the expiring contract
    and the next contract. Otherwise returns just the front month.
    
    Args:
        date: Date string (YYYY-MM-DD)
    
    Returns:
        List of contract codes (e.g., ['ESM5', 'ESU5'])
    """
    dt = datetime.strptime(date, '%Y-%m-%d')
    year = dt.year
    contract_months = [3, 6, 9, 12]
    
    # Find next contract expiration
    expirations = []
    for month in contract_months:
        exp_date = get_third_friday(year, month)
        if exp_date >= dt:
            expirations.append((month, year, exp_date))
            
    if not expirations:
        for month in contract_months:
            exp_date = get_third_friday(year + 1, month)
            expirations.append((month, year + 1, exp_date))
            
    expirations.sort(key=lambda x: x[2])
    
    current_month, current_year, current_exp = expirations[0]
    
    # Calculate rollover start (8 days before expiry)
    roll_start = current_exp - timedelta(days=8)
    
    contracts = []
    
    # Always include the current front month (contracts[0])
    # Note: get_front_month_contract_code switches strictly at roll_date.
    # Here we want inclusive logic.
    
    # 1. Current expiring contract (until it expires)
    if dt <= current_exp:
        code_1 = f"ES{MONTH_CODES[current_month]}{str(current_year)[-1]}"
        contracts.append(code_1)
        
    # 2. If we are in or past rollover window, include NEXT contract
    if dt >= roll_start:
        if len(expirations) > 1:
            next_month, next_year, _ = expirations[1]
        else:
            next_idx = (contract_months.index(current_month) + 1) % 4
            next_month = contract_months[next_idx]
            next_year = current_year + 1 if next_idx == 0 else current_year
            
        code_2 = f"ES{MONTH_CODES[next_month]}{str(next_year)[-1]}"
        contracts.append(code_2)
        
    return contracts


def validate_contract_for_date(contract: str, date: str) -> Tuple[bool, str]:
    """
    Validate if a contract is the expected front-month for a date.
    
    Args:
        contract: Contract code (e.g., 'ESZ5')
        date: Date string (YYYY-MM-DD)
    
    Returns:
        (is_front_month, expected_front_month)
    """
    expected = get_front_month_contract_code(date)
    is_front = (contract == expected)
    return is_front, expected

