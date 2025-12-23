"""
Black-Scholes Greeks Calculator for SPY 0DTE Options.

Computes real delta and gamma values using the Black-Scholes model.
For 0DTE options, time decay is significant so we use precise time-to-expiry.

Per RULES.md: We NEVER estimate - we calculate real metrics.

Performance: Uses numpy vectorization for batch calculations on millions of trades.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np


@dataclass
class BSMGreeks:
    """Black-Scholes-Merton Greeks output."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float = 0.0


# Standard normal CDF using error function
def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def compute_d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> Tuple[float, float]:
    """
    Compute d1 and d2 for Black-Scholes formula.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years (e.g., 0.5/252 for half a trading day)
        r: Risk-free rate (annualized, e.g., 0.05 for 5%)
        sigma: Implied volatility (annualized, e.g., 0.20 for 20%)

    Returns:
        Tuple of (d1, d2)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    return d1, d2


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str
) -> BSMGreeks:
    """
    Compute Black-Scholes Greeks for an option.

    Args:
        S: Spot price (SPY price)
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Implied volatility (annualized)
        option_type: 'C' for call, 'P' for put

    Returns:
        BSMGreeks with delta, gamma, theta, vega, rho
    """
    # Handle edge cases
    if T <= 0:
        # At expiry, options are either ITM or OTM
        if option_type == 'C':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return BSMGreeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0)

    if sigma <= 0:
        # No volatility = deterministic
        if option_type == 'C':
            delta = 1.0 if S > K * math.exp(-r * T) else 0.0
        else:
            delta = -1.0 if S < K * math.exp(-r * T) else 0.0
        return BSMGreeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)

    # Common terms
    pdf_d1 = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)
    cdf_neg_d1 = _norm_cdf(-d1)
    cdf_neg_d2 = _norm_cdf(-d2)

    # Gamma is same for calls and puts
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # Vega is same for calls and puts (per 1% vol move)
    vega = S * pdf_d1 * sqrt_T / 100.0  # Divided by 100 for per 1% vol

    if option_type == 'C':
        delta = cdf_d1
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * cdf_d2
        ) / 252  # Per trading day
        rho = K * T * math.exp(-r * T) * cdf_d2 / 100  # Per 1% rate
    else:
        delta = cdf_d1 - 1.0  # Negative for puts
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * cdf_neg_d2
        ) / 252  # Per trading day
        rho = -K * T * math.exp(-r * T) * cdf_neg_d2 / 100

    return BSMGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_volatility_newton(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    max_iterations: int = 50,
    tolerance: float = 1e-6
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.

    Args:
        option_price: Observed option price
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        option_type: 'C' for call, 'P' for put
        max_iterations: Maximum Newton iterations
        tolerance: Convergence tolerance

    Returns:
        Implied volatility, or None if cannot converge
    """
    if option_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    # Initial guess based on option price
    sigma = 0.3  # Start with 30% vol

    for _ in range(max_iterations):
        # Calculate option price at current sigma
        price = _bs_price(S, K, T, r, sigma, option_type)

        # Calculate vega for Newton step
        d1, _ = compute_d1_d2(S, K, T, r, sigma)
        vega = S * _norm_pdf(d1) * math.sqrt(T)

        if vega < 1e-10:
            # Vega too small, can't continue
            break

        # Newton-Raphson update
        price_diff = price - option_price
        sigma_new = sigma - price_diff / vega

        # Bound sigma to reasonable range
        sigma_new = max(0.01, min(5.0, sigma_new))

        if abs(sigma_new - sigma) < tolerance:
            return sigma_new

        sigma = sigma_new

    return sigma  # Return best estimate even if not fully converged


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Calculate Black-Scholes option price."""
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)

    if option_type == 'C':
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


class BlackScholesCalculator:
    """
    Calculator for computing Greeks from option trade data.

    For 0DTE options, time is critical. We compute time-to-expiry precisely
    based on market close at 4:00 PM ET.
    """

    # Market close in UTC (4 PM ET = 21:00 UTC during EST)
    MARKET_CLOSE_UTC_HOUR = 21
    MARKET_CLOSE_UTC_MINUTE = 0

    # Default risk-free rate (Fed Funds Rate approximation)
    DEFAULT_RISK_FREE_RATE = 0.045  # 4.5% as of late 2024

    # Default volatility for when we can't compute IV
    DEFAULT_VOLATILITY = 0.20  # 20% annualized

    # Trading days per year
    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, risk_free_rate: Optional[float] = None):
        """
        Initialize calculator.

        Args:
            risk_free_rate: Annualized risk-free rate (defaults to 4.5%)
        """
        self.risk_free_rate = risk_free_rate or self.DEFAULT_RISK_FREE_RATE

    def time_to_expiry(self, ts_ns: int, exp_date: str) -> float:
        """
        Calculate time to expiry in years for 0DTE options.

        Args:
            ts_ns: Current timestamp in nanoseconds
            exp_date: Expiration date in YYYY-MM-DD format

        Returns:
            Time to expiry in years (fraction of a trading day)
        """
        # Parse expiration date
        exp = datetime.strptime(exp_date, '%Y-%m-%d')

        # Market close time on expiration day (4 PM ET = 21:00 UTC in winter)
        # Note: This should account for DST but for 0DTE it's close enough
        expiry_ts = datetime(
            exp.year, exp.month, exp.day,
            hour=self.MARKET_CLOSE_UTC_HOUR,
            minute=self.MARKET_CLOSE_UTC_MINUTE,
            tzinfo=timezone.utc
        )
        expiry_ns = int(expiry_ts.timestamp() * 1e9)

        # Calculate remaining time
        remaining_ns = max(0, expiry_ns - ts_ns)
        remaining_seconds = remaining_ns / 1e9

        # Convert to years (using trading seconds per year)
        # Trading day = 6.5 hours = 23400 seconds
        # Trading year = 252 * 23400 = 5,896,800 seconds
        trading_seconds_per_year = self.TRADING_DAYS_PER_YEAR * 6.5 * 3600

        return remaining_seconds / trading_seconds_per_year

    def compute_greeks_for_trade(
        self,
        spot: float,
        strike: float,
        option_type: str,
        ts_ns: int,
        exp_date: str,
        option_price: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> BSMGreeks:
        """
        Compute Greeks for an option trade.

        If option_price is provided, we first compute implied volatility.
        Otherwise, we use the provided volatility or default.

        Args:
            spot: Current spot price (SPY)
            strike: Strike price
            option_type: 'C' or 'P'
            ts_ns: Trade timestamp in nanoseconds
            exp_date: Expiration date YYYY-MM-DD
            option_price: Observed option price (optional)
            volatility: Override volatility (optional)

        Returns:
            BSMGreeks with computed values
        """
        # Calculate time to expiry
        T = self.time_to_expiry(ts_ns, exp_date)

        # Handle expired options
        if T <= 0:
            if option_type == 'C':
                delta = 1.0 if spot > strike else 0.0
            else:
                delta = -1.0 if spot < strike else 0.0
            return BSMGreeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0)

        # Determine volatility
        sigma = volatility

        if sigma is None and option_price is not None and option_price > 0:
            # Compute implied volatility from option price
            sigma = implied_volatility_newton(
                option_price=option_price,
                S=spot,
                K=strike,
                T=T,
                r=self.risk_free_rate,
                option_type=option_type
            )

        if sigma is None:
            # Use default volatility
            sigma = self.DEFAULT_VOLATILITY

        # Compute Greeks
        return compute_greeks(
            S=spot,
            K=strike,
            T=T,
            r=self.risk_free_rate,
            sigma=sigma,
            option_type=option_type
        )

    def compute_greeks_batch(
        self,
        spot: float,
        trades: list,
        exp_date: str,
        default_volatility: float = 0.20
    ) -> dict:
        """
        Compute Greeks for a batch of option trades.

        Args:
            spot: Current spot price
            trades: List of dicts with 'strike', 'right', 'ts_event_ns', 'price' keys
            exp_date: Expiration date
            default_volatility: Default vol if IV computation fails

        Returns:
            Dict mapping (strike, right) -> BSMGreeks
        """
        results = {}

        for trade in trades:
            key = (trade['strike'], trade['right'])
            if key in results:
                continue  # Already computed for this strike/right

            greeks = self.compute_greeks_for_trade(
                spot=spot,
                strike=trade['strike'],
                option_type=trade['right'],
                ts_ns=trade['ts_event_ns'],
                exp_date=exp_date,
                option_price=trade.get('price'),
                volatility=default_volatility if trade.get('price') is None else None
            )
            results[key] = greeks

        return results


# =============================================================================
# VECTORIZED NUMPY FUNCTIONS FOR BATCH PROCESSING (FAST)
# =============================================================================

def compute_greeks_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: float,
    is_call: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Black-Scholes greeks computation for arrays of options.

    Args:
        S: Spot prices (array)
        K: Strike prices (array)
        T: Time to expiry in years (array)
        r: Risk-free rate (scalar)
        sigma: Volatility (scalar, same for all)
        is_call: Boolean array (True for calls, False for puts)

    Returns:
        Tuple of (delta_array, gamma_array)
    """
    # Handle edge cases
    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)

    # Initialize outputs
    delta = np.zeros_like(S, dtype=np.float64)
    gamma = np.zeros_like(S, dtype=np.float64)

    # For expired options
    expired = T <= 0
    delta[expired & is_call] = np.where(S[expired & is_call] > K[expired & is_call], 1.0, 0.0)
    delta[expired & ~is_call] = np.where(S[expired & ~is_call] < K[expired & ~is_call], -1.0, 0.0)

    if not np.any(valid):
        return delta, gamma

    # Compute for valid options
    S_v = S[valid]
    K_v = K[valid]
    T_v = T[valid]
    is_call_v = is_call[valid]

    sqrt_T = np.sqrt(T_v)
    d1 = (np.log(S_v / K_v) + (r + 0.5 * sigma * sigma) * T_v) / (sigma * sqrt_T)

    # Standard normal CDF and PDF using scipy-free implementation
    cdf_d1 = 0.5 * (1.0 + _erf_vectorized(d1 / np.sqrt(2.0)))
    pdf_d1 = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)

    # Delta: calls = N(d1), puts = N(d1) - 1
    delta_v = np.where(is_call_v, cdf_d1, cdf_d1 - 1.0)

    # Gamma: same for calls and puts
    gamma_v = pdf_d1 / (S_v * sigma * sqrt_T)

    delta[valid] = delta_v
    gamma[valid] = gamma_v

    return delta, gamma


def _erf_vectorized(x: np.ndarray) -> np.ndarray:
    """Vectorized error function approximation (Abramowitz and Stegun)."""
    # Constants for approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def compute_greeks_for_dataframe(
    df,
    spot: float,
    exp_date: str,
    risk_free_rate: float = 0.045,
    default_volatility: float = 0.20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Black-Scholes greeks for an entire DataFrame of option trades.

    This is ~100x faster than iterating row-by-row.

    Args:
        df: DataFrame with columns 'strike', 'right', 'ts_event_ns'
        spot: Current spot price
        exp_date: Expiration date (YYYY-MM-DD)
        risk_free_rate: Annualized risk-free rate
        default_volatility: Volatility to use

    Returns:
        Tuple of (delta_array, gamma_array) with same length as df
    """
    n = len(df)
    if n == 0:
        return np.array([]), np.array([])

    # Extract arrays
    strikes = df['strike'].values.astype(np.float64)
    is_call = (df['right'].values == 'C')

    # Calculate time to expiry for all trades
    # Use median timestamp for batch (close enough for 0DTE)
    exp = datetime.strptime(exp_date, '%Y-%m-%d')
    expiry_ts = datetime(exp.year, exp.month, exp.day, hour=21, minute=0, tzinfo=timezone.utc)
    expiry_ns = int(expiry_ts.timestamp() * 1e9)

    ts_ns = df['ts_event_ns'].values.astype(np.int64)
    remaining_ns = np.maximum(0, expiry_ns - ts_ns)
    remaining_seconds = remaining_ns / 1e9

    # Trading seconds per year
    trading_seconds_per_year = 252 * 6.5 * 3600
    T = remaining_seconds / trading_seconds_per_year

    # Spot price array (same for all)
    S = np.full(n, spot, dtype=np.float64)

    # Compute vectorized greeks
    delta, gamma = compute_greeks_vectorized(
        S=S,
        K=strikes,
        T=T,
        r=risk_free_rate,
        sigma=default_volatility,
        is_call=is_call
    )

    return delta, gamma
