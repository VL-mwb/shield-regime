"""
velocity.py — Price Velocity (first derivative of price).

In classical mechanics, velocity = Δposition / Δtime.
Applied to markets:  velocity = Δprice / Δtime  (smoothed).

A stock under normal trading conditions has velocity that fluctuates
within a predictable band (±2σ).  When velocity breaches 3σ it may
indicate artificial price manipulation — either a pump (extreme
positive velocity) or a dump (extreme negative velocity).

This module is intentionally *general-purpose*.  It knows nothing
about specific trading strategies or proprietary parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Core indicator
# ──────────────────────────────────────────────────────────────────────

def price_velocity(
    close: pd.Series,
    window: int = 5,
    smoothing: str = "ema",
) -> pd.Series:
    """Compute smoothed price velocity (first derivative).

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.  Index should be a DatetimeIndex or
        monotonically increasing integers.
    window : int, default 5
        Smoothing window for the EMA / SMA applied *after* differencing.
    smoothing : {"ema", "sma"}, default "ema"
        Smoothing method.  EMA reacts faster to recent changes;
        SMA gives equal weight to all observations in the window.

    Returns
    -------
    pd.Series
        Smoothed daily velocity (same index as *close*).
        Units: currency per day (e.g. TWD/day, USD/day).
    """
    if len(close) < 2:
        return pd.Series(dtype=float, index=close.index)

    raw_velocity = close.diff()  # ΔPrice / Δt (Δt = 1 trading day)

    if smoothing == "ema":
        return raw_velocity.ewm(span=window, adjust=False).mean()
    elif smoothing == "sma":
        return raw_velocity.rolling(window=window, min_periods=1).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing!r}")


def price_velocity_pct(
    close: pd.Series,
    window: int = 5,
    smoothing: str = "ema",
) -> pd.Series:
    """Percentage-based velocity (normalised by price level).

    Useful when comparing stocks at vastly different price levels
    (e.g. TSMC at TWD 900 vs. a penny stock at TWD 5).

    Returns
    -------
    pd.Series
        Daily percentage velocity (decimal form, e.g. 0.03 = 3 %/day).
    """
    if len(close) < 2:
        return pd.Series(dtype=float, index=close.index)

    raw_pct = close.pct_change()  # ΔPrice / Price

    if smoothing == "ema":
        return raw_pct.ewm(span=window, adjust=False).mean()
    elif smoothing == "sma":
        return raw_pct.rolling(window=window, min_periods=1).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing!r}")


# ──────────────────────────────────────────────────────────────────────
# Z-score wrapper  (the key anomaly-detection layer)
# ──────────────────────────────────────────────────────────────────────

def velocity_zscore(
    close: pd.Series,
    velocity_window: int = 5,
    lookback: int = 60,
    smoothing: str = "ema",
    use_pct: bool = True,
) -> pd.Series:
    """Rolling Z-score of velocity against its own history.

    A Z-score > 3.0 means the current velocity is more than 3 standard
    deviations above its recent average — a strong anomaly signal.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    velocity_window : int, default 5
        Window for the underlying velocity EMA.
    lookback : int, default 60
        Rolling window (trading days) for mean / std calculation.
    smoothing : {"ema", "sma"}, default "ema"
        Smoothing method for the velocity.
    use_pct : bool, default True
        If True, use percentage velocity (price-level neutral).

    Returns
    -------
    pd.Series
        Z-score of velocity.  Positive = accelerating upward.
        Negative = accelerating downward.
    """
    vel_fn = price_velocity_pct if use_pct else price_velocity
    vel = vel_fn(close, window=velocity_window, smoothing=smoothing)

    rolling_mean = vel.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    rolling_std = vel.rolling(window=lookback, min_periods=max(lookback // 2, 1)).std()

    # Avoid division by zero in perfectly flat series
    rolling_std = rolling_std.replace(0, np.nan)

    return (vel - rolling_mean) / rolling_std
