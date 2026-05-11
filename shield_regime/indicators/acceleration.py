"""
acceleration.py — Price Acceleration (second derivative of price).

In physics:  acceleration = Δvelocity / Δtime.
Applied to markets:  acceleration tells us whether a trend is
*gaining strength* (positive accel) or *losing steam* (negative accel).

A hallmark of Pump-and-Dump manipulation is an extreme positive
acceleration (the pump) followed immediately by an extreme negative
acceleration (the dump).  Normal trends rarely exhibit such violent
reversals within a short window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shield_regime.indicators.velocity import price_velocity_pct, price_velocity


# ──────────────────────────────────────────────────────────────────────
# Core indicator
# ──────────────────────────────────────────────────────────────────────

def price_acceleration(
    close: pd.Series,
    velocity_window: int = 5,
    accel_window: int = 5,
    smoothing: str = "ema",
    use_pct: bool = True,
) -> pd.Series:
    """Compute smoothed price acceleration (second derivative).

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    velocity_window : int, default 5
        Window for the first-derivative velocity calculation.
    accel_window : int, default 5
        Smoothing window applied to the *change* in velocity.
    smoothing : {"ema", "sma"}, default "ema"
        Smoothing method.
    use_pct : bool, default True
        If True, derive acceleration from percentage velocity
        (price-level neutral).

    Returns
    -------
    pd.Series
        Smoothed daily acceleration (same index as *close*).
    """
    vel_fn = price_velocity_pct if use_pct else price_velocity
    vel = vel_fn(close, window=velocity_window, smoothing=smoothing)

    raw_accel = vel.diff()  # Δvelocity / Δt

    if smoothing == "ema":
        return raw_accel.ewm(span=accel_window, adjust=False).mean()
    elif smoothing == "sma":
        return raw_accel.rolling(window=accel_window, min_periods=1).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing!r}")


# ──────────────────────────────────────────────────────────────────────
# Z-score wrapper
# ──────────────────────────────────────────────────────────────────────

def acceleration_zscore(
    close: pd.Series,
    velocity_window: int = 5,
    accel_window: int = 5,
    lookback: int = 60,
    smoothing: str = "ema",
    use_pct: bool = True,
) -> pd.Series:
    """Rolling Z-score of acceleration against its own history.

    A Z-score > 3.0 suggests the trend is gaining force *far* faster
    than historically normal — potential pump activity.

    A sudden flip from Z > +3 to Z < -3 within a short period is a
    classic Pump-and-Dump kinematic signature.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    velocity_window : int, default 5
        Window for velocity.
    accel_window : int, default 5
        Window for acceleration smoothing.
    lookback : int, default 60
        Rolling window for Z-score mean / std.
    smoothing : {"ema", "sma"}, default "ema"
        Smoothing method.
    use_pct : bool, default True
        Use percentage-based velocity.

    Returns
    -------
    pd.Series
        Z-score of acceleration.
    """
    accel = price_acceleration(
        close,
        velocity_window=velocity_window,
        accel_window=accel_window,
        smoothing=smoothing,
        use_pct=use_pct,
    )

    rolling_mean = accel.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    rolling_std = accel.rolling(window=lookback, min_periods=max(lookback // 2, 1)).std()

    rolling_std = rolling_std.replace(0, np.nan)

    return (accel - rolling_mean) / rolling_std


# ──────────────────────────────────────────────────────────────────────
# Kinematic reversal detector (Pump → Dump signature)
# ──────────────────────────────────────────────────────────────────────

def kinematic_reversal_score(
    close: pd.Series,
    velocity_window: int = 5,
    accel_window: int = 5,
    lookback: int = 60,
    reversal_window: int = 15,
    smoothing: str = "ema",
) -> pd.Series:
    """Detect violent acceleration reversals (Pump → Dump signature).

    For each day *t*, this looks back *reversal_window* days to find the
    maximum acceleration Z-score, then computes:

        reversal_score(t) = max_accel_z(t-w..t) − accel_z(t)

    A high reversal score means the stock recently had extreme upward
    acceleration and has now shifted to deceleration or downward
    acceleration — the kinematic fingerprint of a dump phase.

    Parameters
    ----------
    reversal_window : int, default 15
        How many trading days to look back for the peak acceleration.

    Returns
    -------
    pd.Series
        Reversal score.  Higher = more suspicious.
    """
    accel_z = acceleration_zscore(
        close,
        velocity_window=velocity_window,
        accel_window=accel_window,
        lookback=lookback,
        smoothing=smoothing,
    )

    peak_accel = accel_z.rolling(window=reversal_window, min_periods=1).max()

    return peak_accel - accel_z
