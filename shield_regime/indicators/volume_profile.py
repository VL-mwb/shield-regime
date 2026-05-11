"""
volume_profile.py — Volume anomaly and volume-price relationship analysis.

Volume is the *force* that moves the price particle. In a healthy market,
volume and price changes are correlated. When this breaks, it may indicate
wash trading or controlled insider manipulation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def volume_spike_index(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Ratio of today's volume to its rolling average.

    Returns spike index. Value of 3.0 = today is 3x the recent average.
    Values > 3.0 are considered anomalous.
    """
    avg_vol = volume.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    avg_vol = avg_vol.replace(0, np.nan)
    return volume / avg_vol


def volume_price_divergence(close: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Volume-Price Divergence Index.

    VPD = volume_spike - |price_change_zscore|

    Positive = volume-heavy anomaly (wash trade suspect).
    Negative = price-heavy anomaly (insider pump suspect).
    """
    vol_spike = volume_spike_index(volume, lookback=lookback)

    pct_change = close.pct_change().abs()
    pct_mean = pct_change.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    pct_std = pct_change.rolling(window=lookback, min_periods=max(lookback // 2, 1)).std()
    pct_std = pct_std.replace(0, np.nan)
    price_z = (pct_change - pct_mean) / pct_std

    return vol_spike - price_z.abs()


def wash_trade_index(high: pd.Series, low: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Wash Trade Index — volume spike with minimal price range.

    WTI = volume_spike / (intraday_range_ratio + epsilon)

    High WTI = huge volume but tiny price movement = wash trade suspect.
    """
    vol_spike = volume_spike_index(volume, lookback=lookback)

    mid_price = (high + low) / 2
    mid_price = mid_price.replace(0, np.nan)
    intraday_range = (high - low) / mid_price

    range_mean = intraday_range.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    range_mean = range_mean.replace(0, np.nan)
    range_ratio = intraday_range / range_mean

    epsilon = 0.01
    return vol_spike / (range_ratio + epsilon)
