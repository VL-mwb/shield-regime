"""Unit tests for WashTradeDetector."""

import numpy as np
import pandas as pd
import pytest

from shield_regime.detectors.wash_trade import WashTradeDetector
from shield_regime.models.alert import AlertSeverity


@pytest.fixture
def clean_stock():
    """A stock with healthy volume and price volatility."""
    np.random.seed(42)
    n = 60
    prices = 100 + np.cumsum(np.random.normal(0.05, 0.5, n))
    volume = np.random.normal(500_000, 50_000, n).clip(100_000)
    # Give it healthy daily range
    high = prices + np.random.uniform(0.5, 1.5, n)
    low = prices - np.random.uniform(0.5, 1.5, n)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"close": prices, "volume": volume, "high": high, "low": low}, index=dates)


@pytest.fixture
def wash_trade_stock():
    """A stock showing highly suspicious wash trading behavior (huge volume, frozen price)."""
    np.random.seed(99)
    n = 60
    prices = np.full(n, 100.0)
    volume = np.full(n, 100_000.0)
    high = prices + 1.0
    low = prices - 1.0

    # Phase 0: normal
    for i in range(40):
        prices[i] += np.random.uniform(-1, 1)
        high[i] = prices[i] + np.random.uniform(0.5, 1.5)
        low[i] = prices[i] - np.random.uniform(0.5, 1.5)
        volume[i] = np.random.uniform(80_000, 120_000)

    # Phase 1: Wash Trading (days 40-44) — volume explodes 10x, but high/low are frozen!
    for i in range(40, 45):
        prices[i] = 100.0
        high[i] = 100.01  # extremely narrow range
        low[i] = 99.99
        volume[i] = 1_000_000.0  # 10x average volume

    # Phase 2: back to normal
    for i in range(45, n):
        prices[i] = 100.0 + np.random.uniform(-1, 1)
        high[i] = prices[i] + np.random.uniform(0.5, 1.5)
        low[i] = prices[i] - np.random.uniform(0.5, 1.5)
        volume[i] = np.random.uniform(80_000, 120_000)

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"close": prices, "volume": volume, "high": high, "low": low}, index=dates)


def test_no_false_positives_on_clean_data(clean_stock):
    detector = WashTradeDetector(wti_threshold=3.5, volume_spike_threshold=2.5, lookback=20)
    alerts = detector.scan(clean_stock, ticker="CLEAN")
    assert len(alerts) == 0


def test_detects_wash_trade_anomaly(wash_trade_stock):
    detector = WashTradeDetector(wti_threshold=3.5, volume_spike_threshold=2.5, lookback=20)
    alerts = detector.scan(wash_trade_stock, ticker="WASH")
    
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.ticker == "WASH"
    assert alert.alert_type == "WASH_TRADE"
    assert alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
    assert alert.duration_days == 4  # 4 calendar days (Feb 26 to Mar 1)
    assert alert.metrics["max_wti"] > 5.0
    assert alert.metrics["max_vol_spike"] > 5.0
