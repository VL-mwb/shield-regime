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


def test_first_detection_and_group_extension(wash_trade_stock):
    d=WashTradeDetector()
    first=d.scan(wash_trade_stock.iloc[:41])[0]
    before=first.to_dict()
    full=d.scan(wash_trade_stock)[0]
    assert first.detected_at==wash_trade_stock.index[40]
    assert full.detected_at==first.detected_at
    assert full.end_date>first.end_date
    assert first.to_dict()==before

def test_warmup_not_backdated(monkeypatch,clean_stock):
    import shield_regime.detectors.wash_trade as mod
    v=pd.Series(10.,index=clean_stock.index)
    monkeypatch.setattr(mod,'wash_trade_index',lambda *a,**k:v)
    monkeypatch.setattr(mod,'volume_spike_index',lambda *a,**k:v)
    a=WashTradeDetector().scan(clean_stock)[0]
    assert a.start_date==clean_stock.index[19] and a.detected_at==clean_stock.index[19]


def test_zero_range_prefix_snapshots_and_warmup():
    df=pd.DataFrame(dict(high=100.,low=100.,close=100.,volume=[100.]*20+[1000.]*5),index=pd.bdate_range('2024-01-01',periods=25))
    detector=WashTradeDetector()
    for size in range(1,20):
        from shield_regime import InsufficientDataWarning
        with pytest.warns(InsufficientDataWarning):assert detector.scan(df.iloc[:size])==[]
    assert detector.scan(df.iloc[:20])==[]
    first=detector.scan(df.iloc[:21])[0];saved=first.to_dict()
    for size in range(21,26):
        alert=detector.scan(df.iloc[:size])[0]
        assert alert.detected_at==df.index[20]
        assert alert.metrics['zero_range_days']==float(size-20)
        assert first.to_dict()==saved


def test_existing_positive_range_metrics_unchanged(wash_trade_stock):
    alert=WashTradeDetector().scan(wash_trade_stock)[0]
    assert alert.metrics['zero_range_days']==0.
    assert alert.detected_at==wash_trade_stock.index[40]
