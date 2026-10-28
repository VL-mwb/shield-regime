"""Unit tests for visualization module."""

import numpy as np
import pandas as pd
import pytest

from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.visualizer import plot_alerts


@pytest.fixture
def fake_stock():
    np.random.seed(42)
    n = 60
    prices = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    volume = np.random.normal(100_000, 10_000, n).clip(10_000)
    dates = pd.date_range("2024-01-01", periods=n)
    return pd.DataFrame({"close": prices, "volume": volume}, index=dates)


def test_plot_alerts_runs_successfully(fake_stock):
    # Setup some dummy alerts
    detector = PumpDumpDetector(lookback=20, velocity_z_threshold=1.5)
    alerts = detector.scan(fake_stock, ticker="DUMMY")

    # This should run without throwing any exceptions and return a plotly Figure
    fig = plot_alerts(fake_stock, alerts, ticker="DUMMY", lookback=20)
    
    # Verify return type is a Plotly graph object (if Plotly is installed)
    if fig is not None:
        from plotly.graph_objects import Figure
        assert isinstance(fig, Figure)
        assert len(fig.data) >= 2  # Close price + Velocity Z traces


def test_confirmation_marker_and_legacy(fake_stock):
    from shield_regime import Alert, AlertSeverity
    dt=fake_stock.index[20]
    a=Alert('X','WASH_TRADE',AlertSeverity.HIGH,dt,fake_stock.index[22],'test',detected_at=dt)
    fig=plot_alerts(fake_stock,[a])
    trace=next(t for t in fig.data if t.name=='Alert Confirmed #1')
    assert trace.x[0]==dt and trace.y[0]==fake_stock.loc[dt,'close']
    assert all('Pump Start' not in t.name and 'Dump Low' not in t.name for t in fig.data)
    a.detected_at=None
    assert not any(t.name.startswith('Alert Confirmed') for t in plot_alerts(fake_stock,[a]).data)
