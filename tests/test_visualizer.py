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
