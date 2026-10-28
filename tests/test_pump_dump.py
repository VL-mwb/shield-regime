"""Unit tests for PumpDumpDetector."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.models.alert import AlertSeverity


@pytest.fixture
def clean_data():
    """A clean series representing a stock that does NOT pump and dump."""
    np.random.seed(42)
    n = 100
    prices = 50.0 + np.cumsum(np.random.normal(0.02, 0.3, n))
    volume = np.random.normal(200_000, 20_000, n).clip(50_000)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"close": prices, "volume": volume}, index=dates)




class TestPumpDumpDetector:
    def test_no_false_positives_on_clean_data(self, clean_data):
        detector = PumpDumpDetector(lookback=40)
        alerts = detector.scan(clean_data, ticker="GOOD")
        # A normal stock should generate zero warnings
        assert len(alerts) == 0

    def test_detects_manipulation_on_dirty_data(self, manipulated_data):
        detector = PumpDumpDetector(
            velocity_z_threshold=2.0,
            volume_spike_threshold=3.0,
            dump_drop_threshold=-20.0,
            lookback=40,
        )
        alerts = detector.scan(manipulated_data, ticker="PUMP")
        
        # Should catch at least one severe alert
        assert len(alerts) >= 1
        
        # Verify alert metadata and metrics
        alert = alerts[0]
        assert alert.ticker == "PUMP"
        assert alert.alert_type == "PUMP_AND_DUMP"
        assert alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        assert alert.metrics["dump_drop_pct"] <= -20.0
        assert alert.metrics["pump_max_vol_spike"] >= 3.0
        assert "Pump-and-Dump" in alert.headline
        assert alert.duration_days > 5

    def test_handles_incomplete_or_invalid_columns(self, clean_data):
        detector = PumpDumpDetector()
        broken_df = clean_data.drop(columns=["volume"])
        
        with pytest.raises(ValueError, match="must contain columns"):
            detector.scan(broken_df)
