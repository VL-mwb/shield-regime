"""Unit tests for ShieldRegime indicator modules."""

import numpy as np
import pandas as pd
import pytest

from shield_regime.indicators.velocity import price_velocity, price_velocity_pct, velocity_zscore
from shield_regime.indicators.acceleration import price_acceleration, acceleration_zscore, kinematic_reversal_score
from shield_regime.indicators.volume_profile import volume_spike_index, volume_price_divergence, wash_trade_index
from shield_regime.models.alert import Alert, AlertSeverity


# ── Fixtures ──

@pytest.fixture
def normal_stock():
    """A boring stock with gentle uptrend and stable volume."""
    np.random.seed(42)
    n = 120
    prices = 100 + np.cumsum(np.random.normal(0.05, 0.8, n))
    volume = np.random.normal(1_000_000, 100_000, n).clip(100_000)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"close": prices, "volume": volume, "high": prices + 1, "low": prices - 1}, index=dates)


@pytest.fixture
def pump_dump_stock():
    """Simulated Pump-and-Dump: flat → violent spike → crash."""
    np.random.seed(99)
    n = 100
    prices = np.full(n, 10.0)
    volume = np.full(n, 500_000.0)

    # Phase 0: flat (day 0-59)
    prices[:60] += np.cumsum(np.random.normal(0, 0.05, 60))

    # Phase 1: PUMP (day 60-74) — price rockets, volume explodes
    for i in range(60, 75):
        prices[i] = prices[i - 1] * 1.08  # +8% per day
        volume[i] = volume[0] * 6          # 6x normal volume

    # Phase 2: DUMP (day 75-89) — price crashes, volume stays high
    for i in range(75, 90):
        prices[i] = prices[i - 1] * 0.90  # -10% per day
        volume[i] = volume[0] * 4

    # Phase 3: aftermath (day 90-99)
    for i in range(90, 100):
        prices[i] = prices[i - 1] * (1 + np.random.normal(0, 0.01))
        volume[i] = volume[0]

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "close": prices,
        "volume": volume,
        "high": prices * 1.01,
        "low": prices * 0.99,
    }, index=dates)


# ── Velocity Tests ──

class TestVelocity:
    def test_output_shape(self, normal_stock):
        vel = price_velocity(normal_stock["close"])
        assert len(vel) == len(normal_stock)

    def test_ema_vs_sma(self, normal_stock):
        vel_ema = price_velocity(normal_stock["close"], smoothing="ema")
        vel_sma = price_velocity(normal_stock["close"], smoothing="sma")
        # Both should exist and differ
        assert not vel_ema.equals(vel_sma)

    def test_pump_creates_extreme_velocity(self, pump_dump_stock):
        vel = price_velocity_pct(pump_dump_stock["close"])
        # During pump phase (day 60-74), velocity should be very high
        pump_vel = vel.iloc[62:75]
        normal_vel = vel.iloc[10:55]
        assert pump_vel.mean() > normal_vel.mean() * 5

    def test_zscore_detects_pump(self, pump_dump_stock):
        z = velocity_zscore(pump_dump_stock["close"], lookback=50)
        # Z-score during pump should breach 3.0
        pump_z = z.iloc[65:75].dropna()
        assert pump_z.max() > 3.0, f"Max Z during pump was only {pump_z.max():.2f}"

    def test_invalid_smoothing_raises(self, normal_stock):
        with pytest.raises(ValueError, match="Unknown smoothing"):
            price_velocity(normal_stock["close"], smoothing="invalid")


# ── Acceleration Tests ──

class TestAcceleration:
    def test_output_shape(self, normal_stock):
        accel = price_acceleration(normal_stock["close"])
        assert len(accel) == len(normal_stock)

    def test_pump_creates_positive_acceleration(self, pump_dump_stock):
        accel = price_acceleration(pump_dump_stock["close"])
        # Early pump phase should show strong positive acceleration
        pump_accel = accel.iloc[61:68].dropna()
        assert pump_accel.mean() > 0

    def test_reversal_score_peaks_during_dump(self, pump_dump_stock):
        rev = kinematic_reversal_score(pump_dump_stock["close"], lookback=50)
        # Reversal score should be highest during/after the dump
        dump_rev = rev.iloc[76:88].dropna()
        normal_rev = rev.iloc[10:50].dropna()
        if len(dump_rev) > 0 and len(normal_rev) > 0:
            assert dump_rev.max() > normal_rev.max() * 2


# ── Volume Profile Tests ──

class TestVolumeProfile:
    def test_spike_index_normal(self, normal_stock):
        spike = volume_spike_index(normal_stock["volume"])
        # Normal stock should have spike index near 1.0
        recent = spike.iloc[30:].dropna()
        assert 0.5 < recent.mean() < 2.0

    def test_spike_index_detects_pump_volume(self, pump_dump_stock):
        spike = volume_spike_index(pump_dump_stock["volume"], lookback=20)
        # Check early pump days (60-63) before rolling avg absorbs spike
        early_pump_spike = spike.iloc[60:64].dropna()
        assert early_pump_spike.mean() > 3.0

    def test_wash_trade_index(self, pump_dump_stock):
        wti = wash_trade_index(
            pump_dump_stock["high"],
            pump_dump_stock["low"],
            pump_dump_stock["volume"],
        )
        assert len(wti) == len(pump_dump_stock)
        assert not wti.isna().all()


# ── Alert Model Tests ──

class TestAlert:
    def test_alert_creation(self):
        from datetime import datetime
        alert = Alert(
            ticker="3662",
            alert_type="PUMP_AND_DUMP",
            severity=AlertSeverity.HIGH,
            start_date=datetime(2016, 7, 1),
            end_date=datetime(2016, 8, 15),
            headline="Suspected pump-and-dump on Lesheng Tech",
            metrics={"velocity_z": 4.2, "volume_spike": 5.1},
        )
        assert alert.duration_days == 45
        assert "HIGH" in str(alert)
        d = alert.to_dict()
        assert d["ticker"] == "3662"
        assert d["metrics"]["velocity_z"] == 4.2
