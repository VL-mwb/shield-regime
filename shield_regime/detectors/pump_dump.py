"""
pump_dump.py — Pump-and-Dump Market Manipulation Detector.

This module combines price kinematics (velocity, acceleration) and volume
profiles to detect textbook "Pump-and-Dump" schemes.

The algorithm breaks down the pattern into three kinematic phases:
  1. Base Phase: Steady price, quiet/low volume.
  2. Pump Phase: Extreme upward price velocity/acceleration coupled with massive volume spikes.
  3. Dump Phase: Violent collapse in price, often on continued high volume or complete liquidity dry-up.
"""

from __future__ import annotations

from datetime import datetime
from numbers import Real
from typing import List, Dict

import numpy as np
import pandas as pd

from shield_regime.indicators.velocity import velocity_zscore, price_velocity_pct
from shield_regime.indicators.acceleration import acceleration_zscore
from shield_regime.indicators.volume_profile import volume_spike_index
from shield_regime.models.alert import Alert, AlertSeverity
from shield_regime.numerics import finite_metric, ensure_no_infinity, drawdown_reached

import warnings
from shield_regime.validation import validate_market_data, validate_window, validate_positive, InsufficientDataWarning


from dataclasses import dataclass

@dataclass
class _Candidate:
    start: int
    peak: int
    pump_end: int
    pumping: bool = True


class PumpDumpDetector:
    """Pump-and-Dump Detector based on kinematic anomalies.

    Properties
    ----------
    velocity_z_threshold : float
        Z-score limit for percentage velocity (default 2.5). Breach triggers potential pump.
    volume_spike_threshold : float
        Ratio of volume to recent average (default 3.0). Volume must spike during the pump.
    dump_window : int
        Observation window (trading days) after pump peak to look for a dump (default 15).
    dump_drop_threshold : float
        Minimum percentage price drop from the pump peak to qualify as a dump (default -20.0%).
    lookback : int
        Rolling window for calculating moving averages and standard deviations (default 60).
    """

    def __init__(
        self,
        velocity_z_threshold: float = 2.5,
        volume_spike_threshold: float = 3.0,
        dump_window: int = 15,
        dump_drop_threshold: float = -20.0,
        lookback: int = 60,
    ):
        validate_window('lookback', lookback, 2)
        validate_window('dump_window', dump_window, 1)
        validate_positive('velocity_z_threshold', velocity_z_threshold)
        validate_positive('volume_spike_threshold', volume_spike_threshold)
        if isinstance(dump_drop_threshold, (bool, np.bool_)) or not isinstance(dump_drop_threshold, Real):
            raise ValueError('dump_drop_threshold must have a finite absolute value in (0, 100]')
        if not np.isfinite(dump_drop_threshold) or not 0 < abs(dump_drop_threshold) <= 100:
            raise ValueError('dump_drop_threshold must have a finite absolute value in (0, 100]')
        # 原有 self.dump_drop_threshold = -abs(dump_drop_threshold) 保留。
        self.velocity_z_threshold = velocity_z_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.dump_window = dump_window
        # Ensure drop threshold is negative for comparison convenience
        self.dump_drop_threshold = -abs(dump_drop_threshold)
        self.lookback = lookback

    def scan(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> List[Alert]:
        """Scan historical price & volume data for Pump-and-Dump patterns.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: 'close' and 'volume'.
            Index should be a DatetimeIndex.
        ticker : str, default "UNKNOWN"
            The asset symbol being scanned.

        Returns
        -------
        List[Alert]
            A list of flagged Pump-and-Dump alerts.
        """
        df = validate_market_data(df, ('close', 'volume'))
        if len(df) < self.lookback + 1:
            warnings.warn('PumpDumpDetector requires at least lookback + 1 bars', InsufficientDataWarning, stacklevel=2)
            return []

        # Calculate indicators
        vel_z = velocity_zscore(df["close"], lookback=self.lookback, use_pct=True)
        accel_z = acceleration_zscore(df["close"], lookback=self.lookback, use_pct=True)
        vol_spike = volume_spike_index(df["volume"], lookback=self.lookback)
        for name, values in (("velocity", vel_z), ("acceleration", accel_z), ("volume spike", vol_spike)):
            ensure_no_infinity(name, values)


        alerts = []
        candidates = []
        for i in range(self.lookback, len(df)):
            candidates = [c for c in candidates if i - c.peak <= self.dump_window]
            confirmed = []
            close = float(df['close'].iloc[i])
            for candidate in candidates:
                if candidate.pumping:
                    candidate.pump_end = i
                    if close > df['close'].iloc[candidate.peak]:
                        candidate.peak = i
                    if not (vel_z.iloc[i] >= 1.5):
                        candidate.pumping = False
                age = i - candidate.peak
                peak_price = float(df['close'].iloc[candidate.peak])
                if 1 <= age <= self.dump_window and drawdown_reached(close, peak_price, self.dump_drop_threshold):
                    drawdown_pct = finite_metric('dump_drop_pct', (close / peak_price - 1.0) * 100.0)
                    confirmed.append((candidate, drawdown_pct))
            if confirmed:
                candidate, drawdown_pct = min(confirmed, key=lambda item: item[0].start)
                alerts.append(self._create_alert(df, vel_z, accel_z, vol_spike, candidate, i, drawdown_pct, ticker))
                candidates.clear()
                continue
            is_pump = vel_z.iloc[i] >= self.velocity_z_threshold and vol_spike.iloc[i] >= self.volume_spike_threshold
            if is_pump and not any(c.pumping for c in candidates):
                candidates.append(_Candidate(i, i, i, pumping=bool(vel_z.iloc[i] >= 1.5)))
        return alerts

    def _create_alert(self, df, vel_z, accel_z, vol_spike, candidate, confirmed_idx, drawdown_pct, ticker):
        pump_slice = slice(candidate.start, candidate.pump_end + 1)
        metrics = {}
        for name, values in (
            ('pump_max_vel_z', vel_z.iloc[pump_slice]),
            ('pump_max_accel_z', accel_z.iloc[pump_slice]),
            ('pump_max_vol_spike', vol_spike.iloc[pump_slice]),
        ):
            ensure_no_infinity(name, values)
            finite = values.dropna()
            if not finite.empty:
                metrics[name] = finite_metric(name, finite.max())
        peak_price = float(df['close'].iloc[candidate.peak])
        start_price = float(df['close'].iloc[candidate.start])
        metrics['pump_return_pct'] = finite_metric('pump_return_pct', (peak_price / start_price - 1.0) * 100.0)
        metrics['dump_drop_pct'] = finite_metric('dump_drop_pct', drawdown_pct)
        severity = AlertSeverity.HIGH
        if drawdown_reached(float(df['close'].iloc[confirmed_idx]), peak_price, -40.0) and metrics['pump_max_vol_spike'] >= 5.0:
            severity = AlertSeverity.CRITICAL
        confirmed_at = df.index[confirmed_idx]
        return Alert(
            ticker=ticker,
            alert_type='PUMP_AND_DUMP',
            severity=severity,
            start_date=df.index[candidate.start],
            end_date=confirmed_at,
            detected_at=confirmed_at,
            headline=f'Pump-and-Dump Pattern Detected: {drawdown_pct:.1f}% drawdown confirmed',
            details=(f'Candidate started on {df.index[candidate.start]:%Y-%m-%d}. '
                     f'Observed peak close was {peak_price:.2f} on {df.index[candidate.peak]:%Y-%m-%d}. '
                     f'Drawdown reached {drawdown_pct:.1f}% on {confirmed_at:%Y-%m-%d}, '
                     f'{confirmed_idx - candidate.peak} bars after the peak.'),
            metrics=metrics,
        )
