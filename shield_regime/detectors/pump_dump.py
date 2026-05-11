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
from typing import List, Dict

import numpy as np
import pandas as pd

from shield_regime.indicators.velocity import velocity_zscore, price_velocity_pct
from shield_regime.indicators.acceleration import acceleration_zscore
from shield_regime.indicators.volume_profile import volume_spike_index
from shield_regime.models.alert import Alert, AlertSeverity


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
        # Ensure we have enough data and correct column types
        required_cols = {"close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        if len(df) < self.lookback + self.dump_window:
            return []

        # Ensure index is datetime for alerts
        if not isinstance(df.index, pd.DatetimeIndex):
            # Attempt conversion, fallback to range
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass

        # Calculate indicators
        vel_z = velocity_zscore(df["close"], lookback=self.lookback, use_pct=True)
        accel_z = acceleration_zscore(df["close"], lookback=self.lookback, use_pct=True)
        vol_spike = volume_spike_index(df["volume"], lookback=self.lookback)
        pct_vel = price_velocity_pct(df["close"])

        alerts = []
        n = len(df)
        i = self.lookback

        while i < n - self.dump_window:
            # 1. Look for Pump Phase candidate: Velocity spike + Volume spike
            is_vel_anomaly = vel_z.iloc[i] >= self.velocity_z_threshold
            is_vol_anomaly = vol_spike.iloc[i] >= self.volume_spike_threshold

            if is_vel_anomaly and is_vol_anomaly:
                pump_start_idx = i
                
                # Trace pump peak over consecutive anomaly days
                pump_end_idx = i
                while (
                    pump_end_idx < n - self.dump_window 
                    and vel_z.iloc[pump_end_idx] >= 1.5  # pump momentum continues
                ):
                    pump_end_idx += 1
                
                # Find the peak price date within this pump window
                pump_prices = df["close"].iloc[pump_start_idx:pump_end_idx + 1]
                peak_idx_loc = pump_prices.argmax()
                peak_idx = pump_start_idx + peak_idx_loc
                peak_price = pump_prices.iloc[peak_idx_loc]
                peak_date = df.index[peak_idx]

                # 2. Look for Dump Phase in the subsequent window
                dump_start = peak_idx + 1
                dump_end = min(peak_idx + 1 + self.dump_window, n)
                
                if dump_start < n:
                    dump_prices = df["close"].iloc[dump_start:dump_end]
                    
                    if not dump_prices.empty:
                        min_price_loc = dump_prices.argmin()
                        min_price = dump_prices.iloc[min_price_loc]
                        min_price_idx = dump_start + min_price_loc
                        min_price_date = df.index[min_price_idx]
                        
                        # Calculate price draw-down from the pump peak
                        drawdown_pct = ((min_price - peak_price) / peak_price) * 100.0
                        
                        if drawdown_pct <= self.dump_drop_threshold:
                            # Textbook match! Generate high-fidelity alert
                            severity = AlertSeverity.HIGH
                            if drawdown_pct <= -40.0 and vol_spike.iloc[pump_start_idx:pump_end_idx+1].max() >= 5.0:
                                severity = AlertSeverity.CRITICAL
                            
                            metrics = {
                                "pump_max_vel_z": float(vel_z.iloc[pump_start_idx:pump_end_idx+1].max()),
                                "pump_max_accel_z": float(accel_z.iloc[pump_start_idx:pump_end_idx+1].max()),
                                "pump_max_vol_spike": float(vol_spike.iloc[pump_start_idx:pump_end_idx+1].max()),
                                "pump_return_pct": float((peak_price - df["close"].iloc[pump_start_idx]) / df["close"].iloc[pump_start_idx] * 100.0),
                                "dump_drop_pct": float(drawdown_pct),
                            }

                            headline = (
                                f"Pump-and-Dump Pattern Detected: {drawdown_pct:.1f}% crash "
                                f"following {metrics['pump_return_pct']:.1f}% pump"
                            )
                            
                            details = (
                                f"The asset experienced an extreme price and volume expansion starting on "
                                f"{df.index[pump_start_idx]:%Y-%m-%d}. Price reached a peak of {peak_price:.2f} "
                                f"on {peak_date:%Y-%m-%d} on {vol_spike.iloc[peak_idx]:.1f}x volume. "
                                f"Within {min_price_idx - peak_idx} trading days, the price collapsed by "
                                f"{drawdown_pct:.1f}% to a low of {min_price:.2f} on {min_price_date:%Y-%m-%d}."
                            )

                            alert = Alert(
                                ticker=ticker,
                                alert_type="PUMP_AND_DUMP",
                                severity=severity,
                                start_date=df.index[pump_start_idx],
                                end_date=min_price_date,
                                headline=headline,
                                details=details,
                                metrics=metrics,
                            )
                            alerts.append(alert)
                            
                            # Skip scanning pointer forward to avoid overlapping alerts in the same cycle
                            i = min_price_idx + 1
                            continue

            i += 1

        return alerts
