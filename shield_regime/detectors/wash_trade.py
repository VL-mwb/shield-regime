"""
wash_trade.py — High-fidelity Wash Trading Detector.

Identifies self-dealing or fake volume generation scams where volume
explodes dramatically (high volume spike) but the price remains narrow or
locked in a tight band (abnormally low intraday range).
"""

from __future__ import annotations

import pandas as pd

from shield_regime.indicators.volume_profile import volume_spike_index, wash_trade_index
from shield_regime.models.alert import Alert, AlertSeverity


class WashTradeDetector:
    """Wash Trading Detector using kinematic volume-price footprints."""

    def __init__(
        self,
        wti_threshold: float = 3.5,
        volume_spike_threshold: float = 2.5,
        lookback: int = 20,
    ) -> None:
        """
        Parameters
        ----------
        wti_threshold : float, default 3.5
            Threshold for the Wash Trade Index (WTI).
        volume_spike_threshold : float, default 2.5
            Minimum volume spike ratio (today's volume / rolling avg).
        lookback : int, default 20
            Rolling baseline window.
        """
        self.wti_threshold = wti_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.lookback = lookback

    def scan(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> list[Alert]:
        """Scan historical dataframe for wash trading anomalies.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns 'high', 'low', 'close', and 'volume'.
        ticker : str, default "UNKNOWN"
            Ticker of the asset being scanned.

        Returns
        -------
        list[Alert]
            List of generated wash trading alerts.
        """
        required_cols = ["high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            return []

        if len(df) < self.lookback:
            return []

        # 1. Compute baseline kinematic indicators
        wti = wash_trade_index(df["high"], df["low"], df["volume"], lookback=self.lookback)
        vol_spike = volume_spike_index(df["volume"], lookback=self.lookback)

        # 2. Identify anomaly flags
        is_anomaly = (wti >= self.wti_threshold) & (vol_spike >= self.volume_spike_threshold)

        # 3. Group consecutive anomaly days
        alerts: list[Alert] = []
        in_anomaly = False
        start_idx = None

        for idx, (date, flag) in enumerate(is_anomaly.items()):
            if flag:
                if not in_anomaly:
                    in_anomaly = True
                    start_idx = idx
            else:
                if in_anomaly:
                    in_anomaly = False
                    alerts.append(self._create_alert(df, wti, vol_spike, start_idx, idx - 1, ticker))

        # Handle edge case where anomaly lasts until the end of df
        if in_anomaly:
            alerts.append(self._create_alert(df, wti, vol_spike, start_idx, len(df) - 1, ticker))

        return alerts

    def _create_alert(
        self,
        df: pd.DataFrame,
        wti: pd.Series,
        vol_spike: pd.Series,
        start_idx: int,
        end_idx: int,
        ticker: str,
    ) -> Alert:
        """Helper to package wash trading anomaly stats into an Alert."""
        start_date = df.index[start_idx]
        end_date = df.index[end_idx]
        duration = (end_idx - start_idx) + 1

        sub_wti = wti.iloc[start_idx : end_idx + 1]
        sub_vol_spike = vol_spike.iloc[start_idx : end_idx + 1]

        max_wti = float(sub_wti.max())
        max_vol_spike = float(sub_vol_spike.max())

        # Determine severity
        if max_wti >= 8.0 and duration >= 3:
            severity = AlertSeverity.CRITICAL
        elif max_wti >= 5.0:
            severity = AlertSeverity.HIGH
        else:
            severity = AlertSeverity.MEDIUM

        headline = f"Wash Trading Signature Detected: Peak WTI {max_wti:.2f}"
        details = (
            f"The asset exhibited highly abnormal self-dealing signatures "
            f"over {duration} trading day(s). Volume surged to a peak of "
            f"{max_vol_spike:.2f}x average while the intraday price range remained "
            f"exceptionally locked, pushing the Wash Trade Index (WTI) to {max_wti:.2f}."
        )

        return Alert(
            ticker=ticker,
            alert_type="WASH_TRADE",
            severity=severity,
            start_date=start_date,
            end_date=end_date,
            headline=headline,
            metrics={
                "max_wti": max_wti,
                "max_vol_spike": max_vol_spike,
            },
            details=details,
        )
