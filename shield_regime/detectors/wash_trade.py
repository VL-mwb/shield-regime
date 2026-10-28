"""wash_trade.py — Rule-based volume-range anomaly screening for review."""

from __future__ import annotations

import pandas as pd

from shield_regime.indicators.volume_profile import _prepare_wti_inputs, volume_spike_index, wash_trade_index
from shield_regime.models.alert import Alert, AlertSeverity
from shield_regime.numerics import finite_metric, ensure_no_infinity

import warnings
from shield_regime.validation import validate_market_data, validate_window, validate_positive, InsufficientDataWarning


class WashTradeDetector:
    """Screen volume-range anomalies for review; OHLCV does not identify trading parties."""

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
        validate_window('lookback', lookback, 2)
        validate_positive('wti_threshold', wti_threshold)
        validate_positive('volume_spike_threshold', volume_spike_threshold)
        self.wti_threshold = wti_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.lookback = lookback

    def scan(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> list[Alert]:
        """Scan a historical dataframe for volume-range anomaly candidates.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns 'high', 'low', 'close', and 'volume'.
        ticker : str, default "UNKNOWN"
            Ticker of the asset being scanned.

        Returns
        -------
        list[Alert]
            List of volume-range screening alerts.
        """
        raw_df = df
        df = validate_market_data(raw_df, ('high', 'low', 'close', 'volume'))
        high, low, volume = _prepare_wti_inputs(raw_df['high'], raw_df['low'], raw_df['volume'])
        if len(df) < self.lookback:
            warnings.warn('WashTradeDetector requires at least lookback bars', InsufficientDataWarning, stacklevel=2)
            return []
        wti = wash_trade_index(high, low, volume, lookback=self.lookback)
        vol_spike = volume_spike_index(volume, lookback=self.lookback)

        ensure_no_infinity("WTI", wti)
        ensure_no_infinity("volume spike", vol_spike)

        # 2. Identify anomaly flags
        is_anomaly = (wti >= self.wti_threshold) & (vol_spike >= self.volume_spike_threshold)

        is_anomaly.iloc[:self.lookback - 1] = False

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
        """Package observed volume-range anomaly statistics into an Alert."""
        start_date = df.index[start_idx]
        end_date = df.index[end_idx]
        duration = (end_idx - start_idx) + 1

        sub_wti = wti.iloc[start_idx : end_idx + 1]
        sub_vol_spike = vol_spike.iloc[start_idx : end_idx + 1]

        max_wti = finite_metric("max_wti", sub_wti.max())
        max_vol_spike = finite_metric("max_vol_spike", sub_vol_spike.max())

        zero_range_days = int(df['high'].iloc[start_idx:end_idx + 1].eq(
            df['low'].iloc[start_idx:end_idx + 1]).sum())

        # Determine severity
        if max_wti >= 8.0 and duration >= 3:
            severity = AlertSeverity.CRITICAL
        elif max_wti >= 5.0:
            severity = AlertSeverity.HIGH
        else:
            severity = AlertSeverity.MEDIUM

        headline = f'Volume-Range Anomaly: Peak WTI {max_wti:.2f}'
        details = (
            f'Observed {duration} consecutive volume-range anomaly bar(s). '
            f'Peak volume ratio was {max_vol_spike:.2f}x the rolling mean '
            f'(including the current bar), and peak WTI was {max_wti:.2f}. '
            f'{zero_range_days} bar(s) had high equal to low. '
            'This OHLCV screening signal does not establish self-trading or market manipulation. '
            'Severity indicates review priority, not a probability.'
        )

        return Alert(
            ticker=ticker,
            alert_type="WASH_TRADE",
            severity=severity,
            start_date=start_date,
            detected_at=start_date,
            end_date=end_date,
            headline=headline,
            metrics={
                "max_wti": max_wti,
                "max_vol_spike": max_vol_spike,
                "zero_range_days": float(zero_range_days),
            },
            details=details,
        )
