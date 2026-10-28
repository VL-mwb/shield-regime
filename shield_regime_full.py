"""
==============================================================================
🛡️ ShieldRegime — Physics-Inspired Financial Market Anomaly Detection
==============================================================================

SINGLE-FILE CONSOLIDATED VERSION (shield_regime_full.py)
All models, indicators, detectors, and visualizer in one self-contained file.

This file can be:
  - Pasted directly into any AI chat (ChatGPT, Claude, Gemini) for analysis
  - Dropped into a Jupyter notebook cell for instant use
  - Used as a standalone script with `python shield_regime_full.py`

Project:  https://github.com/VL-mwb/shield-regime
License:  MIT

==============================================================================
MATHEMATICAL FOUNDATION
==============================================================================

ShieldRegime models price as a physical particle and volume as the underlying
force moving it. It computes real-time derivatives to flag manipulative trends.

Price Velocity (V):
    V_t = EMA(ΔP_t, λ)
    The rate of change of price over time, smoothed via EMA.

Price Acceleration (A):
    A_t = EMA(V_t - V_{t-1}, λ)
    The second derivative of price, capturing sudden trend thrusts.

Wash Trade Index (WTI):
    WTI_t = Volume_Spike_Ratio / (Normalized_Intraday_Range + ε)
    Screens volume-range anomalies for review, including exact-zero ranges.
    Apply both WTI and volume-ratio gates. OHLCV does not establish self-trading;
    severity describes review priority, not a probability.

Volume Spike Index:
    vol_spike_t = volume_t / rolling_mean(volume, lookback)
    Ratio of today's volume to its rolling average.

Z-Score Anomaly Detection:
    Z = (indicator - rolling_mean) / rolling_std
    Z ≥ 2.5 → statistical extreme (p < 0.012), indicative of manipulation.

==============================================================================
DETECTOR PARAMETERS & STATISTICAL BASIS
==============================================================================

PumpDumpDetector (拉高倒貨偵測器):
  - velocity_z_threshold = 2.5  → Price velocity > 2.5σ of 60-day history
                                   (p < 0.012, extreme right-tail event)
  - volume_spike_threshold = 3.0 → Volume must be 3x recent 60-day average
  - dump_window = 15             → Look for crash within 15 trading days
                                   (typical institutional dump cycle)
  - dump_drop_threshold = -20.0% → Min crash magnitude to confirm dump
                                   (separates healthy pullback from crash)
  - lookback = 60                → One quarter of trading days as baseline

WashTradeDetector (當沖洗盤偵測器):
  - wti_threshold = 3.5          → WTI > 3.5 means volume/range imbalance
                                   (e.g. 10x volume but only 2.5x range)
  - volume_spike_threshold = 2.5 → Volume must be 2.5x recent 20-day average
  - lookback = 20                → One month of trading days as baseline

==============================================================================
"""

from __future__ import annotations


from numbers import Integral, Real
from typing import Sequence
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

class InsufficientDataWarning(UserWarning):
    """Valid input contains fewer bars than the detector requires."""

def validate_window(name: str, value: int, minimum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')

def validate_positive(name: str, value: float) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite positive number')
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')

def validate_market_data(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')
    if isinstance(df.columns, pd.MultiIndex) or not df.columns.is_unique:
        raise ValueError('columns must be flat and unique')
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f'DataFrame must contain columns: {missing}')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('index must be a DatetimeIndex')
    if df.index.hasnans or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError('DatetimeIndex must contain unique increasing dates without NaT')
    prepared = df.copy(deep=True)
    names = set(required_columns) | (set(df.columns) & {'open', 'high', 'low', 'close', 'volume'})
    for name in sorted(names):
        series = df[name]
        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype) or is_complex_dtype(series.dtype):
            raise ValueError(f'{name} must have a real numeric dtype')
        try:
            with np.errstate(over='raise', invalid='raise'):
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        except (OverflowError, FloatingPointError, TypeError, ValueError) as exc:
            raise ValueError(f'{name} cannot be represented as float64') from exc
        if not np.isfinite(values).all():
            raise ValueError(f'{name} must contain finite values without NaN')
        if name == 'volume' and (values < 0).any():
            raise ValueError('volume must be >= 0')
        if name in {'open', 'high', 'low', 'close'} and (values <= 0).any():
            raise ValueError(f'{name} must be > 0')
        prepared[name] = values
    df = prepared
    if {'high', 'low'}.issubset(df.columns) and (df['high'] < df['low']).any():
        raise ValueError('high must be >= low')
    for name in ('open', 'close'):
        if name in df.columns:
            if 'high' in df.columns and (df[name] > df['high']).any():
                raise ValueError(f'{name} must be <= high')
            if 'low' in df.columns and (df[name] < df['low']).any():
                raise ValueError(f'{name} must be >= low')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if {'high', 'low'}.issubset(df.columns):
                mid_sum = df['high'].to_numpy() + df['low'].to_numpy()
                if not np.isfinite(mid_sum).all():
                    raise ValueError('high + low overflowed')
            if 'close' in df.columns and len(df) >= 2:
                close = df['close'].to_numpy()
                raw_pct = close[1:] / close[:-1] - 1.0
                if not np.isfinite(raw_pct).all():
                    raise ValueError('raw percentage change overflowed')
    except FloatingPointError as exc:
        raise ValueError('OHLCV arithmetic overflowed during preflight') from exc
    return prepared


import math
import numpy as np

_DRAWDOWN_RTOL = 8.0 * np.finfo(np.float64).eps

def finite_metric(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value

def ensure_no_infinity(name, values):
    if np.isinf(values.to_numpy(dtype=np.float64, na_value=np.nan)).any():
        raise ValueError(f'{name} contains infinite indicator values')

def drawdown_reached(close, peak, threshold):
    close = finite_metric('close', close)
    peak = finite_metric('peak', peak)
    threshold = finite_metric('threshold', threshold)
    if close <= 0 or peak <= 0 or not 0 < abs(threshold) <= 100:
        raise ValueError('drawdown requires positive prices and a threshold in (0, 100]')
    if close >= peak or abs(threshold) == 100.0:
        return False
    ratio = finite_metric('close / peak', close / peak)
    limit = 1.0 - abs(threshold) / 100.0
    return ratio <= limit or math.isclose(ratio, limit, rel_tol=_DRAWDOWN_RTOL, abs_tol=0.0)


"""
alert.py — Structured alert model for detected anomalies.

Each detection result is wrapped in an Alert object with severity,
timestamp, and human-readable explanation — ready for dashboards,
logs, or downstream compliance pipelines.
"""


from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class AlertSeverity(Enum):
    """Rule-based review priority for detected anomalies."""

    LOW = "LOW"            # Low review priority
    MEDIUM = "MEDIUM"      # Medium review priority
    HIGH = "HIGH"          # High review priority
    CRITICAL = "CRITICAL"  # Critical review priority


@dataclass
class Alert:
    """A single anomaly detection alert.

    Attributes:
        ticker:     Stock symbol (e.g. "GME", "3662").
        alert_type: Category string such as "PUMP_AND_DUMP" or "WASH_TRADE".
        severity:   Rule-based review priority.
        start_date: First date of the suspicious window.
        end_date:   Observed end; confirmation bar for pump, latest group bar for wash.
        detected_at: First bar whose completed data confirms the alert; None for legacy alerts.
        headline:   One-line human-readable summary.
        details:    Longer explanation with numbers.
        metrics:    Raw indicator values that triggered the alert.
    """

    ticker: str
    alert_type: str
    severity: AlertSeverity
    start_date: datetime
    end_date: datetime
    headline: str
    details: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    detected_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def duration_days(self) -> int:
        """Number of calendar days the suspicious window spans."""
        return max((self.end_date - self.start_date).days, 1)

    def to_dict(self) -> Dict:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "detected_at": self.detected_at.isoformat() if self.detected_at is not None else None,
            "ticker": self.ticker,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "duration_days": self.duration_days,
            "headline": self.headline,
            "details": self.details,
            "metrics": self.metrics,
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] {self.ticker} — {self.headline} "
            f"({self.start_date:%Y-%m-%d} → {self.end_date:%Y-%m-%d})"
        )


"""
velocity.py — Price Velocity (first derivative of price).

In classical mechanics, velocity = Δposition / Δtime.
Applied to markets:  velocity = Δprice / Δtime  (smoothed).

A stock under normal trading conditions has velocity that fluctuates
within a predictable band (±2σ).  When velocity breaches 3σ it may
indicate artificial price manipulation — either a pump (extreme
positive velocity) or a dump (extreme negative velocity).

This module is intentionally *general-purpose*.  It knows nothing
about specific trading strategies or proprietary parameters.
"""


import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Core indicator
# ──────────────────────────────────────────────────────────────────────

def price_velocity(
    close: pd.Series,
    window: int = 5,
    smoothing: str = "ema",
) -> pd.Series:
    """Compute smoothed price velocity (first derivative).

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.  Index should be a DatetimeIndex or
        monotonically increasing integers.
    window : int, default 5
        Smoothing window for the EMA / SMA applied *after* differencing.
    smoothing : {"ema", "sma"}, default "ema"
        Smoothing method.  EMA reacts faster to recent changes;
        SMA gives equal weight to all observations in the window.

    Returns
    -------
    pd.Series
        Smoothed daily velocity (same index as *close*).
        Units: currency per day (e.g. TWD/day, USD/day).
    """
    if len(close) < 2:
        return pd.Series(dtype=float, index=close.index)

    raw_velocity = close.diff()  # ΔPrice / Δt (Δt = 1 trading day)

    if smoothing == "ema":
        return raw_velocity.ewm(span=window, adjust=False).mean()
    elif smoothing == "sma":
        return raw_velocity.rolling(window=window, min_periods=1).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing!r}")


def price_velocity_pct(
    close: pd.Series,
    window: int = 5,
    smoothing: str = "ema",
) -> pd.Series:
    """Percentage-based velocity (normalised by price level).

    Useful when comparing stocks at vastly different price levels
    (e.g. TSMC at TWD 900 vs. a penny stock at TWD 5).

    Returns
    -------
    pd.Series
        Daily percentage velocity (decimal form, e.g. 0.03 = 3 %/day).
    """
    if len(close) < 2:
        return pd.Series(dtype=float, index=close.index)

    raw_pct = close.pct_change()  # ΔPrice / Price

    if smoothing == "ema":
        return raw_pct.ewm(span=window, adjust=False).mean()
    elif smoothing == "sma":
        return raw_pct.rolling(window=window, min_periods=1).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing!r}")


# ──────────────────────────────────────────────────────────────────────
# Z-score wrapper  (the key anomaly-detection layer)
# ──────────────────────────────────────────────────────────────────────

def velocity_zscore(
    close: pd.Series,
    velocity_window: int = 5,
    lookback: int = 60,
    smoothing: str = "ema",
    use_pct: bool = True,
) -> pd.Series:
    """Rolling Z-score of velocity against its own history.

    A Z-score > 3.0 means the current velocity is more than 3 standard
    deviations above its recent average — a strong anomaly signal.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    velocity_window : int, default 5
        Window for the underlying velocity EMA.
    lookback : int, default 60
        Rolling window (trading days) for mean / std calculation.
    smoothing : {"ema", "sma"}, default "ema"
        Smoothing method for the velocity.
    use_pct : bool, default True
        If True, use percentage velocity (price-level neutral).

    Returns
    -------
    pd.Series
        Z-score of velocity.  Positive = accelerating upward.
        Negative = accelerating downward.
    """
    vel_fn = price_velocity_pct if use_pct else price_velocity
    vel = vel_fn(close, window=velocity_window, smoothing=smoothing)

    rolling_mean = vel.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    rolling_std = vel.rolling(window=lookback, min_periods=max(lookback // 2, 1)).std()

    # Avoid division by zero in perfectly flat series
    rolling_std = rolling_std.replace(0, np.nan)

    return (vel - rolling_mean) / rolling_std


"""
acceleration.py — Price Acceleration (second derivative of price).

In physics:  acceleration = Δvelocity / Δtime.
Applied to markets:  acceleration tells us whether a trend is
*gaining strength* (positive accel) or *losing steam* (negative accel).

A hallmark of Pump-and-Dump manipulation is an extreme positive
acceleration (the pump) followed immediately by an extreme negative
acceleration (the dump).  Normal trends rarely exhibit such violent
reversals within a short window.
"""


import numpy as np
import pandas as pd



# ──────────────────────────────────────────────────────────────────────
# Core indicator
# ──────────────────────────────────────────────────────────────────────

def price_acceleration(
    close: pd.Series,
    velocity_window: int = 5,
    accel_window: int = 5,
    smoothing: str = "ema",
    use_pct: bool = True,
) -> pd.Series:
    """Compute smoothed price acceleration (second derivative).

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    velocity_window : int, default 5
        Window for the first-derivative velocity calculation.
    accel_window : int, default 5
        Smoothing window applied to the *change* in velocity.
    smoothing : {"ema", "sma"}, default "ema"
        Smoothing method.
    use_pct : bool, default True
        If True, derive acceleration from percentage velocity
        (price-level neutral).

    Returns
    -------
    pd.Series
        Smoothed daily acceleration (same index as *close*).
    """
    vel_fn = price_velocity_pct if use_pct else price_velocity
    vel = vel_fn(close, window=velocity_window, smoothing=smoothing)

    raw_accel = vel.diff()  # Δvelocity / Δt

    if smoothing == "ema":
        return raw_accel.ewm(span=accel_window, adjust=False).mean()
    elif smoothing == "sma":
        return raw_accel.rolling(window=accel_window, min_periods=1).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing!r}")


# ──────────────────────────────────────────────────────────────────────
# Z-score wrapper
# ──────────────────────────────────────────────────────────────────────

def acceleration_zscore(
    close: pd.Series,
    velocity_window: int = 5,
    accel_window: int = 5,
    lookback: int = 60,
    smoothing: str = "ema",
    use_pct: bool = True,
) -> pd.Series:
    """Rolling Z-score of acceleration against its own history.

    A Z-score > 3.0 suggests the trend is gaining force *far* faster
    than historically normal — potential pump activity.

    A sudden flip from Z > +3 to Z < -3 within a short period is a
    classic Pump-and-Dump kinematic signature.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    velocity_window : int, default 5
        Window for velocity.
    accel_window : int, default 5
        Window for acceleration smoothing.
    lookback : int, default 60
        Rolling window for Z-score mean / std.
    smoothing : {"ema", "sma"}, default "ema"
        Smoothing method.
    use_pct : bool, default True
        Use percentage-based velocity.

    Returns
    -------
    pd.Series
        Z-score of acceleration.
    """
    accel = price_acceleration(
        close,
        velocity_window=velocity_window,
        accel_window=accel_window,
        smoothing=smoothing,
        use_pct=use_pct,
    )

    rolling_mean = accel.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    rolling_std = accel.rolling(window=lookback, min_periods=max(lookback // 2, 1)).std()

    rolling_std = rolling_std.replace(0, np.nan)

    return (accel - rolling_mean) / rolling_std


# ──────────────────────────────────────────────────────────────────────
# Kinematic reversal detector (Pump → Dump signature)
# ──────────────────────────────────────────────────────────────────────

def kinematic_reversal_score(
    close: pd.Series,
    velocity_window: int = 5,
    accel_window: int = 5,
    lookback: int = 60,
    reversal_window: int = 15,
    smoothing: str = "ema",
) -> pd.Series:
    """Detect violent acceleration reversals (Pump → Dump signature).

    For each day *t*, this looks back *reversal_window* days to find the
    maximum acceleration Z-score, then computes:

        reversal_score(t) = max_accel_z(t-w..t) − accel_z(t)

    A high reversal score means the stock recently had extreme upward
    acceleration and has now shifted to deceleration or downward
    acceleration — the kinematic fingerprint of a dump phase.

    Parameters
    ----------
    reversal_window : int, default 15
        How many trading days to look back for the peak acceleration.

    Returns
    -------
    pd.Series
        Reversal score.  Higher = more suspicious.
    """
    accel_z = acceleration_zscore(
        close,
        velocity_window=velocity_window,
        accel_window=accel_window,
        lookback=lookback,
        smoothing=smoothing,
    )

    peak_accel = accel_z.rolling(window=reversal_window, min_periods=1).max()

    return peak_accel - accel_z


"""
volume_profile.py — Volume anomaly and volume-price relationship analysis.

Volume is the *force* that moves the price particle. In a healthy market,
volume and price changes are correlated. When this breaks, it may indicate
wash trading or controlled insider manipulation.
"""

import numpy as np
import pandas as pd


def volume_spike_index(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Ratio of today's volume to its rolling average.

    Returns spike index. Value of 3.0 = today is 3x the recent average.
    Values > 3.0 are considered anomalous.
    """
    avg_vol = volume.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    avg_vol = avg_vol.replace(0, np.nan)
    return volume / avg_vol


def volume_price_divergence(close: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Volume-Price Divergence Index.

    VPD = volume_spike - |price_change_zscore|

    Positive = volume-heavy anomaly (wash trade suspect).
    Negative = price-heavy anomaly (insider pump suspect).
    """
    vol_spike = volume_spike_index(volume, lookback=lookback)

    pct_change = close.pct_change().abs()
    pct_mean = pct_change.rolling(window=lookback, min_periods=max(lookback // 2, 1)).mean()
    pct_std = pct_change.rolling(window=lookback, min_periods=max(lookback // 2, 1)).std()
    pct_std = pct_std.replace(0, np.nan)
    price_z = (pct_change - pct_mean) / pct_std

    return vol_spike - price_z.abs()


def _prepare_wti_inputs(high, low, volume):
    inputs = (high, low, volume)
    if not all(isinstance(value, pd.Series) for value in inputs):
        raise TypeError('high, low and volume must be pandas Series')
    if not high.index.is_unique or not high.index.equals(low.index) or not high.index.equals(volume.index):
        raise ValueError('WTI Series must have identical unique indexes')
    prepared = []
    for name, series in zip(('high', 'low', 'volume'), inputs):
        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype) or is_complex_dtype(series.dtype):
            raise ValueError(f'{name} must have a real numeric dtype')
        try:
            with np.errstate(over='raise', invalid='raise'):
                values = series.to_numpy(dtype=np.float64, na_value=np.nan).copy()
        except (ValueError, TypeError, OverflowError, FloatingPointError) as exc:
            raise ValueError(f'{name} cannot be represented as float64') from exc
        if not np.isfinite(values).all():
            raise ValueError(f'{name} must be finite without missing values')
        if name == 'volume' and series.lt(0).any():
            raise ValueError('volume must be >= 0')
        if name == 'volume' and ((values == 0) & series.gt(0).to_numpy(dtype=bool)).any():
            raise ValueError('positive volume was lost during float64 conversion')
        if name != 'volume' and (values <= 0).any():
            raise ValueError(f'{name} must be > 0')
        prepared.append(pd.Series(values, index=series.index, name=series.name))
    raw_high = high.to_numpy(dtype=object)
    raw_low = low.to_numpy(dtype=object)
    if np.less(raw_high, raw_low).any():
        raise ValueError('high must be >= low')
    raw_zero = np.equal(raw_high, raw_low)
    if (prepared[0] < prepared[1]).any():
        raise ValueError('high must be >= low')
    if not np.array_equal(raw_zero, prepared[0].eq(prepared[1]).to_numpy(dtype=bool)):
        raise ValueError('price range was lost during float64 conversion')
    return tuple(prepared)

def wash_trade_index(high: pd.Series, low: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Screen volume relative to intrabar range; this does not establish self-trading.

    Inputs are identical unique-index numeric Series, copied to float64.
    Rolling means include the current bar; minimum observations are lookback//2
    (at least one). An entirely exact-zero range window defines range ratio zero.
    Undefined all-zero volume remains NaN; arithmetic failures raise ValueError.
    Use the separate volume gate when interpreting scores.
    """
    validate_window('lookback', lookback, 1)
    high, low, volume = _prepare_wti_inputs(high, low, volume)
    minimum = max(lookback // 2, 1)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            mid_values = (high.to_numpy() + low.to_numpy()) / 2.0
            range_values = (high.to_numpy() - low.to_numpy()) / mid_values
    except FloatingPointError as exc:
        raise ValueError('WTI intraday range arithmetic is not finite') from exc
    if not np.isfinite(range_values).all():
        raise ValueError('WTI intraday range must be finite')
    exact_zero = high.eq(low)
    if ((~exact_zero).to_numpy() & (range_values == 0)).any():
        raise ValueError('nonzero intraday range underflowed to zero')
    intraday_range = pd.Series(range_values, index=high.index, name=(high - low).name)
    baseline = intraday_range.rolling(lookback, min_periods=minimum).mean()
    zero_count = exact_zero.astype(float).rolling(lookback, min_periods=1).sum()
    sample_count = pd.Series(1.0, index=high.index).rolling(lookback, min_periods=1).sum()
    ready = sample_count.ge(minimum)
    flat_baseline = ready & zero_count.eq(sample_count)
    if (ready & ~np.isfinite(baseline)).any():
        raise ValueError('WTI range baseline is undefined after warmup')
    if (ready & ~flat_baseline & baseline.le(0)).any():
        raise ValueError('positive range baseline is not representable')
    ensure_no_infinity('WTI range baseline', baseline)
    ratio = intraday_range / baseline.replace(0, np.nan)
    ratio = ratio.mask(flat_baseline, 0.0)
    ensure_no_infinity('WTI range ratio', ratio)
    if (ready & ratio.isna()).any():
        raise ValueError('WTI range ratio is undefined after warmup')
    volume_mean = volume.rolling(lookback, min_periods=minimum).mean()
    positive_count = volume.gt(0).astype(float).rolling(lookback, min_periods=1).sum()
    zero_volume = ready & positive_count.eq(0)
    if (ready & ~np.isfinite(volume_mean)).any():
        raise ValueError('WTI volume baseline is undefined after warmup')
    if (ready & ~zero_volume & volume_mean.le(0)).any():
        raise ValueError('positive volume baseline is not representable')
    spike = volume_spike_index(volume, lookback=lookback)
    ensure_no_infinity('WTI volume spike', spike)
    expected_defined = ready & ~zero_volume
    if (expected_defined & ~np.isfinite(spike)).any():
        raise ValueError('WTI volume spike is undefined for a positive-volume window')
    if (expected_defined & volume.gt(0) & spike.eq(0)).any():
        raise ValueError('positive WTI volume spike underflowed to zero')
    if (zero_volume & spike.notna()).any():
        raise ValueError('zero-volume window has an inconsistent volume spike')
    result = spike / (ratio + 0.01)
    ensure_no_infinity('WTI', result)
    if (expected_defined & result.isna()).any():
        raise ValueError('WTI unexpectedly became undefined after warmup')
    return result


"""
pump_dump.py — Pump-and-Dump Market Manipulation Detector.

This module combines price kinematics (velocity, acceleration) and volume
profiles to detect textbook "Pump-and-Dump" schemes.

The algorithm breaks down the pattern into three kinematic phases:
  1. Base Phase: Steady price, quiet/low volume.
  2. Pump Phase: Extreme upward price velocity/acceleration coupled with massive volume spikes.
  3. Dump Phase: Violent collapse in price, often on continued high volume or complete liquidity dry-up.
"""


from datetime import datetime
from numbers import Real
from typing import List, Dict

import numpy as np
import pandas as pd


import warnings


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


"""
wash_trade.py — Rule-based volume-range anomaly screening for review.
"""


import pandas as pd


import warnings


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


"""
visualizer.py — Plotly-based interactive visualization for anomaly detection.

Creates high-impact, interactive charts highlighting detected market manipulation
phases, price kinematics, and volume spikes. Ready for Jupyter Notebooks or
web dashboards.
"""


from typing import List, Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False



def plot_alerts(
    df: pd.DataFrame,
    alerts: List[Alert],
    ticker: str = "UNKNOWN",
    lookback: int = 40,
) -> Optional[go.Figure]:
    """Generate an interactive Plotly figure visualizing detected anomalies.

    Subplot 1: Price series with shaded anomaly windows and peak flags.
    Subplot 2: Velocity Z-score with trigger thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'close' and 'volume'. Index should be DatetimeIndex.
    alerts : List[Alert]
        List of Alert objects generated by the detector.
    ticker : str, default "UNKNOWN"
        Ticker symbol.
    lookback : int, default 40
        Lookback window used for indicators.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns None if plotly is not installed, otherwise returns the Figure.
    """
    if not HAS_PLOTLY:
        print("[WARN] Plotly is not installed. Visualizer disabled.")
        return None

    # Calculate indicators for visualization
    vel_z = velocity_zscore(df["close"], lookback=lookback, use_pct=True)

    # Create Subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"{ticker} Close Price with Detected Anomaly Windows", "Price Velocity Z-Score"),
    )

    # 1. Plot Close Price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1,
        col=1,
    )

    # Highlight Alerts
    for idx, alert in enumerate(alerts, 1):
        # Shaded background region for the whole alert duration
        fig.add_vrect(
            x0=alert.start_date,
            x1=alert.end_date,
            fillcolor="rgba(239, 83, 80, 0.2)",
            layer="below",
            line_width=0,
            annotation_text=f"Anomaly #{idx}",
            annotation_position="top left",
            row=1,
            col=1,
        )

        # Draw a marker at the start of the pump
        fig.add_trace(
            go.Scatter(
                x=[alert.start_date],
                y=[df.loc[alert.start_date, "close"]],
                mode="markers+text",
                name=f"Event Start #{idx}",
                marker=dict(symbol="triangle-up", size=12, color="#4CAF50"),
                text=[f"Event Start"],
                textposition="bottom center",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Draw a marker at the end/low of the dump
        fig.add_trace(
            go.Scatter(
                x=[alert.end_date],
                y=[df.loc[alert.end_date, "close"]],
                mode="markers+text",
                name=f"Observed End #{idx}",
                marker=dict(symbol="triangle-down", size=12, color="#F44336"),
                text=[f"Observed End ({alert.metrics.get('dump_drop_pct', 0.0):.1f}%)"],
                textposition="top center",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        if alert.detected_at is not None:
            fig.add_trace(
                go.Scatter(
                    x=[alert.detected_at],
                    y=[df.loc[alert.detected_at, 'close']],
                    mode='markers+text',
                    name=f'Alert Confirmed #{idx}',
                    marker=dict(symbol='diamond', size=12, color='#FF9800'),
                    text=['Alert Confirmed'],
                    textposition='top center',
                    showlegend=False,
                ), row=1, col=1,
            )

    # 2. Plot Velocity Z-Score
    fig.add_trace(
        go.Scatter(
            x=vel_z.index,
            y=vel_z,
            mode="lines",
            name="Velocity Z-Score",
            line=dict(color="#e377c2", width=1.5),
        ),
        row=2,
        col=1,
    )

    # Threshold guidelines
    fig.add_hline(y=2.5, line_dash="dash", line_color="#EF5350", annotation_text="+2.5σ Limit", row=2, col=1)
    fig.add_hline(y=-2.5, line_dash="dash", line_color="#EF5350", annotation_text="-2.5σ Limit", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="#7f7f7f", row=2, col=1)

    # Styling and Layout
    fig.update_layout(
        title=dict(
            text=f"<b>🛡️ ShieldRegime Analysis — {ticker}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18, family="Arial"),
        ),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("  ShieldRegime Full — Self-Contained Demo")
    print("=" * 70)

    try:
        import yfinance as yf

        # Download GME data around the 2021 short squeeze
        print("\nDownloading GME historical data (2020-09 to 2021-04)...")
        df = yf.download("GME", start="2020-09-01", end="2021-04-30", progress=False)

        if df.empty:
            print("No data downloaded. Check your internet connection.")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]

            # Run PumpDump Detector
            pd_detector = PumpDumpDetector(
                velocity_z_threshold=2.0,
                volume_spike_threshold=3.0,
                dump_drop_threshold=-25.0,
                lookback=40,
            )
            pd_alerts = pd_detector.scan(df, ticker="GME")

            # Run WashTrade Detector
            wt_detector = WashTradeDetector(wti_threshold=3.5)
            wt_alerts = wt_detector.scan(df, ticker="GME")

            all_alerts = pd_alerts + wt_alerts

            if all_alerts:
                print(f"\n{len(all_alerts)} alert(s) detected:")
                for alert in all_alerts:
                    print(f"  {alert}")
                    print(f"    Details: {alert.details}")
                    print(f"    Metrics: {alert.metrics}\n")
            else:
                print("\nNo anomalies detected.")

    except ImportError:
        print("\nyfinance not installed. Install with: pip install yfinance")
        print("This demo requires yfinance for data download.")
