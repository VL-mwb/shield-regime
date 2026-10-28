"""
volume_profile.py — Volume anomaly and volume-price relationship analysis.

Volume is the *force* that moves the price particle. In a healthy market,
volume and price changes are correlated. When this breaks, it may indicate
wash trading or controlled insider manipulation.
"""

from __future__ import annotations
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


from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_complex_dtype
from shield_regime.validation import validate_window
from shield_regime.numerics import ensure_no_infinity

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
