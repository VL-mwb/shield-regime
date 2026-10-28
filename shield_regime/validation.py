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
