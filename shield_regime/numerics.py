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
