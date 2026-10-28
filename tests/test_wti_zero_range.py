import numpy as np
import pandas as pd
import pytest
from shield_regime.indicators.volume_profile import wash_trade_index, volume_spike_index

def test_flat_baseline_spike_is_finite():
    high = pd.Series([100.0] * 25)
    volume = pd.Series([100.0] * 20 + [1000.0] * 5)
    actual = wash_trade_index(high, high.copy(), volume, lookback=20)
    assert actual.iloc[:9].isna().all()
    assert actual.iloc[20] == pytest.approx(100.0 * (1000.0 / 145.0))
    assert np.isfinite(actual.iloc[9:]).all()

def test_steady_flat_data_requires_volume_gate():
    high = pd.Series([100.0] * 25)
    volume = pd.Series([100.0] * 25)
    assert wash_trade_index(high, high, volume).iloc[-1] == pytest.approx(100.0)
    assert volume_spike_index(volume).iloc[-1] == pytest.approx(1.0)


def test_float32_matches_formula_on_float64_copies():
    rng = np.random.default_rng(42)
    low = pd.Series(rng.uniform(90, 110, 50).astype('float32'))
    high = low + pd.Series(rng.uniform(0.01, 1, 50).astype('float32'))
    volume = pd.Series(rng.uniform(100, 1000, 50).astype('float32'))
    high64, low64, volume64 = (s.astype('float64') for s in (high, low, volume))
    intraday = (high64 - low64) / ((high64 + low64) / 2)
    baseline = intraday.rolling(20, min_periods=10).mean().replace(0, np.nan)
    expected = volume_spike_index(volume64) / (intraday / baseline + 0.01)
    np.testing.assert_allclose(wash_trade_index(high, low, volume), expected,
                               rtol=1e-12, atol=0, equal_nan=True)


@pytest.mark.parametrize('case', ['collapsed', 'mixed', 'reversed', 'unsigned', 'huge_volume', 'tiny_volume'])
def test_wti_rejects_unreliable_arithmetic(case):
    high = pd.Series([100.0] * 25)
    low = high.copy()
    volume = pd.Series([100.0] * 20 + [1000.0] * 5)
    if case in ('collapsed', 'mixed'):
        high = pd.Series([2**53 + 1] * 25, dtype='int64')
        low = pd.Series([2**53] * 25, dtype='int64' if case == 'collapsed' else 'float64')
    elif case == 'reversed':
        high = pd.Series([2**53] * 25, dtype='float64')
        low = pd.Series([2**53 + 1] * 25, dtype='int64')
    elif case == 'unsigned':
        high = pd.Series([2**63] * 25, dtype='uint64')
        low = pd.Series([2**63 - 1] * 25, dtype='int64')
    elif case == 'huge_volume':
        volume *= 1e305
    else:
        volume = pd.Series([0.0] * 19 + [np.nextafter(0.0, 1.0)] + [0.0] * 5)
    before = [s.copy(deep=True) for s in (high, low, volume)]
    with pytest.raises(ValueError):
        wash_trade_index(high, low, volume)
    for actual, original in zip((high, low, volume), before):
        pd.testing.assert_series_equal(actual, original)


@pytest.mark.parametrize('dtype', ['int8', 'int64', 'Int64', 'Float64', 'float32', 'float64'])
def test_dtype_and_input_immutability(dtype):
    high = pd.Series([101]*30, dtype=dtype)
    low = pd.Series([99]*30, dtype=dtype)
    volume = pd.Series([10]*20+[100]*10, dtype=dtype)
    before = [s.copy(deep=True) for s in (high, low, volume)]
    actual = wash_trade_index(high, low, volume)
    expected = volume_spike_index(volume.astype(float)) / 1.01
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0, equal_nan=True)
    for a,b in zip((high,low,volume),before):
        pd.testing.assert_series_equal(a,b)


def test_transitions_single_zero_and_positive_regression():
    low = pd.Series([100.]*80)
    high = low + pd.Series([1.]*25+[0.]*30+[1.]*25)
    volume = pd.Series([100.]*80)
    actual = wash_trade_index(high,low,volume)
    raw = (high-low)/((high+low)/2)
    baseline = raw.rolling(20,min_periods=10).mean().replace(0,np.nan)
    legacy = volume_spike_index(volume)/(raw/baseline+.01)
    mask = baseline.notna()
    np.testing.assert_allclose(actual[mask],legacy[mask],rtol=1e-12,atol=0,equal_nan=True)
    assert actual.iloc[44:55].eq(100.).all()
    assert actual.iloc[:9].isna().all()


def test_zero_volume_and_zero_current():
    h=pd.Series([100.]*25)
    assert wash_trade_index(h,h,pd.Series([0.]*25)).isna().all()
    v=pd.Series([100.]*24+[0.])
    assert wash_trade_index(h,h,v).iloc[-1]==0.


def test_empty_small_range_and_one_bar_window():
    e=pd.Series([],dtype=float)
    assert wash_trade_index(e,e,e).empty
    low=pd.Series([1.]*25)
    high=pd.Series([np.nextafter(1.,2.)]*25)
    volume=pd.Series([100.]*25)
    assert wash_trade_index(high,low,volume).iloc[-1]==pytest.approx(1/1.01)
    assert wash_trade_index(low,low,volume,1).eq(100.).all()


@pytest.mark.parametrize('case', ['nonseries','misaligned','duplicate','string','bool','complex','nan','inf','negative','zero_price','bounds','overflow'])
def test_invalid_inputs_preserve_series(case):
    h=pd.Series([101.]*25);l=pd.Series([99.]*25);v=pd.Series([100.]*25)
    if case=='nonseries':h=h.tolist()
    elif case=='misaligned':l.index=range(1,26)
    elif case=='duplicate':h.index=[0]*25;l.index=h.index;v.index=h.index
    elif case=='string':h=h.astype(str)
    elif case=='bool':h=h.astype(bool)
    elif case=='complex':h=h.astype(complex)
    elif case=='nan':v.iloc[0]=np.nan
    elif case=='inf':v.iloc[-1]=np.inf
    elif case=='negative':v.iloc[-1]=-1.
    elif case=='zero_price':l.iloc[-1]=0.
    elif case=='bounds':l.iloc[-1]=102.
    elif case=='overflow':h[:]=1e308;l[:]=1e308
    before=[s.copy(deep=True) if isinstance(s,pd.Series) else list(s) for s in (h,l,v)]
    with pytest.raises(TypeError if case=='nonseries' else ValueError):wash_trade_index(h,l,v)
    for a,b in zip((h,l,v),before):
        if isinstance(a,pd.Series):pd.testing.assert_series_equal(a,b)
        else:assert a==b


@pytest.mark.parametrize('lookback',[True,0,-1,1.5,np.inf,np.nan])
def test_invalid_window(lookback):
    h=pd.Series([100.]*25)
    with pytest.raises(ValueError):wash_trade_index(h,h,h,lookback)


def test_unsorted_generic_index_and_prefix_causality():
    idx=pd.Index([3,1,2]+list(range(4,26)))
    h=pd.Series(100.,index=idx);v=pd.Series([100.]*20+[1000.]*5,index=idx)
    full=wash_trade_index(h,h,v)
    assert full.index.equals(idx)
    for n in range(1,26):
        pd.testing.assert_series_equal(wash_trade_index(h.iloc[:n],h.iloc[:n],v.iloc[:n]),full.iloc[:n])
