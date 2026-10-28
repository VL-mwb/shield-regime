import importlib.util
import sys
from pathlib import Path
import pytest
import pandas as pd
from shield_regime import PumpDumpDetector, WashTradeDetector

@pytest.fixture
def standalone():
    spec=importlib.util.spec_from_file_location('standalone_probe',Path(__file__).resolve().parents[1]/'shield_regime_full.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def test_pump_parity(standalone,manipulated_data):
    for size in [63,100]:
        expected=PumpDumpDetector(lookback=40).scan(manipulated_data.iloc[:size])
        actual=standalone.PumpDumpDetector(lookback=40).scan(manipulated_data.iloc[:size])
        assert [a.to_dict() for a in actual]==[a.to_dict() for a in expected]

def test_errors_and_short(standalone):
    for cls in [standalone.PumpDumpDetector,standalone.WashTradeDetector]:
        with pytest.raises(ValueError):cls().scan(pd.DataFrame())
        df=pd.DataFrame(dict(close=[100.],high=[101.],low=[99.],volume=[10.]),index=pd.date_range('2024-01-01',periods=1))
        with pytest.warns(standalone.InsufficientDataWarning):assert cls().scan(df)==[]


def test_numeric_and_dtype_parity(standalone):
    from shield_regime.numerics import drawdown_reached
    base=pd.DataFrame(dict(close=[50]*20+[100],high=[51]*20+[101],low=[49]*20+[99],volume=[100]*20+[1000]),index=pd.bdate_range('2024-01-01',periods=21))
    for dtype in ['int8','Int64','float64']:
        d=base.copy();d[['close','high','low']]=d[['close','high','low']].astype(dtype)
        assert [a.to_dict() for a in standalone.WashTradeDetector().scan(d)]==[a.to_dict() for a in WashTradeDetector().scan(d)]
    for price in [80,80+1e-9,80-1e-9,60]:
        assert standalone.drawdown_reached(price,100,-20)==drawdown_reached(price,100,-20)

def test_repeated_pump_parity(standalone):
    import numpy as np
    p=np.full(100,100.);p[50:55]=130;p[55]=200;p[56:]=150
    v=np.full(100,100.);v[[50,55]]=1000
    d=pd.DataFrame(dict(close=p,volume=v),index=pd.bdate_range('2024-01-01',periods=100))
    assert [a.to_dict() for a in standalone.PumpDumpDetector(lookback=40).scan(d)]==[a.to_dict() for a in PumpDumpDetector(lookback=40).scan(d)]


@pytest.mark.parametrize('scenario',['spike','steady','zero','single_zero','positive'])
def test_wti_payload_and_prefix_parity(standalone,scenario):
    import numpy as np
    import warnings
    from shield_regime import wash_trade_index
    high=np.full(25,100.);low=high.copy();volume=np.array([100.]*20+[1000.]*5)
    if scenario=='steady':volume[:]=100.
    elif scenario=='zero':volume[:]=0.
    elif scenario=='single_zero':high[:]=101.;low[:]=99.;high[20]=low[20]=100.
    elif scenario=='positive':high[:]=101.;low[:]=99.
    df=pd.DataFrame(dict(high=high,low=low,close=100.,volume=volume),index=pd.bdate_range('2024-01-01',periods=25))
    np.testing.assert_allclose(standalone.wash_trade_index(df.high,df.low,df.volume),wash_trade_index(df.high,df.low,df.volume),rtol=1e-12,atol=0,equal_nan=True)
    for size in range(1,26):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a=standalone.WashTradeDetector().scan(df.iloc[:size]);b=WashTradeDetector().scan(df.iloc[:size])
        assert [x.to_dict() for x in a]==[x.to_dict() for x in b]
    if scenario=='spike':assert b[0].metrics['zero_range_days']==5.


@pytest.mark.parametrize('case',['collapsed','mixed','reversed','unsigned','huge_volume','tiny_volume','missing','indexes'])
def test_wti_error_parity(standalone,case):
    import numpy as np
    from shield_regime import wash_trade_index
    h=pd.Series([100.]*25);l=h.copy();v=pd.Series([100.]*20+[1000.]*5)
    if case in ('collapsed','mixed'):
        h=pd.Series([2**53+1]*25,dtype='int64');l=pd.Series([2**53]*25,dtype='int64' if case=='collapsed' else 'float64')
    elif case=='reversed':h=pd.Series([2**53]*25,dtype='float64');l=pd.Series([2**53+1]*25,dtype='int64')
    elif case=='unsigned':h=pd.Series([2**63]*25,dtype='uint64');l=pd.Series([2**63-1]*25,dtype='int64')
    elif case=='huge_volume':v*=1e305
    elif case=='tiny_volume':v=pd.Series([0.]*19+[np.nextafter(0.,1.)]+[0.]*5)
    elif case=='missing':v.iloc[0]=np.nan
    else:l.index=range(1,26)
    for fn in [wash_trade_index,standalone.wash_trade_index]:
        with pytest.raises(ValueError):fn(h,l,v)
    df=pd.DataFrame(dict(high=h,low=l,close=l.copy(),volume=v))
    df.index=pd.bdate_range('2024-01-01',periods=len(df))
    for cls in [WashTradeDetector,standalone.WashTradeDetector]:
        with pytest.raises(ValueError):cls().scan(df)
