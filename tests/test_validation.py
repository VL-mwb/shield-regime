import pandas as pd
import numpy as np
import pytest
from shield_regime import PumpDumpDetector, WashTradeDetector

@pytest.fixture
def valid():
    return pd.DataFrame(dict(close=[100.]*80,high=[101.]*80,low=[99.]*80,volume=[100.]*80),index=pd.bdate_range('2024-01-01',periods=80))

@pytest.mark.parametrize('kind',[PumpDumpDetector,WashTradeDetector])
@pytest.mark.parametrize('fault',['missing','index','reverse','duplicate','nat','nan','inf','string','bool','complex','negative_volume','zero_price','bad_high','multi','duplicate_columns'])
def test_invalid(kind,fault,valid):
    d=valid.copy()
    if fault=='missing': d=d.drop(columns='volume')
    elif fault=='index': d.index=range(len(d))
    elif fault=='reverse': d=d.iloc[::-1]
    elif fault=='duplicate': d.index=[d.index[0]]*len(d)
    elif fault=='nat': d.index=pd.DatetimeIndex([pd.NaT]*len(d))
    elif fault in ('nan','inf'): d.iloc[0,0]=np.nan if fault=='nan' else np.inf
    elif fault=='string': d['close']=d.close.astype(str)
    elif fault=='bool': d['close']=True
    elif fault=='complex': d['close']=d.close.astype(complex)
    elif fault=='negative_volume': d.iloc[0,3]=-1
    elif fault=='zero_price': d.iloc[0,0]=0
    elif fault=='bad_high': d.iloc[0,1]=98
    elif fault=='multi': d.columns=pd.MultiIndex.from_product([d.columns,['X']])
    elif fault=='duplicate_columns': d.columns=['close','high','low','close']
    before=d.copy(deep=True)
    with pytest.raises(ValueError): kind().scan(d)
    pd.testing.assert_frame_equal(d,before)

@pytest.mark.parametrize('kind',[PumpDumpDetector,WashTradeDetector])
def test_short_and_schema(kind,valid):
    from shield_regime import InsufficientDataWarning
    with pytest.warns(InsufficientDataWarning): assert kind().scan(valid.iloc[:1])==[]
    with pytest.raises(ValueError): kind().scan(valid.iloc[:1].drop(columns='volume'))
    with pytest.raises(TypeError): kind().scan(None)

@pytest.mark.parametrize('value',[True,0,1,2.5,float('nan'),float('inf')])
@pytest.mark.parametrize('kind',[PumpDumpDetector,WashTradeDetector])
def test_bad_lookback(kind,value):
    with pytest.raises(ValueError): kind(lookback=value)

@pytest.mark.parametrize('value',[0,True,float('nan'),float('inf'),101])
def test_bad_drop(value):
    with pytest.raises(ValueError): PumpDumpDetector(dump_drop_threshold=value)

def test_normalization(valid):
    from shield_regime.validation import validate_market_data
    for dtype in ['int8','int64','Int64','Float64','float64']:
        d=valid.astype(dtype);before=d.copy(deep=True)
        result=validate_market_data(d,('close','volume'))
        pd.testing.assert_frame_equal(d,before)
        assert result is not d
        assert all(str(x)=='float64' for x in result.dtypes)

def test_zero_and_timezone(valid):
    valid['high']=valid['low']=valid['close'];valid['volume']=0
    valid.index=valid.index.tz_localize('Europe/London')
    assert WashTradeDetector().scan(valid)==[]

def test_derived_overflow():
    from shield_regime.validation import validate_market_data
    for d in [pd.DataFrame(dict(close=[1e-308,1e308],volume=[1.,1.]),index=pd.date_range('2024-01-01',periods=2)),pd.DataFrame(dict(close=[1e308],high=[1e308],low=[9e307],volume=[1.]),index=pd.date_range('2024-01-01',periods=1))]:
        with pytest.raises(ValueError):validate_market_data(d,('close','volume'))


@pytest.mark.parametrize('kind,argument',[(PumpDumpDetector,'velocity_z_threshold'),(PumpDumpDetector,'volume_spike_threshold'),(WashTradeDetector,'wti_threshold'),(WashTradeDetector,'volume_spike_threshold')])
@pytest.mark.parametrize('value',[True,'3',0,-1,float('nan'),float('inf')])
def test_threshold_parameters(kind,argument,value):
    with pytest.raises(ValueError):kind(**{argument:value})

@pytest.mark.parametrize('value',[True,0,1.5])
def test_dump_window(value):
    with pytest.raises(ValueError):PumpDumpDetector(dump_window=value)

def test_open_bounds_and_empty(valid):
    from shield_regime import InsufficientDataWarning
    from shield_regime.validation import validate_market_data
    for value in [98.,102.]:
        bad=valid.assign(open=value)
        with pytest.raises(ValueError):validate_market_data(bad,('close','volume'))
    for cls in [PumpDumpDetector,WashTradeDetector]:
        with pytest.warns(InsufficientDataWarning):assert cls().scan(valid.iloc[:0])==[]

def test_dtype_event_invariance():
    from shield_regime.validation import validate_market_data
    from shield_regime.indicators.volume_profile import wash_trade_index
    base=pd.DataFrame(dict(close=[50]*20+[100],high=[51]*20+[101],low=[49]*20+[99],volume=[100]*20+[1000]),index=pd.bdate_range('2024-01-01',periods=21))
    results=[]
    for dtype in ['int8','int64','Int64','Float64','float64']:
        d=base.copy();d[['close','high','low']]=d[['close','high','low']].astype(dtype);before=d.copy(deep=True)
        prepared=validate_market_data(d,('high','low','close','volume'))
        assert wash_trade_index(prepared.high,prepared.low,prepared.volume).iloc[-1]==pytest.approx(13.191050379665485)
        results.append([a.to_dict() for a in WashTradeDetector().scan(d)])
        pd.testing.assert_frame_equal(d,before)
    assert all(r==results[0] for r in results) and len(results[0])==1
