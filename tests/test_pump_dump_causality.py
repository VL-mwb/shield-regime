import numpy as np
import pandas as pd
import pytest
from shield_regime import PumpDumpDetector

def test_confirms_known_tail_crash(manipulated_data):
    detector = PumpDumpDetector(lookback=40, velocity_z_threshold=2.0)
    prefix = manipulated_data.iloc[:63]
    alerts = detector.scan(prefix, ticker='TAIL')
    assert alerts
    assert alerts[0].detected_at <= prefix.index[-1]
    assert alerts[0].end_date == alerts[0].detected_at
    assert alerts[0].metrics['dump_drop_pct'] <= -20

def test_pump_alerts_do_not_repaint(manipulated_data):
    detector = PumpDumpDetector(lookback=40, velocity_z_threshold=2.0)
    full = detector.scan(manipulated_data, ticker='X')
    for size in range(41, len(manipulated_data) + 1):
        cutoff = manipulated_data.index[size - 1]
        actual = detector.scan(manipulated_data.iloc[:size], ticker='X')
        expected = [a for a in full if a.detected_at <= cutoff]
        assert [a.to_dict() for a in actual] == [a.to_dict() for a in expected]

def test_second_pump():
    p=np.full(100,100.);p[50:55]=130;p[55]=200;p[56:]=150
    vol=np.full(100,100.);vol[[50,55]]=1000
    df=pd.DataFrame(dict(close=p,volume=vol),index=pd.bdate_range('2024-01-01',periods=100))
    a=PumpDumpDetector(lookback=40).scan(df)
    assert len(a)==1 and a[0].start_date==df.index[55] and a[0].detected_at==df.index[56]

@pytest.mark.parametrize('prices,start,detect', [([200,190,180,159],20,23),([200,195,190,190,150],22,24),([200,190,220,160],20,23)])
def test_candidate_windows(monkeypatch,prices,start,detect):
    import shield_regime.detectors.pump_dump as mod
    p=np.full(30,100.);p[20:20+len(prices)]=prices;p[20+len(prices):]=prices[-1]
    df=pd.DataFrame(dict(close=p,volume=100.),index=pd.bdate_range('2024-01-01',periods=30))
    v=pd.Series(1.,index=df.index);v.iloc[[20,22]]=3
    vol=pd.Series(1.,index=df.index);vol.iloc[[20,22]]=4
    monkeypatch.setattr(mod,'velocity_zscore',lambda *a,**k:v)
    monkeypatch.setattr(mod,'acceleration_zscore',lambda *a,**k:v*0)
    monkeypatch.setattr(mod,'volume_spike_index',lambda *a,**k:vol)
    alerts=PumpDumpDetector(lookback=20,dump_window=3).scan(df)
    assert len(alerts)==1
    assert alerts[0].start_date==df.index[start] and alerts[0].detected_at==df.index[detect]

@pytest.mark.parametrize('drop_at,price,expected',[(21,80.,1),(23,80.,1),(24,80.,0),(21,80.000000001,0),(21,60.,1)])
def test_exact_window_and_threshold(monkeypatch,drop_at,price,expected):
    import shield_regime.detectors.pump_dump as mod
    p=np.full(30,100.);p[drop_at:]=price
    df=pd.DataFrame(dict(close=p,volume=100.),index=pd.bdate_range('2024-01-01',periods=30))
    v=pd.Series(1.,index=df.index);v.iloc[20]=3
    vol=pd.Series(1.,index=df.index);vol.iloc[20]=6
    monkeypatch.setattr(mod,'velocity_zscore',lambda *a,**k:v)
    monkeypatch.setattr(mod,'acceleration_zscore',lambda *a,**k:v*0)
    monkeypatch.setattr(mod,'volume_spike_index',lambda *a,**k:vol)
    alerts=PumpDumpDetector(lookback=20,dump_window=3).scan(df)
    assert len(alerts)==expected
    if alerts:
        assert alerts[0].detected_at==df.index[drop_at]
        if price==60: assert alerts[0].severity.value=='CRITICAL'


def test_two_separate_events_and_future_extension():
    p=np.full(200,100.);v=np.full(200,100.)
    for offset in [0,100]:
        p[offset+50:offset+55]=130;p[offset+55]=200;p[offset+56:offset+100]=150;v[[offset+50,offset+55]]=1000
    df=pd.DataFrame(dict(close=p,volume=v),index=pd.bdate_range('2024-01-01',periods=200))
    detector=PumpDumpDetector(lookback=40)
    first=detector.scan(df.iloc[:100]);full=detector.scan(df)
    assert len(first)==1 and len(full)==2
    assert first[0].to_dict()==full[0].to_dict()
    assert full[1].detected_at==df.index[156]
    for size in [55,56,57,99,155,156,157,200]:
        assert [a.to_dict() for a in detector.scan(df.iloc[:size])]==[a.to_dict() for a in full if a.detected_at<=df.index[size-1]]


@pytest.mark.parametrize('fall_index,expected',[(21,1),(24,0)])
def test_momentum_continues_and_equal_peak_not_reset(monkeypatch,fall_index,expected):
    import shield_regime.detectors.pump_dump as mod
    df=pd.DataFrame(dict(close=[100.]*30,volume=[100.]*30),index=pd.bdate_range('2024-01-01',periods=30))
    df.iloc[fall_index:,0]=80.
    v=pd.Series(2.,index=df.index);v.iloc[20]=3.
    vol=pd.Series(1.,index=df.index);vol.iloc[20]=4.
    monkeypatch.setattr(mod,'velocity_zscore',lambda *a,**k:v)
    monkeypatch.setattr(mod,'acceleration_zscore',lambda *a,**k:v*0)
    monkeypatch.setattr(mod,'volume_spike_index',lambda *a,**k:vol)
    assert len(PumpDumpDetector(lookback=20,dump_window=3).scan(df))==expected

def test_continuous_pump_has_one_candidate(monkeypatch):
    import shield_regime.detectors.pump_dump as mod
    df=pd.DataFrame(dict(close=np.arange(100.,125.),volume=100.),index=pd.bdate_range('2024-01-01',periods=25))
    v=pd.Series(4.,index=df.index);created=[];original=mod._Candidate
    def candidate(*a,**k):created.append(a[0]);return original(*a,**k)
    monkeypatch.setattr(mod,'_Candidate',candidate)
    monkeypatch.setattr(mod,'velocity_zscore',lambda *a,**k:v)
    monkeypatch.setattr(mod,'acceleration_zscore',lambda *a,**k:v*0)
    monkeypatch.setattr(mod,'volume_spike_index',lambda *a,**k:v)
    assert PumpDumpDetector(lookback=20).scan(df)==[]
    assert created==[20]
