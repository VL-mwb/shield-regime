import importlib
from types import SimpleNamespace
import pandas as pd
import pytest
from shield_regime import Alert, AlertSeverity

def test_replay_keeps_first_snapshot():
    from examples.validate_forward_returns import collect_first_observed_alerts
    df=pd.DataFrame(dict(close=[100.]*4),index=pd.date_range('2024-01-01',periods=4))
    class Detector:
        def scan(self,prefix,ticker):
            if len(prefix)<2:return []
            return [Alert(ticker,'WASH_TRADE',AlertSeverity.HIGH,prefix.index[1],prefix.index[-1],'test',metrics={'size':len(prefix)},detected_at=prefix.index[1])]
    alerts=collect_first_observed_alerts(df,[Detector()])
    assert len(alerts)==1 and alerts[0].metrics['size']==2

def test_legacy_rejected():
    from examples.validate_forward_returns import collect_first_observed_alerts
    df=pd.DataFrame(index=pd.date_range('2024-01-01',periods=2))
    d=SimpleNamespace(scan=lambda prefix,ticker:[Alert(ticker,'X',AlertSeverity.HIGH,prefix.index[0],prefix.index[0],'legacy')])
    with pytest.raises(ValueError): collect_first_observed_alerts(df,[d])


def test_imports_never_download(monkeypatch):
    import sys
    def forbidden(*args,**kwargs):raise AssertionError('import attempted a download')
    monkeypatch.setitem(sys.modules,'yfinance',SimpleNamespace(download=forbidden))
    import examples.validate_forward_returns as forward
    import examples.trace_state_machine_6182 as trace
    importlib.reload(forward)
    importlib.reload(trace)

def test_no_future_window_is_fabricated(monkeypatch,capsys):
    import sys
    import examples.validate_forward_returns as mod
    df=pd.DataFrame(dict(close=[100.]*4,high=[101.]*4,low=[99.]*4,volume=[100.]*4),index=pd.date_range('2024-01-01',periods=4))
    monkeypatch.setitem(sys.modules,'yfinance',SimpleNamespace(download=lambda *a,**k:df.copy().rename(columns=str.title)))
    a=Alert('X','WASH_TRADE',AlertSeverity.HIGH,df.index[-1],df.index[-1],'test',detected_at=df.index[-1])
    monkeypatch.setattr(mod,'collect_first_observed_alerts',lambda *a,**k:[a_alert])
    a_alert=a
    mod.verify_post_alert_performance('X','test','2024-01-01','2024-02-01')
    out=capsys.readouterr().out
    assert 'T+5' not in out and '暫無足夠' in out
