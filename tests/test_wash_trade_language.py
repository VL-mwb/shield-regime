import numpy as np
import pandas as pd
import pytest

def test_zero_range_group_and_volume_gate():
    from shield_regime import WashTradeDetector
    index = pd.bdate_range('2024-01-01', periods=25)
    data = pd.DataFrame({'high': 100.0, 'low': 100.0, 'close': 100.0,
                         'volume': [100.0] * 20 + [1000.0] * 5}, index=index)
    detector = WashTradeDetector()
    alerts = detector.scan(data, ticker='FLAT')
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.start_date == alert.detected_at == index[20]
    assert alert.end_date == index[24]
    assert alert.metrics['zero_range_days'] == 5.0
    assert detector.scan(data.assign(volume=100.0)) == []
    assert detector.scan(data.assign(volume=0.0)) == []

@pytest.mark.parametrize('case', ['collapsed', 'mixed', 'reversed', 'unsigned', 'huge_volume', 'tiny_volume'])
def test_detector_rejects_unreliable_original_data(case):
    from shield_regime import WashTradeDetector
    data = pd.DataFrame({'high': [100.0] * 25, 'low': [100.0] * 25,
                         'close': [100.0] * 25, 'volume': [100.0] * 20 + [1000.0] * 5},
                        index=pd.bdate_range('2024-01-01', periods=25))
    if case in ('collapsed', 'mixed'):
        data['high'] = np.full(25, 2**53 + 1, dtype='int64')
        data['low'] = np.full(25, 2**53, dtype='int64' if case == 'collapsed' else 'float64')
    elif case == 'reversed':
        data['high'] = np.full(25, 2**53, dtype='float64')
        data['low'] = np.full(25, 2**53 + 1, dtype='int64')
    elif case == 'unsigned':
        data['high'] = np.full(25, 2**63, dtype='uint64')
        data['low'] = np.full(25, 2**63 - 1, dtype='int64')
    elif case == 'huge_volume':
        data['volume'] *= 1e305
    else:
        data['volume'] = [0.0] * 19 + [np.nextafter(0.0, 1.0)] + [0.0] * 5
    if case in ('collapsed', 'mixed', 'reversed', 'unsigned'):
        data['close'] = data['low'].copy()
    before = data.copy(deep=True)
    with pytest.raises(ValueError):
        WashTradeDetector().scan(data)
    pd.testing.assert_frame_equal(data, before)
    if case in ('collapsed', 'mixed', 'reversed', 'unsigned'):
        with pytest.raises(ValueError):
            WashTradeDetector().scan(data.iloc[:1])


def test_neutral_alert_observations_and_json():
    import json
    from shield_regime import WashTradeDetector
    prices=np.arange(100.,125.)
    data=pd.DataFrame(dict(high=prices,low=prices,close=prices,volume=[100.]*20+[1000.]*5),index=pd.bdate_range('2024-01-01',periods=25))
    alert=WashTradeDetector().scan(data)[0]
    assert alert.headline.startswith('Volume-Range Anomaly: Peak WTI ')
    assert '5 consecutive volume-range anomaly bar(s)' in alert.details
    assert 'including the current bar' in alert.details
    assert '5 bar(s) had high equal to low' in alert.details
    assert 'does not establish self-trading or market manipulation' in alert.details
    assert 'review priority, not a probability' in alert.details
    assert 'locked' not in alert.details
    assert alert.headline in str(alert)
    assert alert.to_dict()['metrics']['zero_range_days']==5.
    json.dumps(alert.to_dict(),allow_nan=False)


@pytest.mark.parametrize('score,duration,expected',[(3.5,1,'MEDIUM'),(5.,1,'HIGH'),(8.,2,'HIGH'),(8.,3,'CRITICAL')])
def test_priority_rules_preserved(score,duration,expected):
    from shield_regime import WashTradeDetector
    data=pd.DataFrame(dict(high=100.,low=100.,close=100.,volume=100.),index=pd.bdate_range('2024-01-01',periods=duration))
    alert=WashTradeDetector()._create_alert(data,pd.Series(score,index=data.index),pd.Series(3.,index=data.index),0,duration-1,'TEST')
    assert alert.severity.value==expected
