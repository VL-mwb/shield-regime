import json
from datetime import datetime
from shield_regime import Alert, AlertSeverity

def test_detected_at_is_explicit_and_optional():
    dt = datetime(2024, 1, 2)
    legacy = Alert('X', 'PUMP_AND_DUMP', AlertSeverity.HIGH, dt, dt, 'example')
    assert legacy.detected_at is None
    assert legacy.to_dict()['detected_at'] is None
    current = Alert('X', 'PUMP_AND_DUMP', AlertSeverity.HIGH, dt, dt, 'example', detected_at=dt)
    assert current.to_dict()['detected_at'] == dt.isoformat()
    json.dumps(current.to_dict(), allow_nan=False)
