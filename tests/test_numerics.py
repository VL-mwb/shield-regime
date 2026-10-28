import pytest
from shield_regime.numerics import drawdown_reached

@pytest.mark.parametrize('scale', [1e-6, 1.0, 1e6])
def test_exact_drop_and_both_sides(scale):
    assert drawdown_reached(80.0 * scale, 100.0 * scale, -20.0)
    assert not drawdown_reached((80.0 + 1e-9) * scale, 100.0 * scale, -20.0)
    assert drawdown_reached((80.0 - 1e-9) * scale, 100.0 * scale, -20.0)

def test_no_drop_and_positive_price_cannot_lose_one_hundred_percent():
    assert not drawdown_reached(100.0, 100.0, -20.0)
    assert not drawdown_reached(1e-300, 1e300, -100.0)


def test_nonfinite_metrics_and_indicators():
    import pandas as pd
    from shield_regime.numerics import finite_metric, ensure_no_infinity
    for value in [float('nan'),float('inf'),float('-inf')]:
        with pytest.raises(ValueError):finite_metric('test',value)
    with pytest.raises(ValueError):ensure_no_infinity('test',pd.Series([float('nan'),float('inf')]))
    ensure_no_infinity('warmup',pd.Series([float('nan'),1.]))
