import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def manipulated_data():
    """Textbook Pump-and-Dump: flat → sharp pump → violent collapse."""
    n = 100
    prices = np.full(n, 20.0)
    volume = np.full(n, 100_000.0)

    # Base period
    np.random.seed(123)
    prices[:50] += np.cumsum(np.random.normal(0, 0.1, 50))

    # PUMP (days 50-59)
    for i in range(50, 60):
        prices[i] = prices[i - 1] * 1.12  # +12% compounded daily
        volume[i] = volume[0] * 7          # 7x volume surge

    # DUMP (days 60-70)
    for i in range(60, 71):
        prices[i] = prices[i - 1] * 0.85  # -15% daily crash
        volume[i] = volume[0] * 5          # volume remains high during dump

    # Aftermath (days 71-99)
    for i in range(71, n):
        prices[i] = prices[i - 1] * (1 + np.random.normal(0, 0.01))
        volume[i] = volume[0]

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"close": prices, "volume": volume}, index=dates)

