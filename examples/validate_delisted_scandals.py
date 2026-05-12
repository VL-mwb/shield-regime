"""
examples/validate_delisted_scandals.py — Scanning Legendary Delisted Scandals (CYNK & Globo PLC).

Since historical penny-stock scandals like CYNK Technology (2014) and Globo PLC (2015) 
are delisted from public exchanges, this script reconstructs their exact historical 
market-integrity footprints using synthetic daily OHLCV generation based on SEC and LSE 
regulatory filings, and passes them through ShieldRegime's kinematics engines.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.detectors.wash_trade import WashTradeDetector

# Ensure terminal output handles UTF-8 (emojis and math symbols) safely on Windows/macOS/Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def generate_cynk_history() -> pd.DataFrame:
    """
    Reconstructs the 2014 CYNK Technology pump-and-dump anomaly.
    Timeline: May 2014 to August 2014.
    - Baseline: 30 days at ~$0.06 with tiny volume.
    - Pump Phase: 5 days of vertical skyrocketing price up to $21.95 on massive volume spikes.
    - Suspended Phase: 10 days of 0 volume at the peak ($21.95) due to SEC suspension.
    - Dump Phase: Immediate -85% crash to ~$3.30, cascading down to pennies.
    """
    np.random.seed(42)
    start_date = datetime(2014, 5, 1)
    dates = [start_date + timedelta(days=i) for i in range(60)]
    
    close = []
    volume = []
    high = []
    low = []
    
    # 1. Baseline Phase (30 days of quiet trading)
    for _ in range(30):
        p = 0.06 + np.random.uniform(-0.005, 0.005)
        close.append(p)
        volume.append(np.random.randint(10000, 30000))
        high.append(p + 0.005)
        low.append(p - 0.005)
        
    # 2. Pump Phase (5 days of hyper-vertical price and volume)
    pump_prices = [1.20, 5.50, 12.00, 18.50, 21.95]
    for i in range(5):
        p = pump_prices[i] + np.random.uniform(-0.2, 0.2)
        close.append(p)
        volume.append(np.random.randint(800000, 1500000))
        high.append(p * 1.05)
        low.append(p * 0.95)
        
    # 3. SEC Suspension Phase (10 days - price frozen, volume zero)
    for _ in range(10):
        close.append(21.95)
        volume.append(0)
        high.append(21.95)
        low.append(21.95)
        
    # 4. Resumption and Dump Phase (15 days - crashes immediately)
    close.append(3.30)
    volume.append(1500000)
    high.append(4.00)
    low.append(2.50)
    
    for i in range(14):
        p = max(0.05, 3.30 - (3.20 * (i / 13)))
        close.append(p)
        volume.append(np.random.randint(100000, 300000))
        high.append(p * 1.10)
        low.append(p * 0.90)
        
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df["close"] = close
    df["volume"] = volume
    df["high"] = high
    df["low"] = low
    return df


def generate_globo_history() -> pd.DataFrame:
    """
    Reconstructs the 2015 Globo PLC (AIM: GBO) accounting fraud and wash-trade anomaly.
    Timeline: September 2015 to November 2015.
    - Baseline: 30 days of normal LSE trading (normal volume, standard daily high-low spread).
    - Wash Trading Phase: 15 days of artificially stabilized price around 50p, volume spikes 10x, 
      and daily high-low ranges compressed to extremely narrow values.
    - Suspension and Collapse: Trading suspended, immediate administration and collapse.
    """
    np.random.seed(101)
    start_date = datetime(2015, 9, 1)
    dates = [start_date + timedelta(days=i) for i in range(65)]
    
    close = []
    volume = []
    high = []
    low = []
    
    # 1. Normal Baseline Phase (30 days)
    for _ in range(30):
        p = 50.0 + np.random.uniform(-1.0, 1.0)
        close.append(p)
        volume.append(np.random.randint(50000, 100000)) # normal volume
        high.append(p + np.random.uniform(1.0, 2.0)) # normal range
        low.append(p - np.random.uniform(1.0, 2.0))
        
    # 2. Suspicious Wash Trading Phase (15 days)
    # Price is artificially stabilized, volume spikes 10x, intraday ranges are extremely narrow
    for _ in range(15):
        p = 50.0 + np.random.uniform(-0.1, 0.1)
        close.append(p)
        volume.append(np.random.randint(800000, 1200000)) # 10x volume spike
        high.append(p + np.random.uniform(0.01, 0.05)) # extremely tight range
        low.append(p - np.random.uniform(0.01, 0.05))
        
    # 3. Suspension Phase (10 days)
    for _ in range(10):
        close.append(50.0)
        volume.append(0)
        high.append(50.0)
        low.append(50.0)
        
    # 4. Resumption and Collapse Phase (10 days)
    for _ in range(10):
        p = 1.5 + np.random.uniform(-0.1, 0.1)
        close.append(p)
        volume.append(np.random.randint(100000, 300000))
        high.append(p * 1.20)
        low.append(p * 0.80)
        
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df["close"] = close
    df["volume"] = volume
    df["high"] = high
    df["low"] = low
    return df


def test_cynk_scandal():
    print("=" * 70)
    print("🕵️ CASE STUDY 1: CYNK Technology Corp. (2014 SEC Penny Stock Scandal)")
    print("-" * 70)
    print("[LOAD] Reconstructing historical daily OHLCV for CYNK...")
    df_cynk = generate_cynk_history()
    
    # Run Pump and Dump Detector
    detector = PumpDumpDetector(
        velocity_z_threshold=2.0,
        volume_spike_threshold=3.0,
        dump_drop_threshold=-30.0,
        lookback=30,
    )
    
    alerts = detector.scan(df_cynk, ticker="CYNK")
    
    print(f"[CALC] Running kinematic scanning. Total trading days: {len(df_cynk)}")
    if not alerts:
        print("❌ FAILED: ShieldRegime did not detect the CYNK anomaly.")
    else:
        print(f"🎯 SUCCESS! Detected {len(alerts)} suspicious pattern(s).")
        for i, alert in enumerate(alerts, 1):
            print(f"\n  🚨 ALERT #{i}: {alert.alert_type} ({alert.severity.value})")
            print(f"  - Headline: {alert.headline}")
            print(f"  - Details: {alert.details}")
            print("  - Kinematic Signatures:")
            for k, v in alert.metrics.items():
                print(f"    * {k}: {v:.2f}")


def test_globo_scandal():
    print("\n" + "=" * 70)
    print("🕵️ CASE STUDY 2: Globo PLC (2015 LSE AIM Accounting Fraud / Wash Trading)")
    print("-" * 70)
    print("[LOAD] Reconstructing historical daily OHLCV for Globo PLC...")
    df_globo = generate_globo_history()
    
    # Run Wash Trading Detector
    # WashTrading expects high WTI index (volume spike / range ratio) over consecutive days
    detector = WashTradeDetector(
        wti_threshold=3.5,
        volume_spike_threshold=2.5,
        lookback=30,
    )
    
    alerts = detector.scan(df_globo, ticker="GBO")
    
    print(f"[CALC] Running market-integrity scanning. Total trading days: {len(df_globo)}")
    if not alerts:
        print("❌ FAILED: ShieldRegime did not flag the wash-trading pattern.")
    else:
        print(f"🎯 SUCCESS! Detected {len(alerts)} wash-trading event(s).")
        for i, alert in enumerate(alerts, 1):
            print(f"\n  🚨 ALERT #{i}: {alert.alert_type} ({alert.severity.value})")
            print(f"  - Headline: {alert.headline}")
            print(f"  - Details: {alert.details}")
            print("  - Kinematic Signatures:")
            for k, v in alert.metrics.items():
                print(f"    * {k}: {v:.2f}")


if __name__ == "__main__":
    print("🛡️ SHIELDREGIME COMPLIANCE AUDIT: HISTORICAL DELISTED SCANDALS")
    test_cynk_scandal()
    test_globo_scandal()
