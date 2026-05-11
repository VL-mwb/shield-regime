"""
validate_real_cases.py — Scan real-world historical market manipulation cases.

Downloads historical data for GameStop (GME, 2021) and attempts to pull
the delisted Taiwan stock Lesheng (3662.TW, 2016) from Yahoo Finance to
demonstrate the PumpDumpDetector in action.
"""

from datetime import datetime
import pandas as pd
import yfinance as yf

from shield_regime.detectors.pump_dump import PumpDumpDetector


import sys

# Ensure UTF-8 output if possible, to prevent Windows console issues
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def scan_asset(ticker: str, start_date: str, end_date: str) -> None:
    print("\n" + "=" * 70)
    print(f"[SCAN] SCANNING {ticker} ({start_date} to {end_date})")
    print("=" * 70)

    # 1. Fetch data
    print(f"[DATA] Fetching data from Yahoo Finance...")
    try:
        # yfinance columns are capitalized: Open, High, Low, Close, Volume
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            print(f"[WARN] No data returned for {ticker}. It might be delisted or invalid.")
            return

        # Prepare DataFrame to match package requirements (lowercase close and volume)
        df_clean = pd.DataFrame(index=df.index)
        
        # Handle multi-level columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            close_col = ("Close", ticker)
            vol_col = ("Volume", ticker)
            high_col = ("High", ticker)
            low_col = ("Low", ticker)
        else:
            close_col = "Close"
            vol_col = "Volume"
            high_col = "High"
            low_col = "Low"

        df_clean["close"] = df[close_col].astype(float)
        df_clean["volume"] = df[vol_col].astype(float)
        if high_col in df.columns:
            df_clean["high"] = df[high_col].astype(float)
        if low_col in df.columns:
            df_clean["low"] = df[low_col].astype(float)

        print(f"[INFO] Downloaded {len(df_clean)} trading days of data.")

        # 2. Run detector
        detector = PumpDumpDetector(
            velocity_z_threshold=2.0,
            volume_spike_threshold=3.0,
            dump_drop_threshold=-25.0,
            lookback=40,
        )
        
        print("[CALC] Analyzing price kinematics and volume profiles...")
        alerts = detector.scan(df_clean, ticker=ticker)

        # 3. Print results
        if not alerts:
            print("[OK] No suspicious Pump-and-Dump patterns detected.")
        else:
            print(f"[ALERT] DETECTED {len(alerts)} SUSPICIOUS PUMP-AND-DUMP SCHEMES:")
            for idx, alert in enumerate(alerts, 1):
                print(f"\n[{idx}] {alert}")
                print(f"    - Type: {alert.alert_type} (Severity: {alert.severity.value})")
                print(f"    - Headline: {alert.headline}")
                print(f"    - Duration: {alert.duration_days} days")
                print(f"    - Details: {alert.details}")
                print("    - Metrics:")
                for k, v in alert.metrics.items():
                    print(f"      * {k}: {v:.2f}")

    except Exception as e:
        print(f"[ERROR] Error scanning {ticker}: {str(e)}")


if __name__ == "__main__":
    # Case 1: GameStop (GME) Short Squeeze Anomaly (US Market)
    scan_asset("GME", "2020-09-01", "2021-04-30")

    # Case 2: Lesheng (3662.TW) (TWSE Market - Delisted in 2017)
    # Note: Delisted stocks can sometimes be missing on yfinance. If missing, we explain why.
    scan_asset("3662.TW", "2016-01-01", "2016-12-31")
