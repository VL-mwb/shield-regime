"""
crypto_scan.py — Scan crypto markets for speculative Pump-and-Dump cycles (e.g., DOGE-USD).

Downloads Dogecoin (DOGE-USD) historical price and volume data during the legendary
2020-2021 meme-crypto cycle, runs the PumpDumpDetector, and exports the interactive analysis.
"""

import os
import sys
import pandas as pd
import yfinance as yf

from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.visualizer import plot_alerts

# Configure console to support UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    # Dogecoin (DOGE-USD) — The quintessential retail speculation wave
    ticker = "DOGE-USD"
    start_date = "2020-09-01"
    end_date = "2021-07-31"

    print("=" * 70)
    print(f"🪙 [CRYPTO SCAN] SCANNING {ticker} SPECTACULAR MEME CYCLE")
    print("=" * 70)

    # 1. Download Crypto Data from Yahoo Finance (Free, no token required)
    print(f"[DATA] Downloading {ticker} daily candles...")
    try:
        df_raw = yf.download(ticker, start=start_date, end=end_date)
        if df_raw.empty:
            print("[ERROR] No data returned from Yahoo Finance.")
            return
        print(f"[INFO] Downloaded {len(df_raw)} daily candles.")
    except Exception as e:
        print(f"[ERROR] Failed to download data: {str(e)}")
        return

    # 2. Reformat columns for ShieldRegime (must be lowercase index-friendly)
    df = pd.DataFrame(index=df_raw.index)
    if isinstance(df_raw.columns, pd.MultiIndex):
        close_col = ("Close", ticker)
        vol_col = ("Volume", ticker)
        high_col = ("High", ticker)
        low_col = ("Low", ticker)
    else:
        close_col = "Close"
        vol_col = "Volume"
        high_col = "High"
        low_col = "Low"

    df["close"] = df_raw[close_col].astype(float)
    df["volume"] = df_raw[vol_col].astype(float)
    df["high"] = df_raw[high_col].astype(float)
    df["low"] = df_raw[low_col].astype(float)

    # 3. Configure the Kinematic Anomaly Detector
    # Crypto markets are highly volatile, so we use customized thresholds:
    detector = PumpDumpDetector(
        velocity_z_threshold=2.5,      # React on velocity moves > 2.5σ
        volume_spike_threshold=3.0,    # Require volume to exceed 3x normal average
        dump_drop_threshold=-35.0,     # Trigger on > 35% crashes from peak price
        lookback=40,                   # 40-day rolling base period
    )

    print("[CALC] Scanning for hyper-momentum pumps and subsequent system crashes...")
    alerts = detector.scan(df, ticker=ticker)

    # 4. Print detailed results to console
    if not alerts:
        print(f"[OK] No suspicious manipulation patterns detected for {ticker}.")
    else:
        print(f"🚨 [ALERT] DETECTED {len(alerts)} SUSPICIOUS PUMP-AND-DUMP SCHEMES:")
        for idx, alert in enumerate(alerts, 1):
            print(f"\n[{idx}] {alert}")
            print(f"    - Severity: {alert.severity.value}")
            print(f"    - Headline: {alert.headline}")
            print(f"    - Duration: {alert.duration_days} trading days")
            print(f"    - Details: {alert.details}")
            print("    - Metrics:")
            for k, v in alert.metrics.items():
                print(f"      * {k}: {v:.2f}")

    # 5. Export interactive report
    print("\n[PLOT] Generating interactive Plotly visualization...")
    fig = plot_alerts(df, alerts, ticker=ticker, lookback=40)
    
    if fig is not None:
        output_filename = "crypto_analysis.html"
        output_path = os.path.abspath(output_filename)
        fig.write_html(output_path)
        print("=" * 70)
        print("🎉 SUCCESS! Crypto interactive report is ready.")
        print(f"👉 To view, open in any web browser:")
        print(f"   File path: file:///{output_path.replace(os.sep, '/')}")
        print("=" * 70)


if __name__ == "__main__":
    main()
