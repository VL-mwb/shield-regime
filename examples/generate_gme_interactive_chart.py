"""
generate_gme_interactive_chart.py — Generate and save an interactive HTML visualization.

Fetches GameStop (GME) 2021 data, runs the PumpDumpDetector, and exports
an interactive Plotly chart to an HTML file that you can open in any browser.
"""

import os
import sys
import yfinance as yf
import pandas as pd

from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.visualizer import plot_alerts

# Configure console for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    ticker = "GME"
    start_date = "2020-09-01"
    end_date = "2021-04-30"

    print("=" * 70)
    print(f"🎬 [VISUALIZATION] Generating Interactive Chart for {ticker}")
    print("=" * 70)

    # 1. Download Data
    print(f"[1/4] Fetching historical price and volume for {ticker} from Yahoo Finance...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        print("[ERROR] No data fetched. Please check internet connection.")
        return

    # 2. Clean and format DataFrame for ShieldRegime
    df_clean = pd.DataFrame(index=df.index)
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
    df_clean["high"] = df[high_col].astype(float)
    df_clean["low"] = df[low_col].astype(float)

    # 3. Scan for Anomaly Alerts
    print("[2/4] Running Pump-and-Dump Detector Scan...")
    detector = PumpDumpDetector(
        velocity_z_threshold=2.0,
        volume_spike_threshold=3.0,
        dump_drop_threshold=-25.0,
        lookback=40,
    )
    alerts = detector.scan(df_clean, ticker=ticker)
    print(f"[INFO] Scan finished. Found {len(alerts)} suspicious manipulation window(s).")

    # 4. Generate & Save Interactive Plotly Chart
    print("[3/4] Rendering Plotly interactive subplots...")
    fig = plot_alerts(df_clean, alerts, ticker=ticker, lookback=40)

    if fig is None:
        print("[ERROR] Failed to render chart. Plotly might not be installed.")
        return

    # Define output path
    output_filename = "gme_analysis.html"
    output_path = os.path.abspath(output_filename)

    print(f"[4/4] Saving interactive report to: {output_path}")
    fig.write_html(output_path)

    print("\n🎉 SUCCESS! Interactive visualization is ready.")
    print("=" * 70)
    print("👉 HOW TO VIEW THE INTERACTIVE CHART:")
    print(f"   Please open this file in any web browser (Chrome, Edge, Safari):")
    print(f"   File path: file:///{output_path.replace(os.sep, '/')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
