"""
generate_taiwan_interactive_chart.py — Generate and save an interactive HTML visualization for Lesheng (3662.TW).

Fetches 3662.TW daily price and volume from FinMind, runs the PumpDumpDetector with a 
90-day dump window (adapted for Taiwan corporate manipulation cycles), and exports 
an interactive Plotly chart to an HTML file.
"""

import os
import sys
import requests
import pandas as pd

from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.visualizer import plot_alerts

# Configure console for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load .env file safely (use python-dotenv if available, otherwise manually parse)
if os.path.exists(".env"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        with open(".env") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, val = line.strip().split("=", 1)
                    os.environ[key] = val.strip(' "\'')

# Retrieve the FinMind API Token from environment variables (fallback to empty string for free-tier limits)
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")


def fetch_finmind_data(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": FINMIND_TOKEN,
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"FinMind API error: HTTP {response.status_code}")
        
    res_data = response.json()
    if res_data.get("msg") != "success" or not res_data.get("data"):
        raise ValueError(f"Failed to fetch data for {stock_id}: {res_data.get('msg', 'No data')}")
        
    df_raw = pd.DataFrame(res_data["data"])
    
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(df_raw["date"])
    df["close"] = df_raw["close"].astype(float)
    df["volume"] = df_raw["Trading_Volume"].astype(float)
    df["high"] = df_raw["max"].astype(float)
    df["low"] = df_raw["min"].astype(float)
    
    df.set_index("date", inplace=True)
    df.dropna(subset=["close", "volume"], inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def main():
    stock_id = "3662"
    name = "樂陞科技"
    start_date = "2016-01-01"
    end_date = "2016-12-31"

    print("=" * 70)
    print(f"🎬 [VISUALIZATION] Generating Taiwan TWSE {stock_id} ({name}) Anomaly Chart")
    print("=" * 70)

    # 1. Fetch data
    print(f"[1/4] Connecting to FinMind API to retrieve {name} price history...")
    try:
        df = fetch_finmind_data(stock_id, start_date, end_date)
        print(f"[INFO] Successfully retrieved {len(df)} trading days of data.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {str(e)}")
        return

    # 2. Run detector with long window (90 days)
    print("[2/4] Running Pump-and-Dump Detector Scan (dump_window=90)...")
    detector_long = PumpDumpDetector(
        velocity_z_threshold=2.0,
        volume_spike_threshold=2.5,
        dump_drop_threshold=-30.0,
        dump_window=90,  # 90-day window to catch long corporate waves
        lookback=40,
    )
    alerts = detector_long.scan(df, ticker=f"{stock_id}.TW")
    print(f"[INFO] Scan finished. Found {len(alerts)} suspicious manipulation window(s).")

    # 3. Generate Interactive Plotly Chart
    print("[3/4] Rendering Plotly interactive subplots...")
    fig = plot_alerts(df, alerts, ticker=f"{stock_id}.TW ({name})", lookback=40)

    if fig is None:
        print("[ERROR] Failed to render chart. Plotly might not be installed.")
        return

    # Define output path
    output_filename = "taiwan_3662_analysis.html"
    output_path = os.path.abspath(output_filename)

    print(f"[4/4] Saving interactive report to: {output_path}")
    fig.write_html(output_path)

    print("\n🎉 SUCCESS! Taiwan anomaly visualization is ready.")
    print("=" * 70)
    print("👉 HOW TO VIEW THE INTERACTIVE CHART:")
    print(f"   Please open this file in any web browser (Chrome, Edge, Safari):")
    print(f"   File path: file:///{output_path.replace(os.sep, '/')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
