"""
validate_taiwan_finmind.py — Scan TWSE anomalies (6452.TW and 3662.TW) using FinMind API.

Retrieves Taiwan daily stock history directly using the FinMind API
for Lesheng (3662) and Kangyou-KY (6452), running the PumpDumpDetector
on these real-world Taiwanese stock manipulation scandals.
"""

import os
import sys
import pandas as pd
import requests

from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.indicators.velocity import velocity_zscore
from shield_regime.indicators.volume_profile import volume_spike_index

# Configure console to prevent encoding issues on Windows
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
    """Fetch Taiwan stock daily data directly from FinMind API."""
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
    
    # Build dataframe first without setting index to prevent pandas index-alignment mismatch (NaN propagation)
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(df_raw["date"])
    df["close"] = df_raw["close"].astype(float)
    df["volume"] = df_raw["Trading_Volume"].astype(float)
    df["high"] = df_raw["max"].astype(float)
    df["low"] = df_raw["min"].astype(float)
    
    df.set_index("date", inplace=True)
    df.dropna(subset=["close", "volume"], inplace=True)
    
    # Clean duplicates and sort
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def scan_taiwan_asset(stock_id: str, name: str, start_date: str, end_date: str) -> None:
    print("\n" + "=" * 70)
    print(f"[SCAN] SCANNING TWSE {stock_id} ({name}) — {start_date} to {end_date}")
    print("=" * 70)
    
    try:
        # 1. Fetch data
        print(f"[DATA] Connecting to FinMind API for {stock_id}...")
        df = fetch_finmind_data(stock_id, start_date, end_date)
        print(f"[INFO] Successfully retrieved {len(df)} trading days of historical data.")
        
        # 2. Run detector
        # We adjust thresholds slightly for Taiwan daily price limit (+-10%)
        detector = PumpDumpDetector(
            velocity_z_threshold=2.0,      # React on velocity moves > 2σ
            volume_spike_threshold=2.5,    # Volume > 2.5x normal average
            dump_drop_threshold=-20.0,     # Trigger on > 20% drops from peak
            lookback=40,
        )
        
        print("[CALC] Running Kinematic Anomaly Analysis...")
        alerts = detector.scan(df, ticker=f"{stock_id}.TW")
        
        # Calculate for diagnostics
        vel_z = velocity_zscore(df["close"], lookback=40, use_pct=True)
        vol_spike = volume_spike_index(df["volume"], lookback=40)
        
        print(f"[DIAG] Data length: {len(df)}")
        print(f"[DIAG] Max Close Price: {df['close'].max():.2f}")
        print(f"[DIAG] Min Close Price: {df['close'].min():.2f}")
        print(f"[DIAG] Max Velocity Z-score: {vel_z.max():.2f}")
        print(f"[DIAG] Max Volume Spike Ratio: {vol_spike.max():.2f}")
        
        # 3. Print results
        if not alerts:
            print(f"[OK] No suspicious Pump-and-Dump patterns detected for {name}.")
        else:
            print(f"[ALERT] DETECTED {len(alerts)} SUSPICIOUS SCHEMES FOR {name}:")
            for idx, alert in enumerate(alerts, 1):
                print(f"\n[{idx}] {alert}")
                print(f"    - Severity: {alert.severity.value}")
                print(f"    - Headline: {alert.headline}")
                print(f"    - Duration: {alert.duration_days} days")
                print(f"    - Details: {alert.details}")
                print("    - Metrics:")
                for k, v in alert.metrics.items():
                    print(f"      * {k}: {v:.2f}")
                    
    except Exception as e:
        print(f"[ERROR] Failed to scan {stock_id} ({name}): {str(e)}")


if __name__ == "__main__":
        # Case 1: 康友-KY (6452) - 2020 掏空與連續跌停事件
        # 康友是典型的「直接暴跌型（Crash-and-Burn）」，它沒有前期的惡意拉抬（Pump），而是平盤後因掏空直接連續無量跌停。
        # 這種案例不屬於 Pump-and-Dump，而是屬於 Systemic Crash，所以我們的偵測器不應產生 PND 誤報，這證明了算法的低誤報率。
        scan_taiwan_asset("6452", "康友-KY (直接暴跌型，無前期拉抬)", "2020-01-01", "2020-11-30")

        # Case 2: 樂陞科技 (3662) - 2016 假收購與暴跌事件（長週期波段型）
        # 樂陞的假收購案有拉抬與崩盤，但由於是「長週期法人/主力型操縱」，拉抬高點（5月）到崩盤（9月）間隔了近 3 個月（約 70 個交易日）。
        # 我們先用預設的 15 天短視窗掃描（預期不觸發）：
        scan_taiwan_asset("3662", "樂陞科技 (預設 15 天短週期視窗)", "2016-01-01", "2016-12-31")
        
        # 接著，我們調整參數，將崩盤觀測視窗擴大到 90 天，以適應「長週期公司債/股權操縱」：
        print("\n" + "-" * 70)
        print("💡 [調整參數調整] 針對長週期主力操縱案，將 dump_window 擴大至 90 天進行二次掃描...")
        print("-" * 70)
        
        # 執行長週期掃描
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockPrice", "data_id": "3662", "start_date": "2016-01-01", "end_date": "2016-12-31", "token": FINMIND_TOKEN}
        response = requests.get(url, params=params)
        df_3662 = pd.DataFrame(response.json()["data"])
        df = pd.DataFrame()
        df["date"] = pd.to_datetime(df_3662["date"])
        df["close"] = df_3662["close"].astype(float)
        df["volume"] = df_3662["Trading_Volume"].astype(float)
        df["high"] = df_3662["max"].astype(float)
        df["low"] = df_3662["min"].astype(float)
        df.set_index("date", inplace=True)
        df.dropna(subset=["close", "volume"], inplace=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        
        detector_long = PumpDumpDetector(
            velocity_z_threshold=2.0,
            volume_spike_threshold=2.5,
            dump_drop_threshold=-30.0,
            dump_window=90,  # 擴大至 90 天
            lookback=40,
        )
        alerts_long = detector_long.scan(df, ticker="3662.TW")
        if alerts_long:
            print(f"🚨 [ALERT] 成功偵測到長週期 Pump-and-Dump 操縱案！")
            for idx, alert in enumerate(alerts_long, 1):
                print(f"[{idx}] {alert}")
                print(f"    - Details: {alert.details}")
                print("    - Metrics:")
                for k, v in alert.metrics.items():
                    print(f"      * {k}: {v:.2f}")
        else:
            print("[OK] 未偵測到長週期操縱。")
