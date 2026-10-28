import sys
import pandas as pd
import yfinance as yf
from shield_regime.detectors.wash_trade import WashTradeDetector

# Ensure UTF-8 output if possible, to prevent Windows console issues
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 75)
print("[SURVEILLANCE] TWSE Volume-Range Screening Case Studies")
print("=" * 75)

detector = WashTradeDetector(
    wti_threshold=3.5,
    volume_spike_threshold=2.5,
    lookback=20,
)

def run_twse_validation(ticker: str, name: str, start_date: str, end_date: str):
    print(f"\n👉 Checking {ticker} ({name}) from {start_date} to {end_date}...")
    
    # yfinance uses symbol suffix .TW or .TWO
    symbol = f"{ticker}.TWO" if ticker in ["6161", "3537", "6182"] else f"{ticker}.TW"
    
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            # Fallback to .TW if .TWO failed or vice-versa
            alt_symbol = f"{ticker}.TW" if symbol.endswith(".TWO") else f"{ticker}.TWO"
            df = yf.download(alt_symbol, start=start_date, end=end_date, progress=False)
            
        if df.empty:
            print(f"❌ Failed to fetch data for {ticker}")
            return
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        
        alerts = detector.scan(df, ticker=ticker)
        
        if not alerts:
            print("✅ [NO SIGNAL] No volume-range anomaly met the configured rules.")
        else:
            print(f"⚠️ [ALERT] Flagged {len(alerts)} volume-range anomaly candidate(s) for review:")
            for idx, alert in enumerate(alerts, 1):
                print(f"  [{idx}] {alert}")
                print(f"      - Details: {alert.details}")
                print(f"      - Metrics: {alert.metrics}")
    except Exception as e:
        print(f"❌ Error scanning {ticker}: {str(e)}")

# Run validation on the 3 real TWSE cases
run_twse_validation("6161", "捷波 (Volume-Range Candidate on 2026-05-19)", "2025-11-20", "2026-05-25")
run_twse_validation("3537", "保達 (Volume-Range Candidate on 2026-05-25)", "2025-11-26", "2026-05-26")
run_twse_validation("6182", "合晶 (Limit-Up Comparison Case on 2026-05-25)", "2025-11-26", "2026-05-26")
