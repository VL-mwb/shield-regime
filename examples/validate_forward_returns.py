import sys
import pandas as pd
import numpy as np
from shield_regime.detectors.wash_trade import WashTradeDetector
from shield_regime.detectors.pump_dump import PumpDumpDetector

import warnings
from shield_regime import InsufficientDataWarning

def collect_first_observed_alerts(df, detectors, ticker='UNKNOWN'):
    seen = set()
    first = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', InsufficientDataWarning)
        for size in range(1, len(df) + 1):
            prefix = df.iloc[:size]
            observed_at = prefix.index[-1]
            for detector in detectors:
                for alert in detector.scan(prefix, ticker=ticker):
                    key = (alert.ticker, alert.alert_type, alert.start_date)
                    if key in seen:
                        continue
                    if alert.detected_at is None or alert.detected_at != observed_at:
                        raise ValueError('new alert detected_at must equal first observable bar')
                    seen.add(key)
                    first.append(alert)
    return first

from shield_regime.validation import validate_market_data

def verify_post_alert_performance(ticker: str, name: str, start_date: str, end_date: str):
    import yfinance as yf

    symbol = f"{ticker}.TWO" if ticker in ["6161", "3537", "6182"] else (f"{ticker}.TW" if ticker.isdigit() else ticker)
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty:
        alt_symbol = f"{ticker}.TW" if symbol.endswith(".TWO") else f"{ticker}.TWO"
        df = yf.download(alt_symbol, start=start_date, end=end_date, progress=False)
        
    if df.empty:
        print(f"❌ Failed to fetch data for {ticker}")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)

    wt_detector = WashTradeDetector(wti_threshold=3.5, volume_spike_threshold=2.5, lookback=20)
    pd_detector = PumpDumpDetector(velocity_z_threshold=2.0, volume_spike_threshold=2.5, dump_drop_threshold=-20.0, lookback=40)

    df = validate_market_data(df, ('high', 'low', 'close', 'volume'))
    all_alerts = collect_first_observed_alerts(df, [wt_detector, pd_detector], ticker)

    print(f"\n================================================================================")
    print(f"🔍 標的: {ticker} {name} (共 {len(df)} 根日K，分析區間: {start_date} ~ {end_date})")
    print(f"================================================================================")

    if not all_alerts:
        print("  ✅ 期間無任何警報。")
        return

    for alert in all_alerts:
        trigger_date = alert.detected_at
        if trigger_date is None or trigger_date not in df.index:
            raise ValueError('confirmation time is missing from input data')

        trigger_idx = df.index.get_loc(trigger_date)
        trigger_close = df.loc[trigger_date, "close"]

        # Forward windows: T+5, T+10, T+20, T+40, T+60 days
        print(f"\n🚨 警報觸發: [{alert.severity.value}] {alert.alert_type} on {trigger_date.strftime('%Y-%m-%d')}")
        print(f"   觸發日收盤價: {trigger_close:.2f} 元 | 詳情: {alert.headline}")
        print(f"   -----------------------------------------------------------------------------")

        # Forward analysis
        subsequent_df = df.iloc[trigger_idx + 1:]
        if subsequent_df.empty:
            print("   ⚠️ 警報位於資料尾端，暫無足夠後續交易日以供統計。")
            continue

        print(f"   {'時間窗口':<10} | {'收盤價':<10} | {'報酬率 (%)':<12} | {'相對確認日收盤的最低報酬':<18} | {'區間最高價 / 最低價'}")
        print(f"   {'-'*75}")

        for days in [5, 10, 20, 40, 60]:
            if len(subsequent_df) >= days:
                f_df = subsequent_df.iloc[:days]
                target_close = f_df.iloc[-1]["close"]
                ret = ((target_close - trigger_close) / trigger_close) * 100.0
                
                # Lowest return relative to confirmation close
                min_low = f_df["low"].min()
                max_high = f_df["high"].max()
                min_return_from_signal_pct = ((min_low - trigger_close) / trigger_close) * 100.0

                print(f"   T+{days:<7} | {target_close:<10.2f} | {ret:>+10.2f}% | {min_return_from_signal_pct:>+16.2f}% | {max_high:.2f} / {min_low:.2f}")

if __name__ == '__main__':
    print('後續價格變化以確認日收盤為參考，並非可成交策略績效。')
    # Run verification on our main study cases up to August 2026
    verify_post_alert_performance("6161", "捷波", "2025-11-01", "2026-08-20")
    verify_post_alert_performance("3537", "保達", "2025-11-01", "2026-08-20")
    verify_post_alert_performance("6182", "合晶", "2025-11-01", "2026-08-20")
    verify_post_alert_performance("GME", "GameStop (2021 Anomaly)", "2020-09-01", "2021-06-30")
