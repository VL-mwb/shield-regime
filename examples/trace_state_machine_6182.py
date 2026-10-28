import sys
import pandas as pd
import numpy as np

from shield_regime.indicators.velocity import velocity_zscore, price_velocity_pct
from shield_regime.indicators.acceleration import acceleration_zscore
from shield_regime.indicators.volume_profile import volume_spike_index

if __name__ == '__main__':
    import yfinance as yf

    # Ensure UTF-8 output
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 85)
    print("⚙️ PUMPDUMP DETECTOR 逐日警報重播 (Daily Alert Replay for 6182 合晶)")
    print("=" * 85)

    # 1. 下載資料
    df = yf.download("6182.TWO", start="2025-11-01", end="2026-06-30", progress=False)
    if df.empty:
        df = yf.download("6182.TW", start="2025-11-01", end="2026-06-30", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)

    # 2. 狀態機參數
    velocity_z_threshold = 2.5
    volume_spike_threshold = 3.0
    dump_window = 15
    dump_drop_threshold = -20.0
    lookback = 40

    vel_z = velocity_zscore(df["close"], lookback=lookback, use_pct=True)
    accel_z = acceleration_zscore(df["close"], lookback=lookback, use_pct=True)
    vol_spike = volume_spike_index(df["volume"], lookback=lookback)

    print(f"參數設定: Velocity_Z門檻={velocity_z_threshold}, Volume_Spike門檻={volume_spike_threshold}x, Dump觀察期={dump_window}天, Dump跌幅門檻={dump_drop_threshold}%")
    print(f"分析資料筆數: {len(df)} 根日K ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d})\n")

    from shield_regime import PumpDumpDetector
    from examples.validate_forward_returns import collect_first_observed_alerts

    detector = PumpDumpDetector(velocity_z_threshold=velocity_z_threshold,
        volume_spike_threshold=volume_spike_threshold, dump_window=dump_window,
        dump_drop_threshold=dump_drop_threshold, lookback=lookback)
    for alert in collect_first_observed_alerts(df, [detector], ticker='6182'):
        print(f'首次確認: {alert.detected_at} | 事件起點: {alert.start_date} | {alert.headline}')

    print("\n" + "=" * 85)
    print("🔍 針對 5/21 ~ 5/25 暴衝波段的狀態機逐日檢測日誌：")
    print("=" * 85)
    for dt in df.loc["2026-05-20":"2026-05-27"].index:
        v_z = vel_z.loc[dt]
        v_sp = vol_spike.loc[dt]
        c = df.loc[dt, "close"]
        vel_check = "✅ 符合(>=2.5)" if v_z >= 2.5 else f"❌ 不符(={v_z:.2f} < 2.5)"
        vol_check = "✅ 符合(>=3.0x)" if v_sp >= 3.0 else f"❌ 不符(={v_sp:.2f}x < 3.0x)"
        state = "進入 PUMP 狀態機" if (v_z >= 2.5 and v_sp >= 3.0) else "維持 BASE 監控狀態"
        print(f"[{dt:%Y-%m-%d}] 收盤:{c:6.2f} | Vel_Z:{v_z:+5.2f} ({vel_check}) | Vol_Spike:{v_sp:4.2f}x ({vol_check}) ➔ 狀態: {state}")
    print("=" * 85)
