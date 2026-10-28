# 🛡️ ShieldRegime: Physics-Inspired Financial Market Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VL-mwb/shield-regime/blob/main/examples/demo.ipynb)

**ShieldRegime** is an open-source Python library designed to identify market manipulation and suspicious trading activities (such as **Pump-and-Dump schemes** and **Wash Trading**) using **price kinematics**—an innovative, physics-inspired approach to financial time-series analysis.

By modeling price as a physical particle and volume as the underlying force moving it, ShieldRegime computes real-time derivatives (velocity, acceleration, kinetic friction) to flag manipulative trends before they impact market integrity.

> [!TIP]
> **Try it in your browser right now!** Click the **Open In Colab** badge above to launch a fully interactive cloud demo running on real-world GameStop ($GME$) retail-driven mania data.

---

## ⚡ Key Features

*   **📈 Price Kinematics**: Smoothed velocity (1st derivative) and acceleration (2nd derivative) rolling Z-score computations to identify momentum-thrust anomalies.
*   **📊 Volume-Price Profiling**: Advanced indicators (`volume_price_divergence`, `wash_trade_index`) to flag volume-range anomalies for review.
*   **🚨 Multi-Stage Detectors**: High-fidelity detection models (`PumpDumpDetector`) mapping base, pump, and dump stages.
*   **🎨 Stunning Visualization**: Interactive, presentation-ready Plotly charts highlighting anomaly regions, trigger bounds, and drawdowns.

---

## 🚀 Installation

### 1. Via PyPI (Recommended)
You can install the stable release of ShieldRegime directly from PyPI:
```bash
pip install shield-regime
```

### 2. From Source (For Development)
If you want to contribute, run the tests, or build from source:
```bash
# Clone the official repository
git clone https://github.com/VL-mwb/shield-regime.git
cd shield-regime

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 📖 Quick Start & Visualization

The following example downloads historical data for **GameStop (GME)** during the 2021 short-squeeze anomaly, runs the `PumpDumpDetector`, and visualizes the results with Plotly.

```python
import shield_regime as sr
import yfinance as yf

# 1. Download historical market data
df = yf.download("GME", start="2020-09-01", end="2021-04-30")

# Rename columns to match requirements
df_clean = df.copy()
df_clean.index = pd.to_datetime(df.index)
df_clean.columns = [col.lower() for col in df_clean.columns]

# 2. Initialize the detector with default baseline thresholds
detector = sr.PumpDumpDetector(lookback=40)

# 3. Scan the asset
alerts = detector.scan(df_clean, ticker="GME")

for alert in alerts:
    print(f"\n🚨 [DETECTED] {alert}")
    print(f"Details: {alert.details}")
    print(f"Metrics: {alert.metrics}")

# 4. Generate beautiful interactive Plotly chart
fig = sr.plot_alerts(df_clean, alerts, ticker="GME", lookback=40)
if fig:
    # In Jupyter, this displays directly. Or save to HTML:
    fig.write_html("gme_analysis.html")
    print("\n📊 Saved interactive analysis to 'gme_analysis.html'")
```

---

## 🧪 Mathematical Foundation

### Price Velocity ($V$)
The rate of change of price over time, smoothed via Exponential Moving Average (EMA) to filter out high-frequency noise:

$$V_t = \text{EMA}(\Delta P_t, \lambda)$$

### Price Acceleration ($A$)
The second derivative of price, capturing sudden trend thrusts and aggressive manipulation entries:

$$A_t = \text{EMA}(V_t - V_{t-1}, \lambda)$$

### Wash Trade Index ($WTI$)
WTI screens observed volume relative to intrabar range for review:

$$WTI_t = \frac{\text{Volume Spike Ratio}}{\text{Normalized Intraday Range} + \epsilon}$$

Rolling baselines include the current bar and use SMA. Write R=(high-low)/((high+low)/2), B=rolling_mean(R), and VS=volume/rolling_mean(volume). With epsilon=0.01, WTI=VS/(R/B+0.01) when B>0. After m=max(lookback//2,1) observations, an entire window of exact high==low defines R/B=0 and WTI=VS/0.01. Fewer than m observations remain NaN. A zero-volume window has undefined VS/WTI; current zero volume with a positive baseline gives WTI=0. Arithmetic failure is an error, not a no-signal result. The detector still needs the full lookback and both WTI>=3.5 and VS>=2.5 by default. Stable positive volume on flat bars can yield WTI=100 with VS=1 and no alert. WTI must not be used alone. Zero intrabar range does not mean prices are unchanged across dates.


WTI migration: headline/details now describe volume-range anomaly candidates for review. WASH_TRADE, class names, severity values and dates remain compatibility identifiers. The new float metric zero_range_days counts exact high==low bars inside the alert group, rather than calendar duration_days. Severity indicates rule-based review priority, not a probability. OHLCV screening does not establish self-trading, trading identity, intent or market manipulation. Direct WTI calls require real finite numeric Series with identical unique indexes, positive prices, high>=low and nonnegative volume; generic indexes follow row order. Inputs are copied to float64, so float32 and narrow-integer intermediate calculations can differ from older native-dtype results. Positive-baseline regression compares the old formula on identical float64 copies at rtol=1e-12, atol=0. Wash checks original price relationships even before full warmup and rejects lost price ranges or unreliable arithmetic; the shared validator and Pump behavior are unchanged. Text consumers and strict metrics-key consumers must migrate.


---

## 🛠️ Running Unit Tests

The latest verified suite passes 195 tests with 92% statement coverage (Python 3.11.9, pandas 3.0.2, NumPy 2.4.4). Both detector modules have 100% statement coverage; other modules retain coverage gaps. Run tests with:

```bash
pytest tests/ -v
```

---

## 🤖 Autonomous AI Agent Integration & Cloud API

ShieldRegime's mathematical engine is designed to be fully compatible with Model Context Protocol (MCP) and modern Agentic AI frameworks (Claude, ChatGPT, AutoGen).


---

## 🗺️ Roadmap & Future Work

ShieldRegime is actively developed to provide regulatory-grade market surveillance.

- [x] **Phase 1: Core Price Kinematics** — Smoothed velocity & acceleration derivatives.
- [x] **Phase 2: Pump-and-Dump Detection** — Dynamic peak-tracking state machines.
- [x] **Phase 3: Wash Trading & Crypto Scans** — Volume-price profiling (`wash_trade_index`) and crypto speculative scans.
- [ ] **Phase 4 (Future HFT Expansion): High-Frequency Spoofing & Layering Detector** — Modeling book pressure imbalances (requires L2/L3 order book tick feeds).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



`Alert.start_date` marks the event onset, while `detected_at` is the first completed bar that confirms an alert. For Pump-and-Dump, `end_date` equals `detected_at`; the drop and severity are frozen at first confirmation, rather than revised using a later low. For Wash Trade, `detected_at` stays at the first eligible anomaly while the historical group can extend its end and aggregate metrics. Use daily prefix replay and retain the first snapshot when evaluating post-alert price changes. Daily bars are known after completion; reference-close returns do not assume execution at that close.

Detectors require flat unique OHLCV columns, a unique increasing DatetimeIndex without NaT, finite real numeric values, positive prices, nonnegative volume, and consistent supplied OHLC bounds. They compute on a float64 copy. Invalid inputs and identifiable arithmetic overflow raise errors; valid data shorter than the warmup returns an empty list with `InsufficientDataWarning`. Existing manually constructed alerts may have `detected_at=None`; reconstruct timing from original data rather than substituting event dates.

Pump candidates have independent peak-relative observation windows. The final bar in `dump_window` is eligible. A new qualifying pump after momentum tracking ends can coexist with older unexpired candidates. The first confirmation merges overlapping candidates into one alert, using the earliest-starting qualifying candidate consistently. Drop comparisons use price ratios with relative tolerance `8 * numpy.finfo(numpy.float64).eps` and zero absolute tolerance, including the severity boundary. Historical alert dates, drops and counts can change with this migration. WTI zero-baseline handling now follows the piecewise rule above; broader formula-documentation alignment and statistical calibration remain separate work.
