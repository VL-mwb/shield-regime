# 🛡️ ShieldRegime: Physics-Inspired Financial Market Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**ShieldRegime** is an open-source Python library designed to identify market manipulation and suspicious trading activities (such as **Pump-and-Dump schemes** and **Wash Trading**) using **price kinematics**—an innovative, physics-inspired approach to financial time-series analysis.

By modeling price as a physical particle and volume as the underlying force moving it, ShieldRegime computes real-time derivatives (velocity, acceleration, kinetic friction) to flag manipulative trends before they impact market integrity.

---

## ⚡ Key Features

*   **📈 Price Kinematics**: Smoothed velocity (1st derivative) and acceleration (2nd derivative) rolling Z-score computations to identify momentum-thrust anomalies.
*   **📊 Volume-Price Profiling**: Advanced indicators (`volume_price_divergence`, `wash_trade_index`) to catch high-volume, narrow-range wash sales.
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

# 2. Initialize the detector with custom thresholds
detector = sr.PumpDumpDetector(
    velocity_z_threshold=2.0,      # Trigger on +2.0σ price velocity
    volume_spike_threshold=3.0,    # Require volume > 3x recent average
    dump_drop_threshold=-25.0,     # Flag if price crashes > 25% from peak
    lookback=40,                   # 40-day rolling baseline
)

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
Under self-dealing conditions, volume explodes while the intraday price range remains narrow. We flag this anomaly using:

$$WTI_t = \frac{\text{Volume Spike Ratio}}{\text{Normalized Intraday Range} + \epsilon}$$

---

## 🛠️ Running Unit Tests

We maintain 100% test coverage for all kinematic formulas, detectors, and visualization routines. Run tests with:

```bash
pytest tests/ -v
```

---

## 🤖 Agentic AI & MCP Server Support (Autonomous Surveillance)

ShieldRegime is built for the era of autonomous trading and AI agents. It natively supports the **Model Context Protocol (MCP)**, allowing LLM-based assistants (like Claude, ChatGPT, and Gemini) to autonomously run physics-inspired compliance audits.

We provide pre-built integrations for the three main AI Agent paradigms in the [agency/](agency/) directory:

1. **Model Context Protocol (`agency/mcp_server.py`)**: Run a local MCP server so your AI assistants can natively query `scan_ticker_for_manipulation`.
2. **Cursor IDE Rules (`agency/cursor/.cursorrules`)**: Enforce compliance checks inside Cursor or Copilot whenever AI generates trading code.
3. **Claude Code Prompt (`agency/claude_code/market_surveillance.prompt`)**: Instruct Claude Code to act as a terminal-based market surveillance compliance officer.

For setup instructions, see the specialized [agency/README.md](agency/README.md).

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

