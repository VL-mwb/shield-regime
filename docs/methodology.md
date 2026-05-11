# 🛡️ ShieldRegime: Market Kinematics & Manipulation Detection Methodology

## 1. Introduction: From Black-Box Models to White-Box Kinematics

In financial surveillance and regulatory compliance, deep learning models (such as LSTMs or Transformers) often suffer from a critical flaw: **explainability**. When a regulatory body like the **UK Financial Conduct Authority (FCA)** or the **US Securities and Exchange Commission (SEC)** brings a market abuse case to court, a "black-box" neural network prediction is legally insufficient. Regulators and judges require deterministic, mathematically verifiable, and logically transparent evidence.

**ShieldRegime** bridges this gap by moving away from speculative machine learning and returning to first-principles Newtonian physics. We treat the financial price series not as a random walk, but as a **macro-particle moving through a fluid medium under the influence of external forces (volume and liquidity)**. 

By calculating the smoothed first and second derivatives of price (Velocity and Acceleration) and cross-referencing them with volume profile anomalies, we can isolate the exact physical footprint of market manipulation.

---

## 2. The Kinematic-Market Mapping (物理學與金融市場對照)

In classical Newtonian mechanics, a particle's movement is described by its position, mass, velocity, acceleration, and the forces acting upon it. We map these concepts directly to order books and price series:

| Classical Physics Concept | Mathematical Symbol | Financial Market Equivalent | Description / Economic Interpretation |
| :--- | :---: | :--- | :--- |
| **Position** ($x$) | $P(t)$ | **Price Level** | The current coordinate of the asset in currency space. |
| **Velocity** ($v$) | $V_p(t)$ | **Smoothed Price Change Rate** | The rate of price change per unit of time ($dp/dt$). |
| **Acceleration** ($a$) | $A_p(t)$ | **Rate of Velocity Change** | The rate of change of price velocity ($dv/dt$). |
| **Mass** ($m$) | $M(t)$ | **Order Book Depth / Liquidity** | The resistance of the market to price changes. Highly liquid stocks have high "mass" (hard to move). |
| **Force** ($F$) | $F(t)$ | **Transaction Volume (Force)** | The external energy input pushing the price particle ($F = m \cdot a$). |

---

## 3. Core Mathematical Formulations

### 3.1. Smoothed Price Velocity ($V_p$)

Continuous price velocity is the first derivative of price with respect to time:
$$v(t) = \frac{dP(t)}{dt}$$

In discrete financial time series where $\Delta t = 1$ trading day, we define raw price velocity as the daily price difference:
$$\Delta P_t = P_t - P_{t-1}$$

To eliminate daily noise and isolate the underlying momentum, we apply an Exponential Moving Average (EMA) smoothing window of span $w$:
$$V_p(t) = \text{EMA}(\Delta P, w)_t = \alpha \cdot \Delta P_t + (1 - \alpha) \cdot V_p(t-1)$$
where the smoothing multiplier $\alpha$ is:
$$\alpha = \frac{2}{w + 1}$$

#### Price-Level Neutralized (Percentage Velocity)
To compare velocity across assets with vastly different nominal prices (e.g., TSMC at TWD 900 vs. a penny stock at TWD 5), we define percentage velocity $v_{\%}(t)$:
$$v_{\%}(t) = \text{EMA}\left( \frac{P_t - P_{t-1}}{P_{t-1}}, w \right)_t$$

---

### 3.2. Rolling Velocity Z-Score ($Z_{v}$)

A raw velocity of +3% per day might be normal for a highly volatile microcap stock, but an absolute anomaly for a major index. We normalize velocity by calculating its rolling Z-score against its own historical baseline over a lookback window $L$:

$$Z_{v}(t) = \frac{v_{\%}(t) - \mu_{v}(t)}{\sigma_{v}(t)}$$

Where:
*   $\mu_{v}(t)$ is the rolling mean of percentage velocity:
    $$\mu_{v}(t) = \frac{1}{L} \sum_{i=0}^{L-1} v_{\%}(t-i)$$
*   $\sigma_{v}(t)$ is the rolling standard deviation of percentage velocity:
    $$\sigma_{v}(t) = \sqrt{\frac{1}{L-1} \sum_{i=0}^{L-1} (v_{\%}(t-i) - \mu_{v}(t))^2}$$

A $Z_{v}(t) > 3.0$ indicates that the price is moving upward at a velocity more than 3 standard deviations above its recent historical average — a strong statistical outlier.

---

### 3.3. Smoothed Price Acceleration ($A_p$) and Z-Score ($Z_{a}$)

Acceleration is the second derivative of price, representing the rate at which velocity is increasing or decreasing:
$$a(t) = \frac{dv(t)}{dt} = \frac{d^2P(t)}{dt^2}$$

In our discrete framework, raw acceleration is the first difference of smoothed velocity:
$$A_p(t) = V_p(t) - V_p(t-1)$$

We normalize acceleration similarly using a rolling Z-score over lookback window $L$:
$$Z_{a}(t) = \frac{A_p(t) - \mu_{a}(t)}{\sigma_{a}(t)}$$

---

### 3.4. Kinematic Reversal Score (KRS)

The KRS is designed to capture **momentum exhaustion** (the point where a stock is still moving upward, but its brakes are slammed on heavily). 

In physical terms, this occurs when velocity is positive ($Z_v > 0$) but acceleration is highly negative ($Z_a < 0$). This represents an "impending top" due to extreme deceleration:

$$KRS(t) = \begin{cases} 
- (Z_{v}(t) \cdot Z_{a}(t)) & \text{if } Z_{v}(t) > 0 \text{ and } Z_{a}(t) < 0 \\
0 & \text{otherwise}
\end{cases}$$

A high $KRS$ score is a leading indicator that the "pump" phase of a manipulation has lost its acceleration, making a crash imminent.

---

### 3.5. Volume Anomaly Profile

#### Volume Spike Index ($VS$)
Volume represents the transaction "force" entering the system. The Volume Spike Index is the ratio of today's volume to its rolling average over window $L$:
$$VS_t = \frac{\text{Volume}_t}{\frac{1}{L}\sum_{i=0}^{L-1} \text{Volume}_{t-i}}$$
Values of $VS_t > 3.0$ represent a volume spike that is $3\times$ the historical norm.

#### Volume-Price Divergence ($VPD$)
In a healthy market, large volume spikes are accompanied by proportional price moves. If volume spikes heavily but price barely moves, or if price spikes heavily on no volume, we observe a divergence:
$$VPD_t = VS_t - |Z_{price\_change}(t)|$$
*   **$VPD_t \gg 0$** (Volume-heavy anomaly): Extreme trading activity with minimal price displacement. High suspicion of **Wash Trading**.
*   **$VPD_t \ll 0$** (Price-heavy anomaly): Massive price movement on thin volume. High suspicion of illiquid **Insider Pumps**.

---

### 3.6. Wash Trade Index ($WTI$) — The Tire Slipping Formula

Wash trading is a manipulation where scammers trade back and forth with themselves to generate artificial volume while keeping the price locked in a tight range. 

We model this as the **"Tire Slipping Effect"** (high engine RPM/Force, zero forward velocity).

We calculate the normalized Intraday Range ratio $RR_t$:
$$\text{Intraday Range}_t = \frac{\text{High}_t - \text{Low}_t}{(\text{High}_t + \text{Low}_t) / 2}$$
$$RR_t = \frac{\text{Intraday Range}_t}{\text{EMA}(\text{Intraday Range}, L)_t}$$

The **Wash Trade Index ($WTI$)** is formulated as:
$$WTI_t = \frac{VS_t}{RR_t + \epsilon}$$
Where:
*   $VS_t$ is the Volume Spike Index.
*   $RR_t$ is the normalized daily price range ratio.
*   $\epsilon = 0.01$ is a tiny constant preventing division by zero.

When scammers generate $10\times$ average volume ($VS_t = 10$) but lock the daily range to $10\%$ of normal ($RR_t = 0.1$), the WTI explodes:
$$WTI_t = \frac{10}{0.1 + 0.01} \approx 90.9 \quad (\text{Extreme Anomaly!})$$

Under normal market conditions, a volume spike of $10\times$ naturally expands the daily trading range (e.g. $RR_t = 3.0$), keeping $WTI$ low:
$$WTI_t = \frac{10}{3.0 + 0.01} \approx 3.32 \quad (\text{Normal Market Behavior})$$

---

## 4. Multi-Phase Detection State Machines

To convert raw mathematical indicators into actionable compliance alerts, ShieldRegime groups indicators into state machines with asymmetric conditions.

```mermaid
state_chart
state "NORMAL_TRADING" as normal
state "PUMP_PHASE" as pump
state "PEAK_TRACKING" as peak
state "ALERT_TRIGGERED" as alert

[*] --> normal
normal --> pump : Velocity Z-Score >= 2.5\nAND Volume Spike >= 3.0
pump --> peak : Velocity Decelerating\n(Inertia Peak Reached)
peak --> alert : Price Drawdown >= -25%\nwithin dump_window
alert --> normal : Window Ends / Reset
```

### 4.1. Pump-and-Dump Detector (PND)

The detector models the pump-and-dump cycle as an asymmetric three-step process:

1.  **The Pump Phase Trigger (拉抬觸發)**:
    The detector scans for a joint anomaly where velocity and volume explode simultaneously on some day $t_{pump}$:
    $$Z_{v}(t_{pump}) \ge \tau_{v} \quad \text{AND} \quad VS_{t_{pump}} \ge \tau_{vol}$$
    Where $\tau_{v}$ (default 2.0) and $\tau_{vol}$ (default 2.5 or 3.0) are statistical thresholds.
2.  **Peak Price Tracking (峰值追蹤)**:
    Once the pump is triggered, the system starts a trailing peak tracker to identify the maximum close price $P_{peak}$ achieved during this mania phase. It stops tracking when velocity decelerates heavily (KRS peaks).
3.  **The Dump Phase Validation (崩盤驗證)**:
    From the day of $P_{peak}$, the detector opens an observation window of length $W_{dump}$ (e.g., 15 to 90 days). It checks if the asset suffers a dramatic, catastrophic drop $\theta_{dump}$ (e.g., $-25\%$ or $-30\%$) from $P_{peak}$:
    $$\text{Drawdown}_t = \frac{P_t - P_{peak}}{P_{peak}} \le \theta_{dump} \quad \text{for } t \in [t_{peak}, t_{peak} + W_{dump}]$$
    If this condition is met, a **HIGH** or **CRITICAL** severity `PUMP_AND_DUMP` alert is recorded.

---

### 4.2. Wash Trading Detector (WTD)

Unlike the asymmetrical PND cycle, wash trading is a continuous state of artificial friction. The detector triggers a `WASH_TRADE` alert when the asset exhibits the "Tire Slipping" footprint continuously:

$$WTI_t \ge \tau_{wti} \quad \text{AND} \quad VS_t \ge \tau_{vol} \quad \text{for consecutive days } D \ge D_{min}$$

Where:
*   $\tau_{wti}$ is the WTI threshold (default 3.5).
*   $\tau_{vol}$ is the minimum volume spike (default 2.5).
*   $D_{min}$ is the minimum duration (default 1 or 2 trading days).

---

## 5. Regulatory Relevance (UK FCA & GTV Alignment)

This methodology directly maps to the **UK FCA Handbook - Market Conduct (MAR 1.6: Market Abuse - Manipulating Transactions)**:

*   **FCA MAR 1.6.2 R (1)**: *Giving false or misleading signals as to the supply of, demand for, or price of financial instruments.*
    *   *ShieldRegime's Alignment*: The **Wash Trade Detector** mathematically proves the presence of artificial volume signals that do not translate into physical price displacement (force without displacement).
*   **FCA MAR 1.6.2 R (2)**: *Securing the price of one or several financial instruments at an abnormal or artificial level.*
    *   *ShieldRegime's Alignment*: The **Pump-and-Dump Detector** isolates artificial price inflation driven by abnormal velocity and volume surges, tracing the trajectory from initial force application to final collapse.

By grounding its entire surveillance architecture in Newtonian physics and explainable mathematics, **ShieldRegime** provides a robust, defense-grade compliance framework capable of surviving rigorous regulatory audits and court-level scrutiny.
