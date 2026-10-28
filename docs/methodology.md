# 🛡️ ShieldRegime: Market Kinematics & Manipulation Detection Methodology

## 1. Introduction: From Black-Box Models to White-Box Kinematics

In financial surveillance and regulatory compliance, deep learning models (such as LSTMs or Transformers) often suffer from a critical flaw: **explainability**. When a regulatory body like the **UK Financial Conduct Authority (FCA)** or the **US Securities and Exchange Commission (SEC)** brings a market abuse case to court, a "black-box" neural network prediction is legally insufficient. Regulators and judges require deterministic, mathematically verifiable, and logically transparent evidence.

**ShieldRegime** bridges this gap by moving away from speculative machine learning and returning to first-principles Newtonian physics. We treat the financial price series not as a random walk, but as a **macro-particle moving through a fluid medium under the influence of external forces (volume and liquidity)**. 

By calculating the smoothed first and second derivatives of price (Velocity and Acceleration) and cross-referencing them with volume profile anomalies, we can isolate the exact physical footprint of market manipulation.

---

## 2. The Kinematic-Market Mapping 

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

We normalize acceleration similarly using a rolling Z-score over a lookback window $L$:
$$Z_{a}(t) = \frac{A_p(t) - \mu_{a}(t)}{\sigma_{a}(t)}$$

---

### 3.4. Kinematic Reversal Score (KRS)

The KRS is designed to capture **momentum exhaustion** (the point where a stock is still moving upward, but its brakes are slammed on heavily). 

In physical terms, this occurs when velocity is positive ($Z_v > 0$), but acceleration is highly negative ($Z_a < 0$). This represents an "impending top" due to extreme deceleration:

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

WTI is an OHLCV volume-range screening measure. The physical analogy is descriptive and does not establish self-trading or manipulation.

$$R_t=\frac{H_t-L_t}{(H_t+L_t)/2},\quad B_t=\mathrm{SMA}(R,L)_t$$
$$WTI_t=\frac{VS_t}{RR_t+0.01},\quad RR_t=R_t/B_t\text{ when }B_t>0$$

Rolling baselines include the current bar and use SMA. Write R=(high-low)/((high+low)/2), B=rolling_mean(R), and VS=volume/rolling_mean(volume). With epsilon=0.01, WTI=VS/(R/B+0.01) when B>0. After m=max(lookback//2,1) observations, an entire window of exact high==low defines R/B=0 and WTI=VS/0.01. Fewer than m observations remain NaN. A zero-volume window has undefined VS/WTI; current zero volume with a positive baseline gives WTI=0. Arithmetic failure is an error, not a no-signal result. The detector still needs the full lookback and both WTI>=3.5 and VS>=2.5 by default. Stable positive volume on flat bars can yield WTI=100 with VS=1 and no alert. WTI must not be used alone. Zero intrabar range does not mean prices are unchanged across dates.

For a hypothetical VS=10 and RR=0.1, WTI=10/(0.1+0.01), approximately 90.9. With the same hypothetical VS and RR=3, WTI is approximately 3.32. These calculations do not imply that volume determines a particular price response in actual markets.

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

1.  **The Pump Phase Trigger**:
    The detector scans for a joint anomaly where velocity and volume explode simultaneously on some day $t_{pump}$:
    $$Z_{v}(t_{pump}) \ge \tau_{v} \quad \text{AND} \quad VS_{t_{pump}} \ge \tau_{vol}$$
    Where $\tau_{v}$ (default 2.0) and $\tau_{vol}$ (default 2.5 or 3.0) are statistical thresholds.
2.  **Peak Price Tracking**:
    Once the pump is triggered, the system starts a trailing peak tracker to identify the maximum close price $P_{peak}$ achieved during this mania phase. It stops tracking after including the first bar with velocity Z-score below 1.5. Active candidates keep independent observation windows; a new qualifying pump after tracking ends does not replace an older unexpired candidate.
3.  **The Dump Phase Validation**:
    From the day of $P_{peak}$, the detector opens an observation window of length $W_{dump}$ (e.g., 15 to 90 days). It checks if the asset suffers a dramatic, catastrophic drop $\theta_{dump}$ (e.g., $-25\%$ or $-30\%$) from $P_{peak}$:
    $$\text{Drawdown}_t = \frac{P_t - P_{peak}}{P_{peak}} \le \theta_{dump} \quad \text{for } t \in [t_{peak}, t_{peak} + W_{dump}]$$
    If this condition is met, a **HIGH** or **CRITICAL** severity `PUMP_AND_DUMP` alert is recorded.

---

### 4.2. Wash Trading Detector (WTD)

Each eligible bar must satisfy WTI>=3.5 AND VS>=2.5 by default. Consecutive qualifying bars are then grouped into a WASH_TRADE alert. One bar can trigger an alert; there is no D_min parameter. Full lookback warmup applies. Group duration affects the existing review-priority rules only; detected_at stays at the first eligible bar while group end and metrics may extend.

---

## 5. Regulatory Relevance (UK FCA & GTV Alignment)

This methodology directly maps to the **UK FCA Handbook - Market Conduct (MAR 1.6: Market Abuse - Manipulating Transactions)**:

*   **FCA MAR 1.6.2 R (1)**: *Giving false or misleading signals as to the supply of, demand for, or price of financial instruments.*
    *   *ShieldRegime's Alignment*: The **Wash Trade Detector** flags volume-range patterns for further review; the signal alone cannot establish trading identity, intent or unlawful conduct.
*   **FCA MAR 1.6.2 R (2)**: *Securing the price of one or several financial instruments at an abnormal or artificial level.*
    *   *ShieldRegime's Alignment*: The **Pump-and-Dump Detector** isolates artificial price inflation driven by abnormal velocity and volume surges, tracing the trajectory from initial force application to final collapse.

**ShieldRegime** provides inspectable screening rules to support further review. OHLCV patterns alone do not provide court-level proof of self-trading or manipulation, and no regulatory certification is claimed.


`Alert.start_date` marks the event onset, while `detected_at` is the first completed bar that confirms an alert. For Pump-and-Dump, `end_date` equals `detected_at`; the drop and severity are frozen at first confirmation, rather than revised using a later low. For Wash Trade, `detected_at` stays at the first eligible anomaly while the historical group can extend its end and aggregate metrics. Use daily prefix replay and retain the first snapshot when evaluating post-alert price changes. Daily bars are known after completion; reference-close returns do not assume execution at that close.

Detectors require flat unique OHLCV columns, a unique increasing DatetimeIndex without NaT, finite real numeric values, positive prices, nonnegative volume, and consistent supplied OHLC bounds. They compute on a float64 copy. Invalid inputs and identifiable arithmetic overflow raise errors; valid data shorter than the warmup returns an empty list with `InsufficientDataWarning`. Existing manually constructed alerts may have `detected_at=None`; reconstruct timing from original data rather than substituting event dates.

Pump candidates have independent peak-relative observation windows. The final bar in `dump_window` is eligible. A new qualifying pump after momentum tracking ends can coexist with older unexpired candidates. The first confirmation merges overlapping candidates into one alert, using the earliest-starting qualifying candidate consistently. Drop comparisons use price ratios with relative tolerance `8 * numpy.finfo(numpy.float64).eps` and zero absolute tolerance, including the severity boundary. Historical alert dates, drops and counts can change with this migration. WTI zero-baseline handling now follows the piecewise rule above; broader formula-documentation alignment and statistical calibration remain separate work.


WTI migration: headline/details now describe volume-range anomaly candidates for review. WASH_TRADE, class names, severity values and dates remain compatibility identifiers. The new float metric zero_range_days counts exact high==low bars inside the alert group, rather than calendar duration_days. Severity indicates rule-based review priority, not a probability. OHLCV screening does not establish self-trading, trading identity, intent or market manipulation. Direct WTI calls require real finite numeric Series with identical unique indexes, positive prices, high>=low and nonnegative volume; generic indexes follow row order. Inputs are copied to float64, so float32 and narrow-integer intermediate calculations can differ from older native-dtype results. Positive-baseline regression compares the old formula on identical float64 copies at rtol=1e-12, atol=0. Wash checks original price relationships even before full warmup and rejects lost price ranges or unreliable arithmetic; the shared validator and Pump behavior are unchanged. Text consumers and strict metrics-key consumers must migrate.
