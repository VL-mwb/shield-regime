"""
ShieldRegime — Detect market manipulation using price kinematics.

A physics-inspired toolkit for identifying suspicious trading patterns
such as Pump-and-Dump schemes, Wash Trading, and Spoofing in financial
time series data.
"""

__version__ = "0.1.0"

from shield_regime.indicators.velocity import price_velocity, velocity_zscore
from shield_regime.indicators.acceleration import price_acceleration, acceleration_zscore
from shield_regime.indicators.volume_profile import (
    volume_spike_index,
    volume_price_divergence,
    wash_trade_index,
)
from shield_regime.models.alert import Alert, AlertSeverity
from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.detectors.wash_trade import WashTradeDetector
from shield_regime.visualizer import plot_alerts

__all__ = [
    "price_velocity",
    "velocity_zscore",
    "price_acceleration",
    "acceleration_zscore",
    "volume_spike_index",
    "volume_price_divergence",
    "wash_trade_index",
    "Alert",
    "AlertSeverity",
    "PumpDumpDetector",
    "WashTradeDetector",
    "plot_alerts",
]
