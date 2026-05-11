"""
alert.py — Structured alert model for detected anomalies.

Each detection result is wrapped in an Alert object with severity,
timestamp, and human-readable explanation — ready for dashboards,
logs, or downstream compliance pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class AlertSeverity(Enum):
    """Traffic-light severity for detected anomalies."""

    LOW = "LOW"            # Statistical outlier, worth monitoring
    MEDIUM = "MEDIUM"      # Multiple indicators converge — suspicious
    HIGH = "HIGH"          # Strong manipulation signature detected
    CRITICAL = "CRITICAL"  # Textbook manipulation pattern confirmed


@dataclass
class Alert:
    """A single anomaly detection alert.

    Attributes:
        ticker:     Stock symbol (e.g. "GME", "3662").
        alert_type: Category string such as "PUMP_AND_DUMP" or "WASH_TRADE".
        severity:   How confident the detector is.
        start_date: First date of the suspicious window.
        end_date:   Last date of the suspicious window (may equal start_date).
        headline:   One-line human-readable summary.
        details:    Longer explanation with numbers.
        metrics:    Raw indicator values that triggered the alert.
    """

    ticker: str
    alert_type: str
    severity: AlertSeverity
    start_date: datetime
    end_date: datetime
    headline: str
    details: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def duration_days(self) -> int:
        """Number of calendar days the suspicious window spans."""
        return max((self.end_date - self.start_date).days, 1)

    def to_dict(self) -> Dict:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "ticker": self.ticker,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "duration_days": self.duration_days,
            "headline": self.headline,
            "details": self.details,
            "metrics": self.metrics,
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] {self.ticker} — {self.headline} "
            f"({self.start_date:%Y-%m-%d} → {self.end_date:%Y-%m-%d})"
        )
