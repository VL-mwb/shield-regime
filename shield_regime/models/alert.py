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
    """Rule-based review priority for detected anomalies."""

    LOW = "LOW"            # Low review priority
    MEDIUM = "MEDIUM"      # Medium review priority
    HIGH = "HIGH"          # High review priority
    CRITICAL = "CRITICAL"  # Critical review priority


@dataclass
class Alert:
    """A single anomaly detection alert.

    Attributes:
        ticker:     Stock symbol (e.g. "GME", "3662").
        alert_type: Category string such as "PUMP_AND_DUMP" or "WASH_TRADE".
        severity:   Rule-based review priority.
        start_date: First date of the suspicious window.
        end_date:   Observed end; confirmation bar for pump, latest group bar for wash.
        detected_at: First bar whose completed data confirms the alert; None for legacy alerts.
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

    detected_at: Optional[datetime] = None

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
            "detected_at": self.detected_at.isoformat() if self.detected_at is not None else None,
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
