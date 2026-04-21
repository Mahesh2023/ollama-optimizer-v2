"""Prompt/response drift detection."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from math import log2
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    drift_detected: bool
    metric_name: str
    reference_stat: float
    current_stat: float
    delta_pct: float
    details: dict[str, Any] = field(default_factory=dict)


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


def _avg_length(samples: list[str]) -> float:
    if not samples:
        return 0.0
    return sum(len(s) for s in samples) / len(samples)


class DriftDetector:
    """Lightweight statistical drift detector.

    For production, integrate Evidently AI for full drift reports.
    """

    def __init__(self, threshold_pct: float = 20.0) -> None:
        self.threshold_pct = threshold_pct
        self._reference: list[str] = []

    def set_reference(self, samples: list[str]) -> None:
        self._reference = list(samples)

    def check(self, current: list[str]) -> list[DriftReport]:
        if not self._reference:
            return []

        reports = []

        # Length drift
        ref_len = _avg_length(self._reference)
        cur_len = _avg_length(current)
        delta = ((cur_len - ref_len) / ref_len * 100) if ref_len > 0 else 0.0
        reports.append(DriftReport(
            drift_detected=abs(delta) > self.threshold_pct,
            metric_name="avg_length",
            reference_stat=ref_len,
            current_stat=cur_len,
            delta_pct=delta,
        ))

        # Entropy drift
        ref_ent = sum(_shannon_entropy(s) for s in self._reference) / max(len(self._reference), 1)
        cur_ent = sum(_shannon_entropy(s) for s in current) / max(len(current), 1)
        ent_delta = ((cur_ent - ref_ent) / ref_ent * 100) if ref_ent > 0 else 0.0
        reports.append(DriftReport(
            drift_detected=abs(ent_delta) > self.threshold_pct,
            metric_name="shannon_entropy",
            reference_stat=ref_ent,
            current_stat=cur_ent,
            delta_pct=ent_delta,
        ))

        return reports


# Optional Evidently integration
def evidently_drift_report(reference_df, current_df) -> dict[str, Any]:  # type: ignore
    """Run full Evidently drift report. Requires pandas DataFrames."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        return {"error": "Install evidently: pip install 'ollama-optimizer[llmops]'"}

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    return report.as_dict()
