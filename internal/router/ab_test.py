"""A/B testing framework for model variants."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Variant(str, Enum):
    CONTROL = "control"      # Baseline/production model
    TREATMENT = "treatment"  # New model being tested


@dataclass
class ABTestConfig:
    experiment_name: str
    control_model: str
    treatment_model: str
    traffic_split: float  # 0.0..1.0 percentage to treatment
    enabled: bool = True


class ABTestRouter:
    """Consistent-hashing A/B test router (sticky per user)."""

    def __init__(self, config: ABTestConfig) -> None:
        self.config = config

    def _hash_user(self, user_id: str) -> int:
        """Stable hash in [0, 99]."""
        return int(hashlib.md5(user_id.encode(), usedforsecurity=False).hexdigest(), 16) % 100

    def route(self, user_id: str) -> tuple[Variant, str]:
        """Return (variant, model) for the given user."""
        if not self.config.enabled:
            return Variant.CONTROL, self.config.control_model

        threshold = int(self.config.traffic_split * 100)
        if self._hash_user(user_id) < threshold:
            return Variant.TREATMENT, self.config.treatment_model
        return Variant.CONTROL, self.config.control_model

    def record_outcome(
        self,
        user_id: str,
        variant: Variant,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Return structured outcome for downstream logging."""
        return {
            "experiment": self.config.experiment_name,
            "user_id": user_id,
            "variant": variant.value,
            "metrics": metrics,
        }


@dataclass
class ABTestResult:
    control_mean: float
    treatment_mean: float
    lift_pct: float
    sample_size_control: int
    sample_size_treatment: int
    metric_name: str

    @property
    def treatment_wins(self) -> bool:
        return self.lift_pct > 0


def analyze_ab_test(
    control_values: list[float],
    treatment_values: list[float],
    metric_name: str,
) -> ABTestResult:
    """Simple mean comparison; production should use proper stats (scipy ttest)."""
    c_mean = sum(control_values) / len(control_values) if control_values else 0.0
    t_mean = sum(treatment_values) / len(treatment_values) if treatment_values else 0.0
    lift = ((t_mean - c_mean) / c_mean * 100) if c_mean > 0 else 0.0

    return ABTestResult(
        control_mean=c_mean,
        treatment_mean=t_mean,
        lift_pct=lift,
        sample_size_control=len(control_values),
        sample_size_treatment=len(treatment_values),
        metric_name=metric_name,
    )
