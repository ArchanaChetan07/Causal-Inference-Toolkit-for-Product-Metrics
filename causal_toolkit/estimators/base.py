"""Common interface every causal estimator implements.

This is what makes the toolkit a coherent LIBRARY rather than three
disconnected scripts: any estimator can be swapped in wherever a
CausalEstimator is expected (report generation, API, placebo tests).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class EffectEstimate:
    method: str
    point_estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        ci = (
            f"[{self.ci_lower:.3f}, {self.ci_upper:.3f}]"
            if self.ci_lower is not None
            else "n/a"
        )
        p = f"{self.p_value:.4f}" if self.p_value is not None else "n/a"
        return f"{self.method}: effect={self.point_estimate:.3f}, 95% CI={ci}, p={p}"


class CausalEstimator(ABC):
    """Abstract base class. Subclasses implement fit() and effect()."""

    name: str = "base"

    def __init__(self):
        self._fitted = False
        self._result: EffectEstimate | None = None

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        intervention_time: int,
        **kwargs,
    ) -> CausalEstimator:
        ...

    def effect(self) -> EffectEstimate:
        if not self._fitted or self._result is None:
            raise RuntimeError(f"{self.name} estimator has not been fit yet.")
        return self._result

    def plot(self, ax=None):
        raise NotImplementedError(f"{self.name} does not implement plot() yet.")
