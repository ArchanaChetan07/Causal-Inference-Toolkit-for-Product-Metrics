"""Difference-in-Differences estimator.

Implements classic two-group / two-period DiD via OLS of
`outcome ~ treated * post` with HC1 robust standard errors.

Note (documented honestly): this is *not* a full two-way fixed-effects
(TWFE) regression with unit and time dummies. For a single treated unit
and a common intervention time the classic interaction recovers the same
ATT as TWFE; for staggered multi-cohort rollouts prefer Callaway–Sant'Anna
(roadmap item — see README).
"""
from __future__ import annotations

import logging

import pandas as pd
import statsmodels.formula.api as smf

from .base import CausalEstimator, EffectEstimate

logger = logging.getLogger(__name__)


class DiDEstimator(CausalEstimator):
    name = "difference_in_differences"

    def fit(self, df: pd.DataFrame, treated_unit: str, intervention_time: int, **kwargs) -> DiDEstimator:
        data = df.copy()
        # Always derive treatment from treated_unit so placebo / API callers
        # that reassign the treated entity are respected (do not trust a stale
        # `treated` column from the panel schema).
        data["treated"] = (data["unit"] == treated_unit).astype(int)
        data["post"] = (data["time"] >= intervention_time).astype(int)
        if data["treated"].sum() == 0:
            raise ValueError(f"Treated unit '{treated_unit}' not found in panel.")

        model = smf.ols("outcome ~ treated * post", data=data).fit(cov_type="HC1")

        key = "treated:post"
        if key not in model.params:
            raise RuntimeError("DiD interaction term not estimable from this panel.")
        coef = model.params[key]
        ci_low, ci_high = model.conf_int().loc[key]
        pval = model.pvalues[key]

        self._model = model
        self._data = data
        self._treated_unit = treated_unit
        self._result = EffectEstimate(
            method=self.name,
            point_estimate=float(coef),
            ci_lower=float(ci_low),
            ci_upper=float(ci_high),
            p_value=float(pval),
            diagnostics={
                "r_squared": float(model.rsquared),
                "n_obs": int(model.nobs),
                "specification": "classic treated×post OLS (HC1), not full TWFE",
            },
        )
        self._fitted = True
        logger.info("DiD fit complete for %s: effect=%.4f", treated_unit, float(coef))
        return self

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        means = self._data.groupby(["time", "treated"])["outcome"].mean().unstack()
        means.plot(ax=ax)
        ax.set_title("DiD: treated vs. control means over time")
        ax.set_ylabel("outcome")
        return ax
