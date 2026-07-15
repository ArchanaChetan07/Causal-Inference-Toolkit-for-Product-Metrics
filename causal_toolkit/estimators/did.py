"""Difference-in-Differences estimator.

Implements classic two-way-fixed-effects (TWFE) DiD. Note (documented
honestly, per the project's "cross-method disagreement" requirement):
TWFE is known to be biased under staggered treatment timing with
heterogeneous effects (Goodman-Bacon 2021, Callaway-Sant'Anna 2021).
This toolkit's single-treated-unit case is immune to that specific bias
(no staggering), but the docstring flags it so users applying this to
multi-cohort staggered rollouts know to reach for a staggered-DiD
estimator instead (roadmap item, see README).
"""
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from .base import CausalEstimator, EffectEstimate


class DiDEstimator(CausalEstimator):
    name = "difference_in_differences"

    def fit(self, df: pd.DataFrame, treated_unit: str, intervention_time: int, **kwargs) -> "DiDEstimator":
        data = df.copy()
        # Always derive treatment from treated_unit so placebo / API callers
        # that reassign the treated entity are respected (do not trust a stale
        # `treated` column from the panel schema).
        data["treated"] = (data["unit"] == treated_unit).astype(int)
        data["post"] = (data["time"] >= intervention_time).astype(int)
        model = smf.ols("outcome ~ treated * post", data=data).fit(cov_type="HC1")

        key = "treated:post"
        coef = model.params[key]
        se = model.bse[key]
        ci_low, ci_high = model.conf_int().loc[key]
        pval = model.pvalues[key]

        self._model = model
        self._data = data
        self._result = EffectEstimate(
            method=self.name,
            point_estimate=float(coef),
            ci_lower=float(ci_low),
            ci_upper=float(ci_high),
            p_value=float(pval),
            diagnostics={"r_squared": float(model.rsquared), "n_obs": int(model.nobs)},
        )
        self._fitted = True
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
