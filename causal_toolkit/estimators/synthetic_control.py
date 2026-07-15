"""Synthetic Control Method (Abadie-style), implemented via constrained
convex optimization: find nonnegative weights summing to 1 over donor
units that minimize pre-period MSE to the treated unit, then use the
weighted donor average as the post-period counterfactual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import CausalEstimator, EffectEstimate


class SyntheticControlEstimator(CausalEstimator):
    name = "synthetic_control"

    def fit(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        intervention_time: int,
        donor_units: list | None = None,
        n_placebo_for_ci: int = 200,
        **kwargs,
    ) -> "SyntheticControlEstimator":
        wide = df.pivot(index="time", columns="unit", values="outcome")
        if donor_units is None:
            donor_units = [c for c in wide.columns if c != treated_unit]
        else:
            # Never include the (possibly placebo) treated unit in its own donor pool
            donor_units = [c for c in donor_units if c != treated_unit]

        pre_mask = wide.index < intervention_time
        Y_treated_pre = wide.loc[pre_mask, treated_unit].values
        Y_donors_pre = wide.loc[pre_mask, donor_units].values  # (T_pre, n_donors)

        n_donors = len(donor_units)
        if n_donors == 0:
            raise ValueError("Synthetic control requires at least one donor unit.")

        def loss(w):
            synth = Y_donors_pre @ w
            return np.sum((Y_treated_pre - synth) ** 2)

        w0 = np.repeat(1 / n_donors, n_donors)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n_donors
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x

        synth_full = wide[donor_units].values @ weights
        gap = wide[treated_unit].values - synth_full
        post_mask = np.asarray(wide.index >= intervention_time)
        point_estimate = float(gap[post_mask].mean())

        # In-space placebo distribution -> approximate p-value / CI, the
        # standard synthetic-control inference approach (Abadie et al. 2010)
        placebo_gaps = []
        for donor in donor_units:
            alt_donors = [d for d in donor_units if d != donor]
            Y_alt_pre = wide.loc[pre_mask, alt_donors].values
            Y_target_pre = wide.loc[pre_mask, donor].values

            def loss_p(w):
                return np.sum((Y_target_pre - Y_alt_pre @ w) ** 2)

            w0p = np.repeat(1 / len(alt_donors), len(alt_donors))
            resp = minimize(
                loss_p, w0p, method="SLSQP",
                bounds=[(0, 1)] * len(alt_donors),
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            )
            synth_p = wide[alt_donors].values @ resp.x
            gap_p = wide[donor].values - synth_p
            placebo_gaps.append(gap_p[post_mask].mean())

        placebo_gaps = np.array(placebo_gaps)
        p_value = float(np.mean(np.abs(placebo_gaps) >= abs(point_estimate))) if len(placebo_gaps) else None
        # Abadie-style interval: center the placebo null distribution on the
        # treated estimate (percentiles of placebo gaps alone are a null
        # reference, not a CI for the effect).
        if len(placebo_gaps):
            q = float(np.percentile(np.abs(placebo_gaps), 95))
            ci_lower = float(point_estimate - q)
            ci_upper = float(point_estimate + q)
        else:
            ci_lower = ci_upper = None

        self._weights = dict(zip(donor_units, weights))
        self._wide = wide
        self._synth_full = synth_full
        self._donor_units = donor_units
        self._result = EffectEstimate(
            method=self.name,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            diagnostics={
                "pre_period_rmse": float(np.sqrt(np.mean((Y_treated_pre - Y_donors_pre @ weights) ** 2))),
                "nonzero_donors": {k: round(v, 4) for k, v in self._weights.items() if v > 1e-3},
            },
        )
        self._fitted = True
        return self

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        treated_unit = [c for c in self._wide.columns if c not in self._donor_units][0]
        ax.plot(self._wide.index, self._wide[treated_unit], label="actual (treated)")
        ax.plot(self._wide.index, self._synth_full, label="synthetic control", linestyle="--")
        ax.set_title("Synthetic Control: actual vs. synthetic")
        ax.legend()
        return ax
