"""Bayesian Structural Time-Series estimator (CausalImpact-style).

Fits a local-level + regression structural time-series model on the
PRE-period only (treated outcome ~ donor covariates + local trend),
then forecasts the counterfactual into the post-period. The gap between
observed and forecast is the estimated causal effect, with a posterior
predictive interval standing in for the credible interval.

Implemented on statsmodels' UnobservedComponents (Kalman-filter based)
rather than a full PyMC MCMC model — this keeps fit times fast (no GPU
required) while preserving the same "structural time series + synthetic
counterfactual forecast" logic that CausalImpact uses. A `pymc_backend`
flag is left as a documented extension point for a full-MCMC version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .base import CausalEstimator, EffectEstimate


class BSTSEstimator(CausalEstimator):
    name = "bayesian_structural_time_series"

    def fit(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        intervention_time: int,
        donor_units: list | None = None,
        n_sim: int = 1000,
        seed: int = 0,
        **kwargs,
    ) -> "BSTSEstimator":
        wide = df.pivot(index="time", columns="unit", values="outcome")
        if donor_units is None:
            donor_units = [c for c in wide.columns if c != treated_unit]
        else:
            donor_units = [c for c in donor_units if c != treated_unit]

        y = wide[treated_unit]
        X = wide[donor_units]
        pre_mask = wide.index < intervention_time
        post_mask = ~pre_mask

        model = UnobservedComponents(
            endog=y[pre_mask].values,
            exog=X[pre_mask].values,
            level="local level",
        )
        # Prefer a robust optimizer; fall back if MLE still struggles.
        try:
            fit_res = model.fit(disp=False, maxiter=500, method="lbfgs")
        except Exception:
            fit_res = model.fit(disp=False, maxiter=500)

        # Forecast counterfactual over the post period using post-period donor covariates
        fc = fit_res.get_forecast(steps=int(post_mask.sum()), exog=X[post_mask].values)
        forecast_mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)

        observed_post = y[post_mask].values
        pointwise_effect = observed_post - forecast_mean
        point_estimate = float(pointwise_effect.mean())

        # Approximate CI on the average effect from forecast predictive
        # variance (documented approximation — not a full MCMC posterior).
        # Floor SEs with pre-period RMSE and do NOT shrink by 1/sqrt(T_post):
        # independent-step averaging undercovers badly when the Kalman fit
        # is overconfident on rich donor covariates.
        se_per_step = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)
        fitted_pre = np.asarray(fit_res.fittedvalues, dtype=float)
        pre_rmse = float(np.sqrt(np.mean((y[pre_mask].values - fitted_pre) ** 2)))
        se_per_step = np.maximum(np.asarray(se_per_step, dtype=float), max(pre_rmse, 1e-6))
        se_avg = float(np.mean(se_per_step))
        rng = np.random.default_rng(seed)
        avg_sim_effects = point_estimate + rng.normal(0.0, se_avg, size=n_sim)
        ci_lower, ci_upper = np.percentile(avg_sim_effects, [2.5, 97.5])
        p_value = float(np.mean(np.sign(avg_sim_effects) != np.sign(point_estimate))) if point_estimate != 0 else 1.0

        self._y = y
        self._forecast_mean = forecast_mean
        self._pre_mask = pre_mask
        self._post_mask = post_mask
        self._result = EffectEstimate(
            method=self.name,
            point_estimate=point_estimate,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=p_value,
            diagnostics={
                "aic": float(fit_res.aic),
                "n_donors_used": len(donor_units),
                "note": "CI approximated via forecast-variance simulation, not full MCMC.",
            },
        )
        self._fitted = True
        return self

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        idx = self._y.index
        ax.plot(idx, self._y.values, label="actual")
        counterfactual = np.concatenate(
            [self._y[self._pre_mask].values, self._forecast_mean]
        )
        ax.plot(idx, counterfactual, label="model / counterfactual forecast", linestyle="--")
        ax.set_title("Bayesian Structural Time Series: actual vs. counterfactual")
        ax.legend()
        return ax
