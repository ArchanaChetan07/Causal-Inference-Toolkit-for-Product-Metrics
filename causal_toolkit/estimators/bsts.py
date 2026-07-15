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

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .base import CausalEstimator, EffectEstimate

logger = logging.getLogger(__name__)


class BSTSEstimator(CausalEstimator):
    name = "bayesian_structural_time_series"

    def fit(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        intervention_time: int,
        donor_units: list[str] | None = None,
        n_sim: int = 1000,
        seed: int = 0,
        max_donors: int | None = None,
        **kwargs,
    ) -> BSTSEstimator:
        wide = df.pivot(index="time", columns="unit", values="outcome")
        if treated_unit not in wide.columns:
            raise ValueError(f"Treated unit '{treated_unit}' not found in panel.")

        if donor_units is None:
            ever_treated = set(df.loc[df["treated"] == 1, "unit"].unique())
            donor_units = [
                c for c in wide.columns
                if c != treated_unit and c not in ever_treated
            ]
            if not donor_units:
                donor_units = [c for c in wide.columns if c != treated_unit]
        else:
            donor_units = [c for c in donor_units if c != treated_unit and c in wide.columns]

        pre_mask = wide.index < intervention_time
        post_mask = ~pre_mask
        t_pre = int(pre_mask.sum())
        if t_pre < 3:
            raise ValueError("BSTS requires at least 3 pre-intervention periods.")
        if not post_mask.any():
            raise ValueError("BSTS requires at least one post-intervention period.")
        if not donor_units:
            raise ValueError("BSTS requires at least one donor covariate series.")

        # Cap donors to avoid over-parameterized pre-period regressions
        donor_cap = max_donors if max_donors is not None else max(1, t_pre - 2)
        if len(donor_units) > donor_cap:
            # Prefer donors already ranked by caller; otherwise keep first N
            donor_units = list(donor_units)[:donor_cap]
            logger.info("BSTS capped donors to %d (T_pre=%d)", donor_cap, t_pre)

        y = wide[treated_unit]
        X = wide[donor_units]

        model = UnobservedComponents(
            endog=y[pre_mask].values,
            exog=X[pre_mask].values,
            level="local level",
        )
        try:
            fit_res = model.fit(disp=False, maxiter=500, method="lbfgs")
        except Exception as exc:  # noqa: BLE001 — fall back to default optimizer
            logger.warning("BSTS LBFGS fit failed (%s); retrying default optimizer.", exc)
            fit_res = model.fit(disp=False, maxiter=500)

        fc = fit_res.get_forecast(steps=int(post_mask.sum()), exog=X[post_mask].values)
        forecast_mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)

        observed_post = y[post_mask].values
        pointwise_effect = observed_post - forecast_mean
        point_estimate = float(pointwise_effect.mean())

        # Approximate CI on the average effect from forecast predictive
        # variance (documented approximation — not a full MCMC posterior).
        # Floor SEs with pre-period RMSE and do NOT shrink by 1/sqrt(T_post):
        # independent-step averaging undercovers when the Kalman fit is
        # overconfident on rich donor covariates.
        se_per_step = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)
        fitted_pre = np.asarray(fit_res.fittedvalues, dtype=float)
        pre_rmse = float(np.sqrt(np.mean((y[pre_mask].values - fitted_pre) ** 2)))
        se_per_step = np.maximum(np.asarray(se_per_step, dtype=float), max(pre_rmse, 1e-6))
        se_avg = float(np.mean(se_per_step))
        rng = np.random.default_rng(seed)
        avg_sim_effects = point_estimate + rng.normal(0.0, se_avg, size=n_sim)
        ci_lower, ci_upper = np.percentile(avg_sim_effects, [2.5, 97.5])
        if se_avg > 0 and point_estimate != 0:
            z = abs(point_estimate) / se_avg
            p_value = float(2 * (1 - scipy_stats.norm.cdf(z)))
        else:
            p_value = 1.0

        self._y = y
        self._forecast_mean = forecast_mean
        self._pre_mask = pre_mask
        self._post_mask = post_mask
        self._treated_unit = treated_unit
        self._result = EffectEstimate(
            method=self.name,
            point_estimate=point_estimate,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=p_value,
            diagnostics={
                "aic": float(fit_res.aic),
                "n_donors_used": len(donor_units),
                "se_avg": se_avg,
                "pre_rmse": pre_rmse,
                "note": "CI approximated via forecast-variance simulation, not full MCMC.",
            },
        )
        self._fitted = True
        logger.info("BSTS fit for %s: effect=%.4f se=%.4f", treated_unit, point_estimate, se_avg)
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
