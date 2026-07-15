"""Synthetic Control Method (Abadie-style), implemented via constrained
convex optimization: find nonnegative weights summing to 1 over donor
units that minimize pre-period MSE to the treated unit, then use the
weighted donor average as the post-period counterfactual.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import CausalEstimator, EffectEstimate

logger = logging.getLogger(__name__)


class SyntheticControlEstimator(CausalEstimator):
    name = "synthetic_control"

    def fit(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        intervention_time: int,
        donor_units: list[str] | None = None,
        **kwargs,
    ) -> SyntheticControlEstimator:
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
            # Never include the (possibly placebo) treated unit in its own donor pool
            donor_units = [c for c in donor_units if c != treated_unit and c in wide.columns]

        n_donors = len(donor_units)
        if n_donors == 0:
            raise ValueError("Synthetic control requires at least one donor unit.")

        pre_mask = wide.index < intervention_time
        post_mask = np.asarray(wide.index >= intervention_time)
        if not pre_mask.any() or not post_mask.any():
            raise ValueError("Need both pre- and post-intervention periods.")

        Y_treated_pre = wide.loc[pre_mask, treated_unit].values
        Y_donors_pre = wide.loc[pre_mask, donor_units].values  # (T_pre, n_donors)

        def loss(w):
            synth = Y_donors_pre @ w
            return np.sum((Y_treated_pre - synth) ** 2)

        w0 = np.repeat(1 / n_donors, n_donors)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0)] * n_donors
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = np.asarray(res.x, dtype=float)
        if not res.success:
            logger.warning(
                "Synthetic control optimizer did not fully converge (%s); using best weights.",
                res.message,
            )
            # Renormalize to satisfy simplex if needed
            weights = np.clip(weights, 0, None)
            s = weights.sum()
            weights = weights / s if s > 0 else w0

        synth_full = wide[donor_units].values @ weights
        gap = wide[treated_unit].values - synth_full
        point_estimate = float(gap[post_mask].mean())

        # In-space placebo distribution -> approximate p-value / CI
        placebo_gaps = []
        if n_donors >= 2:
            for donor in donor_units:
                alt_donors = [d for d in donor_units if d != donor]
                if not alt_donors:
                    continue
                Y_alt_pre = wide.loc[pre_mask, alt_donors].values
                Y_target_pre = wide.loc[pre_mask, donor].values

                def loss_p(w, Y_alt=Y_alt_pre, Y_tgt=Y_target_pre):
                    return np.sum((Y_tgt - Y_alt @ w) ** 2)

                w0p = np.repeat(1 / len(alt_donors), len(alt_donors))
                resp = minimize(
                    loss_p,
                    w0p,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * len(alt_donors),
                    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                )
                w_p = np.asarray(resp.x, dtype=float)
                if not resp.success:
                    w_p = np.clip(w_p, 0, None)
                    s = w_p.sum()
                    w_p = w_p / s if s > 0 else w0p
                synth_p = wide[alt_donors].values @ w_p
                gap_p = wide[donor].values - synth_p
                placebo_gaps.append(float(gap_p[post_mask].mean()))

        placebo_gaps_arr = np.array(placebo_gaps)
        if len(placebo_gaps_arr):
            p_value = float(np.mean(np.abs(placebo_gaps_arr) >= abs(point_estimate)))
            q = float(np.percentile(np.abs(placebo_gaps_arr), 95))
            ci_lower = float(point_estimate - q)
            ci_upper = float(point_estimate + q)
        else:
            p_value = None
            ci_lower = None
            ci_upper = None

        self._weights = dict(zip(donor_units, weights))
        self._wide = wide
        self._synth_full = synth_full
        self._donor_units = donor_units
        self._treated_unit = treated_unit
        self._result = EffectEstimate(
            method=self.name,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            diagnostics={
                "pre_period_rmse": float(
                    np.sqrt(np.mean((Y_treated_pre - Y_donors_pre @ weights) ** 2))
                ),
                "nonzero_donors": {k: round(float(v), 4) for k, v in self._weights.items() if v > 1e-3},
                "optimizer_success": bool(res.success),
                "n_placebos": len(placebo_gaps_arr),
            },
        )
        self._fitted = True
        logger.info(
            "Synthetic control fit for %s: effect=%.4f (optimizer_success=%s)",
            treated_unit,
            point_estimate,
            res.success,
        )
        return self

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self._wide.index, self._wide[self._treated_unit], label="actual (treated)")
        ax.plot(self._wide.index, self._synth_full, label="synthetic control", linestyle="--")
        ax.set_title("Synthetic Control: actual vs. synthetic")
        ax.legend()
        return ax
