"""Pre-period diagnostics: parallel-trends checks and donor-pool selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dataclasses import dataclass
from typing import List


@dataclass
class ParallelTrendsResult:
    passed: bool
    p_value: float
    interaction_coef: float
    message: str


def parallel_trends_test(df: pd.DataFrame, intervention_time: int, alpha: float = 0.05) -> ParallelTrendsResult:
    """Event-study style test: regress outcome on unit*time trend
    interacted with treatment, restricted to the PRE-period only.
    A significant treated*time interaction suggests trends were already
    diverging before treatment -> DiD/synthetic-control assumption at risk.
    """
    pre = df[df["time"] < intervention_time].copy()
    pre["time_c"] = pre["time"] - pre["time"].mean()
    model = smf.ols("outcome ~ time_c * treated", data=pre).fit()
    key = "time_c:treated"
    if key not in model.params:
        return ParallelTrendsResult(True, 1.0, 0.0, "No interaction term estimable; skipping.")
    coef = model.params[key]
    pval = model.pvalues[key]
    passed = bool(pval > alpha)
    msg = (
        f"Pre-trend interaction coef={coef:.4f}, p={pval:.4f} -> "
        + ("PASS (no significant pre-trend divergence)" if passed else "FAIL (parallel trends assumption violated)")
    )
    return ParallelTrendsResult(passed, float(pval), float(coef), msg)


def select_donor_pool(df: pd.DataFrame, treated_unit: str, intervention_time: int, top_k: int = 10) -> List[str]:
    """Rank candidate donor (control) units by pre-period similarity
    (correlation of outcome trajectories) to the treated unit, returning
    the top_k most similar donors — standard practice for synthetic
    control to avoid a noisy/irrelevant donor pool.
    """
    pre = df[df["time"] < intervention_time]
    wide = pre.pivot(index="time", columns="unit", values="outcome")
    if treated_unit not in wide.columns:
        raise ValueError(f"Treated unit '{treated_unit}' not found in panel.")
    target = wide[treated_unit]
    scores = {}
    for col in wide.columns:
        if col == treated_unit:
            continue
        corr = wide[col].corr(target)
        scores[col] = corr if not np.isnan(corr) else -np.inf
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [unit for unit, _ in ranked[:top_k]]


def covariate_balance(df: pd.DataFrame, intervention_time: int) -> pd.DataFrame:
    """Simple pre-period mean/std balance table between treated and control groups."""
    pre = df[df["time"] < intervention_time]
    return (
        pre.groupby("treated")["outcome"]
        .agg(["mean", "std", "count"])
        .rename(index={0: "control", 1: "treated"})
    )
