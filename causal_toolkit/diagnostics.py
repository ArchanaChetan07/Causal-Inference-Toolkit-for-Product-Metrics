"""Pre-period diagnostics: parallel-trends checks and donor-pool selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


@dataclass
class ParallelTrendsResult:
    passed: bool
    p_value: float
    interaction_coef: float
    message: str


def _mark_treated(df: pd.DataFrame, treated_unit: str | None) -> pd.DataFrame:
    """Return a copy with `treated` aligned to treated_unit when provided."""
    out = df.copy()
    if treated_unit is not None:
        if treated_unit not in set(out["unit"].unique()):
            raise ValueError(f"Treated unit '{treated_unit}' not found in panel.")
        out["treated"] = (out["unit"] == treated_unit).astype(int)
    return out


def parallel_trends_test(
    df: pd.DataFrame,
    intervention_time: int,
    alpha: float = 0.05,
    treated_unit: str | None = None,
) -> ParallelTrendsResult:
    """Event-study style test: regress outcome on unit*time trend
    interacted with treatment, restricted to the PRE-period only.
    A significant treated*time interaction suggests trends were already
    diverging before treatment -> DiD/synthetic-control assumption at risk.
    """
    pre = _mark_treated(df, treated_unit)
    pre = pre[pre["time"] < intervention_time].copy()
    if pre.empty:
        return ParallelTrendsResult(False, 1.0, 0.0, "No pre-period rows; cannot test parallel trends.")
    pre["time_c"] = pre["time"] - pre["time"].mean()
    model = smf.ols("outcome ~ time_c * treated", data=pre).fit()
    key = "time_c:treated"
    if key not in model.params:
        return ParallelTrendsResult(True, 1.0, 0.0, "No interaction term estimable; skipping.")
    coef = float(model.params[key])
    pval = float(model.pvalues[key])
    passed = bool(pval > alpha)
    msg = (
        f"Pre-trend interaction coef={coef:.4f}, p={pval:.4f} -> "
        + ("PASS (no significant pre-trend divergence)" if passed else "FAIL (parallel trends assumption violated)")
    )
    return ParallelTrendsResult(passed, pval, coef, msg)


def select_donor_pool(
    df: pd.DataFrame,
    treated_unit: str,
    intervention_time: int,
    top_k: int = 10,
) -> list[str]:
    """Rank candidate donor (never-treated) units by pre-period similarity
    (correlation of outcome trajectories) to the treated unit, returning
    the top_k most similar donors — standard practice for synthetic
    control to avoid a noisy/irrelevant donor pool.
    """
    if treated_unit not in set(df["unit"].unique()):
        raise ValueError(f"Treated unit '{treated_unit}' not found in panel.")

    # Never-treated units only (exclude schema-treated and the focal unit)
    ever_treated = set(df.loc[df["treated"] == 1, "unit"].unique())
    candidates = [
        u for u in df["unit"].unique()
        if u != treated_unit and u not in ever_treated
    ]
    if not candidates:
        # Fall back: any unit not equal to treated_unit (single-treated panels
        # where the schema flag already matches).
        candidates = [u for u in df["unit"].unique() if u != treated_unit]
    if not candidates:
        raise ValueError("No donor candidates available.")

    pre = df[df["time"] < intervention_time]
    wide = pre.pivot(index="time", columns="unit", values="outcome")
    if treated_unit not in wide.columns:
        raise ValueError(f"Treated unit '{treated_unit}' has no pre-period rows.")
    target = wide[treated_unit]
    scores = {}
    for col in candidates:
        if col not in wide.columns:
            continue
        corr = wide[col].corr(target)
        scores[col] = float(corr) if not np.isnan(corr) else -np.inf
    if not scores:
        raise ValueError("No overlapping pre-period donor series found.")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    pool = [unit for unit, _ in ranked[:top_k]]
    logger.debug("Selected donor pool for %s: %s", treated_unit, pool)
    return pool


def covariate_balance(
    df: pd.DataFrame,
    intervention_time: int,
    treated_unit: str | None = None,
) -> pd.DataFrame:
    """Simple pre-period mean/std balance table between treated and control groups."""
    pre = _mark_treated(df, treated_unit)
    pre = pre[pre["time"] < intervention_time]
    return (
        pre.groupby("treated")["outcome"]
        .agg(["mean", "std", "count"])
        .rename(index={0: "control", 1: "treated"})
    )
