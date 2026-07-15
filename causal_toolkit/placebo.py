"""Placebo-test framework: in-time and in-space placebos.

This is the toolkit's core credibility mechanism (per the project spec's
"placebo test pass rate" success metric): an estimator that finds a
"significant effect" where none should exist is not trustworthy,
regardless of how good its ground-truth backtest looked once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type, List

import numpy as np
import pandas as pd

from .estimators.base import CausalEstimator


@dataclass
class PlaceboResult:
    test_type: str  # "in_time" or "in_space"
    unit_or_time: str
    estimated_effect: float
    passed: bool  # True = no false-positive effect detected


def in_time_placebo(
    estimator_cls: Type[CausalEstimator],
    df: pd.DataFrame,
    treated_unit: str,
    true_intervention_time: int,
    fake_intervention_time: int,
    threshold: float,
    **fit_kwargs,
) -> PlaceboResult:
    """Re-run the estimator with a FAKE intervention time strictly before
    the true one. A well-behaved estimator should find ~0 effect.
    """
    assert fake_intervention_time < true_intervention_time, "Fake time must precede the real intervention."
    pre_only = df[df["time"] < true_intervention_time]
    est = estimator_cls()
    est.fit(pre_only, treated_unit=treated_unit, intervention_time=fake_intervention_time, **fit_kwargs)
    effect = est.effect().point_estimate
    passed = abs(effect) < threshold
    return PlaceboResult("in_time", str(fake_intervention_time), effect, passed)


def in_space_placebo(
    estimator_cls: Type[CausalEstimator],
    df: pd.DataFrame,
    true_treated_unit: str,
    intervention_time: int,
    threshold: float,
    **fit_kwargs,
) -> List[PlaceboResult]:
    """Re-run the estimator pretending each CONTROL unit was treated.
    A well-behaved estimator should find ~0 effect for these fake
    "treated" units, since they were never actually intervened on.
    """
    all_units = df["unit"].unique().tolist()
    control_units = [u for u in all_units if u != true_treated_unit]
    # Drop the truly treated unit so its real post-period shock cannot
    # contaminate donor / control means during in-space placebos.
    panel = df[df["unit"] != true_treated_unit].copy()
    fit_kwargs = dict(fit_kwargs)
    donor_units = fit_kwargs.pop("donor_units", None)
    results = []
    for fake_unit in control_units:
        donors = None
        if donor_units is not None:
            donors = [d for d in donor_units if d != fake_unit and d != true_treated_unit]
        est = estimator_cls()
        est.fit(
            panel,
            treated_unit=fake_unit,
            intervention_time=intervention_time,
            donor_units=donors,
            **fit_kwargs,
        )
        effect = est.effect().point_estimate
        passed = abs(effect) < threshold
        results.append(PlaceboResult("in_space", fake_unit, effect, passed))
    return results


def placebo_pass_rate(results: List[PlaceboResult]) -> float:
    if not results:
        return float("nan")
    return sum(r.passed for r in results) / len(results)
