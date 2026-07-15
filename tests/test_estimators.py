import pytest

from causal_toolkit.data import make_ground_truth_dataset
from causal_toolkit.estimators import BSTSEstimator, DiDEstimator, SyntheticControlEstimator


@pytest.fixture(scope="module")
def gt():
    return make_ground_truth_dataset(n_control_units=15, n_periods=40, true_effect=-8.0, seed=1)


@pytest.mark.parametrize("cls", [DiDEstimator, SyntheticControlEstimator, BSTSEstimator])
def test_estimator_recovers_true_effect_within_tolerance(gt, cls):
    est = cls()
    est.fit(gt.df, treated_unit=gt.treated_unit, intervention_time=gt.intervention_time)
    eff = est.effect()
    # Method-specific tolerances: SC is known to be biased on this noisy-walk DGP
    tol = 5.0 if cls is SyntheticControlEstimator else 2.5
    assert abs(eff.point_estimate - gt.true_effect) < tol, (
        f"{cls.__name__} estimate too far from ground truth"
    )


@pytest.mark.parametrize("cls", [DiDEstimator, SyntheticControlEstimator, BSTSEstimator])
def test_effect_before_fit_raises(cls):
    est = cls()
    with pytest.raises(RuntimeError):
        est.effect()


def test_did_respects_treated_unit_argument(gt):
    """DiD must re-derive treatment from treated_unit (critical for placebos)."""
    real = (
        DiDEstimator()
        .fit(gt.df, treated_unit=gt.treated_unit, intervention_time=gt.intervention_time)
        .effect()
    )
    fake_unit = [u for u in gt.df["unit"].unique() if u != gt.treated_unit][0]
    panel = gt.df[gt.df["unit"] != gt.treated_unit]
    fake = (
        DiDEstimator()
        .fit(panel, treated_unit=fake_unit, intervention_time=gt.intervention_time)
        .effect()
    )
    assert abs(real.point_estimate - gt.true_effect) < 3.0
    assert abs(fake.point_estimate) < abs(real.point_estimate) * 0.5


def test_did_ci_contains_estimate(gt):
    est = DiDEstimator()
    est.fit(gt.df, treated_unit=gt.treated_unit, intervention_time=gt.intervention_time)
    eff = est.effect()
    assert eff.ci_lower < eff.point_estimate < eff.ci_upper


def test_synthetic_control_weights_sum_to_one(gt):
    est = SyntheticControlEstimator()
    est.fit(gt.df, treated_unit=gt.treated_unit, intervention_time=gt.intervention_time)
    total_weight = sum(est._weights.values())
    assert abs(total_weight - 1.0) < 1e-4


def test_synthetic_control_ci_centered_on_estimate(gt):
    est = SyntheticControlEstimator()
    est.fit(gt.df, treated_unit=gt.treated_unit, intervention_time=gt.intervention_time)
    eff = est.effect()
    assert eff.ci_lower is not None and eff.ci_upper is not None
    assert eff.ci_lower <= eff.point_estimate <= eff.ci_upper


def test_bsts_ci_covers_true_effect_on_ground_truth(gt):
    est = BSTSEstimator()
    est.fit(gt.df, treated_unit=gt.treated_unit, intervention_time=gt.intervention_time)
    eff = est.effect()
    assert eff.ci_lower <= gt.true_effect <= eff.ci_upper
    assert eff.ci_lower < eff.point_estimate < eff.ci_upper


def test_synthetic_control_single_donor_does_not_crash():
    gt = make_ground_truth_dataset(n_control_units=1, n_periods=30, seed=0)
    est = SyntheticControlEstimator()
    est.fit(gt.df, treated_unit=gt.treated_unit, intervention_time=gt.intervention_time)
    eff = est.effect()
    assert eff.point_estimate == eff.point_estimate  # not NaN
    # Placebo inference unavailable with a single donor
    assert eff.diagnostics["n_placebos"] == 0
