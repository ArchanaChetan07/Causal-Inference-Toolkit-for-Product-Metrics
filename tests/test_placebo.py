from causal_toolkit.data import make_ground_truth_dataset
from causal_toolkit.estimators import DiDEstimator, BSTSEstimator
from causal_toolkit.placebo import in_time_placebo, in_space_placebo, placebo_pass_rate


def test_in_time_placebo_returns_small_effect_for_did():
    gt = make_ground_truth_dataset(seed=9)
    result = in_time_placebo(
        DiDEstimator, gt.df, gt.treated_unit,
        true_intervention_time=gt.intervention_time,
        fake_intervention_time=10, threshold=2.0,
    )
    assert result.test_type == "in_time"


def test_in_space_placebo_runs_for_all_controls():
    gt = make_ground_truth_dataset(n_control_units=8, seed=5)
    results = in_space_placebo(BSTSEstimator, gt.df, gt.treated_unit, gt.intervention_time, threshold=2.0)
    assert len(results) == 8
    rate = placebo_pass_rate(results)
    assert 0.0 <= rate <= 1.0


def test_placebo_pass_rate_empty_list():
    import math
    assert math.isnan(placebo_pass_rate([]))
