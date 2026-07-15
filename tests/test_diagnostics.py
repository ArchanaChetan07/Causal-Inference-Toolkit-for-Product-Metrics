from causal_toolkit.data import make_ground_truth_dataset
from causal_toolkit.diagnostics import parallel_trends_test, select_donor_pool, covariate_balance


def test_parallel_trends_passes_on_well_behaved_synthetic_data():
    gt = make_ground_truth_dataset(seed=7)
    result = parallel_trends_test(gt.df, gt.intervention_time)
    assert result.passed, "Well-behaved synthetic data should pass parallel trends"


def test_donor_pool_selection_returns_requested_count():
    gt = make_ground_truth_dataset(n_control_units=20, seed=3)
    donors = select_donor_pool(gt.df, gt.treated_unit, gt.intervention_time, top_k=5)
    assert len(donors) == 5
    assert gt.treated_unit not in donors


def test_covariate_balance_has_two_groups():
    gt = make_ground_truth_dataset()
    bal = covariate_balance(gt.df, gt.intervention_time)
    assert set(bal.index) == {"control", "treated"}
