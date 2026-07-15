from causal_toolkit.data import make_ground_truth_dataset
from causal_toolkit.diagnostics import covariate_balance, parallel_trends_test, select_donor_pool


def test_parallel_trends_passes_on_well_behaved_synthetic_data():
    gt = make_ground_truth_dataset(seed=7)
    result = parallel_trends_test(gt.df, gt.intervention_time, treated_unit=gt.treated_unit)
    assert result.passed, "Well-behaved synthetic data should pass parallel trends"


def test_parallel_trends_respects_treated_unit_override():
    gt = make_ground_truth_dataset(n_control_units=10, seed=7)
    # Flip schema flags so treated_unit must be used for a correct test
    df = gt.df.copy()
    df["treated"] = 0
    df.loc[df["unit"] == "control_0", "treated"] = 1
    # When we ask about treated_unit (true treated), should still pass on this DGP
    result = parallel_trends_test(df, gt.intervention_time, treated_unit=gt.treated_unit)
    assert result.passed


def test_donor_pool_selection_returns_requested_count():
    gt = make_ground_truth_dataset(n_control_units=20, seed=3)
    donors = select_donor_pool(gt.df, gt.treated_unit, gt.intervention_time, top_k=5)
    assert len(donors) == 5
    assert gt.treated_unit not in donors


def test_donor_pool_excludes_other_treated_units():
    gt = make_ground_truth_dataset(n_control_units=10, seed=3)
    df = gt.df.copy()
    # Mark control_0 as also treated in schema
    df.loc[df["unit"] == "control_0", "treated"] = 1
    donors = select_donor_pool(df, gt.treated_unit, gt.intervention_time, top_k=20)
    assert "control_0" not in donors
    assert gt.treated_unit not in donors


def test_covariate_balance_has_two_groups():
    gt = make_ground_truth_dataset()
    bal = covariate_balance(gt.df, gt.intervention_time, treated_unit=gt.treated_unit)
    assert set(bal.index) == {"control", "treated"}
