import pytest

from causal_toolkit.data import SchemaError, assert_fit_inputs, make_ground_truth_dataset, validate_panel


def test_assert_fit_inputs_rejects_bad_unit():
    gt = make_ground_truth_dataset(n_control_units=3, n_periods=20, seed=0)
    with pytest.raises(SchemaError, match="not found"):
        assert_fit_inputs(gt.df, "missing", gt.intervention_time)


def test_assert_fit_inputs_rejects_out_of_range_time():
    gt = make_ground_truth_dataset(n_control_units=3, n_periods=20, seed=0)
    with pytest.raises(SchemaError, match="outside"):
        assert_fit_inputs(gt.df, gt.treated_unit, 999)


def test_validate_panel_rejects_empty():
    import pandas as pd

    df = pd.DataFrame(columns=["unit", "time", "outcome", "treated", "post"])
    with pytest.raises(SchemaError, match="empty"):
        validate_panel(df)


def test_load_csv_roundtrip(tmp_path):
    from causal_toolkit.data import load_csv

    gt = make_ground_truth_dataset(n_control_units=3, n_periods=10, seed=0)
    path = tmp_path / "panel.csv"
    gt.df.to_csv(path, index=False)
    loaded = load_csv(str(path))
    assert len(loaded) == len(gt.df)
