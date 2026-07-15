import pandas as pd
import pytest

from causal_toolkit.data import SchemaError, make_ground_truth_dataset, validate_panel


def test_ground_truth_dataset_shape():
    gt = make_ground_truth_dataset(n_control_units=5, n_periods=20)
    assert gt.df["unit"].nunique() == 6
    assert set(gt.df.columns) == {"unit", "time", "outcome", "treated", "post"}
    assert gt.true_effect != 0


def test_validate_panel_passes_on_good_data():
    gt = make_ground_truth_dataset()
    validated = validate_panel(gt.df)
    assert len(validated) == len(gt.df)


def test_validate_panel_missing_column():
    df = pd.DataFrame({"unit": ["a"], "time": [0], "outcome": [1.0], "treated": [1]})
    with pytest.raises(SchemaError):
        validate_panel(df)


def test_validate_panel_nan_outcome():
    gt = make_ground_truth_dataset()
    df = gt.df.copy()
    df.loc[0, "outcome"] = float("nan")
    with pytest.raises(SchemaError):
        validate_panel(df)


def test_validate_panel_requires_untreated_donors():
    df = pd.DataFrame(
        {
            "unit": ["a", "a"],
            "time": [0, 1],
            "outcome": [1.0, 2.0],
            "treated": [1, 1],
            "post": [0, 1],
        }
    )
    with pytest.raises(SchemaError):
        validate_panel(df)


def test_validate_panel_rejects_duplicate_unit_time():
    df = pd.DataFrame(
        {
            "unit": ["a", "a", "b", "b"],
            "time": [0, 0, 0, 1],
            "outcome": [1.0, 2.0, 3.0, 4.0],
            "treated": [1, 1, 0, 0],
            "post": [0, 0, 0, 1],
        }
    )
    with pytest.raises(SchemaError, match="duplicate"):
        validate_panel(df)


def test_validate_panel_rejects_inconsistent_treated_flag():
    df = pd.DataFrame(
        {
            "unit": ["a", "a", "b", "b"],
            "time": [0, 1, 0, 1],
            "outcome": [1.0, 2.0, 3.0, 4.0],
            "treated": [1, 0, 0, 0],
            "post": [0, 1, 0, 1],
        }
    )
    with pytest.raises(SchemaError, match="constant"):
        validate_panel(df)
