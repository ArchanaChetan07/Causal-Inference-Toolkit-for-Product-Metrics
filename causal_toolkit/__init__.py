"""Causal Inference Toolkit for Product Metrics.

A validated multi-method causal impact library: difference-in-differences,
synthetic control, and Bayesian structural time series, with built-in
pre-period diagnostics, placebo-test validation, and cross-method
disagreement reporting.
"""
from .data import GroundTruthDataset, load_csv, make_ground_truth_dataset, validate_panel
from .diagnostics import covariate_balance, parallel_trends_test, select_donor_pool
from .estimators.bsts import BSTSEstimator
from .estimators.did import DiDEstimator
from .estimators.synthetic_control import SyntheticControlEstimator
from .placebo import in_space_placebo, in_time_placebo, placebo_pass_rate
from .report import render_html_report, run_all_methods, run_placebo_suite

__version__ = "0.2.0"

__all__ = [
    "make_ground_truth_dataset",
    "load_csv",
    "validate_panel",
    "GroundTruthDataset",
    "DiDEstimator",
    "SyntheticControlEstimator",
    "BSTSEstimator",
    "parallel_trends_test",
    "select_donor_pool",
    "covariate_balance",
    "in_time_placebo",
    "in_space_placebo",
    "placebo_pass_rate",
    "run_all_methods",
    "run_placebo_suite",
    "render_html_report",
    "__version__",
]
