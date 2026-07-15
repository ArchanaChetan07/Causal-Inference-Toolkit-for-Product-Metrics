"""Causal Inference Toolkit for Product Metrics.

A validated multi-method causal impact library: difference-in-differences,
synthetic control, and Bayesian structural time series, with built-in
pre-period diagnostics, placebo-test validation, and cross-method
disagreement reporting.
"""
from .data import make_ground_truth_dataset, load_csv, validate_panel, GroundTruthDataset
from .estimators.did import DiDEstimator
from .estimators.synthetic_control import SyntheticControlEstimator
from .estimators.bsts import BSTSEstimator
from .diagnostics import parallel_trends_test, select_donor_pool, covariate_balance
from .placebo import in_time_placebo, in_space_placebo, placebo_pass_rate
from .report import run_all_methods, run_placebo_suite, render_html_report

__version__ = "0.1.0"

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
