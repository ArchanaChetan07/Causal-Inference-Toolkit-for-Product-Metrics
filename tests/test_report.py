import os

from causal_toolkit.data import make_ground_truth_dataset
from causal_toolkit.report import run_all_methods, run_placebo_suite, render_html_report


def test_run_all_methods_returns_three_results():
    gt = make_ground_truth_dataset(n_control_units=10, seed=2)
    comp = run_all_methods(gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect)
    assert len(comp["results"]) == 3
    assert comp["ground_truth_check"] is not None


def test_render_html_report_creates_file(tmp_path):
    gt = make_ground_truth_dataset(n_control_units=8, seed=4)
    comp = run_all_methods(gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect)
    placebo = run_placebo_suite(gt.df, gt.treated_unit, gt.intervention_time, fake_intervention_time=10, threshold=2.0)
    out = tmp_path / "report.html"
    render_html_report(comp, placebo, str(out))
    assert out.exists()
    assert out.stat().st_size > 1000
