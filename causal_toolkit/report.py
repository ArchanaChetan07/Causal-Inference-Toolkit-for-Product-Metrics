"""Cross-method comparison and disagreement-reporting module.

Runs all three estimators on the same panel, quantifies disagreement,
runs placebo validation, and renders a self-contained HTML report —
the "methods-comparison report" deliverable from the project spec.
"""
from __future__ import annotations

import base64
import html
import io
import logging
from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from .data import assert_fit_inputs, validate_panel
from .diagnostics import parallel_trends_test, select_donor_pool
from .estimators.bsts import BSTSEstimator
from .estimators.did import DiDEstimator
from .estimators.synthetic_control import SyntheticControlEstimator
from .placebo import in_space_placebo, in_time_placebo, placebo_pass_rate

logger = logging.getLogger(__name__)

ESTIMATORS: dict[str, type] = {
    "difference_in_differences": DiDEstimator,
    "synthetic_control": SyntheticControlEstimator,
    "bayesian_structural_time_series": BSTSEstimator,
}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _fmt_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


def run_all_methods(
    df: pd.DataFrame,
    treated_unit: str,
    intervention_time: int,
    true_effect: float | None = None,
) -> dict[str, Any]:
    """Fit all three estimators and return a structured results dict."""
    df = validate_panel(df)
    assert_fit_inputs(df, treated_unit, intervention_time)

    trend_check = parallel_trends_test(df, intervention_time, treated_unit=treated_unit)
    donor_pool = select_donor_pool(df, treated_unit, intervention_time, top_k=15)

    results = {}
    figs = {}
    for name, cls in ESTIMATORS.items():
        est = cls()
        est.fit(
            df,
            treated_unit=treated_unit,
            intervention_time=intervention_time,
            donor_units=donor_pool,
        )
        eff = est.effect()
        results[name] = eff
        try:
            figs[name] = est.plot()
        except NotImplementedError:
            pass

    estimates = {k: v.point_estimate for k, v in results.items()}
    spread = max(estimates.values()) - min(estimates.values())
    mean_est = sum(estimates.values()) / len(estimates)
    rel_disagreement = spread / abs(mean_est) if mean_est != 0 else float("inf")

    ground_truth_check = None
    if true_effect is not None:
        ground_truth_check = {
            name: {
                "estimate": v.point_estimate,
                "true_effect": true_effect,
                "abs_error": abs(v.point_estimate - true_effect),
                "within_ci": (
                    v.ci_lower <= true_effect <= v.ci_upper
                    if v.ci_lower is not None and v.ci_upper is not None
                    else None
                ),
            }
            for name, v in results.items()
        }

    return {
        "results": results,
        "figs": figs,
        "trend_check": trend_check,
        "donor_pool": donor_pool,
        "disagreement": {"spread": spread, "relative_spread": rel_disagreement, "estimates": estimates},
        "ground_truth_check": ground_truth_check,
    }


def run_placebo_suite(
    df: pd.DataFrame,
    treated_unit: str,
    intervention_time: int,
    fake_intervention_time: int,
    threshold: float,
) -> dict[str, Any]:
    """Run in-time + in-space placebo tests for every estimator."""
    df = validate_panel(df)
    assert_fit_inputs(df, treated_unit, intervention_time)
    if fake_intervention_time >= intervention_time:
        raise ValueError(
            f"fake_intervention_time ({fake_intervention_time}) must be "
            f"< intervention_time ({intervention_time})."
        )

    donor_pool = select_donor_pool(df, treated_unit, intervention_time, top_k=15)
    suite: dict[str, Any] = {}
    for name, cls in ESTIMATORS.items():
        it = in_time_placebo(
            cls,
            df,
            treated_unit,
            intervention_time,
            fake_intervention_time,
            threshold,
            donor_units=donor_pool,
        )
        isp = in_space_placebo(
            cls, df, treated_unit, intervention_time, threshold, donor_units=donor_pool
        )
        suite[name] = {
            "in_time": it,
            "in_space": isp,
            "in_space_pass_rate": placebo_pass_rate(isp),
        }
    return suite


def render_html_report(
    comparison: dict,
    placebo: dict,
    out_path: str,
    title: str = "Causal Inference Toolkit Report",
) -> str:
    rows = []
    for name, eff in comparison["results"].items():
        pval = f"{eff.p_value:.4f}" if eff.p_value is not None else "n/a"
        rate = placebo[name]["in_space_pass_rate"]
        rate_s = f"{rate * 100:.1f}%" if rate == rate else "n/a"  # NaN check
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{eff.point_estimate:.3f}</td>"
            f"<td>{_fmt_ci(eff.ci_lower, eff.ci_upper)}</td>"
            f"<td>{pval}</td>"
            f"<td>{rate_s}</td>"
            "</tr>"
        )

    imgs_html = ""
    for name, ax in comparison["figs"].items():
        fig = ax.figure
        b64 = _fig_to_base64(fig)
        imgs_html += (
            f"<h3>{html.escape(name)}</h3>"
            f"<img src='data:image/png;base64,{b64}' style='max-width:700px;'/><br/>"
        )

    gt_html = ""
    if comparison.get("ground_truth_check"):
        gt_rows = "".join(
            "<tr>"
            f"<td>{html.escape(k)}</td>"
            f"<td>{v['estimate']:.3f}</td>"
            f"<td>{v['true_effect']:.3f}</td>"
            f"<td>{v['abs_error']:.3f}</td>"
            f"<td>{v['within_ci']}</td>"
            "</tr>"
            for k, v in comparison["ground_truth_check"].items()
        )
        gt_html = f"""
        <h2>Ground-Truth Validation</h2>
        <table border=1 cellpadding=6>
        <tr><th>Method</th><th>Estimate</th><th>True Effect</th><th>Abs Error</th><th>True effect within 95% CI?</th></tr>
        {gt_rows}
        </table>
        """

    disagreement = comparison["disagreement"]
    agree_msg = (
        "Methods are in reasonable agreement."
        if disagreement["relative_spread"] < 0.5
        else (
            "Methods disagree substantially — inspect assumptions "
            "(parallel trends for DiD, donor fit for synthetic control, "
            "model specification for BSTS) before trusting any single number."
        )
    )
    donor_list = ", ".join(html.escape(str(u)) for u in comparison["donor_pool"])

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>{html.escape(title)}</title></head>
<body style="font-family: sans-serif; max-width: 900px; margin: auto;">
<h1>{html.escape(title)}</h1>
<p>Generated: {datetime.now(timezone.utc).isoformat()}</p>

<h2>Pre-period Diagnostics</h2>
<p>{html.escape(comparison['trend_check'].message)}</p>
<p>Donor pool ({len(comparison['donor_pool'])} units): {donor_list}</p>

<h2>Cross-Method Comparison</h2>
<table border=1 cellpadding=6>
<tr><th>Method</th><th>Effect</th><th>95% CI</th><th>p-value</th><th>In-space placebo pass rate</th></tr>
{''.join(rows)}
</table>
<p><b>Disagreement:</b> spread={disagreement['spread']:.3f},
relative spread={disagreement['relative_spread']:.2%}. {agree_msg}</p>

{gt_html}

<h2>Estimator Plots</h2>
{imgs_html}
</body></html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    logger.info("Wrote HTML report to %s", out_path)
    return out_path
