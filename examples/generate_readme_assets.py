"""Regenerate README figures under assets/ from the ground-truth demo panel.

Run:  python examples/generate_readme_assets.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from causal_toolkit import make_ground_truth_dataset, parallel_trends_test, run_all_methods, run_placebo_suite
from causal_toolkit.diagnostics import select_donor_pool
from causal_toolkit.estimators import BSTSEstimator, DiDEstimator, SyntheticControlEstimator


def main() -> None:
    out = Path("assets")
    out.mkdir(exist_ok=True)

    gt = make_ground_truth_dataset(n_control_units=20, n_periods=40, true_effect=-8.0, seed=42)
    comp = run_all_methods(gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect)
    placebo = run_placebo_suite(
        gt.df, gt.treated_unit, gt.intervention_time, fake_intervention_time=12, threshold=1.5
    )
    trend = parallel_trends_test(gt.df, gt.intervention_time, treated_unit=gt.treated_unit)

    keys = list(comp["results"].keys())
    names = ["DiD", "Synthetic\nControl", "BSTS"]
    colors = ["#1f4e79", "#2e7d4f", "#8a4b08"]
    ests = [comp["results"][k].point_estimate for k in keys]
    los = [comp["results"][k].ci_lower for k in keys]
    his = [comp["results"][k].ci_upper for k in keys]
    yerr = np.array([[ests[i] - los[i], his[i] - ests[i]] for i in range(3)]).T

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, ests, color=colors, width=0.55)
    ax.errorbar(names, ests, yerr=yerr, fmt="none", ecolor="#333333", capsize=6, linewidth=1.4)
    ax.axhline(-8.0, color="#c0392b", linestyle="--", linewidth=1.6, label="True effect = -8.0")
    ax.set_ylabel("Estimated ATT")
    ax.set_title("Cross-method causal estimates vs ground truth")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "effect_comparison.png", dpi=160)
    plt.close(fig)

    rates = [placebo[k]["in_space_pass_rate"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(names, rates, color=colors, width=0.55)
    ax.set_ylabel("In-space placebo pass rate (%)")
    ax.set_title("Placebo credibility: share of control units with |effect| < 1.5")
    ax.set_ylim(0, 110)
    for i, r in enumerate(rates):
        ax.text(i, r + 2, f"{r:.0f}%", ha="center", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "placebo_pass_rates.png", dpi=160)
    plt.close(fig)

    errs = [comp["ground_truth_check"][k]["abs_error"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(names, errs, color=colors, width=0.55)
    ax.set_ylabel("|estimate − true effect|")
    ax.set_title("Ground-truth absolute error (true ATT = -8.0)")
    for i, e in enumerate(errs):
        ax.text(i, e + 0.08, f"{e:.2f}", ha="center", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "ground_truth_error.png", dpi=160)
    plt.close(fig)

    donors = select_donor_pool(gt.df, gt.treated_unit, gt.intervention_time, top_k=15)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, (title, cls) in zip(
        axes,
        [
            ("DiD", DiDEstimator),
            ("Synthetic Control", SyntheticControlEstimator),
            ("BSTS", BSTSEstimator),
        ],
    ):
        est = cls().fit(gt.df, gt.treated_unit, gt.intervention_time, donor_units=donors)
        est.plot(ax=ax)
        ax.set_title(title, fontsize=11)
        ax.axvline(gt.intervention_time, color="#c0392b", linestyle=":", linewidth=1.2, alpha=0.9)
    fig.suptitle(
        "Actual vs counterfactual / control trajectories (intervention at t=25)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Pre-trend: {trend.message}")
    print(f"Disagreement: {comp['disagreement']['relative_spread']:.1%}")
    print(f"Wrote figures to {out.resolve()}")


if __name__ == "__main__":
    main()
