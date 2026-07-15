"""Worked example: generate a ground-truth dataset, run diagnostics, all
three estimators, the full placebo suite, and render the methods-comparison
report -- exactly what `Success Metrics` in the project spec asks for.

Run with:  python examples/run_demo.py
"""
from causal_toolkit import (
    make_ground_truth_dataset,
    parallel_trends_test,
    run_all_methods,
    run_placebo_suite,
    render_html_report,
)


def main():
    gt = make_ground_truth_dataset(n_control_units=20, n_periods=40, true_effect=-8.0, seed=42)
    print(gt.description)
    print()

    trend = parallel_trends_test(gt.df, gt.intervention_time)
    print("Pre-period diagnostic:", trend.message)
    print()

    comp = run_all_methods(gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect)
    print("=== Cross-method comparison ===")
    for name, eff in comp["results"].items():
        print(" ", eff.summary())
    print(f"Relative disagreement across methods: {comp['disagreement']['relative_spread']:.1%}")
    print()

    print("=== Ground-truth validation ===")
    for name, check in comp["ground_truth_check"].items():
        status = "PASS" if check["within_ci"] else "MISS"
        print(f"  {name}: est={check['estimate']:.2f} true={check['true_effect']:.2f} "
              f"abs_error={check['abs_error']:.2f} [{status}]")
    print()

    print("=== Placebo validation ===")
    placebo = run_placebo_suite(
        gt.df, gt.treated_unit, gt.intervention_time,
        fake_intervention_time=gt.intervention_time // 2, threshold=1.5,
    )
    for name, res in placebo.items():
        print(f"  {name}: in-time effect={res['in_time'].estimated_effect:.2f} "
              f"({'PASS' if res['in_time'].passed else 'FAIL'}), "
              f"in-space pass rate={res['in_space_pass_rate']:.0%}")

    out = render_html_report(comp, placebo, "reports/comparison_report.html")
    print(f"\nFull HTML report written to: {out}")


if __name__ == "__main__":
    main()
