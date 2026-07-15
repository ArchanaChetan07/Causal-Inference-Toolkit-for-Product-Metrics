"""Streamlit dashboard: upload a panel CSV (or use the built-in
ground-truth demo dataset), run diagnostics, all three estimators,
and the placebo-test suite, with live plots.
"""
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

from causal_toolkit.data import make_ground_truth_dataset, validate_panel, SchemaError
from causal_toolkit.diagnostics import parallel_trends_test, select_donor_pool, covariate_balance
from causal_toolkit.report import run_all_methods, run_placebo_suite

st.set_page_config(page_title="Causal Inference Toolkit", layout="wide")
st.title("📊 Causal Inference Toolkit for Product Metrics")
st.caption("Difference-in-Differences · Synthetic Control · Bayesian Structural Time Series")

with st.sidebar:
    st.header("Data")
    source = st.radio("Data source", ["Built-in ground-truth demo", "Upload CSV"])

    df, treated_unit, intervention_time, true_effect = None, None, None, None

    if source == "Built-in ground-truth demo":
        n_controls = st.slider("Number of control units", 5, 40, 20)
        n_periods = st.slider("Number of periods", 20, 80, 40)
        eff = st.slider("Documented true effect", -20.0, 0.0, -8.0)
        gt = make_ground_truth_dataset(n_control_units=n_controls, n_periods=n_periods, true_effect=eff)
        df, treated_unit, intervention_time, true_effect = gt.df, gt.treated_unit, gt.intervention_time, gt.true_effect
        st.info(gt.description)
    else:
        upload = st.file_uploader("Panel CSV (unit, time, outcome, treated, post)", type="csv")
        if upload is not None:
            raw = pd.read_csv(upload)
            try:
                df = validate_panel(raw)
                treated_unit = st.selectbox("Treated unit", sorted(df.loc[df.treated == 1, "unit"].unique()))
                intervention_time = st.number_input(
                    "Intervention time", value=int(df.loc[df.post == 1, "time"].min())
                )
            except SchemaError as e:
                st.error(str(e))

if df is not None and treated_unit is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["Diagnostics", "Estimators", "Placebo Tests", "Raw Data"])

    with tab1:
        st.subheader("Pre-period diagnostics")
        trend = parallel_trends_test(df, intervention_time)
        (st.success if trend.passed else st.error)(trend.message)
        donors = select_donor_pool(df, treated_unit, intervention_time, top_k=10)
        st.write("Top donor pool candidates:", donors)
        st.dataframe(covariate_balance(df, intervention_time))

    with tab2:
        st.subheader("Cross-method comparison")
        comp = run_all_methods(df, treated_unit, intervention_time, true_effect=true_effect)
        cols = st.columns(3)
        for col, (name, eff) in zip(cols, comp["results"].items()):
            with col:
                st.metric(name.replace("_", " ").title(), f"{eff.point_estimate:.2f}",
                          help=f"95% CI [{eff.ci_lower:.2f}, {eff.ci_upper:.2f}], p={eff.p_value:.4f}")
        if comp["ground_truth_check"]:
            st.write("Ground-truth validation:")
            st.dataframe(pd.DataFrame(comp["ground_truth_check"]).T)
        st.write(
            f"Relative disagreement across methods: **{comp['disagreement']['relative_spread']:.1%}**"
        )
        for name, ax in comp["figs"].items():
            st.pyplot(ax.figure)

    with tab3:
        st.subheader("Placebo validation")
        fake_time = st.slider("Fake intervention time (pre-period)", 1, int(intervention_time) - 2, max(1, int(intervention_time) // 2))
        threshold = st.number_input("Pass threshold (abs effect)", value=1.5)
        if st.button("Run placebo suite"):
            with st.spinner("Running in-time and in-space placebo tests..."):
                suite = run_placebo_suite(df, treated_unit, intervention_time, fake_time, threshold)
            for name, res in suite.items():
                st.write(f"**{name}**")
                st.write(f"- In-time placebo effect: {res['in_time'].estimated_effect:.3f} "
                         f"({'PASS' if res['in_time'].passed else 'FAIL'})")
                st.write(f"- In-space placebo pass rate: {res['in_space_pass_rate']:.1%}")

    with tab4:
        st.dataframe(df)
else:
    st.info("Choose the demo dataset or upload a panel CSV in the sidebar to begin.")
