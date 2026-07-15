"""FastAPI service exposing the causal toolkit over HTTP.

Endpoints:
    POST /estimate           - run all 3 estimators on uploaded panel data
    POST /diagnostics        - pre-period parallel-trends + donor pool
    POST /placebo-test       - run the placebo suite
    GET  /demo               - run everything on the built-in ground-truth dataset
    GET  /health             - liveness probe
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .data import validate_panel, make_ground_truth_dataset, SchemaError
from .diagnostics import parallel_trends_test, select_donor_pool
from .report import run_all_methods, run_placebo_suite, render_html_report

app = FastAPI(
    title="Causal Inference Toolkit API",
    description="Difference-in-differences, synthetic control, and Bayesian "
    "structural time series over observational panel data, with placebo "
    "validation and cross-method disagreement reporting.",
    version="0.1.0",
)


class EstimateResponse(BaseModel):
    method: str
    point_estimate: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    p_value: Optional[float]


def _read_upload(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    try:
        return validate_panel(df)
    except SchemaError as e:
        raise HTTPException(422, str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demo")
def demo():
    """Run the full pipeline on the built-in synthetic ground-truth dataset."""
    gt = make_ground_truth_dataset()
    comp = run_all_methods(gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect)
    return {
        "description": gt.description,
        "true_effect": gt.true_effect,
        "results": {k: v.summary() for k, v in comp["results"].items()},
        "ground_truth_check": comp["ground_truth_check"],
        "trend_check": comp["trend_check"].message,
    }


@app.post("/estimate", response_model=dict)
def estimate(
    file: UploadFile = File(...),
    treated_unit: str = Query(...),
    intervention_time: int = Query(...),
):
    df = _read_upload(file)
    comp = run_all_methods(df, treated_unit, intervention_time)
    return {k: v.__dict__ for k, v in comp["results"].items()}


@app.post("/diagnostics")
def diagnostics(
    file: UploadFile = File(...),
    treated_unit: str = Query(...),
    intervention_time: int = Query(...),
):
    df = _read_upload(file)
    trend = parallel_trends_test(df, intervention_time)
    donors = select_donor_pool(df, treated_unit, intervention_time)
    return {"parallel_trends": trend.__dict__, "donor_pool": donors}


@app.post("/placebo-test")
def placebo_test(
    file: UploadFile = File(...),
    treated_unit: str = Query(...),
    intervention_time: int = Query(...),
    fake_intervention_time: int = Query(...),
    threshold: float = Query(1.0),
):
    df = _read_upload(file)
    suite = run_placebo_suite(df, treated_unit, intervention_time, fake_intervention_time, threshold)
    return {
        name: {
            "in_time_effect": v["in_time"].estimated_effect,
            "in_time_passed": v["in_time"].passed,
            "in_space_pass_rate": v["in_space_pass_rate"],
        }
        for name, v in suite.items()
    }


@app.get("/demo/report", response_class=HTMLResponse)
def demo_report():
    """Render the full HTML comparison report for the demo dataset."""
    gt = make_ground_truth_dataset()
    comp = run_all_methods(gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect)
    placebo = run_placebo_suite(gt.df, gt.treated_unit, gt.intervention_time, fake_intervention_time=12, threshold=1.5)
    fd, path = tempfile.mkstemp(suffix=".html", prefix="causal_demo_report_")
    os.close(fd)
    try:
        render_html_report(comp, placebo, path)
        with open(path, encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
