"""FastAPI service exposing the causal toolkit over HTTP.

Endpoints:
    POST /estimate           - run all 3 estimators on uploaded panel data
    POST /diagnostics        - pre-period parallel-trends + donor pool
    POST /placebo-test       - run the placebo suite
    GET  /demo               - run everything on the built-in ground-truth dataset
    GET  /demo/report        - HTML comparison report for the demo dataset
    GET  /health             - liveness probe
    GET  /ready              - readiness probe (deps importable)
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .data import MAX_UPLOAD_BYTES, SchemaError, make_ground_truth_dataset, validate_panel
from .diagnostics import covariate_balance, parallel_trends_test, select_donor_pool
from .report import render_html_report, run_all_methods, run_placebo_suite

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Causal Inference Toolkit API",
    description=(
        "Difference-in-differences, synthetic control, and Bayesian "
        "structural time series over observational panel data, with placebo "
        "validation and cross-method disagreement reporting."
    ),
    version=__version__,
)


class EffectOut(BaseModel):
    method: str
    point_estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EstimateResponse(BaseModel):
    results: dict[str, EffectOut]
    disagreement: dict[str, Any]
    trend_check: str
    donor_pool: list[str]


def _read_upload(file: UploadFile) -> pd.DataFrame:
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"Upload exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.",
        )
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse CSV: {e}") from e
    try:
        return validate_panel(df)
    except SchemaError as e:
        raise HTTPException(422, str(e)) from e


def _effect_out(eff) -> EffectOut:
    return EffectOut(
        method=eff.method,
        point_estimate=eff.point_estimate,
        ci_lower=eff.ci_lower,
        ci_upper=eff.ci_upper,
        p_value=eff.p_value,
        diagnostics=eff.diagnostics or {},
    )


def _map_client_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (SchemaError, ValueError)):
        return HTTPException(422, str(exc))
    logger.exception("Unhandled server error")
    return HTTPException(500, "Internal server error while running causal analysis.")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready():
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import statsmodels  # noqa: F401
    except ImportError as e:
        raise HTTPException(503, f"Dependency unavailable: {e}") from e
    return {"status": "ready", "version": __version__}


@app.get("/demo")
def demo():
    """Run the full pipeline on the built-in synthetic ground-truth dataset."""
    gt = make_ground_truth_dataset()
    try:
        comp = run_all_methods(
            gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect
        )
    except Exception as e:  # noqa: BLE001
        raise _map_client_error(e) from e
    return {
        "description": gt.description,
        "true_effect": gt.true_effect,
        "results": {k: v.summary() for k, v in comp["results"].items()},
        "ground_truth_check": comp["ground_truth_check"],
        "trend_check": comp["trend_check"].message,
    }


@app.post("/estimate", response_model=EstimateResponse)
def estimate(
    file: UploadFile = File(...),
    treated_unit: str = Query(...),
    intervention_time: int = Query(...),
):
    df = _read_upload(file)
    try:
        comp = run_all_methods(df, treated_unit, intervention_time)
    except Exception as e:  # noqa: BLE001
        raise _map_client_error(e) from e
    return EstimateResponse(
        results={k: _effect_out(v) for k, v in comp["results"].items()},
        disagreement=comp["disagreement"],
        trend_check=comp["trend_check"].message,
        donor_pool=comp["donor_pool"],
    )


@app.post("/diagnostics")
def diagnostics(
    file: UploadFile = File(...),
    treated_unit: str = Query(...),
    intervention_time: int = Query(...),
):
    df = _read_upload(file)
    try:
        trend = parallel_trends_test(df, intervention_time, treated_unit=treated_unit)
        donors = select_donor_pool(df, treated_unit, intervention_time)
        balance = covariate_balance(df, intervention_time, treated_unit=treated_unit)
    except Exception as e:  # noqa: BLE001
        raise _map_client_error(e) from e
    return {
        "parallel_trends": trend.__dict__,
        "donor_pool": donors,
        "covariate_balance": balance.reset_index().to_dict(orient="records"),
    }


@app.post("/placebo-test")
def placebo_test(
    file: UploadFile = File(...),
    treated_unit: str = Query(...),
    intervention_time: int = Query(...),
    fake_intervention_time: int = Query(...),
    threshold: float = Query(1.0, gt=0),
):
    df = _read_upload(file)
    try:
        suite = run_placebo_suite(
            df, treated_unit, intervention_time, fake_intervention_time, threshold
        )
    except Exception as e:  # noqa: BLE001
        raise _map_client_error(e) from e
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
    try:
        comp = run_all_methods(
            gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect
        )
        placebo = run_placebo_suite(
            gt.df,
            gt.treated_unit,
            gt.intervention_time,
            fake_intervention_time=12,
            threshold=1.5,
        )
    except Exception as e:  # noqa: BLE001
        raise _map_client_error(e) from e

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
