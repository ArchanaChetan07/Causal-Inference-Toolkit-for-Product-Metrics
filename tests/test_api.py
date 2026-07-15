import io

import pandas as pd
from fastapi.testclient import TestClient

from causal_toolkit.api import app
from causal_toolkit.data import make_ground_truth_dataset

client = TestClient(app)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_demo_endpoint():
    r = client.get("/demo")
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) == 3
    assert body["true_effect"] == -8.0


def test_demo_report_endpoint():
    r = client.get("/demo/report")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert b"Cross-Method Comparison" in r.content


def test_estimate_endpoint_with_valid_csv():
    gt = make_ground_truth_dataset(n_control_units=8, seed=11)
    files = {"file": ("panel.csv", _csv_bytes(gt.df), "text/csv")}
    r = client.post(
        "/estimate",
        files=files,
        params={"treated_unit": gt.treated_unit, "intervention_time": gt.intervention_time},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3


def test_diagnostics_endpoint():
    gt = make_ground_truth_dataset(n_control_units=8, seed=12)
    files = {"file": ("panel.csv", _csv_bytes(gt.df), "text/csv")}
    r = client.post(
        "/diagnostics",
        files=files,
        params={"treated_unit": gt.treated_unit, "intervention_time": gt.intervention_time},
    )
    assert r.status_code == 200
    assert "parallel_trends" in r.json()
    assert "donor_pool" in r.json()


def test_placebo_endpoint():
    gt = make_ground_truth_dataset(n_control_units=8, seed=13)
    files = {"file": ("panel.csv", _csv_bytes(gt.df), "text/csv")}
    r = client.post(
        "/placebo-test",
        files=files,
        params={
            "treated_unit": gt.treated_unit,
            "intervention_time": gt.intervention_time,
            "fake_intervention_time": 10,
            "threshold": 2.0,
        },
    )
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_estimate_rejects_bad_csv():
    bad_csv = b"unit,time,outcome\na,0,1.0\n"
    files = {"file": ("bad.csv", bad_csv, "text/csv")}
    r = client.post("/estimate", files=files, params={"treated_unit": "a", "intervention_time": 1})
    assert r.status_code == 422
