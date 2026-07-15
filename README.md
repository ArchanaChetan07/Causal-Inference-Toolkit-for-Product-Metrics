# Causal Inference Toolkit for Product Metrics

[![CI](https://github.com/ArchanaChetan07/Causal-Inference-Toolkit-for-Product-Metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/ArchanaChetan07/Causal-Inference-Toolkit-for-Product-Metrics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Version](https://img.shields.io/badge/version-0.2.0-blue)

**Estimate causal impact on observational product data — and prove the estimate is credible.**

A production-oriented Python library that unifies **Difference-in-Differences**, **Synthetic Control**, and **Bayesian Structural Time Series** (CausalImpact-style) behind one estimator API, with pre-period diagnostics, placebo validation, and honest cross-method disagreement reporting.

Built for the questions FAANG product / growth / ads / marketplace science teams actually ask when randomized experiments are unavailable, contaminated, or ethically constrained.

---

## The problem this solves

Shipping a regression coefficient labeled “causal effect” is easy. What Meta Core Data Science, Google Ads / Search causal teams, and Amazon Central Science optimize for is harder:

| Failure mode | How this toolkit responds |
|---|---|
| Curve-fitting masquerading as causality | Ground-truth backtests with a **documented known effect** before trusting unknown data |
| False positives on noisy panels | **In-time** and **in-space placebo tests** that should find ~0 when nothing happened |
| Cherry-picking the flattering method | Runs DiD + SC + BSTS together and **surfaces disagreement explicitly** |
| Silent assumption violations | Parallel-trends diagnostics, donor-pool selection, covariate balance |
| Notebook-only science | FastAPI service, Streamlit UI, Docker, CI matrix, ≥90% coverage gate |

---

## What you get

- **Three estimators, one interface** — `fit()` / `effect()` / `plot()` for DiD, Synthetic Control, and BSTS
- **Credibility layer** — parallel trends, donor ranking, in-time / in-space placebos, HTML comparison report
- **Production surface** — FastAPI (`/estimate`, `/diagnostics`, `/placebo-test`, `/demo`), Streamlit dashboard, multi-stage non-root Docker image
- **Engineering bar** — panel schema validation, upload size caps, structured 422 errors, `/health` + `/ready`, ruff + mypy + pytest CI on Python 3.9–3.12

---

## 60-second demo

```bash
pip install -e ".[api,dashboard,dev]"
python examples/run_demo.py          # writes reports/comparison_report.html
pytest --cov=causal_toolkit          # CI fails under 90% coverage
```

```python
from causal_toolkit import make_ground_truth_dataset, run_all_methods

gt = make_ground_truth_dataset()  # documented true_effect = -8.0
comp = run_all_methods(
    gt.df,
    treated_unit=gt.treated_unit,
    intervention_time=gt.intervention_time,
    true_effect=gt.true_effect,
)

for name, effect in comp["results"].items():
    print(effect.summary())

print(f"Relative disagreement: {comp['disagreement']['relative_spread']:.1%}")
```

Illustrative output on the built-in ground-truth panel (re-run locally for exact floats):

```text
difference_in_differences:           effect ≈ -7.3   (true effect in 95% CI)
synthetic_control:                   effect ≈ -3.8   (biased on this DGP — reported)
bayesian_structural_time_series:     effect ≈ -8.2   (closest to truth on this panel)
Relative disagreement across methods: ~69%
```

That disagreement is a **feature**: the toolkit refuses to silently average incompatible answers.

---

## Architecture

```text
causal_toolkit/
├── data.py                  # Panel schema, validation, synthetic ground-truth generator
├── diagnostics.py           # Parallel trends, donor pool, covariate balance
├── estimators/
│   ├── base.py              # CausalEstimator + EffectEstimate
│   ├── did.py               # Classic treated × post DiD (HC1 robust SE)
│   ├── synthetic_control.py # Abadie-style convex weights + placebo inference
│   └── bsts.py              # Kalman local-level + donor regression (CausalImpact-style)
├── placebo.py               # In-time / in-space placebo framework
├── report.py                # Cross-method comparison + HTML report
└── api.py                   # FastAPI service

dashboard/app.py             # Streamlit interactive UI
examples/run_demo.py         # End-to-end worked example
tests/                       # Pytest suite (CI-gated)
```

Every estimator implements the same contract so reporting, placebos, and the API are method-agnostic.

---

## Methods (and when to trust them)

| Method | Estimator | Inference | Best when | Honest limitation |
|---|---|---|---|---|
| **DiD** | Classic `treated × post` OLS | HC1 robust SE | Clear treated / control groups, parallel trends plausible | Not Callaway–Sant’Anna; not full unit+time FE TWFE |
| **Synthetic Control** | Nonnegative weights summing to 1 | In-space placebo permutation (Abadie) | One treated unit, strong pre-fit donors | Can bias under idiosyncratic noise-walks; conservative with small pools |
| **BSTS** | Local-level + donor covariates (Kalman) | Forecast-variance CI (documented approx.) | Rich pre-period + donor time series | Not full MCMC posterior (PyMC roadmap) |

---

## API & dashboard

```bash
# API
uvicorn causal_toolkit.api:app --reload
# Docs:   http://127.0.0.1:8000/docs
# Live:   /health  /ready  /demo  /demo/report
# Upload: POST /estimate | /diagnostics | /placebo-test
#         (CSV: unit, time, outcome, treated, post)

# Dashboard
streamlit run dashboard/app.py
```

Upload limits: **25 MiB** / **500k rows**. The HTTP API has **no auth** — bind to localhost or put it behind a reverse proxy for anything beyond local demos.

```bash
docker build -t causal-toolkit .
docker run --rm -p 8000:8000 causal-toolkit
docker compose up --build
```

---

## Design principles (portfolio signal)

1. **Validate before you believe** — synthetic panels with known ATT, then placebos that must fail to “find” effects
2. **Disagree in public** — cross-method spread is quantified and written into the HTML report
3. **Name limitations** — approximate CIs, classic DiD scope, and SC bias modes are documented, not papered over
4. **Ship like an internal library** — schema validation, typed package (`py.typed`), CI gates, non-root containers, readiness probes

---

## Production checklist (v0.2)

| Area | Status |
|---|---|
| Panel validation (schema, duplicates, treated constancy, size caps) | Done |
| Diagnostics honor `treated_unit` | Done |
| Donor pools exclude other treated units | Done |
| Estimator / API error mapping (422 vs 500) | Done |
| Upload size limits + logging | Done |
| CI: ruff + mypy + pytest (3.9–3.12), coverage ≥90% | Done |
| Multi-stage non-root Docker + HEALTHCHECK | Done |
| Liveness `/health` + readiness `/ready` | Done |

---

## Project structure for reviewers

| Path | Why it matters in an interview |
|---|---|
| `estimators/` | Shared abstraction + three real methods, not three scripts |
| `placebo.py` | Credibility engineering, not just point estimates |
| `report.py` | Forces honest multi-method comparison |
| `api.py` + `Dockerfile` | Research code that can run as a service |
| `tests/` + `.github/workflows/ci.yml` | Regression safety and reproducible quality bar |

---

## Roadmap

- [ ] Callaway–Sant’Anna staggered DiD
- [ ] Full PyMC / NumPyro MCMC backend for BSTS
- [ ] California Prop 99 (or similar) public demo panel alongside the synthetic generator
- [ ] Generalized synthetic control for multiple treated units
- [ ] PyPI release

---

## License

MIT — see [LICENSE](LICENSE).

---

### Author

Built as a portfolio demonstration of **applied causal inference + production Python** for product-metrics and experiment-science roles.
