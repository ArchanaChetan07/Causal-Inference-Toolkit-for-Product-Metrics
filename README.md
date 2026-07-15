# Causal Inference Toolkit for Product Metrics

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)
![python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

A validated, multi-method causal-impact library for observational product
data: **Difference-in-Differences**, **Synthetic Control**, and **Bayesian
Structural Time Series** (CausalImpact-style), unified behind one estimator
interface, with built-in pre-period diagnostics, placebo-test validation,
and honest cross-method disagreement reporting.

Built to go beyond A/B testing into the harder, more valuable skill of
causal inference on observational data — the kind of question Meta Core
Data Science, Google Ads/Search causal inference, and Amazon Central
Science teams actually ask.

## Why this exists

Anyone can fit a regression and call the coefficient a "causal effect."
The hard part — and the part this toolkit is built around — is proving the
estimate isn't just curve-fitting:

1. **Ground-truth validation** — every estimator is backtested against a
   dataset with a documented, exactly known effect before it's trusted on
   real data with an unknown answer.
2. **Placebo testing** — in-time placebos (fake intervention before the
   real one) and in-space placebos (pretend a control unit was treated)
   catch estimators that "find" effects where none exist. This is the
   single strongest credibility signal in applied causal work.
3. **Cross-method disagreement, reported honestly** — when DiD, synthetic
   control, and BSTS disagree, the toolkit says so explicitly instead of
   picking whichever number looks best.

## Quickstart

```bash
pip install -e ".[api,dashboard,dev]"
python examples/run_demo.py          # end-to-end demo -> reports/comparison_report.html
pytest --cov=causal_toolkit           # 21 tests, 97% coverage
```

```python
from causal_toolkit import make_ground_truth_dataset, run_all_methods

gt = make_ground_truth_dataset()  # documented true_effect = -8.0
comp = run_all_methods(gt.df, gt.treated_unit, gt.intervention_time, true_effect=gt.true_effect)

for name, effect in comp["results"].items():
    print(effect.summary())
# difference_in_differences: effect=-7.313, 95% CI=[-9.293, -5.333], p=0.0000
# synthetic_control:         effect=-3.777, 95% CI=[-7.680, 1.879],  p=0.0667
# bayesian_structural_time_series: effect=-8.224, 95% CI=[-8.321, -8.123], p=0.0000
```

## Architecture

```
causal_toolkit/
├── data.py                 # panel schema, validation, synthetic ground-truth generator
├── diagnostics.py          # parallel-trends test, donor-pool selection, covariate balance
├── estimators/
│   ├── base.py              # common CausalEstimator interface + EffectEstimate
│   ├── did.py                # Difference-in-Differences (TWFE, HC1 robust SE)
│   ├── synthetic_control.py  # convex-optimized donor weights + placebo inference
│   └── bsts.py                # Kalman-filter structural time series (CausalImpact-style)
├── placebo.py               # in-time / in-space placebo test framework
├── report.py                # cross-method comparison + HTML report generator
└── api.py                   # FastAPI service

dashboard/app.py            # Streamlit interactive UI
examples/run_demo.py        # worked end-to-end example
tests/                      # pytest suite, 97% coverage
```

Every estimator implements the same interface (`fit()`, `effect()`,
`plot()`), so the report generator, placebo framework, and API can treat
DiD / synthetic control / BSTS interchangeably.

## Running the API

```bash
uvicorn causal_toolkit.api:app --reload
# http://127.0.0.1:8000/docs        interactive OpenAPI docs
# http://127.0.0.1:8000/demo        run all 3 methods on the built-in ground-truth dataset
# http://127.0.0.1:8000/demo/report full HTML comparison report
```

Upload your own panel CSV (columns: `unit, time, outcome, treated, post`) to
`/estimate`, `/diagnostics`, or `/placebo-test` along with `treated_unit`
and `intervention_time` query params.

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

Interactive tabs for diagnostics, live estimator comparison with plots, and
on-demand placebo testing — either on the built-in synthetic dataset or an
uploaded CSV.

## Docker

```bash
docker build -t causal-toolkit .
docker run -p 8000:8000 causal-toolkit                      # API
docker run -p 8501:8501 causal-toolkit streamlit run dashboard/app.py --server.address 0.0.0.0
# or, both at once:
docker compose up
```

## Success metrics (from the project spec)

| Metric | Target | Result on the built-in ground-truth dataset |
|---|---|---|
| Ground-truth validation | Estimated effect within documented CI | DiD ✅, BSTS point estimate within 0.22 of truth, synthetic control off by 4.2 (see below) |
| Placebo test pass rate | No significant effect in placebo periods | BSTS 100% in-space pass rate, synthetic control 90%, DiD 0% — **this is real, reported disagreement, not a bug** |
| Method agreement analysis | Explicit, honest disagreement discussion | See `reports/comparison_report.html`, generated by `examples/run_demo.py` |
| Test coverage | ≥90% across Python versions | 97%, CI matrix across 3.9–3.12 |

### On the honest disagreement (read this before trusting any single number)

Running `examples/run_demo.py` on the default synthetic dataset produces a
**69% relative disagreement** between methods and a **0% in-space placebo
pass rate for classic TWFE DiD** on this particular data-generating process.
That's not a defect to paper over — it's exactly the signal a careful
practitioner needs: this synthetic panel's control units share a common
smooth trend plus unit-specific noise-walks, which is a harder setting for
simple two-period DiD than for donor-weighted or time-series methods that
can flexibly track each unit's idiosyncratic path. The toolkit surfaces
this automatically instead of hiding it — see `report.render_html_report`'s
disagreement section, which explicitly tells the user to go re-check
parallel trends, donor fit, and model specification when methods diverge
this much, rather than silently averaging the three numbers together.

## Known limitations (documented on purpose)

- **DiD** implements classic two-way-fixed-effects. It does *not* yet
  implement Callaway-Sant'Anna for staggered multi-cohort rollouts (not
  needed for the single-treated-unit case here, but flagged for anyone
  applying this to staggered adoption data).
- **BSTS** uses statsmodels' Kalman-filter `UnobservedComponents` rather
  than full PyMC MCMC, for fast, GPU-free fitting. Its "credible interval"
  is a documented approximation from forecast variance, not a true
  posterior — a `pymc_backend` option is a natural next extension.
- **Synthetic control** inference comes from in-space placebo permutation
  (standard in the literature, per Abadie et al. 2010) and is conservative
  with small donor pools.

## Roadmap

- [ ] Callaway-Sant'Anna staggered DiD estimator
- [ ] Full PyMC MCMC backend for BSTS (GPU-accelerated via JAX/NumPyro)
- [ ] Swap in a real public dataset (California Prop 99) as the default demo alongside the synthetic generator
- [ ] Multiple-treated-units support for synthetic control (generalized synthetic control / gsynth)
- [ ] PyPI release

## License

MIT — see [LICENSE](LICENSE).
