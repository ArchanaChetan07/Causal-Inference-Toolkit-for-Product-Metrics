# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-07-14
### Added
- Panel data schema + validation (`causal_toolkit.data`)
- Synthetic ground-truth dataset generator with documented, exact causal effect
- Difference-in-Differences estimator (TWFE, HC1 robust SEs)
- Synthetic Control estimator (convex-optimized donor weights + in-space placebo inference)
- Bayesian Structural Time Series estimator (Kalman-filter local-level model, CausalImpact-style)
- Pre-period diagnostics: parallel-trends test, donor-pool selection, covariate balance
- Placebo-test framework: in-time and in-space placebos with pass-rate reporting
- Cross-method comparison + disagreement quantification
- HTML methods-comparison report generator
- FastAPI service (`/estimate`, `/diagnostics`, `/placebo-test`, `/demo`, `/demo/report`)
- Streamlit interactive dashboard
- Dockerfile + docker-compose for API and dashboard
- GitHub Actions CI: lint, type-check, test matrix (Python 3.9-3.12), coverage gate, package build, Docker build
- Test suite, 97% coverage

### Known limitations (documented, not silent)
- DiD implementation is classic TWFE; does not yet implement Callaway-Sant'Anna
  for staggered multi-cohort treatment timing (single-treated-unit case here is unaffected).
- BSTS estimator uses statsmodels' Kalman-filter UnobservedComponents rather than
  full PyMC MCMC; credible intervals are a documented approximation via forecast-variance
  simulation, not a true posterior.
- Synthetic control p-values come from in-space placebo permutation, standard in the
  literature but conservative with small donor pools (<10 units).
