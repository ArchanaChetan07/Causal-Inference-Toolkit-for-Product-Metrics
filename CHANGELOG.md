# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] - 2026-07-15
### Fixed
- DiD now re-derives treatment from `treated_unit` (placebos no longer return the real ATT)
- Diagnostics (`parallel_trends_test`, `covariate_balance`) honor `treated_unit`
- Donor pools exclude other treated units (prevents SC/BSTS contamination)
- Synthetic-control CI centered on the treated estimate; single-donor placebo path no longer crashes
- BSTS average-effect CIs no longer undercover via 1/√T independent-step shrink; donor count capped by T_pre
- API `/demo/report` works on Windows (tempfile + UTF-8); HTML report escapes content
- Placebo fake-time validation raises `ValueError` (HTTP 422) instead of assert/500

### Added
- Stricter panel validation: unique `(unit, time)`, constant `treated` within unit, row/upload caps
- `assert_fit_inputs`, structured API responses, `/ready` probe, request logging
- Upload size limit (25 MiB) and panel row cap (500k)
- Multi-stage non-root Docker image with HEALTHCHECK
- CI hard-fails on ruff/mypy; coverage gate raised to 90%; wheel smoke-install
- Streamlit result caching; dashboard no longer mutates `sys.path`

### Changed
- Version bumped to 0.2.0
- DiD docs clarify classic `treated×post` (not full TWFE)
- Removed unused `scikit-learn` / `plotly` dependencies

## [0.1.0] - 2026-07-14
### Added
- Panel data schema + validation (`causal_toolkit.data`)
- Synthetic ground-truth dataset generator with documented, exact causal effect
- Difference-in-Differences estimator (classic treated×post, HC1 robust SEs)
- Synthetic Control estimator (convex-optimized donor weights + in-space placebo inference)
- Bayesian Structural Time Series estimator (Kalman-filter local-level model, CausalImpact-style)
- Pre-period diagnostics: parallel-trends test, donor-pool selection, covariate balance
- Placebo-test framework: in-time and in-space placebos with pass-rate reporting
- Cross-method comparison + disagreement quantification
- HTML methods-comparison report generator
- FastAPI service (`/estimate`, `/diagnostics`, `/placebo-test`, `/demo`, `/demo/report`)
- Streamlit interactive dashboard
- Dockerfile + docker-compose for API and dashboard
- GitHub Actions CI matrix (Python 3.9-3.12)
- Initial test suite

### Known limitations (documented, not silent)
- DiD is classic treated×post; Callaway–Sant'Anna staggered DiD is roadmap.
- BSTS uses statsmodels Kalman-filter UnobservedComponents rather than full PyMC MCMC.
- Synthetic control p-values come from in-space placebo permutation (conservative with small pools).
