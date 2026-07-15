"""
Data layer: panel schema, validation, and a documented synthetic
ground-truth dataset generator (stand-in for a real public dataset such
as California Prop 99 tobacco tax or a marketing geo-experiment).

Panel schema (long format), required columns:
    unit        : str  - entity identifier (state, region, store, etc.)
    time        : int  - time period index (or datetime)
    outcome     : float- outcome metric (sales, deaths, revenue, etc.)
    treated     : int  - 1 if this unit is ever treated, else 0
    post        : int  - 1 if this time period is post-intervention
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["unit", "time", "outcome", "treated", "post"]

# Soft caps for API / production workloads (rows = units × periods)
MAX_PANEL_ROWS = 500_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB


class SchemaError(ValueError):
    pass


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a panel dataframe against the toolkit's required schema.

    Raises SchemaError with an actionable message on failure.
    Returns the dataframe (sorted) if valid.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Panel is missing required columns: {missing}")

    if len(df) == 0:
        raise SchemaError("Panel is empty.")

    if len(df) > MAX_PANEL_ROWS:
        raise SchemaError(
            f"Panel has {len(df)} rows; maximum allowed is {MAX_PANEL_ROWS}."
        )

    if df["outcome"].isna().any():
        n = int(df["outcome"].isna().sum())
        raise SchemaError(f"Panel has {n} missing outcome values; impute or drop first.")

    if not pd.api.types.is_numeric_dtype(df["outcome"]):
        raise SchemaError("`outcome` must be numeric.")

    if not pd.api.types.is_numeric_dtype(df["time"]):
        raise SchemaError("`time` must be numeric.")

    treated_vals = set(pd.Series(df["treated"]).dropna().unique().tolist())
    if not treated_vals.issubset({0, 1}):
        raise SchemaError("`treated` column must be binary (0/1).")

    post_vals = set(pd.Series(df["post"]).dropna().unique().tolist())
    if not post_vals.issubset({0, 1}):
        raise SchemaError("`post` column must be binary (0/1).")

    if df.duplicated(subset=["unit", "time"]).any():
        n = int(df.duplicated(subset=["unit", "time"]).sum())
        raise SchemaError(f"Panel has {n} duplicate (unit, time) rows.")

    # treated flag must be constant within each unit
    treated_nunique = df.groupby("unit")["treated"].nunique()
    bad_units = treated_nunique[treated_nunique > 1].index.tolist()
    if bad_units:
        raise SchemaError(
            f"`treated` must be constant within each unit; inconsistent units: {bad_units[:5]}"
        )

    n_units = df["unit"].nunique()
    n_treated = df.loc[df["treated"] == 1, "unit"].nunique()
    if n_treated == 0:
        raise SchemaError("No treated units found (treated==1 for at least one unit is required).")
    if n_treated == n_units:
        raise SchemaError("All units are treated; synthetic control / DiD require untreated donors.")

    return df.sort_values(["unit", "time"]).reset_index(drop=True)


def assert_fit_inputs(
    df: pd.DataFrame,
    treated_unit: str,
    intervention_time: int | float,
) -> None:
    """Validate estimator fit arguments against an already-validated panel."""
    if treated_unit not in set(df["unit"].unique()):
        raise SchemaError(f"Treated unit '{treated_unit}' not found in panel.")
    times = df["time"]
    t_min, t_max = times.min(), times.max()
    if not (t_min <= intervention_time <= t_max):
        raise SchemaError(
            f"intervention_time={intervention_time} is outside panel time range [{t_min}, {t_max}]."
        )
    n_pre = int((df["time"] < intervention_time).sum())
    n_post = int((df["time"] >= intervention_time).sum())
    if n_pre == 0:
        raise SchemaError("No pre-intervention observations; cannot fit counterfactual.")
    if n_post == 0:
        raise SchemaError("No post-intervention observations; cannot estimate an effect.")


@dataclass
class GroundTruthDataset:
    """A synthetic panel with a documented, known causal effect.

    This mirrors the structure of real quasi-experiments (e.g. Prop 99)
    so the toolkit's estimators can be backtested against a KNOWN answer
    before being trusted on real, unknown-effect data.
    """

    df: pd.DataFrame
    true_effect: float
    treated_unit: str
    intervention_time: int
    description: str


def make_ground_truth_dataset(
    n_control_units: int = 20,
    n_periods: int = 40,
    intervention_time: int = 25,
    true_effect: float = -8.0,
    noise_sd: float = 1.5,
    seed: int = 42,
) -> GroundTruthDataset:
    """Generate a panel with one treated unit and N controls, where the
    treated unit receives a documented, exact additive effect after
    `intervention_time`. Used for placebo tests and ground-truth
    validation (Success Metric #1 in the project spec).
    """
    rng = np.random.default_rng(seed)
    units = [f"control_{i}" for i in range(n_control_units)] + ["treated_unit"]
    time = np.arange(n_periods)

    # Shared latent trend + unit-specific fixed effects + AR-ish noise
    common_trend = np.linspace(0, 10, n_periods)
    rows = []
    for unit in units:
        fixed_effect = rng.normal(loc=50, scale=10)
        unit_noise = rng.normal(0, noise_sd, size=n_periods)
        # small idiosyncratic slope so donors aren't perfectly parallel
        idio_slope = rng.normal(0, 0.05)
        outcome = fixed_effect + common_trend + idio_slope * time + unit_noise.cumsum() * 0.1
        treated = 1 if unit == "treated_unit" else 0
        for t, y in zip(time, outcome):
            post = int(t >= intervention_time)
            y_final = y + (true_effect if (treated and post) else 0.0)
            rows.append(
                dict(unit=unit, time=int(t), outcome=float(y_final), treated=treated, post=post)
            )

    df = pd.DataFrame(rows)
    return GroundTruthDataset(
        df=df,
        true_effect=true_effect,
        treated_unit="treated_unit",
        intervention_time=intervention_time,
        description=(
            f"Synthetic panel: {n_control_units} control units + 1 treated unit, "
            f"{n_periods} periods, documented additive effect={true_effect} "
            f"starting at t={intervention_time}. Stand-in for a real dataset "
            f"(e.g. Prop 99) with an exactly known ground truth, used to "
            f"validate estimator honesty before trusting real data."
        ),
    )


def load_csv(path: str) -> pd.DataFrame:
    """Load and validate a user-provided panel CSV."""
    df = pd.read_csv(path)
    return validate_panel(df)
