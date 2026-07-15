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

import numpy as np
import pandas as pd
from dataclasses import dataclass


REQUIRED_COLUMNS = ["unit", "time", "outcome", "treated", "post"]


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

    if df["outcome"].isna().any():
        n = df["outcome"].isna().sum()
        raise SchemaError(f"Panel has {n} missing outcome values; impute or drop first.")

    if not set(df["treated"].unique()).issubset({0, 1}):
        raise SchemaError("`treated` column must be binary (0/1).")

    if not set(df["post"].unique()).issubset({0, 1}):
        raise SchemaError("`post` column must be binary (0/1).")

    n_units = df["unit"].nunique()
    n_treated = df.loc[df["treated"] == 1, "unit"].nunique()
    if n_treated == 0:
        raise SchemaError("No treated units found (treated==1 for at least one unit is required).")
    if n_treated == n_units:
        raise SchemaError("All units are treated; synthetic control / DiD require untreated donors.")

    return df.sort_values(["unit", "time"]).reset_index(drop=True)


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
