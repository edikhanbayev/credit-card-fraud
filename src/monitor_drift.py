import numpy as np
import pandas as pd
import json
from pathlib import Path
from sqlalchemy import text
from src.db import engine
from src.features import build_features

def calculate_psi(
    reference,
    current,
    bins=10
):

    reference = np.asarray(
        reference
    )

    current = np.asarray(
        current
    )

    quantiles = np.linspace(
        0,
        1,
        bins + 1
    )

    edges = np.unique(
        np.quantile(
            reference,
            quantiles
        )
    )

    if len(edges) < 3:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_hist, _ = (
        np.histogram(
            reference,
            bins=edges
        )
    )

    current_hist, _ = (
        np.histogram(
            current,
            bins=edges
        )
    )

    reference_pct = (
        reference_hist
        / reference_hist.sum()
    )

    current_pct = (
        current_hist
        / current_hist.sum()
    )

    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None
    )

    psi = np.sum(
        (
            current_pct
            -
            reference_pct
        )
        *
        np.log(
            current_pct
            /
            reference_pct
        )
    )

    return float(psi)

def interpret_psi(
    psi: float
) -> str:

    if np.isnan(psi):
        return "insufficient_data"

    if psi < 0.10:
        return "low"

    if psi < 0.25:
        return "moderate"

    return "high"

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


REFERENCE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "reference_sample.csv"
)


reference = pd.read_csv(
    REFERENCE_PATH
)
query = text(
    """
    SELECT
        features,
        fraud_score,
        predicted_fraud
    FROM prediction_log
    WHERE created_at >=
          NOW() - INTERVAL '7 days'
    """
)
with engine.connect() as connection:

    production = pd.read_sql(
        query,
        connection
    )
if production.empty:

    print(
        "No recent production data."
    )

    raise SystemExit(0)
production_features = (
    pd.json_normalize(
        production[
            "features"
        ]
    )
)
production_fe = build_features(
    production_features
)
drift_results = []

for column in reference.columns:

    if column not in production_fe.columns:
        continue

    psi = calculate_psi(
        reference[column],
        production_fe[column]
    )

    drift_results.append(
        {
            "feature": column,
            "psi": psi
        }
    )

drift_df = (
    pd.DataFrame(
        drift_results
    )
    .sort_values(
        "psi",
        ascending=False
    )
)
print(drift_df)

drift_df["drift_level"] = drift_df[
    "psi"
].apply(
    interpret_psi
)
print(
    drift_df.head(20)
)
print(
    "\nFraud score:"
)

print(
    production[
        "fraud_score"
    ].describe()
)
predicted_fraud_rate = (
    production[
        "predicted_fraud"
    ].mean()
)

print(
    "\nPredicted fraud rate:",
    predicted_fraud_rate
)