from pathlib import Path
from src.db import (
    log_prediction
)
import joblib
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException
)

from pydantic import BaseModel

from src.features import (
    build_features
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "fraud_model.joblib"
)


artifact = joblib.load(
    MODEL_PATH
)


model = artifact["model"]

threshold = artifact["threshold"]

raw_columns = artifact[
    "raw_columns"
]


app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0"
)
class TransactionRequest(
    BaseModel
):
    features: dict[str, float]

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model":
            artifact["model_name"],
        "version":
            artifact["version"]
    }
@app.post("/predict")
def predict(
    request: TransactionRequest
):

    missing = [
        column
        for column in raw_columns
        if column
        not in request.features
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features":
                    missing
            }
        )

    row = pd.DataFrame(
        [
            {
                column:
                    request.features[
                        column
                    ]
                for column
                in raw_columns
            }
        ]
    )

    row_fe = build_features(
        row
    )

    fraud_score = float(
        model.predict_proba(
            row_fe
        )[0, 1]
    )

    predicted_fraud = int(
        fraud_score >= threshold
    )
    log_prediction(
        raw_features=
        row.iloc[0].to_dict(),

        fraud_score=
        fraud_score,

        threshold=
        threshold,

        predicted_fraud=
        predicted_fraud,

        model_version=
        artifact["version"]
    )
    return {
        "fraud_score":
            fraud_score,

        "threshold":
            threshold,

        "predicted_fraud":
            predicted_fraud,

        "model":
            artifact[
                "model_name"
            ],

        "model_version":
            artifact[
                "version"
            ]
    }