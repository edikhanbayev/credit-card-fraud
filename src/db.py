import json
import os

from dotenv import (
    load_dotenv
)

from sqlalchemy import (
    create_engine,
    text
)


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured"
    )


engine = create_engine(
    DATABASE_URL
)
def log_prediction(
    raw_features: dict,
    fraud_score: float,
    threshold: float,
    predicted_fraud: int,
    model_version: str
):

    query = text(
        """
        INSERT INTO prediction_log
        (
            features,
            amount,
            fraud_score,
            threshold,
            predicted_fraud,
            model_version
        )
        VALUES
        (
            CAST(:features AS jsonb),
            :amount,
            :fraud_score,
            :threshold,
            :predicted_fraud,
            :model_version
        )
        """
    )

    parameters = {
        "features":
            json.dumps(
                raw_features
            ),

        "amount":
            raw_features.get(
                "Amount"
            ),

        "fraud_score":
            fraud_score,

        "threshold":
            threshold,

        "predicted_fraud":
            bool(
                predicted_fraud
            ),

        "model_version":
            model_version
    }

    with engine.begin() as connection:

        connection.execute(
            query,
            parameters
        )