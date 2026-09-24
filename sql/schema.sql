CREATE TABLE IF NOT EXISTS prediction_log
(
    id BIGSERIAL PRIMARY KEY,

    created_at TIMESTAMPTZ
        NOT NULL
        DEFAULT NOW(),

    features JSONB
        NOT NULL,

    amount NUMERIC,

    fraud_score DOUBLE PRECISION
        NOT NULL,

    threshold DOUBLE PRECISION
        NOT NULL,

    predicted_fraud BOOLEAN
        NOT NULL,

    model_version VARCHAR(50)
        NOT NULL,

    actual_class SMALLINT
        NULL
);