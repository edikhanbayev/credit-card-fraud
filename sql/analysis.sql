SELECT
    COUNT(*) AS total_transactions,
    SUM(actual_class) AS fraud_transactions,
    ROUND(
        100.0 * SUM(actual_class) / COUNT(*),
        4
    ) AS fraud_rate_percent
FROM fraud_scored_transactions;

SELECT
    predicted_fraud,
    COUNT(*) AS transactions,
    ROUND(
        AVG("Amount")::numeric,
        2
    ) AS avg_amount
FROM fraud_scored_transactions
GROUP BY predicted_fraud
ORDER BY predicted_fraud;

SELECT
    "Amount",
    actual_class,
    predicted_fraud,
    fraud_probability
FROM fraud_scored_transactions
ORDER BY fraud_probability DESC
LIMIT 20;

SELECT
    actual_class,
    predicted_fraud,
    COUNT(*) AS transactions
FROM fraud_scored_transactions
GROUP BY
    actual_class,
    predicted_fraud
ORDER BY
    actual_class,
    predicted_fraud;