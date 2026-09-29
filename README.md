# Credit Card Fraud Detection

An end-to-end educational project for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The project covers the full workflow: data analysis → model training and comparison → threshold selection → model explainability → experiment tracking and artifact persistence → API → PostgreSQL → Docker → basic drift monitoring.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## What Was Implemented

- EDA: missing values, duplicates, class distribution, and transaction amount analysis.
- Removal of fully duplicated rows.
- Train / validation / test split using `stratify`.
- Feature engineering: `LogAmount`, `TimeHours`.
- Model training and comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- Evaluation metrics:
  - Precision
  - Recall
  - F1
  - ROC-AUC
  - Average Precision
  - PR-AUC
- Separate classification threshold selection for each model.
- XGBoost hyperparameter tuning using `RandomizedSearchCV`.
- Final model selection based only on validation data.
- Final evaluation on a separate test set.
- SHAP for feature impact explainability.
- MLflow for tracking experiment parameters and metrics.
- Model and configuration persistence using `joblib`.
- PostgreSQL for storing model results and SQL-based analysis.
- FastAPI for serving predictions through an HTTP API.
- Docker / Docker Compose for local deployment of the API and PostgreSQL.
- Basic PSI-based data drift monitoring.

## Final Model

The final selected model is **XGBoost** with a classification threshold of **0.324**.

Test set results:

| Metric | Value |
|---|---:|
| Precision | 0.7549 |
| Recall | 0.8105 |
| F1 | 0.7817 |
| ROC-AUC | 0.9695 |
| Average Precision | 0.8286 |
| PR-AUC | 0.8285 |
| False Positives | 25 |
| False Negatives | 18 |
| True Positives | 77 |

For demonstration of business-oriented evaluation, the following hypothetical error costs were used:

- False Negative: 100,000 KZT
- False Positive: 2,000 KZT

The resulting hypothetical total error cost on the test set was **1,850,000 KZT**.

> The error costs are educational assumptions and do not represent actual banking costs.

Hyperparameter tuning did not improve the business-oriented result. The tuned XGBoost model achieved the same Recall on the validation set but produced more False Positives. Therefore, the original XGBoost configuration was retained as the final model.

## PostgreSQL and SQL Analysis

Model results are exported to PostgreSQL.

For the test dataset, the following information is stored:

- original transaction features;
- actual class;
- fraud score;
- final prediction.

Example table:

```text
fraud_scored_transactions
```

Example SQL queries:

```sql
-- Overall fraud rate
SELECT
    COUNT(*) AS total_transactions,
    SUM(actual_class) AS fraud_transactions,
    ROUND(100.0 * SUM(actual_class) / COUNT(*), 4) AS fraud_rate_percent
FROM fraud_scored_transactions;
```

```sql
-- Number of transactions by predicted class
SELECT
    predicted_fraud,
    COUNT(*) AS transactions,
    ROUND(AVG("Amount")::numeric, 2) AS avg_amount
FROM fraud_scored_transactions
GROUP BY predicted_fraud
ORDER BY predicted_fraud;
```

```sql
-- Transactions with the highest fraud scores
SELECT
    "Amount",
    actual_class,
    predicted_fraud,
    fraud_probability
FROM fraud_scored_transactions
ORDER BY fraud_probability DESC
LIMIT 20;
```

SQL is not used for model training. It is used for analysing saved predictions and validating model behaviour on persisted data.

## FastAPI

The trained model is saved as `fraud_model.joblib` and loaded by the FastAPI application.

Main endpoints:

```text
GET  /health
POST /predict
```

`/health` checks whether the service is running.

`/predict` accepts the features of a single transaction, applies the same feature engineering used during training, calculates the fraud score, and compares it with the stored threshold.

Example response:

```json
{
  "fraud_score": 0.81,
  "threshold": 0.324,
  "predicted_fraud": 1,
  "model": "XGBoost",
  "model_version": "1.0.0"
}
```

API predictions can also be stored in PostgreSQL for further analysis and monitoring.

FastAPI Swagger documentation is available locally at:

```text
http://127.0.0.1:8000/docs
```

## MLflow

MLflow is used for experiment tracking.

For the final run, `xgboost_champion`, the following artifacts and metrics are stored:

- model;
- threshold;
- Precision / Recall / F1;
- Average Precision / PR-AUC;
- False Positives / False Negatives;
- hypothetical business cost.

The local MLflow metadata store uses SQLite (`mlflow.db`).

## Docker

FastAPI and PostgreSQL can be started as separate containers using Docker Compose:

```powershell
docker compose up --build
```

This provides a repeatable local environment for running both the application and database.

## Drift Monitoring

For basic monitoring, the distribution of production features is compared with a reference sample taken from the training data.

PSI (Population Stability Index) is used as a simple indicator of distribution shift.

This is a lightweight drift indicator and should not be considered a complete production monitoring system.

## Main Technologies

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- MLflow
- FastAPI
- PostgreSQL
- SQLAlchemy
- psycopg
- joblib
- Docker / Docker Compose
- Jupyter Notebook

## Main Files

```text
notebooks/
├── 01_eda.ipynb
└── 02_modeling.ipynb

src/
├── features.py
├── api.py
├── db.py
└── monitor_drift.py

sql/
├── schema.sql
└── analysis.sql

artifacts/
├── fraud_model.joblib
└── reference_sample.csv
```

## Limitations

- The dataset covers only a short time period.
- Features `V1–V28` are anonymised, so their business meaning is unknown.
- The fraud class is extremely rare.
- False Positive and False Negative costs are hypothetical.
- The project is not a production-ready banking anti-fraud system. It does not include:
  - real online labels;
  - automated retraining;
  - full production monitoring;
  - authentication / authorization;
  - production deployment.
