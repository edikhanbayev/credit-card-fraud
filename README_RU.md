# Credit Card Fraud Detection

Учебный end-to-end проект по выявлению мошеннических операций по банковским картам на сильно несбалансированных данных.

Проект охватывает полный путь: анализ данных → обучение и сравнение моделей → подбор порога → объяснение модели → сохранение и отслеживание экспериментов → API → PostgreSQL → Docker → базовый мониторинг drift.

Данные - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Что реализовано

- EDA: пропуски, дубликаты, распределение классов и суммы операций.
- Удаление полностью дублирующихся строк.
- Разделение данных на train / validation / test с `stratify`.
- Feature engineering: `LogAmount`, `TimeHours`.
- Обучение и сравнение:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- Метрики: Precision, Recall, F1, ROC-AUC, Average Precision, PR-AUC.
- Подбор отдельного classification threshold для каждой модели.
- Hyperparameter tuning XGBoost через `RandomizedSearchCV`.
- Выбор итоговой модели только по validation data.
- Финальная проверка на отдельной test выборке.
- SHAP для объяснения влияния признаков.
- MLflow для хранения параметров и метрик эксперимента.
- Сохранение модели и настроек через `joblib`.
- PostgreSQL для хранения результатов и SQL-анализа.
- FastAPI для получения предсказаний через HTTP API.
- Docker / Docker Compose для локального запуска API и PostgreSQL.
- Базовый PSI-based data drift monitoring.

## Итоговая модель

Итоговой моделью выбран **XGBoost** с порогом **0.324**.

Результаты на test выборке:

| Метрика | Значение |
|---|---:|
| Precision | 0.7549 |
| Recall | 0.8105 |
| F1 | 0.7817 |
| ROC-AUC | 0.9695 |
| Average Precision | 0.8286 |
| PR-AUC | 0.8285 |
| False Positive | 25 |
| False Negative | 18 |
| True Positive | 77 |

Для демонстрации бизнес-логики использованы условные стоимости ошибок:

- False Negative: 100 000 KZT
- False Positive: 2 000 KZT

Итоговая условная стоимость ошибок на test выборке: **1 850 000 KZT**.

> Стоимости ошибок являются учебными допущениями и не представляют реальные банковские затраты.

Hyperparameter tuning не улучшил бизнес-ориентированный результат: настроенный XGBoost сохранил тот же Recall на validation выборке, но дал больше False Positive. Поэтому итоговой моделью оставлена исходная конфигурация XGBoost.

## PostgreSQL и SQL-анализ

Результаты модели выгружаются в PostgreSQL. Для test-выборки сохраняются исходные признаки, фактический класс, fraud score и итоговое предсказание.

Пример таблицы:

```text
fraud_scored_transactions
```

Примеры SQL-запросов:

```sql
-- Общая доля мошеннических операций
SELECT
    COUNT(*) AS total_transactions,
    SUM(actual_class) AS fraud_transactions,
    ROUND(100.0 * SUM(actual_class) / COUNT(*), 4) AS fraud_rate_percent
FROM fraud_scored_transactions;
```

```sql
-- Количество операций по предсказанному классу
SELECT
    predicted_fraud,
    COUNT(*) AS transactions,
    ROUND(AVG("Amount")::numeric, 2) AS avg_amount
FROM fraud_scored_transactions
GROUP BY predicted_fraud
ORDER BY predicted_fraud;
```

```sql
-- Операции с наибольшим fraud score
SELECT
    "Amount",
    actual_class,
    predicted_fraud,
    fraud_probability
FROM fraud_scored_transactions
ORDER BY fraud_probability DESC
LIMIT 20;
```

SQL используется не для обучения модели, а для анализа результатов и проверки поведения модели на сохраненных данных.

## FastAPI

Обученная модель сохранена как `fraud_model.joblib` и загружается приложением FastAPI.

Основные endpoints:

```text
GET  /health
POST /predict
```

`/health` проверяет, что сервис работает.

`/predict` принимает признаки одной транзакции, выполняет тот же feature engineering, который использовался при обучении, рассчитывает fraud score и сравнивает его с сохраненным threshold.

Пример ответа:

```json
{
  "fraud_score": 0.81,
  "threshold": 0.324,
  "predicted_fraud": 1,
  "model": "XGBoost",
  "model_version": "1.0.0"
}
```

Предсказания API могут сохраняться в PostgreSQL для последующего анализа и мониторинга.

Swagger-документация FastAPI доступна локально по адресу:

```text
http://127.0.0.1:8000/docs
```

## MLflow

MLflow используется для отслеживания экспериментов.

Для итогового run `xgboost_champion` сохраняются:

- модель;
- threshold;
- Precision / Recall / F1;
- Average Precision / PR-AUC;
- False Positive / False Negative;
- условная business cost.

Локальное хранилище метаданных MLflow использует SQLite (`mlflow.db`).

## Docker

FastAPI и PostgreSQL можно запускать как отдельные контейнеры через Docker Compose:

```powershell
docker compose up --build
```

Это упрощает повторяемый локальный запуск приложения и базы данных.

## Drift monitoring

Для базового мониторинга сравнивается распределение production-признаков с reference sample из обучающих данных.

Используется PSI (Population Stability Index). Это простой индикатор изменения распределения данных, а не полноценная production-система мониторинга.

## Основные технологии

- Python
- pandas, NumPy, matplotlib
- scikit-learn
- XGBoost, LightGBM, CatBoost
- SHAP
- MLflow
- FastAPI
- PostgreSQL
- SQLAlchemy, psycopg
- joblib
- Docker / Docker Compose
- Jupyter Notebook

## Основные файлы

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

## Ограничения

- Данные покрывают короткий временной период.
- `V1–V28` анонимизированы, поэтому их бизнес-смысл неизвестен.
- Fraud-класс крайне редкий.
- Стоимости False Positive и False Negative заданы условно.
- Проект не является готовой банковской anti-fraud системой: нет реальных online labels, автоматического retraining, полноценного мониторинга, authentication/authorization и production deployment.


