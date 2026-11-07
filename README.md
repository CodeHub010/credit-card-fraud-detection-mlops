# Credit Card Fraud Detection

A portfolio MLOps project serving an already-trained Random Forest model through a FastAPI backend and Streamlit dashboard.

**[Open the live dashboard](https://credit-card-fraud-detection-mlops-pkjv6565lv6plygkaapkvt.streamlit.app/)** · [API health](https://credit-card-fraud-api-u5gy.onrender.com/health) · [Interactive API docs](https://credit-card-fraud-api-u5gy.onrender.com/docs)

> **Demo only:** Use synthetic or otherwise non-sensitive data. The public API has no authentication or rate limiting and is not for real fraud decisions. The free API service may take 50 seconds or more to wake after inactivity.

## How it works

The Streamlit dashboard sends transactions to the FastAPI service. The API loads the saved Random Forest and its matching Phase 3 `StandardScaler`. It scales `Time` and `Amount`; PCA features `V1`–`V28` pass through unchanged. The service returns a fraud probability and a class using a 0.5 decision threshold.

The project serves the existing model; the API does not train or tune it.

## Held-out test results

| Metric | Result |
|---|---:|
| Precision | 97.14% |
| Recall | 71.58% |
| F1 score | 82.42% |
| PR-AUC | 79.63% |
| ROC-AUC | 92.46% |

Results are from one evaluation on a held-out test split of 56,746 transactions, including 95 fraud cases. The test set was not used for model selection or threshold tuning.

## Run locally

Use two PowerShell terminals from the project root.

**Terminal 1 — API**

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Check the API at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). Interactive docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

**Terminal 2 — dashboard**

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open the Local URL printed by Streamlit, usually [http://localhost:8501](http://localhost:8501). The dashboard’s **FastAPI base URL** should be `http://127.0.0.1:8000`.

## API

- `GET /health` — confirms the model and preprocessing artifacts loaded.
- `POST /predict` — predicts one transaction.
- `POST /predict/batch` — predicts a batch of up to 5,000 transactions.

Each transaction must contain exactly 30 finite numeric fields: `Time`, `V1`–`V28`, and `Amount`. The dashboard validates uploaded CSVs against this schema. Its result tables and downloads contain row numbers and predictions, not input features.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests use synthetic values and do not modify the source or processed datasets.

## Deployment

The API is hosted on Render using [`render.yaml`](render.yaml). The dashboard is hosted on Streamlit Community Cloud from `dashboard/app.py`.

The Streamlit app reads its API address from the `FRAUD_API_BASE_URL` secret:

```toml
FRAUD_API_BASE_URL = "https://credit-card-fraud-api-u5gy.onrender.com"
```

## Data and privacy

Raw and processed transaction CSVs, local MLflow databases and runs, virtual environments, and logs are excluded from the repository. The saved inference model, matching preprocessing artifact, and aggregate reports are included.

Uploaded CSVs are processed in memory by the dashboard and sent to the configured API. The app does not intentionally persist inputs. Hosting providers may process request metadata.

## Tech stack

Python · scikit-learn · FastAPI · Streamlit · MLflow
