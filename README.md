# Credit Card Fraud Detection

This project serves an already-trained Random Forest for credit-card fraud inference. The API does not train or tune a model. It applies the saved Phase 3 `StandardScaler` to `Time` and `Amount`; PCA features `V1`–`V28` pass through unchanged.

## Run the API

From the project root in PowerShell, start the local server with the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open interactive API documentation at <http://127.0.0.1:8000/docs>.

## Endpoints

- `GET /` — service information.
- `GET /health` — confirms the saved model and preprocessing artifacts loaded.
- `POST /predict` — accepts one JSON object containing numeric `Time`, `V1` through `V28`, and `Amount`. All 30 fields are required; extra fields and non-finite values are rejected.

The prediction response includes `predicted_class` (`0` or `1`), `prediction_label` (`Legitimate` or `Potential Fraud`), `fraud_probability`, and the initial `decision_threshold` of `0.5`.

Example request body (send all 30 fields):

```json
{
  "Time": 0,
  "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0,
  "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0,
  "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0,
  "V19": 0, "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0,
  "V25": 0, "V26": 0, "V27": 0, "V28": 0,
  "Amount": 0
}
```

## Run API tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use synthetic values and do not modify the source or processed datasets.

## Run the Streamlit dashboard

Start FastAPI in one PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then start the dashboard from the project root in another window:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The dashboard defaults to `http://127.0.0.1:8000` locally. For a hosted deployment, configure `FRAUD_API_BASE_URL` in the Streamlit app settings with the public FastAPI URL. Set `FRAUD_API_BASE_URL` or edit the sidebar field to use a different FastAPI base URL. It provides a JSON form for one transaction and CSV batch predictions; CSVs must contain exactly `Time`, `V1`–`V28`, and `Amount` as finite numeric columns. Predictions always go through `POST /predict`; the Streamlit process does not load the model. The API status is shown in the sidebar. Batch results and downloads contain transaction row numbers and prediction fields, not input features.

## Privacy and deployment

This public portfolio demo is for synthetic or otherwise non-sensitive data only. Do not submit real cardholder or transaction data. Uploaded CSVs are processed in memory by the dashboard and sent to the configured API; the app does not intentionally persist inputs. Free hosting providers operate the public services and may process request metadata. The API validates and predicts without logging transaction values.

The repository intentionally excludes raw and processed transaction CSVs, local MLflow databases and runs, virtual environments, and local logs. It includes only the fitted inference model, its matching preprocessing artifact, and aggregate reports. Keep your current datasets and training outputs on your computer.

### Free portfolio deployment

1. Create a new, dedicated GitHub repository for this project. Do not use the existing remote configured in the enclosing user-folder repository; it points to an unrelated project. Set the new repository to **Public** only after reviewing the files Git would publish. The two required `.joblib` inference artifacts are included; CSV datasets remain excluded.
2. Deploy the API from the new repository with Render using the included `render.yaml`. Render’s free web service can sleep after inactivity, so the first request may take about a minute to wake.
3. Deploy `dashboard/app.py` from the same GitHub repo with Streamlit Community Cloud. In the Streamlit app settings under **Secrets**, add `FRAUD_API_BASE_URL = "https://<service>.onrender.com"` using your Render service URL.
4. Check `/health`, then test the single and batch forms with synthetic records only. Avoid uploading the project’s real dataset to either public service.

The dashboard is public and the API currently has no authentication or rate limiting. Treat it as a portfolio demo, not a production fraud decision system.

Local API dependencies are in `requirements-api.txt`; Streamlit Community Cloud dependencies are in `dashboard/requirements.txt`. `requirements.txt` remains the convenience list for local development.
