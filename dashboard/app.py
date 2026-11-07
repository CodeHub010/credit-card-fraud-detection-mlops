"""Privacy-conscious Streamlit frontend for the FastAPI fraud service."""

from __future__ import annotations

import json
import hashlib
import math
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_TIMEOUT_SECONDS = 15
BATCH_API_TIMEOUT = (5, 120)
BATCH_SIZE = 5000
MAX_BATCH_SIZE = 5000


def configured_api_base_url() -> str:
    """Read the public API URL from environment or Streamlit Cloud secrets."""
    value = os.getenv("FRAUD_API_BASE_URL")
    if value:
        return value
    try:
        value = st.secrets.get("FRAUD_API_BASE_URL")
    except Exception:
        value = None
    return value or DEFAULT_API_BASE_URL

def normalize_api_url(url: str) -> str:
    """Trim whitespace and trailing slashes from the configured API URL."""
    return url.strip().rstrip("/")


def validate_transaction(payload: Any) -> dict[str, float]:
    """Require exactly the 30 API features and finite numeric values."""
    if not isinstance(payload, dict):
        raise ValueError("The transaction must be a JSON object.")
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    extra = [column for column in payload if column not in FEATURE_COLUMNS]
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing {len(missing)} required field(s)")
        if extra:
            problems.append(f"contains {len(extra)} unexpected field(s)")
        raise ValueError("Transaction fields " + " and ".join(problems) + ".")

    cleaned: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        value = payload[column]
        if isinstance(value, bool):
            raise ValueError(f"{column} must be a finite number.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be a finite number.") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{column} must be a finite number.")
        cleaned[column] = numeric
    return cleaned


def validate_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the exact input schema and return numeric features in API order."""
    validate_csv_columns(frame.columns)
    if frame.empty:
        raise ValueError("The uploaded CSV contains no transactions.")

    ordered = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    if ordered.isna().any().any():
        raise ValueError("CSV contains blank or non-numeric feature values.")
    if not all(math.isfinite(float(value)) for value in ordered.to_numpy().ravel()):
        raise ValueError("CSV contains non-finite feature values.")
    return ordered


def validate_csv_columns(columns: Any) -> None:
    """Require exactly the 30 prediction features, with no target or extra fields."""
    column_names = list(columns)
    missing = [column for column in FEATURE_COLUMNS if column not in column_names]
    extra = [str(column) for column in column_names if column not in FEATURE_COLUMNS]
    if missing or extra or len(column_names) != len(FEATURE_COLUMNS):
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(extra)}")
        raise ValueError("CSV must contain exactly the 30 required feature columns" + (f" ({'; '.join(details)})." if details else "."))


def count_csv_rows(uploaded: Any, chunksize: int = 50_000) -> int:
    """Validate CSV contents in bounded chunks and return its transaction count."""
    total = 0
    try:
        uploaded.seek(0)
        header = pd.read_csv(uploaded, nrows=0)
        validate_csv_columns(header.columns)
        uploaded.seek(0)
        for chunk in pd.read_csv(uploaded, chunksize=chunksize):
            validate_csv_frame(chunk)
            total += len(chunk)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty.") from exc
    finally:
        uploaded.seek(0)
    if total == 0:
        raise ValueError("The uploaded CSV contains no transactions.")
    return total


def request_prediction(api_base_url: str, payload: dict[str, float]) -> dict[str, Any]:
    """Call the existing backend. Never logs or returns submitted feature values."""
    response = requests.post(
        f"{normalize_api_url(api_base_url)}/predict",
        json=payload,
        timeout=API_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Prediction service returned HTTP {response.status_code}.")
    result = response.json()
    required = {"predicted_class", "prediction_label", "fraud_probability", "decision_threshold"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise RuntimeError("Prediction service returned an unexpected response.")
    return result


def request_prediction_batch(
    api_base_url: str, transactions: list[dict[str, float]]
) -> list[dict[str, Any]]:
    """Send one bounded batch to FastAPI and validate its compact response."""
    if not transactions or len(transactions) > MAX_BATCH_SIZE:
        raise ValueError(f"Each API batch must contain between 1 and {MAX_BATCH_SIZE:,} rows.")
    try:
        response = requests.post(
            f"{normalize_api_url(api_base_url)}/predict/batch",
            json={"transactions": transactions},
            timeout=BATCH_API_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ConnectionError("Could not reach the prediction API for this batch.") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"Prediction API returned HTTP {response.status_code} for a batch.")
    try:
        body = response.json()
        predictions = body["predictions"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Prediction API returned an invalid batch response.") from exc
    required = {"predicted_class", "prediction_label", "fraud_probability", "decision_threshold"}
    if (
        not isinstance(predictions, list)
        or len(predictions) != len(transactions)
        or any(not isinstance(item, dict) or not required.issubset(item) for item in predictions)
    ):
        raise RuntimeError("Prediction API returned an incomplete batch response.")
    return predictions


class BatchPredictionError(RuntimeError):
    """Batch error that retains safe partial prediction results for recovery."""

    def __init__(self, message: str, processed: int, total: int, partial: pd.DataFrame):
        super().__init__(message)
        self.processed = processed
        self.total = total
        self.partial = partial


def predict_frames(
    api_base_url: str,
    frames: Any,
    total_rows: int,
    progress: Any = None,
    on_chunk: Any = None,
    chunk_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Predict CSV chunks via bounded API requests, retaining partial results on failure."""
    if chunk_size < 1 or chunk_size > MAX_BATCH_SIZE:
        raise ValueError(f"chunk_size must be between 1 and {MAX_BATCH_SIZE:,}.")
    result_chunks: list[pd.DataFrame] = []
    processed = 0
    try:
        for source_frame in frames:
            checked = validate_csv_frame(source_frame)
            for start in range(0, len(checked), chunk_size):
                batch_frame = checked.iloc[start : start + chunk_size]
                transactions = batch_frame.to_dict(orient="records")
                predictions = request_prediction_batch(api_base_url, transactions)
                batch_results = [
                    {"transaction_number": processed + index + 1, **prediction}
                    for index, prediction in enumerate(predictions)
                ]
                result_chunks.append(pd.DataFrame.from_records(batch_results))
                processed += len(batch_results)
                if progress is not None:
                    progress.progress(
                        min(processed / total_rows, 1.0),
                        text=f"Processed {processed:,} of {total_rows:,} transactions",
                    )
                if on_chunk is not None:
                    on_chunk(processed, total_rows, pd.DataFrame.from_records(batch_results))
    except Exception as exc:
        partial = pd.concat(result_chunks, ignore_index=True) if result_chunks else pd.DataFrame()
        message = (
            f"Batch prediction stopped after {processed:,} of {total_rows:,} transactions. "
            "Check API availability and retry the upload."
        )
        raise BatchPredictionError(message, processed, total_rows, partial) from exc

    if processed != total_rows:
        partial = pd.concat(result_chunks, ignore_index=True) if result_chunks else pd.DataFrame()
        raise BatchPredictionError(
            f"Batch ended after {processed:,} of {total_rows:,} transactions.",
            processed,
            total_rows,
            partial,
        )
    return pd.concat(result_chunks, ignore_index=True) if result_chunks else pd.DataFrame()


def predict_frame(
    api_base_url: str,
    frame: pd.DataFrame,
    progress: Any = None,
    on_chunk: Any = None,
    chunk_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Predict an in-memory frame using bounded batches (use CSV chunks for large files)."""
    checked = validate_csv_frame(frame)
    chunks = (checked.iloc[start : start + chunk_size] for start in range(0, len(checked), chunk_size))
    return predict_frames(api_base_url, chunks, len(checked), progress, on_chunk, chunk_size)


def summarize_predictions(results: pd.DataFrame) -> dict[str, float | int]:
    total = len(results)
    fraud_count = int((results["predicted_class"] == 1).sum())
    legitimate_count = int((results["predicted_class"] == 0).sum())
    return {
        "total": total,
        "legitimate": legitimate_count,
        "fraud": fraud_count,
        "fraud_percentage": (100.0 * fraud_count / total) if total else 0.0,
    }


def prediction_results_csv(results: pd.DataFrame) -> bytes:
    """Serialize only prediction output fields for privacy-conscious download."""
    return results.to_csv(index=False).encode("utf-8")


def render_summary(results: pd.DataFrame, key: str) -> None:
    summary = summarize_predictions(results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analyzed", f"{summary['total']:,}")
    c2.metric("Legitimate predictions", f"{summary['legitimate']:,}")
    c3.metric("Potential fraud", f"{summary['fraud']:,}")
    c4.metric("Fraud prediction rate", f"{summary['fraud_percentage']:.2f}%")

    chart_data = pd.DataFrame(
        {"Prediction count": [summary["legitimate"], summary["fraud"]]},
        index=["Legitimate", "Potential Fraud"],
    )
    st.subheader("Prediction distribution")
    st.bar_chart(chart_data, color="#20c997", height=220)
    csv_bytes = prediction_results_csv(results)
    st.download_button(
        "Download prediction results (CSV)",
        data=csv_bytes,
        file_name="fraud_prediction_results.csv",
        mime="text/csv",
        key=f"download_{key}",
        help="Downloads prediction results and transaction row numbers only; feature values are excluded.",
    )


def main() -> None:
    st.set_page_config(
        page_title="Fraud Intelligence | Transaction Monitor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
            background: #0B1220 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"] { background: #0B1220 !important; }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background: #111827 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] * { color: #FFFFFF; }
        .stApp, .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stWidgetLabel"] p { color: #FFFFFF; }
        h1, h2, h3, h4, h5, h6, [data-testid="stHeading"] { color: #FFFFFF !important; }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
        small, .secondary-text { color: #CBD5E1 !important; }

        .hero { padding: 1.5rem 1.8rem; border-radius: 18px; color: #FFFFFF;
                background: #1E293B; border: 1px solid #334155; margin-bottom: 1.2rem; }
        .hero h1 { color: #FFFFFF !important; margin: 0 0 .25rem 0; }
        .hero p { color: #CBD5E1 !important; margin: 0; }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: #1E293B !important; border: 1px solid #334155; border-radius: 12px;
            gap: .35rem; padding: .3rem;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] {
            background: #1E293B !important; color: #CBD5E1 !important; border-radius: 9px;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: #22D3EE !important; border-bottom: 2px solid #22D3EE !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-panel"] { color: #FFFFFF !important; }

        input, textarea, [data-baseweb="input"], [data-baseweb="base-input"],
        [data-baseweb="select"] > div, [data-testid="stNumberInput"] button {
            background: #1E293B !important; color: #FFFFFF !important;
            border-color: #334155 !important; caret-color: #22D3EE !important;
        }
        input::placeholder, textarea::placeholder { color: #CBD5E1 !important; opacity: 1; }
        [role="listbox"], [role="option"], [data-baseweb="popover"],
        [data-baseweb="menu"], [data-baseweb="select"] * {
            background-color: #1E293B !important; color: #FFFFFF !important;
        }
        [data-testid="stFileUploader"] section {
            background: #1E293B !important; color: #FFFFFF !important;
            border: 1px dashed #334155 !important;
        }
        [data-testid="stFileUploader"] section *, [data-testid="stFileUploaderDropzone"] * {
            color: #CBD5E1 !important;
        }
        [data-testid="stFileUploader"] button {
            background: #0891B2 !important; color: #FFFFFF !important;
            border: 1px solid #0891B2 !important;
        }

        .stButton > button, [data-testid="stDownloadButton"] button {
            background: #0891B2 !important; color: #FFFFFF !important;
            border: 1px solid #0891B2 !important; border-radius: 9px !important;
        }
        .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
            background: #0E7490 !important; border-color: #22D3EE !important; color: #FFFFFF !important;
        }
        button:focus-visible, input:focus, textarea:focus {
            outline: 2px solid #22D3EE !important; outline-offset: 2px !important;
        }

        div[data-testid="stMetric"], [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
            background: #1E293B !important; color: #FFFFFF !important;
        }
        div[data-testid="stMetric"] {
            padding: 1rem; border-radius: 14px; border: 1px solid #334155;
        }
        [data-testid="stAlert"], [data-testid="stAlert"] * {
            color: #FFFFFF !important; border-color: #334155 !important;
        }
        [data-testid="stAlert"] { background: #1E293B !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"],
        [data-testid="stDataFrame"] iframe, table, thead, tbody, tr, th, td {
            background-color: #1E293B !important; color: #FFFFFF !important;
            border-color: #334155 !important;
        }
        [data-testid="stProgress"] > div > div { background: #22D3EE !important; }
        hr { border-color: #334155 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🛡️ Fraud Intelligence")
        st.caption("Credit Card Fraud Detection · MLOps MVP")
        api_base_url = st.text_input(
            "FastAPI base URL",
            value=configured_api_base_url(),
            help="The dashboard sends requests to this backend; it never loads the model itself.",
        )
        health_slot = st.empty()
        if st.button("Check API connection", use_container_width=True):
            try:
                response = requests.get(f"{normalize_api_url(api_base_url)}/health", timeout=5)
                response.raise_for_status()
                health = response.json()
                if health.get("status") == "healthy":
                    health_slot.success("API connected · model ready")
                else:
                    health_slot.warning("API responded, but health status is unexpected.")
            except requests.RequestException:
                health_slot.error("API unavailable. Start FastAPI and check the configured URL.")
        else:
            try:
                response = requests.get(f"{normalize_api_url(api_base_url)}/health", timeout=3)
                if response.ok and response.json().get("status") == "healthy":
                    health_slot.success("API connected · model ready")
                else:
                    health_slot.warning("API responded but is not reporting healthy.")
            except (requests.RequestException, ValueError):
                health_slot.error("API offline · predictions unavailable")
        st.divider()
        st.caption("Prediction data is sent to the configured API. Feature values are not shown in result tables or included in default downloads.")

    st.markdown(
        """
        <div class="hero">
          <h1>Transaction Fraud Monitor</h1>
          <p>Review individual transactions or analyze a batch with the deployed Random Forest service.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project overview")
    st.write("A lightweight MLOps demonstration for card transaction screening. The dashboard is a client for the existing FastAPI inference service and does not load or retrain the model.")

    single_tab, batch_tab = st.tabs(["Individual transaction", "Batch CSV analysis"])
    with single_tab:
        st.subheader("Submit a transaction as JSON")
        st.caption("Provide all 30 numeric fields: Time, V1–V28, and Amount. Inputs are validated before sending to the API.")
        sample_json = json.dumps({column: 0 for column in FEATURE_COLUMNS}, indent=2)
        raw_json = st.text_area("Transaction JSON", value=sample_json, height=260, label_visibility="collapsed")
        if st.button("Get prediction", type="primary", key="single_predict"):
            try:
                payload = validate_transaction(json.loads(raw_json))
                result = request_prediction(api_base_url, payload)
                st.session_state["single_prediction"] = result
                st.success("Prediction received from FastAPI.")
            except json.JSONDecodeError:
                st.error("Enter valid JSON. The transaction must be a JSON object.")
            except ValueError as exc:
                st.error(str(exc))
            except requests.RequestException:
                st.error("Could not reach the prediction API. Check its URL and health status.")
            except RuntimeError as exc:
                st.error(str(exc))
        if "single_prediction" in st.session_state:
            result = st.session_state["single_prediction"]
            label = result["prediction_label"]
            (st.error if label == "Potential Fraud" else st.success)(f"{label} · class {result['predicted_class']}")
            m1, m2 = st.columns(2)
            m1.metric("Fraud probability", f"{float(result['fraud_probability']):.2%}")
            m2.metric("Decision threshold", f"{float(result['decision_threshold']):.2f}")

    with batch_tab:
        st.subheader("Analyze transactions from CSV")
        st.caption("Upload a CSV with exactly the 30 required feature columns. Row contents remain hidden; the downloaded results contain row numbers and predictions only.")
        # Versioned key intentionally resets previously selected uploader content from older runs.
        uploaded = st.file_uploader("Choose transaction CSV", type=["csv"], key="batch_csv_v2")
        if uploaded is not None:
            upload_signature = hashlib.sha256(uploaded.getbuffer()).hexdigest()
            if st.session_state.get("batch_source_signature") != upload_signature:
                st.session_state.pop("batch_results", None)
                st.session_state.pop("batch_partial_chunks", None)
                st.session_state.pop("batch_total_rows", None)
            st.session_state["batch_source_signature"] = upload_signature
            st.caption(f"Selected upload: `{uploaded.name}`")
            try:
                uploaded.seek(0)
                header = pd.read_csv(uploaded, nrows=0)
                validate_csv_columns(header.columns)
                uploaded.seek(0)
                if "batch_total_rows" not in st.session_state:
                    with st.spinner("Validating CSV rows in memory-bounded chunks…"):
                        total_rows = count_csv_rows(uploaded)
                    st.session_state["batch_total_rows"] = total_rows
                else:
                    total_rows = int(st.session_state["batch_total_rows"])
                uploaded.seek(0)
                st.info(
                    f"CSV validated · {total_rows:,} transaction(s) ready. "
                    f"Predictions will be sent in chunks of up to {BATCH_SIZE:,}."
                )
            except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError):
                total_rows = None
                st.error(f"Upload `{uploaded.name}` could not be parsed as a valid CSV.")
            except ValueError as exc:
                total_rows = None
                st.error(f"Upload `{uploaded.name}`: {exc}")
            if total_rows is not None and st.button("Analyze CSV", type="primary", key="batch_predict"):
                st.session_state.pop("batch_results", None)
                st.session_state["batch_partial_chunks"] = []
                progress = st.progress(0.0, text=f"Processed 0 of {total_rows:,} transactions")

                def preserve_partial(processed: int, total: int, new_results: pd.DataFrame) -> None:
                    partial_chunks = list(st.session_state.get("batch_partial_chunks", []))
                    partial_chunks.append(new_results)
                    st.session_state["batch_partial_chunks"] = partial_chunks

                try:
                    uploaded.seek(0)
                    results = predict_frames(
                        api_base_url,
                        pd.read_csv(uploaded, chunksize=BATCH_SIZE),
                        total_rows=total_rows,
                        progress=progress,
                        on_chunk=preserve_partial,
                        chunk_size=BATCH_SIZE,
                    )
                    st.session_state["batch_results"] = results
                    st.session_state.pop("batch_partial_chunks", None)
                    st.success("Batch predictions completed.")
                except BatchPredictionError as exc:
                    st.error(str(exc))
                except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error("Batch processing stopped because the uploaded CSV could not be read completely.")
        if "batch_results" in st.session_state:
            results = st.session_state["batch_results"]
            render_summary(results, key="batch")
            st.dataframe(results, width="stretch", hide_index=True)
        elif st.session_state.get("batch_partial_chunks"):
            partial = pd.concat(st.session_state["batch_partial_chunks"], ignore_index=True)
            st.warning(
                f"Processing was interrupted after {len(partial):,} of "
                f"{st.session_state.get('batch_total_rows', 0):,} transactions. "
                "Partial predictions are available below; click Analyze CSV to restart."
            )
            render_summary(partial, key="partial_batch")
            st.dataframe(partial, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
