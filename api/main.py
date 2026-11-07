"""FastAPI inference service for the saved credit-card fraud model."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

LOGGER = logging.getLogger("fraud_api")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessing.joblib"
DECISION_THRESHOLD = 0.5
BATCH_MAX_SIZE = 5000

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
SCALED_COLUMNS = ["Time", "Amount"]


def _finite_number(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Input values must be finite numbers.")
    return value


InputPayload = create_model(
    "TransactionInput",
    __config__=ConfigDict(extra="forbid"),
    __validators__={
        "finite_values": field_validator(*FEATURE_COLUMNS, mode="after")(_finite_number)
    },
    **{name: (float, ...) for name in FEATURE_COLUMNS},
)


class BatchInputPayload(BaseModel):
    """Bound each API request to a manageable, validated transaction chunk."""

    model_config = ConfigDict(extra="forbid")
    transactions: list[InputPayload] = Field(min_length=1, max_length=BATCH_MAX_SIZE)


def _load_artifacts() -> tuple[Any, Any]:
    """Load and validate the already-fitted estimator and Phase 3 scaler."""
    if not MODEL_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
        raise RuntimeError("Required model artifacts are unavailable.")
    model_bundle = joblib.load(MODEL_PATH)
    preprocessing_bundle = joblib.load(PREPROCESSOR_PATH)
    if not isinstance(model_bundle, dict) or "estimator" not in model_bundle:
        raise RuntimeError("The saved model artifact has an unsupported format.")
    if model_bundle.get("model_name") != "Random Forest":
        raise RuntimeError("The saved model is not the selected Random Forest.")
    estimator = model_bundle["estimator"]
    scaler = preprocessing_bundle.get("scaler") if isinstance(preprocessing_bundle, dict) else None
    if scaler is None or list(preprocessing_bundle.get("scaled_columns", [])) != SCALED_COLUMNS:
        raise RuntimeError("The saved Phase 3 scaler metadata is incompatible.")
    if list(model_bundle.get("feature_columns", [])) != FEATURE_COLUMNS:
        raise RuntimeError("The saved model feature order is incompatible.")
    if list(preprocessing_bundle.get("feature_columns", [])) != FEATURE_COLUMNS:
        raise RuntimeError("The Phase 3 feature order is incompatible.")
    if list(getattr(estimator, "classes_", [])) != [0, 1]:
        raise RuntimeError("The saved model does not expose expected binary classes.")
    return estimator, scaler


MODEL, SCALER = _load_artifacts()

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Inference API using the saved Random Forest model and Phase 3 scaler.",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def sanitized_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return validation locations/types without echoing submitted transaction values."""
    errors = [
        {"location": list(error.get("loc", [])), "type": error.get("type", "validation_error")}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Credit Card Fraud Detection API",
        "status": "ready",
        "health": "/health",
        "prediction": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "model": "Random Forest", "artifacts": "loaded"}


@app.post("/predict")
def predict(payload: InputPayload) -> dict[str, float | int | str]:
    """Predict a single transaction; only Time and Amount are standardized."""
    try:
        values = payload.model_dump()
        frame = pd.DataFrame([[values[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        frame.loc[:, SCALED_COLUMNS] = SCALER.transform(frame[SCALED_COLUMNS])

        probabilities = MODEL.predict_proba(frame)
        fraud_probability = float(probabilities[0, list(MODEL.classes_).index(1)])
        predicted_class = int(fraud_probability >= DECISION_THRESHOLD)
        return {
            "predicted_class": predicted_class,
            "prediction_label": "Potential Fraud" if predicted_class else "Legitimate",
            "fraud_probability": fraud_probability,
            "decision_threshold": DECISION_THRESHOLD,
        }
    except Exception:
        # Never log request data or exception text that could contain submitted values.
        LOGGER.exception("Prediction failed for a submitted transaction.")
        return JSONResponse(status_code=500, content={"detail": "Prediction failed."})


@app.post("/predict/batch")
def predict_batch(payload: BatchInputPayload) -> dict[str, Any]:
    """Predict one bounded chunk in a vectorized call using the same scaler and threshold."""
    try:
        frame = pd.DataFrame(
            [transaction.model_dump() for transaction in payload.transactions],
            columns=FEATURE_COLUMNS,
        )
        frame.loc[:, SCALED_COLUMNS] = SCALER.transform(frame[SCALED_COLUMNS])
        fraud_index = list(MODEL.classes_).index(1)
        fraud_probabilities = MODEL.predict_proba(frame)[:, fraud_index]
        predicted_classes = (fraud_probabilities >= DECISION_THRESHOLD).astype(int)
        predictions = [
            {
                "predicted_class": int(predicted_class),
                "prediction_label": "Potential Fraud" if predicted_class else "Legitimate",
                "fraud_probability": float(probability),
                "decision_threshold": DECISION_THRESHOLD,
            }
            for predicted_class, probability in zip(predicted_classes, fraud_probabilities)
        ]
        return {"processed": len(predictions), "predictions": predictions}
    except Exception:
        # Do not include submitted transaction values in server logs.
        LOGGER.exception("Batch prediction failed for a submitted chunk.")
        return JSONResponse(status_code=500, content={"detail": "Batch prediction failed."})
