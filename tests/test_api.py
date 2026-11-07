"""API contract tests using synthetic transaction values only."""

import math

from fastapi.testclient import TestClient

from api.main import FEATURE_COLUMNS, app

client = TestClient(app)


def synthetic_payload() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_COLUMNS}


def test_root_and_health() -> None:
    assert client.get("/").status_code == 200
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_response_contract() -> None:
    response = client.post("/predict", json=synthetic_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"] in (0, 1)
    assert result["prediction_label"] in ("Legitimate", "Potential Fraud")
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision_threshold"] == 0.5


def test_missing_and_extra_features_are_rejected() -> None:
    missing = synthetic_payload()
    missing.pop("V28")
    assert client.post("/predict", json=missing).status_code == 422
    extra = synthetic_payload()
    extra["unexpected"] = 1.0
    assert client.post("/predict", json=extra).status_code == 422


def test_invalid_and_nonfinite_values_are_rejected() -> None:
    invalid = synthetic_payload()
    invalid["Amount"] = "not-a-number"  # type: ignore[assignment]
    assert client.post("/predict", json=invalid).status_code == 422
    nonfinite = synthetic_payload()
    nonfinite["Time"] = math.inf
    # Strict JSON clients reject Infinity; this verifies the API input path rejects it.
    response = client.post("/predict", content='{"Time": Infinity}')
    assert response.status_code == 422


def test_batch_prediction_matches_single_prediction_contract() -> None:
    transactions = [synthetic_payload(), {**synthetic_payload(), "Time": 15.0, "Amount": 3.5}]
    response = client.post("/predict/batch", json={"transactions": transactions})
    assert response.status_code == 200
    result = response.json()
    assert result["processed"] == 2
    assert len(result["predictions"]) == 2
    assert result["predictions"] == [
        client.post("/predict", json=transaction).json() for transaction in transactions
    ]


def test_batch_prediction_rejects_extra_features() -> None:
    transaction = synthetic_payload()
    transaction["Class"] = 0
    response = client.post("/predict/batch", json={"transactions": [transaction]})
    assert response.status_code == 422
