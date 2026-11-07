"""Dashboard validation and workflow tests with synthetic transactions."""

import pandas as pd
import pytest

from dashboard import app as dashboard


def sample_transaction(value: float = 0.0) -> dict[str, float]:
    return {column: value for column in dashboard.FEATURE_COLUMNS}


def test_individual_json_validation_and_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = dashboard.validate_transaction(sample_transaction())
    assert len(payload) == 30
    assert payload["Time"] == 0.0

    expected = {
        "predicted_class": 1,
        "prediction_label": "Potential Fraud",
        "fraud_probability": 0.83,
        "decision_threshold": 0.5,
    }

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return expected

    monkeypatch.setattr(dashboard.requests, "post", lambda *args, **kwargs: FakeResponse())
    assert dashboard.request_prediction("http://localhost:8000/", payload) == expected


def test_csv_schema_and_batch_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame([sample_transaction(), sample_transaction(1.0)])
    validated = dashboard.validate_csv_frame(frame)
    assert list(validated.columns) == dashboard.FEATURE_COLUMNS

    def fake_predictions(_url: str, payloads: list[dict[str, float]]) -> list[dict[str, object]]:
        results = []
        for payload in payloads:
            fraud = int(payload["Time"] == 1)
            results.append({
                "predicted_class": fraud,
                "prediction_label": "Potential Fraud" if fraud else "Legitimate",
                "fraud_probability": float(fraud),
                "decision_threshold": 0.5,
            })
        return results

    monkeypatch.setattr(dashboard, "request_prediction_batch", fake_predictions)
    results = dashboard.predict_frame("http://localhost:8000", validated)
    assert len(results) == 2
    assert list(results["predicted_class"]) == [0, 1]
    summary = dashboard.summarize_predictions(results)
    assert summary == {"total": 2, "legitimate": 1, "fraud": 1, "fraud_percentage": 50.0}


def test_chunking_bounds_api_requests_and_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame([sample_transaction()] * 10_000)
    chunk_sizes: list[int] = []

    def fake_predictions(_url: str, payloads: list[dict[str, float]]) -> list[dict[str, object]]:
        chunk_sizes.append(len(payloads))
        return [{
            "predicted_class": 0,
            "prediction_label": "Legitimate",
            "fraud_probability": 0.01,
            "decision_threshold": 0.5,
        } for _ in payloads]

    class ProgressRecorder:
        calls: list[tuple[float, str]] = []

        def progress(self, ratio: float, text: str) -> None:
            self.calls.append((ratio, text))

    progress = ProgressRecorder()
    monkeypatch.setattr(dashboard, "request_prediction_batch", fake_predictions)
    results = dashboard.predict_frame("http://localhost:8000", frame, progress=progress)
    assert len(results) == 10_000
    assert chunk_sizes == [5000, 5000]
    assert progress.calls[-1] == (1.0, "Processed 10,000 of 10,000 transactions")


def test_batch_failure_keeps_partial_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame([sample_transaction()] * 10_000)
    calls = 0

    def fail_second_chunk(_url: str, payloads: list[dict[str, float]]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Synthetic API failure")
        return [{
            "predicted_class": 0,
            "prediction_label": "Legitimate",
            "fraud_probability": 0.01,
            "decision_threshold": 0.5,
        } for _ in payloads]

    monkeypatch.setattr(dashboard, "request_prediction_batch", fail_second_chunk)
    with pytest.raises(dashboard.BatchPredictionError) as error:
        dashboard.predict_frame("http://localhost:8000", frame)
    assert error.value.processed == 5000
    assert error.value.total == 10_000
    assert len(error.value.partial) == 5000


def test_csv_rejects_missing_extra_and_nonfinite_features() -> None:
    frame = pd.DataFrame([sample_transaction()])
    with pytest.raises(ValueError, match="exactly the 30"):
        dashboard.validate_csv_frame(frame.drop(columns=["V28"]))
    with pytest.raises(ValueError, match="exactly the 30"):
        dashboard.validate_csv_frame(frame.assign(unexpected=1))
    bad = frame.copy()
    bad.loc[0, "Amount"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        dashboard.validate_csv_frame(bad)


def test_sample_transactions_file_has_exact_prediction_schema() -> None:
    from pathlib import Path

    sample_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "sample_transactions.csv"
    sample = pd.read_csv(sample_path)
    validated = dashboard.validate_csv_frame(sample)
    assert validated.shape == (100, 30)
    assert validated.columns.tolist() == dashboard.FEATURE_COLUMNS
    assert "Class" not in validated.columns


def test_download_csv_contains_prediction_results_only() -> None:
    results = pd.DataFrame(
        [{"transaction_number": 1, "predicted_class": 0, "prediction_label": "Legitimate",
          "fraud_probability": 0.0, "decision_threshold": 0.5}]
    )
    content = dashboard.prediction_results_csv(results).decode("utf-8")
    assert "transaction_number" in content
    assert "fraud_probability" in content
    assert "Amount" not in content
    assert len(pd.read_csv(pd.io.common.StringIO(content))) == 1
