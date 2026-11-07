"""Shared MLflow logging for the fraud detection training workflow.

This module logs existing validation results and the already-selected model; it
never trains estimators or evaluates the holdout test set.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "Credit-Card-Fraud-Detection"
DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"


def _tracking_uri() -> str:
    """Use a project-local SQLite tracking database."""
    # SQLAlchemy's Windows SQLite URL uses forward slashes after sqlite:///.
    return f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"


def configure_experiment() -> tuple[MlflowClient, str]:
    """Configure the local store and create the experiment if needed."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient(tracking_uri=_tracking_uri())
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=ARTIFACT_ROOT.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    return client, experiment_id


def _model_params(model_name: str, config: dict[str, Any],
                  comparison: dict[str, Any]) -> dict[str, Any]:
    """Flatten model and validation configuration into MLflow parameters."""
    key = "logistic_regression" if model_name == "Logistic Regression" else "random_forest"
    parameters = dict(config[key])
    parameters.update({
        "validation_fraction": comparison["validation_setup"]["validation_fraction_of_training_data"],
        "validation_random_state": comparison["validation_setup"]["random_state"],
        "decision_threshold": config["decision_threshold_for_precision_recall_f1"],
        "selection_metric": "validation_pr_auc_average_precision",
        "training_rows_before_validation_split": comparison["validation_setup"][
            "training_rows_before_validation_split"
        ],
    })
    # MLflow parameters accept scalar strings/numbers; stringify any future compound values.
    return {
        str(key): value if isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in parameters.items()
    }


def log_experiment_runs(
    comparison_report: dict[str, Any],
    model_artifact: dict[str, Any],
    report_artifact_paths: list[Path],
    chart_artifact_paths: list[Path],
    model_source_path: Path,
    preprocessing_source_path: Path,
) -> dict[str, Any]:
    """Create one run per candidate and log only existing validation metrics.

    ``final_test_metrics.json`` is passed as a report artifact path and is never
    opened or logged as an MLflow metric. The estimator and saved preprocessing
    artifacts are read-only inputs.
    """
    client, experiment_id = configure_experiment()
    selected_name = comparison_report["selected_model"]
    selection_reason = comparison_report["selection_reason"]
    if model_artifact.get("model_name") != selected_name:
        raise ValueError("Saved model name does not match the comparison report selection.")
    if selected_name not in comparison_report["models"]:
        raise ValueError(f"Selected model {selected_name!r} has no validation metrics.")

    records: dict[str, dict[str, str]] = {}
    for model_name, model_result in comparison_report["models"].items():
        metrics = model_result["metrics"]
        with mlflow.start_run(experiment_id=experiment_id, run_name=model_name) as run:
            mlflow.log_params(_model_params(
                model_name, comparison_report["configuration"], comparison_report
            ))
            mlflow.log_metrics({
                "validation_precision": float(metrics["precision"]),
                "validation_recall": float(metrics["recall"]),
                "validation_f1_score": float(metrics["f1_score"]),
                "validation_pr_auc": float(metrics["pr_auc"]),
                "validation_roc_auc": float(metrics["roc_auc"]),
            })
            mlflow.set_tags({
                "model_name": model_name,
                "selected_model_name": selected_name,
                "is_selected_model": str(model_name == selected_name).lower(),
                "selection_rationale": selection_reason,
                "validation_only_metrics": "true",
                "test_metrics_logged_as_metrics": "false",
                "random_state": str(comparison_report["configuration"][
                    "logistic_regression"]["random_state"]),
            })

            for path in report_artifact_paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Expected report artifact is missing: {path}")
                mlflow.log_artifact(str(path), artifact_path="reports")
            for path in chart_artifact_paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Expected chart artifact is missing: {path}")
                mlflow.log_artifact(str(path), artifact_path="evaluation_charts")
            for path in (model_source_path, preprocessing_source_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Expected saved model artifact is missing: {path}")
                mlflow.log_artifact(str(path), artifact_path="source_artifacts")

            model_uri = ""
            if model_name == selected_name:
                estimator = model_artifact["estimator"]
                feature_columns = list(model_artifact["feature_columns"])
                # Use a synthetic already-preprocessed example to describe the
                # model interface; no train/test rows or labels are accessed.
                input_example = pd.DataFrame(
                    [{column: 0.0 for column in feature_columns}],
                    columns=feature_columns,
                )
                signature = infer_signature(input_example, estimator.predict(input_example))
                model_info = mlflow.sklearn.log_model(
                    sk_model=estimator,
                    name="selected_fraud_classifier",
                    signature=signature,
                    input_example=input_example,
                    # This Random Forest was trained and saved by this project.
                    # Trust only scikit-learn's tree node type required by skops.
                    skops_trusted_types=(
                        ["sklearn.tree._tree.Tree"] if model_name == "Random Forest" else None
                    ),
                    metadata={
                        "feature_columns": feature_columns,
                        "target_column": model_artifact.get("target_column", "Class"),
                        "expects_phase3_preprocessed_features": True,
                        "scaled_columns": model_artifact.get("scaled_columns", ["Time", "Amount"]),
                        "preprocessing_artifact": "source_artifacts/preprocessing.joblib",
                    },
                )
                model_uri = model_info.model_uri
                mlflow.set_tag("logged_model_uri", model_uri)

            records[model_name] = {
                "run_id": run.info.run_id,
                "run_name": model_name,
                "selected": str(model_name == selected_name).lower(),
                "model_uri": model_uri,
            }
            logging.info("Recorded MLflow run %s (%s)", model_name, run.info.run_id)

    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "tracking_uri": _tracking_uri(),
        "database_path": str(DATABASE_PATH),
        "artifact_root": str(ARTIFACT_ROOT),
        "selected_model": selected_name,
        "selection_rationale": selection_reason,
        "runs": records,
        "test_metrics_logged_as_metrics": False,
        "test_metrics_json_logged_as_artifact": any(
            path.name == "final_test_metrics.json" for path in report_artifact_paths
        ),
    }
