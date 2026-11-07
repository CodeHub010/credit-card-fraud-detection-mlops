"""Train and compare imbalanced fraud classifiers without using test data for selection.

Run from the project root:
    .venv\\Scripts\\python.exe src\\train.py

Existing outputs are protected. Pass --overwrite only when regeneration is intended.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEFAULT_TEST = PROJECT_ROOT / "data" / "processed" / "test.csv"
DEFAULT_PREPROCESSOR = PROJECT_ROOT / "models" / "preprocessing.joblib"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"
DEFAULT_COMPARISON_JSON = PROJECT_ROOT / "reports" / "model_comparison.json"
DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "reports" / "model_comparison.csv"
DEFAULT_TEST_REPORT = PROJECT_ROOT / "reports" / "final_test_metrics.json"
DEFAULT_FIGURES = PROJECT_ROOT / "reports" / "figures"
TARGET = "Class"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    """Parse file locations and explicit output replacement permission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--preprocessor-path", type=Path, default=DEFAULT_PREPROCESSOR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--comparison-json", type=Path, default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--test-report", type=Path, default=DEFAULT_TEST_REPORT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing generated model and report files")
    return parser.parse_args()


def class_counts(y: pd.Series) -> dict[str, int]:
    """Summarize labels without changing the class proportions."""
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    return {"legitimate_0": int(counts.loc[0]), "fraud_1": int(counts.loc[1])}


def validate_dataset(frame: pd.DataFrame, expected_features: list[str], name: str) -> None:
    """Fail early on missing, misordered, or invalid processed data."""
    expected_columns = expected_features + [TARGET]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"{name} columns do not match the Phase 3 preprocessing bundle. "
            f"Expected {expected_columns}, got {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError(f"{name} dataset is empty.")
    if frame.isna().any().any():
        raise ValueError(f"{name} dataset contains missing values.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError(f"{name} must contain both Class values 0 and 1.")
    if not all(pd.api.types.is_numeric_dtype(frame[col]) for col in expected_columns):
        raise ValueError(f"{name} contains nonnumeric feature or target columns.")


def calculate_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, Any]:
    """Compute threshold metrics and ranking metrics from positive-class scores."""
    predictions = (scores >= THRESHOLD).astype("int64")
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        # Average Precision is the standard step-wise PR-AUC summary used here.
        "pr_auc": float(average_precision_score(y_true, scores)),
        "pr_auc_method": "average_precision_score",
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "decision_threshold": THRESHOLD,
        "confusion_matrix_labels_0_1": matrix.astype(int).tolist(),
    }


def plot_confusion_matrix(matrix: list[list[int]], output_path: Path, model_name: str) -> None:
    """Save the final holdout confusion matrix as a PNG."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
        xlabel="Predicted class", ylabel="True class",
        title=f"Test confusion matrix: {model_name}",
    )
    threshold = max(max(row) for row in matrix) / 2 if matrix else 0
    for row in range(2):
        for col in range(2):
            value = matrix[row][col]
            ax.text(col, row, f"{value:,}", ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=12)
    # Reserve enough room for the full y-axis category labels.
    fig.subplots_adjust(left=0.34, right=0.91, bottom=0.16, top=0.88)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_precision_recall(y_true: pd.Series, scores: np.ndarray,
                          output_path: Path, model_name: str, pr_auc: float) -> None:
    """Save the selected model's final-test precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    prevalence = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(recall, precision, color="#7b2cbf", lw=2,
            label=f"{model_name} (AP = {pr_auc:.4f})")
    ax.axhline(prevalence, color="#666666", ls="--", lw=1,
               label=f"Fraud prevalence = {prevalence:.4f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
           title="Precision-recall curve on untouched test set")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Compare on validation data, refit once, then evaluate on the test set once."""
    train_path = args.train_path.resolve()
    test_path = args.test_path.resolve()
    preprocessor_path = args.preprocessor_path.resolve()
    output_paths = [args.model_path.resolve(), args.comparison_json.resolve(),
                    args.comparison_csv.resolve(), args.test_report.resolve(),
                    (args.figures_dir / "confusion_matrix_test.png").resolve(),
                    (args.figures_dir / "precision_recall_curve_test.png").resolve()]
    if len({str(path).lower() for path in output_paths}) != len(output_paths):
        raise ValueError("Each generated output must have a distinct path.")
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing model/report files. Review them and "
            "rerun with --overwrite if replacement is intended:\n" +
            "\n".join(f"  {path}" for path in existing)
        )
    for path in (train_path, test_path, preprocessor_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required project artifact not found: {path}")

    # The Phase 3 artifact is used for its schema and fitted scaler. The CSV
    # inputs already contain scaled Time/Amount, so this script does not rescale.
    preprocessing = joblib.load(preprocessor_path)
    if not isinstance(preprocessing, dict) or "feature_columns" not in preprocessing:
        raise ValueError("Phase 3 preprocessing artifact has an unexpected format.")
    expected_features = list(preprocessing["feature_columns"])
    scaled_columns = list(preprocessing.get("scaled_columns", []))
    if set(scaled_columns) != {"Time", "Amount"}:
        raise ValueError("Expected Phase 3 scaler to apply to Time and Amount only.")

    logging.info("Loading processed training data: %s", train_path)
    train_frame = pd.read_csv(train_path)
    validate_dataset(train_frame, expected_features, "Training")
    X = train_frame.loc[:, expected_features]
    y = train_frame[TARGET].astype("int64")
    if min(class_counts(y).values()) < 2:
        raise ValueError("Training data needs at least two rows from each class for validation.")

    # The test CSV is deliberately not read until model selection and refitting
    # are complete. Validation is drawn only from the training CSV.
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logging.info("Validation split: fit=%s, validation=%s; class counts=%s",
                 X_fit.shape, X_validation.shape, class_counts(y_validation))

    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=RANDOM_STATE,
            max_iter=2000, solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample", random_state=RANDOM_STATE,
            n_estimators=100, n_jobs=-1,
        ),
    }
    validation_results: dict[str, dict[str, Any]] = {}
    candidate_warnings: dict[str, list[str]] = {}
    for name, estimator in candidates.items():
        logging.info("Fitting %s on training-only fit partition", name)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator.fit(X_fit, y_fit)
        candidate_warnings[name] = [str(item.message) for item in captured]
        for message in candidate_warnings[name]:
            logging.warning("%s warning: %s", name, message)
        if not hasattr(estimator, "predict_proba"):
            raise RuntimeError(f"{name} does not support probability scoring.")
        scores = estimator.predict_proba(X_validation)[:, list(estimator.classes_).index(1)]
        validation_results[name] = {
            "metrics": calculate_metrics(y_validation, scores),
            "validation_rows": int(len(y_validation)),
            "validation_class_counts": class_counts(y_validation),
            "warnings": candidate_warnings[name],
        }
        logging.info("%s validation metrics: %s", name, validation_results[name]["metrics"])

    # PR-AUC is primary; fraud recall and precision break exact PR-AUC ties.
    selected_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["metrics"]["pr_auc"],
            validation_results[name]["metrics"]["recall"],
            validation_results[name]["metrics"]["precision"],
        ),
    )
    selected_estimator = candidates[selected_name]
    selection_reason = (
        "Highest validation PR-AUC (average precision); fraud recall and precision "
        "were used as tie-breakers. The test set was not used for selection."
    )
    logging.info("Selected model: %s", selected_name)

    # Refit the selected model on the complete processed training dataset.
    logging.info("Refitting %s on all %s training rows", selected_name, len(y))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        selected_estimator.fit(X, y)
    final_fit_warnings = [str(item.message) for item in captured]
    for message in final_fit_warnings:
        logging.warning("Final fit warning: %s", message)

    # Only now load the untouched holdout. Produce one score vector and derive
    # every final metric and plot from that single set of predictions.
    logging.info("Loading untouched test data for final evaluation: %s", test_path)
    test_frame = pd.read_csv(test_path)
    validate_dataset(test_frame, expected_features, "Test")
    X_test = test_frame.loc[:, expected_features]
    y_test = test_frame[TARGET].astype("int64")
    test_scores = selected_estimator.predict_proba(X_test)[:, list(selected_estimator.classes_).index(1)]
    final_metrics = calculate_metrics(y_test, test_scores)

    # Save all artifacts. Include the Phase 3 scaler so later serving code can
    # transform raw Time/Amount values; estimator inputs here are preprocessed.
    model_artifact = {
        "estimator": selected_estimator,
        "model_name": selected_name,
        "feature_columns": expected_features,
        "target_column": TARGET,
        "phase3_scaler": preprocessing["scaler"],
        "scaled_columns": scaled_columns,
        "expects_phase3_preprocessed_features": True,
        "validation_metrics": validation_results[selected_name]["metrics"],
        "final_test_metrics": final_metrics,
        "random_state": RANDOM_STATE,
    }
    comparison_report = {
        "validation_setup": {
            "source": str(train_path),
            "validation_fraction_of_training_data": VALIDATION_SIZE,
            "random_state": RANDOM_STATE,
            "stratified_by": TARGET,
            "training_rows_before_validation_split": int(len(y)),
            "model_selection_used_test_data": False,
            "class_counts_in_full_training_data": class_counts(y),
        },
        "models": validation_results,
        "selected_model": selected_name,
        "selection_reason": selection_reason,
        "configuration": {
            "logistic_regression": {"class_weight": "balanced", "random_state": RANDOM_STATE,
                                     "max_iter": 2000, "solver": "lbfgs"},
            "random_forest": {"class_weight": "balanced_subsample", "random_state": RANDOM_STATE,
                              "n_estimators": 100, "n_jobs": -1},
            "decision_threshold_for_precision_recall_f1": THRESHOLD,
        },
    }
    final_report = {
        "model_name": selected_name,
        "test_data_source": str(test_path),
        "test_rows": int(len(y_test)),
        "test_class_counts": class_counts(y_test),
        "test_used_for_model_selection": False,
        "final_test_evaluations": 1,
        "metrics": final_metrics,
        "selection_reason": selection_reason,
        "final_refit_warnings": final_fit_warnings,
        "threshold_was_tuned_on_test": False,
    }

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, output_paths[0])
    output_paths[1].write_text(json.dumps(comparison_report, indent=2, allow_nan=False), encoding="utf-8")
    comparison_rows = []
    for name, result in validation_results.items():
        row = {"model": name, **result["metrics"]}
        row["validation_rows"] = result["validation_rows"]
        row["validation_class_counts"] = json.dumps(result["validation_class_counts"])
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(output_paths[2], index=False)
    output_paths[3].write_text(json.dumps(final_report, indent=2, allow_nan=False), encoding="utf-8")
    plot_confusion_matrix(final_metrics["confusion_matrix_labels_0_1"], output_paths[4], selected_name)
    plot_precision_recall(y_test, test_scores, output_paths[5], selected_name, final_metrics["pr_auc"])

    # Log this completed training workflow after its existing reports are saved.
    # The shared helper logs validation metrics; final test metrics remain a
    # JSON artifact only and are never added as MLflow metric values.
    from mlflow_tracking import log_experiment_runs

    mlflow_summary = log_experiment_runs(
        comparison_report=comparison_report,
        model_artifact=model_artifact,
        report_artifact_paths=[output_paths[1], output_paths[2], output_paths[3]],
        chart_artifact_paths=[output_paths[4], output_paths[5]],
        model_source_path=output_paths[0],
        preprocessing_source_path=preprocessor_path,
    )
    mlflow_summary_path = PROJECT_ROOT / "reports" / "mlflow_tracking_summary.json"
    mlflow_summary_path.write_text(json.dumps(mlflow_summary, indent=2), encoding="utf-8")
    logging.info("Recorded MLflow experiment runs: %s", mlflow_summary_path)

    # Smoke-check the serialized model on a synthetic processed row, avoiding
    # any second prediction/evaluation on the real test set.
    loaded = joblib.load(output_paths[0])
    smoke_row = pd.DataFrame([{column: 0.0 for column in expected_features}], columns=expected_features)
    smoke_scores = loaded["estimator"].predict_proba(smoke_row)
    if smoke_scores.shape != (1, 2) or not np.isfinite(smoke_scores).all():
        raise RuntimeError("Saved model failed its synthetic prediction smoke check.")
    logging.info("Saved model reload/prediction smoke check passed: %s", smoke_scores[0].tolist())

    print("Validation metrics:")
    for name, result in validation_results.items():
        print(f"  {name}: {json.dumps(result['metrics'])}")
    print(f"Selected model: {selected_name}")
    print(f"Selection reason: {selection_reason}")
    print(f"Final test metrics: {json.dumps(final_metrics)}")
    print(f"Saved-model smoke prediction probabilities: {smoke_scores[0].tolist()}")
    print("Saved outputs:")
    for path in output_paths:
        print(f"  {path}")
    return {"comparison": comparison_report, "final_test": final_report}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        run_training(args)
    except (FileNotFoundError, FileExistsError, ValueError, OSError, KeyError, RuntimeError) as exc:
        logging.error("Training/evaluation failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
