"""Log Phase 4 results to MLflow without retraining or evaluating test data.

Run from the project root:
    .venv\\Scripts\\python.exe src\\track_mlflow.py

The script reads validation metrics and the selected saved model. The existing
final-test JSON is uploaded as a file artifact only; its values are not read.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib

from mlflow_tracking import EXPERIMENT_NAME, log_experiment_runs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_JSON = PROJECT_ROOT / "reports" / "model_comparison.json"
DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "reports" / "model_comparison.csv"
DEFAULT_FINAL_TEST_JSON = PROJECT_ROOT / "reports" / "final_test_metrics.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"
DEFAULT_PREPROCESSOR = PROJECT_ROOT / "models" / "preprocessing.joblib"
DEFAULT_CONFUSION_MATRIX = PROJECT_ROOT / "reports" / "figures" / "confusion_matrix_test.png"
DEFAULT_PR_CURVE = PROJECT_ROOT / "reports" / "figures" / "precision_recall_curve_test.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "mlflow_tracking_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", type=Path, default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--final-test-json", type=Path, default=DEFAULT_FINAL_TEST_JSON)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--preprocessor-path", type=Path, default=DEFAULT_PREPROCESSOR)
    parser.add_argument("--confusion-matrix", type=Path, default=DEFAULT_CONFUSION_MATRIX)
    parser.add_argument("--pr-curve", type=Path, default=DEFAULT_PR_CURVE)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--overwrite-summary", action="store_true",
                        help="Replace the local run-ID summary after creating new runs")
    return parser.parse_args()


def run_tracking(args: argparse.Namespace) -> dict[str, Any]:
    """Log two validation runs and the already-selected Random Forest model."""
    summary_path = args.summary_path.resolve()
    if summary_path.exists() and not args.overwrite_summary:
        raise FileExistsError(
            f"Run summary already exists: {summary_path}. Use --overwrite-summary "
            "to replace the summary (new MLflow runs will still be created)."
        )

    required_files = [
        args.comparison_json, args.comparison_csv, args.final_test_json,
        args.model_path, args.preprocessor_path, args.confusion_matrix, args.pr_curve,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing existing Phase 4 artifacts:\n  " + "\n  ".join(missing))

    # Read validation comparison and the saved model only. The test report is
    # deliberately passed to MLflow as a file path without parsing its metrics.
    comparison = json.loads(args.comparison_json.read_text(encoding="utf-8"))
    saved_model = joblib.load(args.model_path)
    if comparison.get("selected_model") != "Random Forest":
        raise ValueError("Existing report does not select Random Forest; refusing to change selection.")
    if saved_model.get("model_name") != comparison["selected_model"]:
        raise ValueError("Existing selected model artifact does not match the saved comparison report.")
    if set(comparison.get("models", {})) != {"Logistic Regression", "Random Forest"}:
        raise ValueError("Expected validation results for Logistic Regression and Random Forest.")

    logging.info("Using existing experiment results; no model fitting or test evaluation will run.")
    summary = log_experiment_runs(
        comparison_report=comparison,
        model_artifact=saved_model,
        report_artifact_paths=[
            args.comparison_json.resolve(),
            args.comparison_csv.resolve(),
            args.final_test_json.resolve(),
        ],
        chart_artifact_paths=[
            args.confusion_matrix.resolve(),
            args.pr_curve.resolve(),
        ],
        model_source_path=args.model_path.resolve(),
        preprocessing_source_path=args.preprocessor_path.resolve(),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Experiment: {summary['experiment_name']} (ID {summary['experiment_id']})")
    print(f"Tracking store: {summary['tracking_uri']}")
    for model_name, info in summary["runs"].items():
        print(f"{model_name}: run_id={info['run_id']}; selected={info['selected']}")
        metrics = comparison["models"][model_name]["metrics"]
        print("  validation metrics: " + json.dumps({
            key: metrics[key] for key in ("precision", "recall", "f1_score", "pr_auc", "roc_auc")
        }))
        if info["model_uri"]:
            print(f"  MLflow model URI: {info['model_uri']}")
    print(f"Selection rationale: {summary['selection_rationale']}")
    print(f"Run summary: {summary_path}")
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_tracking(parse_args())
    except (FileNotFoundError, FileExistsError, ValueError, OSError, KeyError, RuntimeError) as exc:
        logging.error("MLflow tracking failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
