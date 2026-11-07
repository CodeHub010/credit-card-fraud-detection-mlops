"""Prepare reproducible train/test data for the fraud detection project.

Run from the project root:
    .venv\\Scripts\\python.exe src\\preprocessing.py

Generated files are protected from accidental replacement. To intentionally
regenerate them after reviewing the existing artifacts, pass --overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEFAULT_TEST = PROJECT_ROOT / "data" / "processed" / "test.csv"
DEFAULT_PREPROCESSOR = PROJECT_ROOT / "models" / "preprocessing.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "preprocessing" / "preprocessing_report.json"
TARGET = "Class"
SCALED_COLUMNS = ["Time", "Amount"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def parse_args() -> argparse.Namespace:
    """Parse paths and require an explicit flag before replacing outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA,
                        help="Input CSV (default: data/raw/creditcard.csv)")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN,
                        help="Processed training CSV output")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST,
                        help="Processed testing CSV output")
    parser.add_argument("--preprocessor-path", type=Path, default=DEFAULT_PREPROCESSOR,
                        help="Fitted preprocessing bundle output")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT,
                        help="Preprocessing report JSON output")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing generated outputs")
    return parser.parse_args()


def distribution(series: pd.Series) -> dict[str, dict[str, float | int]]:
    """Return counts and percentages for legitimate (0) and fraud (1)."""
    counts = series.value_counts().reindex([0, 1], fill_value=0).sort_index()
    total = int(counts.sum())
    return {
        str(int(label)): {
            "label": "legitimate" if int(label) == 0 else "fraud",
            "count": int(count),
            "percentage": (float(count / total * 100) if total else 0.0),
        }
        for label, count in counts.items()
    }


def missing_by_column(frame: pd.DataFrame) -> dict[str, int]:
    """Count null cells in each column for the report."""
    return {str(column): int(count) for column, count in frame.isna().sum().items()}


def run_preprocessing(args: argparse.Namespace) -> dict[str, Any]:
    """Deduplicate, split, scale selected columns, and save reproducible outputs."""
    data_path = args.data_path.resolve()
    output_paths = [args.train_path.resolve(), args.test_path.resolve(),
                    args.preprocessor_path.resolve(), args.report_path.resolve()]
    if len({str(path).lower() for path in output_paths}) != len(output_paths):
        raise ValueError("Each output must have a distinct path.")
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        paths = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing generated files. Review them, then "
            "rerun with --overwrite if replacement is intended:\n" + paths
        )
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if data_path in output_paths:
        raise ValueError("Input dataset path cannot also be an output path.")

    # Load the source unchanged and validate the columns and target labels.
    original = pd.read_csv(data_path)
    original_rows = len(original)
    required = {TARGET, *SCALED_COLUMNS}
    missing_columns = sorted(required.difference(original.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if original.empty:
        raise ValueError("Dataset has no rows.")
    if original[TARGET].isna().any():
        raise ValueError("Target Class contains missing values; cannot stratify safely.")
    if not pd.api.types.is_numeric_dtype(original[TARGET]):
        raise ValueError("Target Class must be numeric with values 0 and 1.")
    if set(original[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected both target classes with values 0 (legitimate) and 1 (fraud).")

    # Count excess exact-duplicate rows before dropping any, grouped by label.
    duplicate_mask = original.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    duplicate_class_counts = (
        original.loc[duplicate_mask, TARGET].value_counts()
        .reindex([0, 1], fill_value=0).sort_index()
    )
    duplicate_group_mask = original.duplicated(keep=False)
    duplicate_group_rows = int(duplicate_group_mask.sum())
    deduplicated = original.drop_duplicates(keep="first").reset_index(drop=True)
    rows_remaining = len(deduplicated)

    # Report missing data before processing; fail rather than silently impute.
    original_missing = missing_by_column(original)
    if sum(original_missing.values()) > 0:
        raise ValueError(
            "Input contains missing cells. No imputation was requested; "
            "inspect missing_values_by_column before continuing."
        )
    deduplicated_missing = missing_by_column(deduplicated)

    # Keep the original features intact except Time and Amount, then split with
    # stratification. The index sets provide an explicit train/test leakage check.
    feature_columns = [column for column in deduplicated.columns if column != TARGET]
    X = deduplicated.loc[:, feature_columns]
    y = deduplicated[TARGET].astype("int64")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    if not set(X_train.index).isdisjoint(set(X_test.index)):
        raise RuntimeError("Train/test row overlap detected.")

    # Fit only on training rows. The PCA-derived V1-V28 columns pass through
    # unchanged; test data are transformed with the training-fitted scaler.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.loc[:, SCALED_COLUMNS] = scaler.fit_transform(X_train[SCALED_COLUMNS])
    X_test.loc[:, SCALED_COLUMNS] = scaler.transform(X_test[SCALED_COLUMNS])

    # Balanced class weights are metadata for later training, not resampling.
    training_classes = np.array([0, 1], dtype="int64")
    weight_values = compute_class_weight(
        class_weight="balanced", classes=training_classes, y=y_train.to_numpy()
    )
    class_weights = {str(cls): float(weight) for cls, weight in zip(training_classes, weight_values)}

    # Merge target back into each output file for convenient later use.
    train_data = X_train.assign(**{TARGET: y_train}).loc[:, feature_columns + [TARGET]]
    test_data = X_test.assign(**{TARGET: y_test}).loc[:, feature_columns + [TARGET]]
    train_missing = missing_by_column(train_data)
    test_missing = missing_by_column(test_data)
    if sum(train_missing.values()) or sum(test_missing.values()):
        raise RuntimeError("Missing values appeared in processed train/test data.")

    # Save the scaler and metadata as a reusable preprocessing bundle.
    bundle = {
        "scaler": scaler,
        "scaled_columns": list(SCALED_COLUMNS),
        "feature_columns": list(feature_columns),
        "target_column": TARGET,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "class_weights_for_future_training": class_weights,
        "scaler_fit_rows": int(len(X_train)),
    }

    report: dict[str, Any] = {
        "source_file": str(data_path),
        "original_row_count": int(original_rows),
        "duplicate_count_removed": duplicate_count,
        "duplicate_class_distribution_removed": {
            str(int(cls)): int(count) for cls, count in duplicate_class_counts.items()
        },
        "rows_in_duplicate_groups_including_first_occurrences": duplicate_group_rows,
        "rows_remaining_after_deduplication": int(rows_remaining),
        "class_distribution_before_deduplication": distribution(original[TARGET]),
        "class_distribution_after_deduplication": distribution(y),
        "train_test_split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified_by": TARGET,
            "train_dimensions_features_only": [int(X_train.shape[0]), int(X_train.shape[1])],
            "test_dimensions_features_only": [int(X_test.shape[0]), int(X_test.shape[1])],
            "train_dimensions_including_target": [int(train_data.shape[0]), int(train_data.shape[1])],
            "test_dimensions_including_target": [int(test_data.shape[0]), int(test_data.shape[1])],
            "train_class_distribution": distribution(y_train),
            "test_class_distribution": distribution(y_test),
            "train_test_indices_disjoint": True,
        },
        "scaler": {
            "type": "sklearn.preprocessing.StandardScaler",
            "columns": list(SCALED_COLUMNS),
            "fit_on": "training data only",
            "fit_rows": int(scaler.n_samples_seen_),
            "mean": {column: float(value) for column, value in zip(SCALED_COLUMNS, scaler.mean_)},
            "scale": {column: float(value) for column, value in zip(SCALED_COLUMNS, scaler.scale_)},
            "variance": {column: float(value) for column, value in zip(SCALED_COLUMNS, scaler.var_)},
            "PCA_columns_scaled": False,
        },
        "class_weights_for_future_training": class_weights,
        "resampling_applied": False,
        "missing_value_checks": {
            "original_by_column": original_missing,
            "deduplicated_by_column": deduplicated_missing,
            "train_by_column": train_missing,
            "test_by_column": test_missing,
            "passed": True,
        },
        "outputs": {
            "train_csv": str(output_paths[0]),
            "test_csv": str(output_paths[1]),
            "preprocessing_bundle": str(output_paths[2]),
            "report_json": str(output_paths[3]),
        },
    }

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(output_paths[0], index=False)
    test_data.to_csv(output_paths[1], index=False)
    joblib.dump(bundle, output_paths[2])
    output_paths[3].write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Original rows: {original_rows:,}")
    print(f"Exact duplicate rows removed: {duplicate_count:,}")
    print("Duplicate rows removed by class: " + ", ".join(
        f"Class {cls}: {count:,}" for cls, count in duplicate_class_counts.items()
    ))
    print(f"Rows remaining: {rows_remaining:,}")
    print(f"Training shape (including target): {train_data.shape}")
    print(f"Testing shape (including target): {test_data.shape}")
    print("Train class distribution: " + json.dumps(report["train_test_split"]["train_class_distribution"]))
    print("Test class distribution: " + json.dumps(report["train_test_split"]["test_class_distribution"]))
    print(f"Scaler fit rows: {scaler.n_samples_seen_:,}; columns: {', '.join(SCALED_COLUMNS)}")
    print(f"Class weights prepared (no resampling): {class_weights}")
    print("Missing-value checks: passed")
    print("Saved outputs:")
    for path in output_paths:
        print(f"  {path}")
    return report


def main() -> int:
    args = parse_args()
    try:
        run_preprocessing(args)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
