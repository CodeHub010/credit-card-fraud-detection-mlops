"""Explore the Kaggle credit card fraud dataset without changing its source file.

Run from any directory with the project environment:
    .venv\\Scripts\\python.exe src\\eda.py

Optional paths are useful for checks or alternate copies of the dataset:
    .venv\\Scripts\\python.exe src\\eda.py --data-path data\\raw\\creditcard.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DEFAULT_REPORTS = PROJECT_ROOT / "reports" / "eda"
DEFAULT_FIGURES = PROJECT_ROOT / "reports" / "figures"
KAGGLE_PAGE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


def parse_args() -> argparse.Namespace:
    """Read optional input and output paths; project paths are the defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA,
                        help="CSV input (default: data/raw/creditcard.csv)")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS,
                        help="EDA tables and JSON output directory")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES,
                        help="PNG chart output directory")
    return parser.parse_args()


def make_charts(df: pd.DataFrame, class_counts: pd.Series,
                class_percentages: pd.Series, figures_dir: Path) -> list[Path]:
    """Save class balance and amount distribution charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = {0: "Legitimate", 1: "Fraud"}
    ordered = class_counts.reindex([0, 1], fill_value=0)
    percentages = class_percentages.reindex([0, 1], fill_value=0)

    # A log axis keeps the rare fraud class visible beside the much larger
    # legitimate class. Bars start above zero because zero is undefined on log.
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = 1
    bars = ax.bar([labels[k] for k in ordered.index], ordered.values,
                  bottom=baseline, color=["#3977a8", "#d94f4f"])
    ax.set_yscale("log")
    ax.set_ylim(baseline, baseline + max(int(ordered.max()), 1) * 3)
    ax.set_title("Transaction class distribution", pad=14)
    ax.set_ylabel("Number of transactions (log scale)")
    ax.grid(axis="y", alpha=0.25)
    for bar, cls in zip(bars, ordered.index):
        ax.annotate(f"{ordered.loc[cls]:,}\n{percentages.loc[cls]:.3f}%",
                    (bar.get_x() + bar.get_width() / 2,
                     bar.get_y() + bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom")
    fig.tight_layout()
    class_chart = figures_dir / "class_distribution.png"
    fig.savefig(class_chart, dpi=150)
    plt.close(fig)

    # log1p compresses the long-tailed amounts for display only; df stays unchanged.
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls, color in [(0, "#3977a8"), (1, "#d94f4f")]:
        amounts = np.log1p(df.loc[df["Class"] == cls, "Amount"].astype(float))
        ax.hist(amounts, bins=60, alpha=0.55, density=True,
                label=f"{labels[cls]} (n={len(amounts):,})", color=color)
    ax.set_title("Transaction amount distribution by class")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    amount_chart = figures_dir / "transaction_amount_distribution.png"
    fig.savefig(amount_chart, dpi=150)
    plt.close(fig)
    return [class_chart, amount_chart]


def run_eda(data_path: Path, report_dir: Path,
            figures_dir: Path) -> dict[str, object]:
    """Load, summarize, visualize, and save results without editing the CSV."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n\n"
            "Download it from Kaggle's official dataset page:\n"
            f"  {KAGGLE_PAGE}\n"
            "Sign in if Kaggle asks, click Download, extract creditcard.csv, "
            "and place it at data/raw/creditcard.csv. Then rerun this script."
        )

    # Read-only input. Pandas operations below create summaries and charts only.
    df = pd.read_csv(data_path)
    if "Class" not in df.columns or "Amount" not in df.columns:
        raise ValueError("Expected Kaggle columns 'Class' and 'Amount' were not found.")
    if df.empty:
        raise ValueError("The CSV contains no transaction rows.")
    if not pd.api.types.is_numeric_dtype(df["Class"]):
        raise ValueError("The 'Class' column must be numeric (0=legitimate, 1=fraud).")

    # Print the requested inspection details to the terminal.
    print(f"Dataset: {data_path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    if df.shape[0] != 284_807:
        print(f"NOTE: expected 284,807 rows for the Kaggle dataset; found {df.shape[0]:,}.")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nColumn names:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes.to_string())

    missing = df.isna().sum()
    duplicates = int(df.duplicated().sum())
    class_counts = df["Class"].value_counts().reindex([0, 1], fill_value=0).sort_index()
    if int(class_counts.sum()) != len(df):
        raise ValueError("Class column contains values other than 0 and 1.")
    class_percentages = class_counts / len(df) * 100
    class_distribution = pd.DataFrame({
        "class": [0, 1],
        "label": ["legitimate", "fraud"],
        "count": [int(class_counts.loc[0]), int(class_counts.loc[1])],
        "percentage": [float(class_percentages.loc[0]), float(class_percentages.loc[1])],
    })

    print("\nMissing values by column:")
    print(missing.to_string())
    print(f"\nTotal missing cells: {int(missing.sum()):,}")
    print(f"Duplicate rows: {duplicates:,}")
    print("\nClass distribution (0=legitimate, 1=fraud):")
    print(class_distribution.to_string(index=False, formatters={"percentage": "{:.6f}%".format}))

    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    distribution_csv = report_dir / "class_distribution.csv"
    class_distribution.to_csv(distribution_csv, index=False)
    charts = make_charts(df, class_counts, class_percentages, figures_dir)

    amount_stats = df.groupby("Class")["Amount"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    result: dict[str, object] = {
        "source_file": str(data_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values_by_column": {str(col): int(value) for col, value in missing.items()},
        "total_missing_cells": int(missing.sum()),
        "duplicate_rows": duplicates,
        "class_distribution": class_distribution.to_dict(orient="records"),
        "amount_stats_by_class": {
            str(class_id): {str(key): (None if pd.isna(value) else float(value))
                            for key, value in row.items()}
            for class_id, row in amount_stats.iterrows()
        },
        "charts": [str(path) for path in charts],
    }
    summary_path = report_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    print("\nAmount summary by class:")
    print(amount_stats.to_string())
    print("\nSaved EDA outputs:")
    print(f"  {summary_path}")
    print(f"  {distribution_csv}")
    for chart in charts:
        print(f"  {chart}")
    return result


def main() -> int:
    args = parse_args()
    try:
        run_eda(args.data_path, args.report_dir, args.figures_dir)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
