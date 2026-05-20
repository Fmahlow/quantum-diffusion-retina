from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


METRIC_SPECS = {
    "FID": ("FID lower is better", "lower"),
    "KID_Mean": ("KID lower is better", "lower"),
    "IS_Mean": ("Inception Score higher is better", "higher"),
    "Feature_FID": ("Feature FID lower is better", "lower"),
    "Feature_MMD_RBF": ("Feature MMD lower is better", "lower"),
    "Fake_Target_Accuracy": ("Generated target-class accuracy higher is better", "higher"),
    "Fake_Target_Confidence": ("Generated target-class confidence higher is better", "higher"),
    "Fake_Prediction_MAE": ("Generated ordinal MAE lower is better", "lower"),
    "Real_Target_Accuracy": ("Real target-class accuracy higher is better", "higher"),
    "Fake_To_Real_NN_Distance": ("Fake-to-real nearest-neighbor feature distance", "context"),
    "Real_Within_NN_Distance": ("Real within-class nearest-neighbor feature distance", "context"),
    "Fake_Within_NN_Distance": ("Fake within-class nearest-neighbor feature distance", "context"),
    "Fake_Diversity_Ratio": ("Fake/real within-class diversity ratio close to 1 is better", "target"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready tables and plots from metric CSV files.")
    parser.add_argument("--metrics-csv", type=Path, default=Path("outputs/ddpm_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ddpm"))
    parser.add_argument("--title", default="RetinaMNIST synthetic image metrics")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metrics CSV: {sorted(missing)}")


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    for column in ("FID", "IS_Mean", "IS_Std"):
        if column in table:
            table[column] = table[column].astype(float).round(4)
    return table


def to_markdown_table(df: pd.DataFrame) -> str:
    rows = []
    columns = [str(column) for column in df.columns]
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(rows) + "\n"


def save_tables(df: pd.DataFrame, output_dir: Path) -> None:
    table = format_table(df)
    table.to_csv(output_dir / "metrics_table.csv", index=False)
    (output_dir / "metrics_table.md").write_text(to_markdown_table(table), encoding="utf-8")
    (output_dir / "metrics_table.tex").write_text(table.to_latex(index=False, escape=True), encoding="utf-8")


def x_labels(df: pd.DataFrame) -> list[str]:
    if "Model" in df.columns and df["Model"].nunique() > 1:
        return [f"{model}\n{label}" for model, label in zip(df["Model"], df["Generator_Label"])]
    return [str(label) for label in df["Generator_Label"]]


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path, title: str) -> None:
    ylabel, _ = METRIC_SPECS.get(metric, (metric, "context"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x_labels(df), df[metric], color="#3b6ea8")
    ax.set_title(f"{title}: {metric}")
    ax.set_xlabel("Class")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric.lower()}_by_class.png", dpi=200)
    plt.close(fig)


def plot_inception_score(df: pd.DataFrame, output_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    yerr = df["IS_Std"] if "IS_Std" in df else None
    ax.bar(x_labels(df), df["IS_Mean"], yerr=yerr, color="#4f8f68", capsize=4)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Inception Score higher is better")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "inception_score_by_class.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics_csv)
    require_columns(df, {"Generator_Label"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["Model", "Generator_Label"] if "Model" in df else ["Generator_Label"])

    save_tables(df, args.output_dir)
    for metric in METRIC_SPECS:
        if metric not in df.columns:
            continue
        if metric == "IS_Mean" and "IS_Std" in df.columns:
            plot_inception_score(df, args.output_dir, f"{args.title}: IS")
        else:
            plot_metric(df, metric, args.output_dir, args.title)

    print(f"Wrote tables and plots to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"plot_metrics.py failed: {exc}", file=sys.stderr)
        raise
