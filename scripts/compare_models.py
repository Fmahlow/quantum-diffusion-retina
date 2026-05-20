"""Compare classical DDPM and hybrid quantum DDPM (QDiffusion) metrics.

Loads metric CSVs from evaluate_ddpm_task_metrics.py and
evaluate_qdiffusion_task_metrics.py, prints a side-by-side comparison table,
and saves a bar-chart figure per metric grouped by class.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py \\
        --ddpm-csv outputs/ddpm_task_metrics.csv \\
        --qdiffusion-csv outputs/qdiffusion_task_metrics.csv \\
        --output-dir outputs/comparison
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_synthesis.data import label_names


METRIC_COLS = [
    "Feature_FID",
    "Feature_MMD_RBF",
    "Fake_Target_Accuracy",
    "Fake_Target_Confidence",
    "Fake_Prediction_MAE",
    "Real_Target_Accuracy",
    "Fake_To_Real_NN_Distance",
    "Fake_Diversity_Ratio",
]

LOWER_BETTER = {"Feature_FID", "Feature_MMD_RBF", "Fake_Prediction_MAE", "Fake_To_Real_NN_Distance"}
HIGHER_BETTER = {"Fake_Target_Accuracy", "Fake_Target_Confidence", "Real_Target_Accuracy", "Fake_Diversity_Ratio"}

MODEL_COLORS = {
    "DDPM": "#2196F3",
    "QDiffusion": "#FF9800",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DDPM and QDiffusion metric CSVs.")
    parser.add_argument("--ddpm-csv", type=Path, default=Path("outputs/ddpm_task_metrics.csv"))
    parser.add_argument("--qdiffusion-csv", type=Path, default=Path("outputs/qdiffusion_task_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARNING] Not found: {path}")
        return None
    return pd.read_csv(path)


def print_comparison_table(combined: pd.DataFrame) -> None:
    names = label_names()
    available = [c for c in METRIC_COLS if c in combined.columns]
    models = sorted(combined["Model"].unique())

    print("\n" + "=" * 90)
    print("Classical DDPM vs Hybrid Quantum DDPM (QDiffusion) — Task Metric Comparison")
    print("=" * 90)

    for metric in available:
        direction = "↓ lower better" if metric in LOWER_BETTER else "↑ higher better"
        print(f"\n{metric}  ({direction})")
        pivot = combined.pivot_table(
            index="Class_ID", columns="Model", values=metric, aggfunc="mean"
        )
        if "DDPM" in pivot.columns and "QDiffusion" in pivot.columns:
            pivot["Δ QDiff−DDPM"] = pivot["QDiffusion"] - pivot["DDPM"]
        for class_id, row in pivot.iterrows():
            class_name = names.get(int(class_id), f"class{class_id}")
            parts = [f"  class {int(class_id)} ({class_name:24s})"]
            for col in pivot.columns:
                val = row[col]
                if "Δ" in col:
                    parts.append(f"  {col}: {val:+.4f}")
                else:
                    parts.append(f"  {col}: {val:.4f}")
            print("".join(parts))

    print("\n" + "-" * 90)
    print("Mean across all classes:")
    for metric in available:
        parts = [f"  {metric}"]
        for model in models:
            vals = combined[combined["Model"] == model][metric].dropna()
            if not vals.empty:
                parts.append(f"  {model}={vals.mean():.4f}")
        print("".join(parts))


def plot_comparison(combined: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = label_names()
    models = sorted(combined["Model"].unique())
    classes = sorted(combined["Class_ID"].dropna().unique().astype(int))
    class_labels = [f"c{c}\n{names.get(c, '?')[:6]}" for c in classes]
    available = [c for c in METRIC_COLS if c in combined.columns]

    n_cols = 2
    n_rows = math.ceil(len(available) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, metric in enumerate(available):
        ax = axes[ax_idx]
        x = list(range(len(classes)))
        width = 0.8 / len(models)

        for model_idx, model in enumerate(models):
            subset = combined[combined["Model"] == model].set_index("Class_ID")
            values = [
                float(subset.loc[c, metric]) if c in subset.index else float("nan")
                for c in classes
            ]
            offsets = [xi + model_idx * width - 0.4 + width / 2 for xi in x]
            color = MODEL_COLORS.get(model, f"C{model_idx}")
            ax.bar(offsets, values, width=width * 0.9, label=model, color=color)

        direction = "↓" if metric in LOWER_BETTER else "↑"
        ax.set_title(f"{metric} {direction}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for ax_idx in range(len(available), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        "Classical DDPM vs Hybrid Quantum DDPM — RetinaMNIST Task Metrics",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = output_dir / "comparison_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {plot_path}")


def main() -> None:
    args = parse_args()

    frames = []
    ddpm_df = load_if_exists(args.ddpm_csv)
    qdiff_df = load_if_exists(args.qdiffusion_csv)

    if ddpm_df is not None:
        ddpm_df["Model"] = "DDPM"
        frames.append(ddpm_df)
    if qdiff_df is not None:
        qdiff_df["Model"] = "QDiffusion"
        frames.append(qdiff_df)

    if not frames:
        print("No CSVs found. Run evaluate_ddpm_task_metrics.py and/or evaluate_qdiffusion_task_metrics.py first.")
        return

    combined = pd.concat(frames, ignore_index=True)
    if "Class_ID" in combined.columns:
        combined["Class_ID"] = combined["Class_ID"].astype(int)

    print_comparison_table(combined)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "combined_task_metrics.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined CSV saved to {combined_path}")

    if not args.no_plot:
        try:
            plot_comparison(combined, args.output_dir)
        except Exception as exc:
            print(f"[WARNING] Plot failed: {exc}")


if __name__ == "__main__":
    main()
