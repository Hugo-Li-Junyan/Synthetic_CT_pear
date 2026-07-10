import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RUN_NAME_RE = re.compile(
    r"^seed(?P<seed>\d+)_binary_(?P<source>syn|nosyn)_(?P<proportion>0\.[1-5])$"
)
PROPORTION_ORDER = [0.1, 0.2, 0.3, 0.4, 0.5]
METRIC_ORDER = ["accuracy", "precision", "recall"]


def find_latest_file(run_dir, filename):
    candidates = sorted(
        run_dir.rglob(filename),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_accuracy(metrics_path, accuracy_key):
    with open(metrics_path, "r") as file:
        metrics = json.load(file)

    for key in (accuracy_key, "accuracy", "rounded_accuracy", "test_accuracy"):
        if key in metrics:
            return float(metrics[key])

    raise KeyError(
        f"{metrics_path} does not contain {accuracy_key}, accuracy, "
        "rounded_accuracy, or test_accuracy"
    )


def read_precision_recall(confusion_path, positive_label):
    matrix = pd.read_csv(confusion_path, index_col=0)
    matrix.index = [str(label) for label in matrix.index]
    matrix.columns = [str(label) for label in matrix.columns]
    positive_label = str(positive_label)

    if positive_label not in matrix.index or positive_label not in matrix.columns:
        raise KeyError(f"{confusion_path} does not contain positive label {positive_label}")

    true_positive = float(matrix.loc[positive_label, positive_label])
    false_positive = float(matrix[positive_label].sum() - true_positive)
    false_negative = float(matrix.loc[positive_label].sum() - true_positive)

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    return precision, recall


def collect_results(results_dir, accuracy_key, positive_label):
    rows = []
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        match = RUN_NAME_RE.match(run_dir.name)
        if not match:
            continue

        metrics_path = find_latest_file(run_dir, "clf_3d_test_metrics.json")
        if metrics_path is None:
            print(f"Skipping {run_dir.name}: no clf_3d_test_metrics.json found")
            continue

        confusion_path = find_latest_file(run_dir, "clf_3d_test_confusion_matrix.csv")
        if confusion_path is None:
            print(f"Skipping {run_dir.name}: no clf_3d_test_confusion_matrix.csv found")
            continue

        precision, recall = read_precision_recall(confusion_path, positive_label)
        rows.append(
            {
                "seed": int(match.group("seed")),
                "source": match.group("source"),
                "proportion": float(match.group("proportion")),
                "accuracy": read_accuracy(metrics_path, accuracy_key),
                "precision": precision,
                "recall": recall,
                "metrics_path": str(metrics_path),
                "confusion_path": str(confusion_path),
            }
        )

    if not rows:
        raise ValueError(
            f"No matching runs found in {results_dir}. Expected folder names like "
            "seed0_binary_syn_0.1 or seed369_binary_nosyn_0.5."
        )
    return pd.DataFrame(rows)


def summarize_metrics(df):
    long_df = df.melt(
        id_vars=["seed", "source", "proportion"],
        value_vars=METRIC_ORDER,
        var_name="metric",
        value_name="value",
    )
    return (
        long_df.groupby(["metric", "proportion", "source"], as_index=False)
        .agg(
            mean_value=("value", "mean"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            n=("value", "size"),
        )
        .sort_values(["metric", "proportion", "source"])
    )


def add_min_max_markers(ax, metric_summary, y_limit):
    bars = [patch for patch in ax.patches if patch.get_width() > 0]
    indexed_summary = metric_summary.set_index(["proportion", "source"])
    ordered_rows = [
        indexed_summary.loc[(proportion, source)]
        for source in ["nosyn", "syn"]
        for proportion in PROPORTION_ORDER
        if (proportion, source) in indexed_summary.index
    ]

    for bar, row in zip(bars, ordered_rows):
        center = bar.get_x() + bar.get_width() / 2
        y_min = row["min_value"]
        y_max = row["max_value"]
        ax.vlines(center, y_min, y_max, color="black", linewidth=1.4, zorder=4)
        ax.scatter([center, center], [y_min, y_max], color="black", s=24, zorder=5)


def plot_metrics(df, output_path):
    summary = summarize_metrics(df)
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(18, 6), sharey=True)
    hue_order = ["nosyn", "syn"]
    palette = {"nosyn": "#0072B2", "syn": "#E69F00"}
    y_limit = min(1.05, max(summary["max_value"].max() + 0.08, 0.1))

    for ax, metric in zip(axes, METRIC_ORDER):
        metric_summary = summary[summary["metric"] == metric]
        sns.barplot(
            data=metric_summary,
            x="proportion",
            y="mean_value",
            hue="source",
            order=PROPORTION_ORDER,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            errorbar=None,
        )

        add_min_max_markers(ax, metric_summary, y_limit)
        ax.set_xlabel("Data proportion")
        ax.set_ylabel("Average" if metric == METRIC_ORDER[0] else "")
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, y_limit)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, _ = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        ["Without synthetic", "Synthetic"],
        title="Training data",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot binary classification accuracy, precision, and recall for seed*_binary_{syn,nosyn}_0.x runs."
    )
    parser.add_argument("results_dir", type=Path, help="folder containing seed*_binary_*_* result folders")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("binary_seed_metrics.png"),
        help="path for the output plot",
    )
    parser.add_argument(
        "--accuracy_key",
        type=str,
        default="accuracy",
        help="metric key to read from clf_3d_test_metrics.json",
    )
    parser.add_argument(
        "--positive_label",
        type=str,
        default="1",
        help="positive class label used to calculate precision and recall from the confusion matrix",
    )
    args = parser.parse_args()

    df = collect_results(args.results_dir, args.accuracy_key, args.positive_label)
    summary = plot_metrics(df, args.output)

    print(summary.to_string(index=False))
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
