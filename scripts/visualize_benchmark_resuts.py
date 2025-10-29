"""
visualize_benchmark_results.py
================================
Generate publication-quality plots from benchmark results.

Usage:
    python visualize_benchmark_results.py --input benchmark_results
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("colorblind")


def load_results(results_dir):
    """Load accuracy and stability results."""
    acc_path = os.path.join(results_dir, "accuracy_metrics.csv")
    stab_path = os.path.join(results_dir, "stability_metrics.csv")

    df_acc = pd.read_csv(acc_path)
    df_stab = pd.read_csv(stab_path)

    return df_acc, df_stab


def plot_accuracy_comparison(df_acc, output_dir):
    """Plot MSE and R² comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSE comparison
    ax1 = axes[0]
    sns.barplot(data=df_acc, x="dataset", y="mse", hue="model", ax=ax1)
    ax1.set_title("Mean Squared Error (Lower = Better)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Dataset", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # R² comparison
    ax2 = axes[1]
    sns.barplot(data=df_acc, x="dataset", y="r2", hue="model", ax=ax2)
    ax2.set_title("R² Score (Higher = Better)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Dataset", fontsize=12)
    ax2.set_ylabel("R²", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_stability_comparison(df_stab, output_dir):
    """Plot stability metrics across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean prediction std
    ax1 = axes[0]
    sns.barplot(data=df_stab, x="dataset", y="mean_pred_std", hue="model", ax=ax1)
    ax1.set_title("Mean Prediction Std Dev (Lower = More Stable)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Dataset", fontsize=12)
    ax1.set_ylabel("Mean Prediction Std", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # P90 prediction std
    ax2 = axes[1]
    sns.barplot(data=df_stab, x="dataset", y="p90_pred_std", hue="model", ax=ax2)
    ax2.set_title("P90 Prediction Std Dev (Lower = More Stable)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Dataset", fontsize=12)
    ax2.set_ylabel("P90 Prediction Std", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "stability_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_accuracy_stability_tradeoff(df_acc, df_stab, output_dir):
    """Scatter plot showing accuracy-stability tradeoff."""
    # Merge dataframes
    df = pd.merge(df_acc, df_stab, on=["dataset", "model"])

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each model
    models = df["model"].unique()
    markers = ["o", "s", "^", "D", "v"]

    for i, model in enumerate(models):
        model_data = df[df["model"] == model]
        ax.scatter(
            model_data["mean_pred_std"],
            model_data["r2"],
            label=model,
            marker=markers[i % len(markers)],
            s=100,
            alpha=0.7,
        )

    ax.set_xlabel("Mean Prediction Std Dev (Lower = More Stable)", fontsize=12)
    ax.set_ylabel("R² Score (Higher = Better)", fontsize=12)
    ax.set_title("Accuracy-Stability Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(title="Model", fontsize=10)
    ax.grid(alpha=0.3)

    # Add ideal region annotation
    ax.annotate(
        "Ideal Region\n(High R², Low Variance)",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "accuracy_stability_tradeoff.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_relative_improvements(df_stab, output_dir):
    """Plot stability improvement relative to CART baseline."""
    # Compute relative improvements
    datasets = df_stab["dataset"].unique()
    models = [m for m in df_stab["model"].unique() if m != "CART"]

    improvements = []
    for dataset in datasets:
        cart_var = df_stab[(df_stab.dataset == dataset) & (df_stab.model == "CART")][
            "mean_pred_std"
        ].values[0]

        for model in models:
            model_var = df_stab[(df_stab.dataset == dataset) & (df_stab.model == model)][
                "mean_pred_std"
            ].values[0]

            pct_improvement = (1 - model_var / cart_var) * 100
            improvements.append(
                {"dataset": dataset, "model": model, "improvement_pct": pct_improvement}
            )

    df_impr = pd.DataFrame(improvements)

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.2

    for i, model in enumerate(models):
        model_data = df_impr[df_impr["model"] == model]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, model_data["improvement_pct"], width, label=model, alpha=0.8)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Stability Improvement vs CART (%)", fontsize=12)
    ax.set_title(
        "Prediction Stability Improvement Over CART Baseline", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend(title="Model", fontsize=9)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "stability_improvements.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_tree_size_comparison(df_acc, output_dir):
    """Plot tree size (number of leaves) comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(data=df_acc, x="dataset", y="leaves", hue="model", ax=ax)
    ax.set_title("Tree Size Comparison (Number of Leaves)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Number of Leaves", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "tree_size_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def generate_summary_table(df_acc, df_stab, output_dir):
    """Generate summary statistics table."""
    # Compute means across datasets
    acc_summary = (
        df_acc.groupby("model")
        .agg({"mse": "mean", "r2": "mean", "leaves": "mean", "fit_time_sec": "mean"})
        .round(4)
    )

    stab_summary = (
        df_stab.groupby("model").agg({"mean_pred_std": "mean", "p90_pred_std": "mean"}).round(4)
    )

    # Merge summaries
    summary = pd.merge(acc_summary, stab_summary, left_index=True, right_index=True)

    # Add relative stability column
    cart_stability = summary.loc["CART", "mean_pred_std"]
    summary["stability_vs_cart_pct"] = (
        (1 - summary["mean_pred_std"] / cart_stability) * 100
    ).round(1)

    # Reorder columns
    summary = summary[
        ["mse", "r2", "mean_pred_std", "stability_vs_cart_pct", "leaves", "fit_time_sec"]
    ]

    # Save to CSV
    save_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(save_path)
    print(f"\nSaved: {save_path}")

    # Print formatted table
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (mean across all datasets)")
    print("=" * 80)
    print(summary.to_string())
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default="./benchmark_results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: same as input)",
    )

    args = parser.parse_args()

    output_dir = args.output if args.output else args.input
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading results from: {args.input}")
    df_acc, df_stab = load_results(args.input)

    print(f"\nGenerating visualizations...")
    plot_accuracy_comparison(df_acc, output_dir)
    plot_stability_comparison(df_stab, output_dir)
    plot_accuracy_stability_tradeoff(df_acc, df_stab, output_dir)
    plot_relative_improvements(df_stab, output_dir)
    plot_tree_size_comparison(df_acc, output_dir)

    print(f"\nGenerating summary statistics...")
    generate_summary_table(df_acc, df_stab, output_dir)

    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
