"""
benchmark_report.py
==================
Generate comprehensive markdown reports from benchmark results.

Creates publication-ready reports with tables, analysis, and recommendations.
"""

import os
from datetime import datetime
from typing import Any

import pandas as pd


def format_number(value: float, precision: int = 3) -> str:
    """Format numbers for display in tables."""
    if pd.isna(value):
        return "N/A"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def create_performance_table(results_df: pd.DataFrame, task: str) -> str:
    """Create performance comparison table for a specific task."""
    task_data = results_df[results_df["task"] == task].copy()

    if task_data.empty:
        return f"No {task} results available.\n\n"

    # Pivot to get models as columns
    if task == "regression":
        metrics = ["pred_variance_mean", "mse", "r2"]
        metric_names = ["Pred Variance", "MSE", "R²"]
    else:
        metrics = ["pred_variance_mean", "accuracy", "f1_macro"]
        metric_names = ["Pred Variance", "Accuracy", "F1-Macro"]
        if "auc" in task_data.columns and not task_data["auc"].isna().all():
            metrics.append("auc")
            metric_names.append("AUC")

    # Create table header
    models = sorted(task_data["model"].unique())
    table = "| Dataset | " + " | ".join(models) + " |\n"
    table += "|" + "---|" * (len(models) + 1) + "\n"

    # Add rows for each metric
    for metric, metric_name in zip(metrics, metric_names, strict=True):
        if metric not in task_data.columns:
            continue

        table += f"| **{metric_name}** | " + " | ".join([""] * len(models)) + " |\n"

        for dataset in sorted(task_data["dataset"].unique()):
            dataset_data = task_data[task_data["dataset"] == dataset]
            row_values = []

            for model in models:
                model_data = dataset_data[dataset_data["model"] == model]
                if not model_data.empty:
                    value = model_data[metric].iloc[0]
                    formatted = format_number(value)

                    # Highlight best values
                    if metric in ["pred_variance_mean", "mse"]:  # Lower is better
                        best_val = dataset_data[metric].min()
                        if abs(value - best_val) < 1e-6:
                            formatted = f"**{formatted}**"
                    elif metric in [
                        "r2",
                        "accuracy",
                        "f1_macro",
                        "auc",
                    ]:  # Higher is better
                        best_val = dataset_data[metric].max()
                        if abs(value - best_val) < 1e-6:
                            formatted = f"**{formatted}**"

                    row_values.append(formatted)
                else:
                    row_values.append("N/A")

            table += f"| {dataset} | " + " | ".join(row_values) + " |\n"

        table += "| | " + " | ".join([""] * len(models)) + " |\n"  # Spacer row

    return table + "\n"


def create_stability_summary_table(stability_df: pd.DataFrame) -> str:
    """Create stability improvement summary table."""
    if stability_df.empty:
        return "No stability analysis available.\n\n"

    # Calculate average variance reduction by model
    model_summary = (
        stability_df.groupby("model")
        .agg(
            {
                "variance_reduction_pct": ["mean", "std", "count"],
                "relative_variance": "mean",
            }
        )
        .round(2)
    )

    model_summary.columns = [
        "Avg_Reduction_%",
        "Std_Reduction",
        "N_Datasets",
        "Relative_Variance",
    ]
    model_summary = model_summary.reset_index().sort_values(
        "Avg_Reduction_%", ascending=False
    )

    # Create markdown table
    table = "| Model | Avg Variance Reduction (%) | Std Dev | Datasets | Relative to CART |\n"
    table += "|-------|---------------------------|---------|----------|------------------|\n"

    for _, row in model_summary.iterrows():
        model = row["model"]
        avg_red = format_number(row["Avg_Reduction_%"], 1)
        std_red = format_number(row["Std_Reduction"], 1)
        n_datasets = int(row["N_Datasets"])
        rel_var = format_number(row["Relative_Variance"], 2)

        # Highlight positive improvements
        if row["Avg_Reduction_%"] > 0:
            avg_red = f"**+{avg_red}**"
        elif row["Avg_Reduction_%"] < 0:
            avg_red = f"{avg_red}"

        table += f"| {model} | {avg_red} | ±{std_red} | {n_datasets} | {rel_var}× |\n"

    return table + "\n"


def create_model_characteristics_table(results_df: pd.DataFrame) -> str:
    """Create model characteristics summary table."""
    # Aggregate characteristics by model
    char_metrics = ["n_leaves", "fit_time_sec"]
    available_metrics = [m for m in char_metrics if m in results_df.columns]

    if not available_metrics:
        return "No model characteristics available.\n\n"

    summary = (
        results_df.groupby("model")[available_metrics].agg(["mean", "std"]).round(2)
    )
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary = summary.reset_index()

    # Create table
    table = "| Model | "
    if "n_leaves_mean" in summary.columns:
        table += "Avg Leaves | "
    if "fit_time_sec_mean" in summary.columns:
        table += "Avg Fit Time (s) | "
    table = table.rstrip(" |") + " |\n"

    table += (
        "|" + "---|" * (len([c for c in summary.columns if c != "model"]) + 1) + "\n"
    )

    for _, row in summary.iterrows():
        model = row["model"]
        row_data = [model]

        if "n_leaves_mean" in summary.columns:
            leaves = f"{row['n_leaves_mean']:.0f} ± {row['n_leaves_std']:.0f}"
            row_data.append(leaves)

        if "fit_time_sec_mean" in summary.columns:
            time_val = f"{row['fit_time_sec_mean']:.2f} ± {row['fit_time_sec_std']:.2f}"
            row_data.append(time_val)

        table += "| " + " | ".join(row_data) + " |\n"

    return table + "\n"


def create_dataset_insights(
    results_df: pd.DataFrame, stability_df: pd.DataFrame
) -> str:
    """Generate insights about which methods work best on which datasets."""
    insights = []

    if stability_df.empty:
        return "Dataset-specific insights not available without stability analysis.\n\n"

    # Find datasets where stable methods show largest improvements
    dataset_improvements = stability_df.groupby("dataset")[
        "variance_reduction_pct"
    ].max()
    best_datasets = dataset_improvements.nlargest(3)
    worst_datasets = dataset_improvements.nsmallest(3)

    insights.append("### Best Datasets for Stability Improvements")
    for dataset, improvement in best_datasets.items():
        best_model = stability_df[
            (stability_df["dataset"] == dataset)
            & (stability_df["variance_reduction_pct"] == improvement)
        ]["model"].iloc[0]
        insights.append(
            f"- **{dataset}**: {improvement:.1f}% reduction with {best_model}"
        )

    insights.append("\n### Challenging Datasets")
    for dataset, improvement in worst_datasets.items():
        if improvement < 0:
            insights.append(
                f"- **{dataset}**: Stable methods show {abs(improvement):.1f}% worse variance"
            )
        else:
            insights.append(
                f"- **{dataset}**: Limited improvement ({improvement:.1f}%)"
            )

    # Find most consistent methods
    insights.append("\n### Most Consistent Methods")
    if len(stability_df) > 0:
        model_consistency = stability_df.groupby("model").agg(
            {"variance_reduction_pct": ["mean", "std"]}
        )
        model_consistency.columns = ["mean_improvement", "std_improvement"]
        model_consistency["consistency_score"] = (
            model_consistency["mean_improvement"] - model_consistency["std_improvement"]
        )

        top_consistent = model_consistency.sort_values(
            "consistency_score", ascending=False
        ).head(3)
        for model, row in top_consistent.iterrows():
            if model != "CART":  # Skip baseline
                insights.append(
                    f"- **{model}**: {row['mean_improvement']:.1f}% ± {row['std_improvement']:.1f}% improvement"
                )

    return "\n".join(insights) + "\n\n"


def generate_markdown_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    output_dir: str,
    benchmark_config: dict[str, Any],
) -> str:
    """
    Generate comprehensive markdown benchmark report.

    Parameters
    ----------
    results_df : pd.DataFrame
        Detailed benchmark results
    summary_df : pd.DataFrame
        Summary statistics
    stability_df : pd.DataFrame
        Stability analysis
    output_dir : str
        Output directory
    benchmark_config : Dict[str, Any]
        Benchmark configuration parameters

    Returns
    -------
    report_path : str
        Path to generated report
    """

    # Generate report content
    report_lines = []

    # Header
    report_lines.extend(
        [
            "# Stable CART Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Random Seed:** {benchmark_config.get('random_state', 'N/A')}",
            f"**Bootstrap Samples:** {benchmark_config.get('n_bootstrap', 'N/A')}",
            "",
            "## Executive Summary",
            "",
            "This report compares stable CART methods against standard CART and ensemble baselines ",
            "across multiple datasets. The primary focus is on **out-of-sample prediction variance** ",
            "as a measure of model stability, complemented by standard discrimination metrics.",
            "",
        ]
    )

    # Key findings
    if not stability_df.empty:
        # Calculate overall improvement
        stable_models = stability_df[stability_df["model"] != "CART"]["model"].unique()
        if len(stable_models) > 0:
            avg_improvement = stability_df[stability_df["model"].isin(stable_models)][
                "variance_reduction_pct"
            ].mean()

            report_lines.extend(
                [
                    "### Key Findings",
                    "",
                    f"- **Average variance reduction:** {avg_improvement:.1f}% across stable methods",
                    f"- **Datasets evaluated:** {len(results_df['dataset'].unique())}",
                    f"- **Models compared:** {len(results_df['model'].unique())}",
                    "",
                ]
            )

            # Best performing stable method
            best_stable = (
                stability_df[stability_df["model"] != "CART"]
                .groupby("model")["variance_reduction_pct"]
                .mean()
                .idxmax()
            )
            best_improvement = stability_df[stability_df["model"] == best_stable][
                "variance_reduction_pct"
            ].mean()

            report_lines.extend(
                [
                    f"- **Best stable method:** {best_stable} ({best_improvement:.1f}% average reduction)",
                    "",
                ]
            )

    # Detailed results by task
    tasks = results_df["task"].unique()

    for task in sorted(tasks):
        task_title = task.capitalize()
        report_lines.extend(
            [
                f"## {task_title} Results",
                "",
                f"Performance comparison for {task} tasks. **Bold** values indicate best performance ",
                "within each dataset.",
                "",
                create_performance_table(results_df, task),
            ]
        )

    # Stability analysis
    if not stability_df.empty:
        report_lines.extend(
            [
                "## Stability Analysis",
                "",
                "Prediction variance reduction compared to CART baseline. Positive values indicate ",
                "more stable predictions (lower variance).",
                "",
                create_stability_summary_table(stability_df),
            ]
        )

    # Model characteristics
    report_lines.extend(
        [
            "## Model Characteristics",
            "",
            "Computational and structural properties of the models.",
            "",
            create_model_characteristics_table(results_df),
        ]
    )

    # Dataset insights
    report_lines.extend(
        ["## Dataset Insights", "", create_dataset_insights(results_df, stability_df)]
    )

    # Methodology
    report_lines.extend(
        [
            "## Methodology",
            "",
            "### Stability Measurement",
            "- **Bootstrap prediction variance**: Models trained on bootstrap samples of training data",
            "- **Test set consistency**: All models evaluated on same held-out test set",
            f"- **Bootstrap samples**: {benchmark_config.get('n_bootstrap', 'N/A')} per model",
            "",
            "### Datasets",
            f"- **Selected datasets**: {', '.join(benchmark_config.get('datasets', []))[:200]}...",
            "- **Train/test split**: 70/30 with stratification for classification",
            "- **Feature standardization**: Applied to real-world datasets",
            "",
            "### Models",
            "- **CART**: Standard sklearn DecisionTreeRegressor/Classifier",
            "- **CART_Pruned**: Cost-complexity pruning with CV-selected alpha",
            "- **RandomForest**: 100-tree ensemble baseline",
            "- **LessGreedyHybrid**: Honest splits + lookahead + oblique root",
            "- **BootstrapVariancePenalized**: Explicit variance penalty in splitting",
            "- **RobustPrefixHonest**: Robust prefix + honest leaves (classification)",
            "",
        ]
    )

    # Recommendations
    if not stability_df.empty:
        stable_models = stability_df[stability_df["model"] != "CART"]
        positive_models = stable_models[stable_models["variance_reduction_pct"] > 0][
            "model"
        ].unique()

        report_lines.extend(
            [
                "## Recommendations",
                "",
                "### When to Use Stable CART Methods",
                "",
            ]
        )

        if len(positive_models) > 0:
            report_lines.extend(
                [
                    "**Use stable methods when:**",
                    "- Prediction consistency is more important than marginal accuracy gains",
                    "- Model will be retrained frequently with new data",
                    "- Predictions are used for critical decision-making requiring reliability",
                    "",
                    "**Recommended stable methods:**",
                ]
            )

            for model in positive_models:
                avg_improvement = stable_models[stable_models["model"] == model][
                    "variance_reduction_pct"
                ].mean()
                report_lines.append(
                    f"- **{model}**: {avg_improvement:.1f}% average variance reduction"
                )
        else:
            report_lines.extend(
                [
                    "**Note**: Stable methods did not show consistent variance reduction across ",
                    "the evaluated datasets. Consider:",
                    "- Using different hyperparameters",
                    "- Testing on datasets with higher noise levels",
                    "- Focusing on specific application domains",
                ]
            )

        report_lines.extend(
            [
                "",
                "### Trade-offs",
                "- Stable methods may have slightly higher computational cost",
                "- Accuracy differences are typically small (< 5%)",
                "- Stability benefits are most apparent with limited training data",
                "",
            ]
        )

    # Footer
    report_lines.extend(
        [
            "---",
            "",
            f"*Report generated by stable-cart benchmark suite v{benchmark_config.get('version', '0.1.0')}*",
            "",
        ]
    )

    # Write report
    report_content = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "comprehensive_benchmark_report.md")

    with open(report_path, "w") as f:
        f.write(report_content)

    return report_path


if __name__ == "__main__":
    # Test with dummy data
    print("Testing markdown report generation...")

    # Create dummy results
    dummy_results = pd.DataFrame(
        [
            {
                "dataset": "test",
                "model": "CART",
                "task": "regression",
                "pred_variance_mean": 0.5,
                "mse": 0.1,
                "r2": 0.8,
            },
            {
                "dataset": "test",
                "model": "LessGreedyHybrid",
                "task": "regression",
                "pred_variance_mean": 0.3,
                "mse": 0.12,
                "r2": 0.78,
            },
        ]
    )

    dummy_stability = pd.DataFrame(
        [
            {
                "dataset": "test",
                "model": "CART",
                "variance_reduction_pct": 0.0,
                "relative_variance": 1.0,
            },
            {
                "dataset": "test",
                "model": "LessGreedyHybrid",
                "variance_reduction_pct": 40.0,
                "relative_variance": 0.6,
            },
        ]
    )

    config = {"random_state": 42, "n_bootstrap": 20, "datasets": ["test"]}

    report_path = generate_markdown_report(
        dummy_results, pd.DataFrame(), dummy_stability, "./test_output", config
    )

    print(f"Test report generated: {report_path}")
