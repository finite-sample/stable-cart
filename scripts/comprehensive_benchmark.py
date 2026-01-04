"""
comprehensive_benchmark.py
==========================
Main orchestrator for comprehensive tree algorithm benchmarking.

Unified benchmark comparing CART vs stable-CART methods across multiple datasets,
focusing on out-of-sample prediction variance as the primary stability metric.
"""

import argparse
import os
from datetime import datetime

import pandas as pd
from benchmark_datasets import ALL_DATASETS, get_dataset_recommendations, load_dataset
from benchmark_evaluation import evaluate_dataset


def run_comprehensive_benchmark(
    datasets: list[str],
    models: list[str] | None = None,
    output_dir: str = "./benchmark_results",
    n_bootstrap: int = 20,
    random_state: int = 42,
    quick_mode: bool = False,
) -> pd.DataFrame:
    """
    Run comprehensive benchmark across specified datasets and models.

    Parameters
    ----------
    datasets : List[str]
        Dataset names to benchmark
    models : List[str], optional
        Model names to include (default: all available)
    output_dir : str, default="./benchmark_results"
        Output directory for results
    n_bootstrap : int, default=20
        Bootstrap samples for stability measurement
    random_state : int, default=42
        Random seed for reproducibility
    quick_mode : bool, default=False
        Use fewer bootstrap samples for faster execution

    Returns
    -------
    results_df : pd.DataFrame
        Combined results across all datasets and models
    """
    os.makedirs(output_dir, exist_ok=True)

    # Adjust bootstrap samples for quick mode
    if quick_mode:
        n_bootstrap = max(5, n_bootstrap // 4)

    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE TREE ALGORITHM BENCHMARK")
    print(f"{'=' * 80}")
    print(f"Datasets: {len(datasets)} selected")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Random seed: {random_state}")
    print(f"Output directory: {output_dir}")
    print(f"Quick mode: {quick_mode}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    all_results = []

    for i, dataset_name in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] Processing {dataset_name}")
        print("-" * 60)

        try:
            # Load dataset
            X_train, X_test, y_train, y_test, task = load_dataset(
                dataset_name, test_size=0.3, random_state=random_state
            )

            print(f"  Task: {task}")
            print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

            # Evaluate all models on this dataset
            dataset_results = evaluate_dataset(
                dataset_name=dataset_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                task=task,
                models_to_run=models,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )

            all_results.append(dataset_results)
            print()

        except Exception as e:
            print(f"  ✗ Failed to process {dataset_name}: {str(e)}\n")
            continue

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
    else:
        print("No results to combine - all datasets failed!")
        return pd.DataFrame()

    # Save detailed results
    detailed_path = os.path.join(output_dir, "detailed_results.csv")
    combined_results.to_csv(detailed_path, index=False)

    print(f"{'=' * 80}")
    print("BENCHMARK COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total datasets processed: {len(all_results)}")
    print(f"Total model evaluations: {len(combined_results)}")
    print(f"Detailed results saved: {detailed_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    return combined_results


def create_summary_statistics(
    results_df: pd.DataFrame, output_dir: str
) -> pd.DataFrame:
    """Create summary statistics across all datasets."""
    if results_df.empty:
        return pd.DataFrame()

    # Group by model and compute summary statistics
    summary_metrics = []

    for task in results_df["task"].unique():
        task_data = results_df[results_df["task"] == task]

        if task == "regression":
            metrics_to_summarize = ["pred_variance_mean", "mse", "rmse", "r2"]
        else:
            metrics_to_summarize = ["pred_variance_mean", "accuracy", "f1_macro"]

        # Add AUC if available
        if "auc" in task_data.columns and not task_data["auc"].isna().all():
            metrics_to_summarize.append("auc")

        # Add model characteristics
        char_metrics = ["n_leaves", "fit_time_sec"]
        available_chars = [m for m in char_metrics if m in task_data.columns]
        metrics_to_summarize.extend(available_chars)

        # Compute summary stats by model
        task_summary = (
            task_data.groupby("model")[metrics_to_summarize]
            .agg(["mean", "std"])
            .round(4)
        )
        task_summary.columns = [f"{col[0]}_{col[1]}" for col in task_summary.columns]
        task_summary["task"] = task
        task_summary["n_datasets"] = task_data.groupby("model").size()

        summary_metrics.append(task_summary.reset_index())

    if summary_metrics:
        summary_df = pd.concat(summary_metrics, ignore_index=True)

        # Save summary
        summary_path = os.path.join(output_dir, "summary_statistics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary statistics saved: {summary_path}")

        return summary_df

    return pd.DataFrame()


def create_stability_analysis(
    results_df: pd.DataFrame, output_dir: str
) -> pd.DataFrame:
    """Create focused stability analysis comparing to CART baseline."""
    if results_df.empty or "CART" not in results_df["model"].values:
        print("Cannot create stability analysis - no CART baseline found")
        return pd.DataFrame()

    stability_analysis = []

    for dataset in results_df["dataset"].unique():
        dataset_data = results_df[results_df["dataset"] == dataset]

        # Get CART baseline variance
        cart_data = dataset_data[dataset_data["model"] == "CART"]
        if cart_data.empty:
            continue

        cart_variance = cart_data["pred_variance_mean"].iloc[0]

        # Compare all models to CART
        for _, row in dataset_data.iterrows():
            model_variance = row["pred_variance_mean"]
            variance_reduction = (cart_variance - model_variance) / cart_variance * 100

            stability_analysis.append(
                {
                    "dataset": dataset,
                    "task": row["task"],
                    "model": row["model"],
                    "pred_variance": model_variance,
                    "cart_variance": cart_variance,
                    "variance_reduction_pct": variance_reduction,
                    "relative_variance": model_variance / cart_variance,
                }
            )

    if stability_analysis:
        stability_df = pd.DataFrame(stability_analysis)

        # Save stability analysis
        stability_path = os.path.join(output_dir, "stability_analysis.csv")
        stability_df.to_csv(stability_path, index=False)
        print(f"Stability analysis saved: {stability_path}")

        return stability_df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive tree algorithm benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Selection Examples:
  --datasets quick                    # Fast benchmark with key datasets
  --datasets comprehensive            # All available datasets
  --datasets friedman1,breast_cancer  # Specific datasets
  --datasets stability_showcase       # Datasets that highlight stability differences

Model Selection Examples:
  --models CART,LessGreedyHybrid     # Compare specific models
  --models all                       # All available models (default)

Quick Mode:
  --quick                            # Reduced bootstrap samples for faster execution
        """,
    )

    # Dataset selection
    parser.add_argument(
        "--datasets",
        type=str,
        default="quick",
        help="Dataset selection: 'quick', 'comprehensive', 'regression_focus', "
        "'classification_focus', 'stability_showcase', 'real_world_only', "
        "or comma-separated list of dataset names",
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Model selection: 'all' or comma-separated list of model names",
    )

    # Output directory
    parser.add_argument(
        "--output",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )

    # Bootstrap samples
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20,
        help="Number of bootstrap samples for stability measurement",
    )

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Quick mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer bootstrap samples for faster execution",
    )

    # Report generation
    parser.add_argument(
        "--no-report", action="store_true", help="Skip markdown report generation"
    )

    args = parser.parse_args()

    # Parse dataset selection
    recommendations = get_dataset_recommendations()
    if args.datasets in recommendations:
        selected_datasets = recommendations[args.datasets]
    elif args.datasets in ALL_DATASETS:
        selected_datasets = [args.datasets]
    else:
        # Comma-separated list
        selected_datasets = [d.strip() for d in args.datasets.split(",")]
        # Validate datasets
        invalid = [d for d in selected_datasets if d not in ALL_DATASETS]
        if invalid:
            print(f"Error: Invalid datasets: {invalid}")
            print(f"Available: {list(ALL_DATASETS.keys())}")
            return

    # Parse model selection
    if args.models == "all":
        selected_models = None  # Use all available models
    else:
        selected_models = [m.strip() for m in args.models.split(",")]

    # Run benchmark
    results_df = run_comprehensive_benchmark(
        datasets=selected_datasets,
        models=selected_models,
        output_dir=args.output,
        n_bootstrap=args.bootstrap_samples,
        random_state=args.seed,
        quick_mode=args.quick,
    )

    if results_df.empty:
        print("No results generated - exiting")
        return

    # Generate summary statistics
    summary_df = create_summary_statistics(results_df, args.output)

    # Generate stability analysis
    stability_df = create_stability_analysis(results_df, args.output)

    # Generate markdown report
    if not args.no_report:
        try:
            from benchmark_report import generate_markdown_report

            report_path = generate_markdown_report(
                results_df=results_df,
                summary_df=summary_df,
                stability_df=stability_df,
                output_dir=args.output,
                benchmark_config={
                    "datasets": selected_datasets,
                    "models": selected_models,
                    "n_bootstrap": args.bootstrap_samples,
                    "random_state": args.seed,
                    "quick_mode": args.quick,
                },
            )
            print(f"Markdown report generated: {report_path}")
        except ImportError:
            print("Markdown report module not available - skipping report generation")
        except Exception as e:
            print(f"Report generation failed: {str(e)}")

    print(f"\n{'=' * 80}")
    print(f"ALL BENCHMARK OUTPUTS SAVED TO: {args.output}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
