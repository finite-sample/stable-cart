"""
Experiment: Evaluate CentroidTree against stable-cart methods and baselines.

This script tests whether selecting the tree closest to ensemble mean
improves stability compared to existing stable-cart methods.

Hypotheses:
1. CentroidTree reduces prediction variance vs single CART
2. Variance reduction smaller for trees than neural nets (trees are deterministic)
3. Combining CentroidTree with stable-cart methods may be redundant
4. CentroidTree achieves RF-like stability with single-tree interpretability
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import (
    BootstrapVariancePenalizedTree,
    CentroidTree,
    LessGreedyHybridTree,
    RobustPrefixHonestTree,
)

warnings.filterwarnings("ignore")


def get_datasets(subset: str = "all") -> dict:
    """
    Load synthetic and real-world datasets.

    Parameters
    ----------
    subset
        Which datasets to load: 'synthetic', 'real', or 'all'.

    Returns
    -------
    dict
        Dictionary of {name: (X, y, task)} tuples.
    """
    datasets = {}

    if subset in ("synthetic", "all"):
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        datasets["synth_classification_easy"] = (X, y, "classification")

        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=3,
            n_redundant=5,
            class_sep=0.5,
            flip_y=0.1,
            random_state=42,
        )
        datasets["synth_classification_hard"] = (X, y, "classification")

        X, y = make_regression(
            n_samples=500,
            n_features=10,
            n_informative=5,
            noise=10.0,
            random_state=42,
        )
        datasets["synth_regression_friedman"] = (X, y, "regression")

    if subset in ("real", "all"):
        data = load_breast_cancer()
        datasets["breast_cancer"] = (data.data, data.target, "classification")

        data = load_diabetes()
        datasets["diabetes"] = (data.data, data.target, "regression")

        data = load_wine()
        y_binary = (data.target == 0).astype(int)
        datasets["wine_binary"] = (data.data, y_binary, "classification")

        try:
            data = fetch_california_housing()
            X_cal = data.data[:2000]
            y_cal = data.target[:2000]
            datasets["california_housing"] = (X_cal, y_cal, "regression")
        except Exception:
            pass

    return datasets


def create_methods(task: str, n_candidates_list: list[int] | None = None) -> dict:
    """
    Create dictionary of methods to evaluate.

    Parameters
    ----------
    task
        Either 'classification' or 'regression'.
    n_candidates_list
        List of n_candidates values to test for CentroidTree.

    Returns
    -------
    dict
        Dictionary of {method_name: estimator_factory}.
    """
    if n_candidates_list is None:
        n_candidates_list = [10, 20]

    methods = {}

    if task == "classification":
        methods["CART"] = lambda seed: DecisionTreeClassifier(random_state=seed)

        methods["LessGreedyHybridTree"] = lambda seed: LessGreedyHybridTree(
            task="classification", max_depth=5, random_state=seed
        )

        methods["BootstrapVariancePenalizedTree"] = lambda seed: (
            BootstrapVariancePenalizedTree(
                task="classification", max_depth=5, random_state=seed
            )
        )

        methods["RobustPrefixHonestTree"] = lambda seed: RobustPrefixHonestTree(
            task="classification", max_depth=5, random_state=seed
        )

        for n_cand in n_candidates_list:
            methods[f"CentroidTree-CART-{n_cand}"] = lambda seed, n=n_cand: (
                CentroidTree(
                    task="classification",
                    n_candidates=n,
                    random_state=seed,
                )
            )

            methods[f"CentroidTree-LessGreedy-{n_cand}"] = lambda seed, n=n_cand: (
                CentroidTree(
                    base_tree_class=LessGreedyHybridTree,
                    task="classification",
                    n_candidates=n,
                    base_params={"task": "classification", "max_depth": 5},
                    random_state=seed,
                )
            )

        methods["RandomForest-20"] = lambda seed: RandomForestClassifier(
            n_estimators=20, random_state=seed
        )

    else:
        methods["CART"] = lambda seed: DecisionTreeRegressor(random_state=seed)

        methods["LessGreedyHybridTree"] = lambda seed: LessGreedyHybridTree(
            task="regression", max_depth=5, random_state=seed
        )

        methods["BootstrapVariancePenalizedTree"] = lambda seed: (
            BootstrapVariancePenalizedTree(
                task="regression", max_depth=5, random_state=seed
            )
        )

        methods["RobustPrefixHonestTree"] = lambda seed: RobustPrefixHonestTree(
            task="regression", max_depth=5, random_state=seed
        )

        for n_cand in n_candidates_list:
            methods[f"CentroidTree-CART-{n_cand}"] = lambda seed, n=n_cand: (
                CentroidTree(
                    task="regression",
                    n_candidates=n,
                    random_state=seed,
                )
            )

            methods[f"CentroidTree-LessGreedy-{n_cand}"] = lambda seed, n=n_cand: (
                CentroidTree(
                    base_tree_class=LessGreedyHybridTree,
                    task="regression",
                    n_candidates=n,
                    base_params={"task": "regression", "max_depth": 5},
                    random_state=seed,
                )
            )

        methods["RandomForest-20"] = lambda seed: RandomForestRegressor(
            n_estimators=20, random_state=seed
        )

    return methods


def run_experiment(
    datasets: dict,
    n_seeds: int = 30,
    test_fraction: float = 0.3,
    n_candidates_list: list[int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full experiment.

    Parameters
    ----------
    datasets
        Dictionary of {name: (X, y, task)}.
    n_seeds
        Number of random seeds for stability measurement.
    test_fraction
        Fraction of data for test set.
    n_candidates_list
        List of n_candidates values to test.
    verbose
        Print progress.

    Returns
    -------
    pd.DataFrame
        Results dataframe.
    """
    if n_candidates_list is None:
        n_candidates_list = [10, 20]

    results = []

    for dataset_name, (X, y, task) in datasets.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_name} (task={task}, n={len(X)})")
            print(f"{'=' * 60}")

        methods = create_methods(task, n_candidates_list)

        for method_name, method_factory in methods.items():
            if verbose:
                print(f"  {method_name}...", end=" ", flush=True)

            method_results = {
                "dataset": dataset_name,
                "method": method_name,
                "task": task,
                "n_samples": len(X),
            }

            seed_predictions = []
            seed_times = []
            seed_scores = []

            for seed in range(n_seeds):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_fraction, random_state=seed
                )

                try:
                    model = method_factory(seed)
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_time

                    preds = model.predict(X_test)
                    score = model.score(X_test, y_test)

                    seed_predictions.append(preds)
                    seed_times.append(train_time)
                    seed_scores.append(score)
                except Exception as e:
                    if verbose:
                        print(f"Error: {e}")
                    break
            else:
                seed_predictions_array = np.array(seed_predictions)

                method_results["mean_score"] = np.mean(seed_scores)
                method_results["std_score"] = np.std(seed_scores)
                method_results["mean_train_time"] = np.mean(seed_times)

                if task == "regression":
                    pred_variance = np.mean(np.var(seed_predictions_array, axis=0))
                    pred_std = np.sqrt(pred_variance)
                    method_results["prediction_variance"] = pred_variance
                    method_results["prediction_std"] = pred_std
                else:
                    mode_preds = np.zeros(seed_predictions_array.shape[1])
                    for j in range(seed_predictions_array.shape[1]):
                        values, counts = np.unique(
                            seed_predictions_array[:, j], return_counts=True
                        )
                        mode_preds[j] = values[np.argmax(counts)]

                    disagreement_rates = []
                    for i in range(n_seeds):
                        disagreement = np.mean(seed_predictions_array[i] != mode_preds)
                        disagreement_rates.append(disagreement)

                    method_results["mean_disagreement"] = np.mean(disagreement_rates)
                    method_results["std_disagreement"] = np.std(disagreement_rates)

                results.append(method_results)

                if verbose:
                    if task == "regression":
                        print(
                            f"score={method_results['mean_score']:.3f}, "
                            f"pred_std={method_results['prediction_std']:.3f}, "
                            f"time={method_results['mean_train_time']:.3f}s"
                        )
                    else:
                        print(
                            f"score={method_results['mean_score']:.3f}, "
                            f"disagreement={method_results['mean_disagreement']:.3f}, "
                            f"time={method_results['mean_train_time']:.3f}s"
                        )

    return pd.DataFrame(results)


def compute_relative_improvements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative improvements over CART baseline.

    Parameters
    ----------
    df
        Results dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with relative improvements.
    """
    improvements = []

    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        cart_row = dataset_df[dataset_df["method"] == "CART"]

        if len(cart_row) == 0:
            continue

        cart_row = cart_row.iloc[0]
        task = cart_row["task"]

        for _, row in dataset_df.iterrows():
            if row["method"] == "CART":
                continue

            improvement = {
                "dataset": dataset,
                "method": row["method"],
                "task": task,
            }

            if task == "regression":
                cart_var = cart_row["prediction_variance"]
                method_var = row["prediction_variance"]
                if cart_var > 0:
                    improvement["variance_reduction_pct"] = (
                        (cart_var - method_var) / cart_var * 100
                    )
            else:
                cart_disagree = cart_row["mean_disagreement"]
                method_disagree = row["mean_disagreement"]
                if cart_disagree > 0:
                    improvement["disagreement_reduction_pct"] = (
                        (cart_disagree - method_disagree) / cart_disagree * 100
                    )

            score_diff = row["mean_score"] - cart_row["mean_score"]
            improvement["score_improvement"] = score_diff

            improvements.append(improvement)

    return pd.DataFrame(improvements)


def print_summary(df: pd.DataFrame, improvements_df: pd.DataFrame):
    """Print experiment summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print("\n--- Mean Performance by Method ---")
    if "prediction_std" in df.columns:
        summary = (
            df.groupby("method")
            .agg(
                {
                    "mean_score": ["mean", "std"],
                    "prediction_std": ["mean"],
                    "mean_train_time": ["mean"],
                }
            )
            .round(4)
        )
        print(summary)

    if "mean_disagreement" in df.columns:
        summary = (
            df.groupby("method")
            .agg(
                {
                    "mean_score": ["mean", "std"],
                    "mean_disagreement": ["mean"],
                    "mean_train_time": ["mean"],
                }
            )
            .round(4)
        )
        print(summary)

    if len(improvements_df) > 0:
        print("\n--- Relative Improvements over CART ---")
        if "variance_reduction_pct" in improvements_df.columns:
            improvement_summary = (
                improvements_df.groupby("method")["variance_reduction_pct"]
                .agg(["mean", "std", "count"])
                .round(2)
            )
            improvement_summary = improvement_summary.sort_values(
                "mean", ascending=False
            )
            print(improvement_summary)

        if "disagreement_reduction_pct" in improvements_df.columns:
            improvement_summary = (
                improvements_df.groupby("method")["disagreement_reduction_pct"]
                .agg(["mean", "std", "count"])
                .round(2)
            )
            improvement_summary = improvement_summary.sort_values(
                "mean", ascending=False
            )
            print(improvement_summary)


def main():
    parser = argparse.ArgumentParser(
        description="Run CentroidTree stability experiment"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10, help="Number of random seeds"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="synthetic",
        choices=["synthetic", "real", "all"],
        help="Dataset subset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/centroid_experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        nargs="+",
        default=[10, 20],
        help="n_candidates values to test",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    datasets = get_datasets(args.subset)
    print(f"Loaded {len(datasets)} datasets")

    print(f"\nRunning experiment with {args.n_seeds} seeds...")
    df = run_experiment(
        datasets,
        n_seeds=args.n_seeds,
        n_candidates_list=args.n_candidates,
        verbose=not args.quiet,
    )

    results_path = output_dir / "raw_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    improvements_df = compute_relative_improvements(df)
    if len(improvements_df) > 0:
        improvements_path = output_dir / "improvements.csv"
        improvements_df.to_csv(improvements_path, index=False)
        print(f"Improvements saved to {improvements_path}")

    print_summary(df, improvements_df)


if __name__ == "__main__":
    main()
