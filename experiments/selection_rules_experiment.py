"""Compare representative selection with rules using the same tree pool.

Every outer bootstrap draw produces one ``RepresentativeEstimator`` candidate
pool. Random, representative, validation-score, and ensemble rules all use that
identical pool, so a difference can be attributed to selection rather than to
different fitted trees.
The ensemble is a non-tree reference, not a competitor for interpretability.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import RepresentativeEstimator

RULES = ("random", "representative", "best_validation", "ensemble")


def datasets() -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """Return offline binary-classification and regression test problems."""
    easy_X, easy_y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        class_sep=1.5,
        random_state=42,
    )
    hard_X, hard_y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=3,
        n_redundant=5,
        class_sep=0.5,
        flip_y=0.1,
        random_state=42,
    )
    regression_X, regression_y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42,
    )
    cancer = load_breast_cancer()
    diabetes = load_diabetes()
    wine = load_wine()
    return {
        "classification_easy": (easy_X, easy_y, "classification"),
        "classification_hard": (hard_X, hard_y, "classification"),
        "regression_synthetic": (regression_X, regression_y, "regression"),
        "breast_cancer": (cancer.data, cancer.target, "classification"),
        "diabetes": (diabetes.data, diabetes.target, "regression"),
        "wine_binary": (wine.data, (wine.target == 0).astype(int), "classification"),
    }


def _stratified_bootstrap(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap within classes so every candidate retains every class."""
    parts = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(parts)
    rng.shuffle(result)
    return result


def select(
    rule: str,
    selector: RepresentativeEstimator,
    rng: np.random.Generator,
) -> int | None:
    """Apply one rule to a shared candidate pool."""
    if rule == "ensemble":
        return None
    if rule == "random":
        return int(rng.integers(len(selector.candidates_)))
    if rule == "representative":
        return selector.selected_index_
    if rule == "best_validation":
        return int(np.argmax(selector.candidate_performance_scores_))
    raise ValueError(f"Unknown rule: {rule!r}.")


def predict_rule(rule: str, pool: list, selected: int | None, X: np.ndarray, task: str):
    """Predict with a selected tree or with the non-tree ensemble reference."""
    if selected is not None:
        return pool[selected].predict(X)
    if task == "classification":
        probabilities = np.mean([tree.predict_proba(X) for tree in pool], axis=0)
        return pool[0].classes_[np.argmax(probabilities, axis=1)]
    return np.mean([tree.predict(X) for tree in pool], axis=0)


def pair_values(predictions: list[np.ndarray], task: str) -> np.ndarray:
    """Return distances for disjoint pairs of independent outer replicates."""
    array = np.stack(predictions)
    left = array[0::2]
    right = array[1::2]
    if task == "classification":
        return np.mean(left != right, axis=1)
    return np.mean((left - right) ** 2, axis=1)


def _mean_mcse(values: np.ndarray) -> tuple[float, float]:
    """Return a mean and its Monte Carlo standard error."""
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def summarize(predictions: list[np.ndarray], scores: list[float], task: str) -> dict:
    """Report pairwise instability and score with Monte Carlo standard errors."""
    pairs = pair_values(predictions, task)
    score_values = np.asarray(scores)
    pairwise, pairwise_mcse = _mean_mcse(pairs)
    score, score_mcse = _mean_mcse(score_values)
    return {
        "pairwise_instability": pairwise,
        "pairwise_mcse": pairwise_mcse,
        "score": score,
        "score_mcse": score_mcse,
    }


def paired_comparison(
    predictions: dict[str, list[np.ndarray]],
    scores: dict[str, list[float]],
    task: str,
    rule: str,
    baseline: str,
) -> dict[str, float]:
    """Compare two rules on their paired pools and replicate pairs."""
    baseline_pairs = pair_values(predictions[baseline], task)
    deltas = pair_values(predictions[rule], task) - baseline_pairs
    score_deltas = np.asarray(scores[rule]) - np.asarray(scores[baseline])
    delta, delta_mcse = _mean_mcse(deltas)
    score_delta, score_delta_mcse = _mean_mcse(score_deltas)
    relative = (
        100.0 * delta / float(baseline_pairs.mean())
        if float(baseline_pairs.mean()) != 0.0
        else float("nan")
    )
    return {
        "instability_delta": delta,
        "instability_delta_mcse": delta_mcse,
        "instability_relative_change_percent": relative,
        "score_delta": score_delta,
        "score_delta_mcse": score_delta_mcse,
    }


def run_dataset(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    n_replicates: int,
    n_candidates: int,
    max_depth: int,
    seed: int,
) -> dict[str, dict]:
    """Run paired outer bootstrap draws for every selection rule."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )
    predictions = {rule: [] for rule in RULES}
    scores = {rule: [] for rule in RULES}

    for replicate in range(n_replicates):
        rng = np.random.default_rng(seed * 100_000 + replicate)
        if task == "classification":
            outer = _stratified_bootstrap(y_train, rng)
        else:
            outer = rng.integers(0, len(y_train), len(y_train))
        tree_class = (
            DecisionTreeClassifier
            if task == "classification"
            else DecisionTreeRegressor
        )
        selector = RepresentativeEstimator(
            estimator=tree_class(max_depth=max_depth, min_samples_leaf=10),
            task=task,
            n_candidates=n_candidates,
            random_state=int(rng.integers(2**31 - 1)),
        ).fit(
            X_train[outer],
            y_train[outer],
        )
        pool = selector.candidates_
        for rule in RULES:
            selected = select(
                rule,
                selector,
                np.random.default_rng(replicate),
            )
            prediction = predict_rule(rule, pool, selected, X_test, task)
            predictions[rule].append(prediction)
            metric = accuracy_score if task == "classification" else r2_score
            scores[rule].append(float(metric(y_test, prediction)))

    summaries = {
        rule: summarize(predictions[rule], scores[rule], task) for rule in RULES
    }
    for rule in RULES:
        summaries[rule]["vs_random"] = paired_comparison(
            predictions, scores, task, rule, "random"
        )
        summaries[rule]["vs_best_validation"] = paired_comparison(
            predictions, scores, task, rule, "best_validation"
        )
    return summaries


def write_summary_csv(results: dict[str, dict], destination: Path) -> None:
    """Write the evidence table in a directly inspectable format."""
    task_lookup = {name: task for name, (_, _, task) in datasets().items()}
    fields = [
        "dataset",
        "task",
        "rule",
        "pairwise_instability",
        "pairwise_mcse",
        "score",
        "score_mcse",
        "instability_change_vs_random_percent",
        "instability_delta_vs_random",
        "instability_delta_vs_random_mcse",
        "score_delta_vs_random",
        "score_delta_vs_random_mcse",
        "instability_change_vs_best_validation_percent",
        "instability_delta_vs_best_validation",
        "instability_delta_vs_best_validation_mcse",
        "score_delta_vs_best_validation",
        "score_delta_vs_best_validation_mcse",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, rules in results.items():
            task = task_lookup[dataset]
            for rule, values in rules.items():
                random = values["vs_random"]
                validation = values["vs_best_validation"]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "task": task,
                        "rule": rule,
                        "pairwise_instability": values["pairwise_instability"],
                        "pairwise_mcse": values["pairwise_mcse"],
                        "score": values["score"],
                        "score_mcse": values["score_mcse"],
                        "instability_change_vs_random_percent": random[
                            "instability_relative_change_percent"
                        ],
                        "instability_delta_vs_random": random["instability_delta"],
                        "instability_delta_vs_random_mcse": random[
                            "instability_delta_mcse"
                        ],
                        "score_delta_vs_random": random["score_delta"],
                        "score_delta_vs_random_mcse": random["score_delta_mcse"],
                        "instability_change_vs_best_validation_percent": validation[
                            "instability_relative_change_percent"
                        ],
                        "instability_delta_vs_best_validation": validation[
                            "instability_delta"
                        ],
                        "instability_delta_vs_best_validation_mcse": validation[
                            "instability_delta_mcse"
                        ],
                        "score_delta_vs_best_validation": validation["score_delta"],
                        "score_delta_vs_best_validation_mcse": validation[
                            "score_delta_mcse"
                        ],
                    }
                )


def main() -> None:
    """Run the paired comparison and write machine-readable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=100)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/selection_rules")
    args = parser.parse_args()
    if args.n_replicates < 4 or args.n_replicates % 2:
        raise ValueError("n_replicates must be an even integer of at least 4.")

    results = {}
    for name, (X, y, task) in datasets().items():
        result = run_dataset(
            X,
            y,
            task,
            args.n_replicates,
            args.n_candidates,
            args.max_depth,
            args.seed,
        )
        results[name] = result
        print(f"\n{name} ({task})")
        print(
            f"  {'rule':16s} {'pairwise':>12s} {'MCSE':>10s} {'vs random':>11s} {'score':>9s}"
        )
        for rule in RULES:
            row = result[rule]
            gain = -row["vs_random"]["instability_relative_change_percent"]
            print(
                f"  {rule:16s} {row['pairwise_instability']:12.5g} "
                f"{row['pairwise_mcse']:10.3g} {gain:10.1f}% {row['score']:9.3f}"
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    destination.write_text(json.dumps(results, indent=2))
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    write_summary_csv(results, output / "summary.csv")
    print(f"\nCreated: {destination}")


if __name__ == "__main__":
    main()
