"""Prospective validation of representative single-model selection.

The frozen design is documented in ``REPRESENTATIVE_CONFIRMATORY_PLAN.md``.
This driver compares selection rules on identical candidate pools and treats
independently generated datasets, rather than test cases, as the synthetic
uncertainty units.
"""

import argparse
import csv
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_classification,
    make_friedman1,
    make_regression,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import RepresentativeEstimator

PLAN_PATH = (
    Path(__file__).parents[1]
    / "experiments/designs/REPRESENTATIVE_CONFIRMATORY_PLAN.md"
)
RULES = ("random", "representative", "best_validation", "ensemble")
SYNTHETIC_PROCESSES = (
    "classification_easy",
    "classification_hard",
    "classification_multiclass",
    "regression_linear",
    "regression_friedman1",
    "regression_heteroscedastic",
)
REAL_DATASETS = ("breast_cancer", "wine", "diabetes")
FAMILIES = ("tree",)


def _stable_seed(master_seed: int, *parts: object) -> int:
    """Derive a deterministic 32-bit seed without Python's randomized hash."""
    material = ":".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _task(process: str) -> str:
    return "classification" if process.startswith("classification") else "regression"


def generate_synthetic(
    process: str,
    seed: int,
    n_train: int = 500,
    n_test: int = 1_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate one independent train/test dataset from a frozen process."""
    n_samples = n_train + n_test
    if process == "classification_easy":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=12,
            n_informative=6,
            n_redundant=2,
            class_sep=1.5,
            flip_y=0.02,
            random_state=seed,
        )
    elif process == "classification_hard":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=12,
            n_informative=6,
            n_redundant=2,
            class_sep=0.5,
            flip_y=0.10,
            random_state=seed,
        )
    elif process == "classification_multiclass":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=12,
            n_informative=8,
            n_redundant=2,
            n_classes=4,
            n_clusters_per_class=1,
            class_sep=0.8,
            flip_y=0.05,
            random_state=seed,
        )
    elif process == "regression_linear":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=12,
            n_informative=8,
            noise=20,
            random_state=seed,
        )
    elif process == "regression_friedman1":
        X, y = make_friedman1(
            n_samples=n_samples,
            n_features=10,
            noise=1,
            random_state=seed,
        )
    elif process == "regression_heteroscedastic":
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, 10))
        mean = 3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] * X[:, 3]
        noise_sd = 0.5 + 1.5 * np.abs(X[:, 0])
        y = mean + rng.normal(scale=noise_sd)
    else:
        raise ValueError(f"Unknown synthetic process: {process!r}.")
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:], _task(process)


def load_real(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one frozen offline scikit-learn dataset."""
    loaders: dict[str, Callable[[], Any]] = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "diabetes": load_diabetes,
    }
    if name not in loaders:
        raise ValueError(f"Unknown real dataset: {name!r}.")
    bunch = loaders[name]()
    task = "regression" if name == "diabetes" else "classification"
    return np.asarray(bunch.data), np.asarray(bunch.target), task


def prototype(task: str, family: str) -> Any:
    """Construct one frozen base estimator."""
    if task == "classification" and family == "tree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
    if task == "regression" and family == "tree":
        return DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
    raise ValueError(f"Unknown task/family combination: {task!r}/{family!r}.")


def stratified_bootstrap(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap within classes, preserving every observed class."""
    parts = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(parts)
    rng.shuffle(result)
    return result


def aligned_probabilities(
    estimator: Any, X: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Align a classifier's probability columns to declared classes."""
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    candidate_classes = np.asarray(estimator.classes_)
    aligned = np.zeros((len(X), len(classes)))
    locations = {label: index for index, label in enumerate(classes)}
    for source, label in enumerate(candidate_classes):
        aligned[:, locations[label]] = probabilities[:, source]
    return aligned


def all_pair_squared_distances(predictions: np.ndarray) -> np.ndarray:
    """Return squared prediction distances for every unordered fit pair."""
    if predictions.ndim not in {2, 3}:
        raise ValueError("predictions must have shape (fits, cases[, outputs]).")
    values = []
    for left in range(len(predictions) - 1):
        for right in range(left + 1, len(predictions)):
            squared = (predictions[left] - predictions[right]) ** 2
            if squared.ndim == 2:
                squared = squared.sum(axis=1)
            values.append(float(np.mean(squared)))
    return np.asarray(values)


def fast_all_pair_mean(predictions: np.ndarray) -> float:
    """Compute mean all-pairs squared distance by the variance identity."""
    n_fits = len(predictions)
    if n_fits < 2:
        raise ValueError("At least two fitted predictions are required.")
    centered = predictions - predictions.mean(axis=0)
    squared = centered**2
    if squared.ndim == 3:
        squared = squared.sum(axis=2)
    return float(2 * n_fits / (n_fits - 1) * squared.mean())


def disjoint_squared_distances(predictions: np.ndarray) -> np.ndarray:
    """Return distances for independent, disjoint pairs of outer fits."""
    if len(predictions) % 2:
        raise ValueError("The number of outer fits must be even.")
    squared = (predictions[0::2] - predictions[1::2]) ** 2
    if squared.ndim == 3:
        squared = squared.sum(axis=2)
    return np.mean(squared, axis=1)


def all_pair_disagreements(labels: np.ndarray) -> np.ndarray:
    """Return mean label disagreement for every unordered fit pair."""
    values = []
    for left in range(len(labels) - 1):
        for right in range(left + 1, len(labels)):
            values.append(float(np.mean(labels[left] != labels[right])))
    return np.asarray(values)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return Spearman rank correlation with average ranks for ties."""
    ranks_a = _average_ranks(np.asarray(values_a))
    ranks_b = _average_ranks(np.asarray(values_b))
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def symmetric_effect(method: float, baseline: float) -> float:
    """Return the bounded symmetric percentage difference."""
    denominator = method + baseline
    if denominator == 0:
        return 0.0
    return float(200 * (method - baseline) / denominator)


def mean_mcse(values: np.ndarray) -> float:
    """Return the Monte Carlo standard error of a sample mean."""
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _pool_predictions(
    selector: RepresentativeEstimator,
    X_test: np.ndarray,
    task: str,
    random_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    pool = selector.candidates_
    best_index = int(np.argmax(selector.candidate_performance_scores_))
    indices = {
        "random": random_index,
        "representative": selector.selected_index_,
        "best_validation": best_index,
    }
    if task == "classification":
        classes = np.asarray(selector.classes_)
        candidate_probabilities = np.stack(
            [aligned_probabilities(candidate, X_test, classes) for candidate in pool]
        )
        probabilities = {
            rule: candidate_probabilities[index] for rule, index in indices.items()
        }
        probabilities["ensemble"] = candidate_probabilities.mean(axis=0)
        labels = {
            rule: np.asarray(pool[index].predict(X_test))
            for rule, index in indices.items()
        }
        labels["ensemble"] = classes[np.argmax(probabilities["ensemble"], axis=1)]
        test_centroid = candidate_probabilities.mean(axis=0)
        test_centrality = np.mean(
            (candidate_probabilities - test_centroid) ** 2, axis=(1, 2)
        )
        association = spearman(selector.candidate_scores_, test_centrality)
        return probabilities, labels, association

    candidate_predictions = np.stack(
        [np.asarray(candidate.predict(X_test)) for candidate in pool]
    )
    predictions = {
        rule: candidate_predictions[index] for rule, index in indices.items()
    }
    predictions["ensemble"] = candidate_predictions.mean(axis=0)
    test_centroid = candidate_predictions.mean(axis=0)
    test_centrality = np.sqrt(
        np.mean((candidate_predictions - test_centroid) ** 2, axis=1)
    )
    association = spearman(selector.candidate_scores_, test_centrality)
    return predictions, predictions, association


def run_dataset(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    family: str,
    n_outer: int,
    n_candidates: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate all selection rules on one paired train/test dataset."""
    if n_outer < 4 or n_outer % 2:
        raise ValueError("n_outer must be an even integer of at least 4.")
    prediction_rows = {rule: [] for rule in RULES}
    label_rows = {rule: [] for rule in RULES}
    score_rows = {rule: [] for rule in RULES}
    associations = []

    for outer_index in range(n_outer):
        outer_seed = _stable_seed(seed, "outer", outer_index)
        rng = np.random.default_rng(outer_seed)
        if task == "classification":
            indices = stratified_bootstrap(y_train, rng)
        else:
            indices = rng.integers(0, len(y_train), len(y_train))
        selector = RepresentativeEstimator(
            estimator=prototype(task, family),
            task=task,
            n_candidates=n_candidates,
            validation_fraction=0.2,
            random_state=_stable_seed(seed, "selector", outer_index),
        ).fit(X_train[indices], y_train[indices])
        random_index = int(
            np.random.default_rng(
                _stable_seed(seed, "random-rule", outer_index)
            ).integers(n_candidates)
        )
        predictions, labels, association = _pool_predictions(
            selector, X_test, task, random_index
        )
        associations.append(association)
        for rule in RULES:
            prediction_rows[rule].append(predictions[rule])
            label_rows[rule].append(labels[rule])
            metric = accuracy_score if task == "classification" else r2_score
            score_rows[rule].append(float(metric(y_test, labels[rule])))

    results: dict[str, float] = {}
    pair_samples: dict[str, np.ndarray] = {}
    for rule in RULES:
        prediction_array = np.stack(prediction_rows[rule])
        pair_samples[rule] = disjoint_squared_distances(prediction_array)
        results[f"{rule}_instability"] = fast_all_pair_mean(prediction_array)
        results[f"{rule}_score"] = float(np.mean(score_rows[rule]))
        if task == "classification":
            results[f"{rule}_label_disagreement"] = float(
                all_pair_disagreements(np.stack(label_rows[rule])).mean()
            )

    representative = results["representative_instability"]
    best = results["best_validation_instability"]
    random = results["random_instability"]
    paired_delta = pair_samples["representative"] - pair_samples["best_validation"]
    results["instability_delta_vs_best"] = representative - best
    results["instability_effect_vs_best_percent"] = symmetric_effect(
        representative, best
    )
    results["instability_delta_vs_best_mcse"] = mean_mcse(paired_delta)
    results["score_delta_vs_best"] = (
        results["representative_score"] - results["best_validation_score"]
    )
    results["instability_effect_vs_random_percent"] = symmetric_effect(
        representative, random
    )
    finite_associations = np.asarray(associations)[np.isfinite(associations)]
    results["validation_test_centrality_spearman"] = (
        float(finite_associations.mean()) if len(finite_associations) else float("nan")
    )
    return results


def percentile_mean_summary(
    values: Sequence[float], seed: int, n_bootstrap: int
) -> dict[str, float]:
    """Summarize a mean using resampling of the supplied uncertainty units."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    means = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_units": len(array),
    }


def stratified_mean_summary(
    groups: Sequence[Sequence[float]], seed: int, n_bootstrap: int
) -> dict[str, float]:
    """Summarize an equally weighted mean while resampling within cells."""
    rng = np.random.default_rng(seed)
    bootstrap_cell_means = []
    observed = []
    for values in groups:
        array = np.asarray(values, dtype=float)
        observed.append(float(array.mean()))
        draws = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
        bootstrap_cell_means.append(array[draws].mean(axis=1))
    bootstrap_means = np.mean(bootstrap_cell_means, axis=0)
    return {
        "mean": float(np.mean(observed)),
        "ci_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci_high": float(np.quantile(bootstrap_means, 0.975)),
        "n_cells": len(groups),
        "n_units": int(sum(len(group) for group in groups)),
    }


def summarize_rows(
    rows: list[dict[str, Any]], seed: int, n_bootstrap: int
) -> dict[str, Any]:
    """Create cell, family, task-score, and grand summaries."""
    cells: dict[str, Any] = {}
    cell_keys = sorted({(row["process"], row["family"]) for row in rows})
    for process, family in cell_keys:
        selected = [
            row for row in rows if row["process"] == process and row["family"] == family
        ]
        cell_seed = _stable_seed(seed, process, family, "cell-summary")
        cells[f"{process}:{family}"] = {
            "task": selected[0]["task"],
            "family": family,
            "instability_effect_vs_best_percent": percentile_mean_summary(
                [row["instability_effect_vs_best_percent"] for row in selected],
                cell_seed,
                n_bootstrap,
            ),
            "score_delta_vs_best": percentile_mean_summary(
                [row["score_delta_vs_best"] for row in selected],
                cell_seed + 1,
                n_bootstrap,
            ),
            "validation_test_centrality_spearman": percentile_mean_summary(
                [row["validation_test_centrality_spearman"] for row in selected],
                cell_seed + 2,
                n_bootstrap,
            ),
        }

    family_summaries = {}
    for family in FAMILIES:
        family_groups = [
            [
                row["instability_effect_vs_best_percent"]
                for row in rows
                if row["process"] == process and row["family"] == family
            ]
            for process in sorted({row["process"] for row in rows})
        ]
        family_summaries[family] = stratified_mean_summary(
            family_groups,
            _stable_seed(seed, family, "family-summary"),
            n_bootstrap,
        )

    score_summaries = {}
    for family in FAMILIES:
        for task in ("classification", "regression"):
            process_names = sorted(
                {
                    row["process"]
                    for row in rows
                    if row["family"] == family and row["task"] == task
                }
            )
            groups = [
                [
                    row["score_delta_vs_best"]
                    for row in rows
                    if row["process"] == process and row["family"] == family
                ]
                for process in process_names
            ]
            score_summaries[f"{task}:{family}"] = stratified_mean_summary(
                groups,
                _stable_seed(seed, task, family, "score-summary"),
                n_bootstrap,
            )

    grand_groups = [
        [
            row["instability_effect_vs_best_percent"]
            for row in rows
            if row["process"] == process and row["family"] == family
        ]
        for process, family in cell_keys
    ]
    return {
        "grand": stratified_mean_summary(
            grand_groups, _stable_seed(seed, "grand-summary"), n_bootstrap
        ),
        "families": family_summaries,
        "task_family_score_guardrails": score_summaries,
        "cells": cells,
    }


def apply_decision_rules(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen broad and family-specific decision rules."""
    cells = summary["cells"]

    def decision(family: str | None) -> dict[str, Any]:
        relevant = [
            cell
            for cell in cells.values()
            if family is None or cell["family"] == family
        ]
        effect_summary = (
            summary["grand"] if family is None else summary["families"][family]
        )
        negative_cells = sum(
            cell["instability_effect_vs_best_percent"]["mean"] < 0 for cell in relevant
        )
        maximum_cell = max(
            cell["instability_effect_vs_best_percent"]["mean"] for cell in relevant
        )
        families = FAMILIES if family is None else (family,)
        guardrails = []
        for selected_family in families:
            classification = summary["task_family_score_guardrails"][
                f"classification:{selected_family}"
            ]
            regression = summary["task_family_score_guardrails"][
                f"regression:{selected_family}"
            ]
            guardrails.extend(
                [classification["ci_low"] >= -0.01, regression["ci_low"] >= -0.02]
            )
        required_negative = 8 if family is None else 4
        checks = {
            "effect_interval_below_zero": effect_summary["ci_high"] < 0,
            "enough_negative_cells": negative_cells >= required_negative,
            "no_cell_above_ten_percent": maximum_cell <= 10,
            "performance_guardrails_pass": all(guardrails),
        }
        return {
            "supported": all(checks.values()),
            "checks": checks,
            "negative_cells": negative_cells,
            "required_negative_cells": required_negative,
            "maximum_cell_effect_percent": maximum_cell,
        }

    return {
        "tree_specific": decision("tree"),
    }


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    if not rows:
        return
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summaries(summary: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for cell, values in summary["cells"].items():
        for metric in (
            "instability_effect_vs_best_percent",
            "score_delta_vs_best",
            "validation_test_centrality_spearman",
        ):
            rows.append(
                {
                    "source": source,
                    "scope": cell,
                    "metric": metric,
                    **values[metric],
                }
            )
    return rows


def run_synthetic(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict]]:
    """Run all frozen synthetic cells, retaining dataset-level rows."""
    rows: list[dict[str, Any]] = []
    failures = []
    for process in SYNTHETIC_PROCESSES:
        for family in FAMILIES:
            print(f"synthetic {process}:{family}", flush=True)
            try:
                for dataset_index in range(args.synthetic_datasets):
                    dataset_seed = _stable_seed(
                        args.seed, "synthetic", process, family, dataset_index
                    )
                    X_train, X_test, y_train, y_test, task = generate_synthetic(
                        process, dataset_seed, args.n_train, args.n_test
                    )
                    result = run_dataset(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        task,
                        family,
                        args.n_outer,
                        args.n_candidates,
                        dataset_seed,
                    )
                    rows.append(
                        {
                            "source": "synthetic",
                            "process": process,
                            "family": family,
                            "task": task,
                            "unit_index": dataset_index,
                            "unit_seed": dataset_seed,
                            **result,
                        }
                    )
            except Exception as error:
                failures.append(
                    {
                        "source": "synthetic",
                        "process": process,
                        "family": family,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                rows = [
                    row
                    for row in rows
                    if not (row["process"] == process and row["family"] == family)
                ]
    return rows, failures


def run_real(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict]]:
    """Run the frozen repeated-split descriptive replication."""
    rows: list[dict[str, Any]] = []
    failures = []
    for name in REAL_DATASETS:
        X, y, task = load_real(name)
        for family in FAMILIES:
            print(f"real {name}:{family}", flush=True)
            try:
                for split_index in range(args.real_splits):
                    split_seed = _stable_seed(
                        args.seed, "real", name, family, split_index
                    )
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.3,
                        random_state=split_seed,
                        stratify=y if task == "classification" else None,
                    )
                    result = run_dataset(
                        np.asarray(X_train),
                        np.asarray(X_test),
                        np.asarray(y_train),
                        np.asarray(y_test),
                        task,
                        family,
                        args.n_outer,
                        args.n_candidates,
                        split_seed,
                    )
                    rows.append(
                        {
                            "source": "real_repeated_split",
                            "process": name,
                            "family": family,
                            "task": task,
                            "unit_index": split_index,
                            "unit_seed": split_seed,
                            **result,
                        }
                    )
            except Exception as error:
                failures.append(
                    {
                        "source": "real_repeated_split",
                        "process": name,
                        "family": family,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                rows = [
                    row
                    for row in rows
                    if not (row["process"] == name and row["family"] == family)
                ]
    return rows, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-datasets", type=int, default=24)
    parser.add_argument("--real-splits", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=1_000)
    parser.add_argument("--n-outer", type=int, default=20)
    parser.add_argument("--n-candidates", type=int, default=12)
    parser.add_argument("--interval-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_817)
    parser.add_argument("--output", default="results/representative_confirmatory")
    parser.add_argument("--skip-real", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the frozen study and write auditable JSON and CSV evidence."""
    args = parse_args()
    if args.synthetic_datasets < 2:
        raise ValueError("synthetic_datasets must be at least 2.")
    if args.real_splits < 2 and not args.skip_real:
        raise ValueError("real_splits must be at least 2.")
    if args.n_outer < 4 or args.n_outer % 2:
        raise ValueError("n_outer must be an even integer of at least 4.")
    if args.n_candidates < 1:
        raise ValueError("n_candidates must be positive.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    synthetic_rows, synthetic_failures = run_synthetic(args)
    real_rows: list[dict[str, Any]] = []
    real_failures: list[dict] = []
    if not args.skip_real:
        real_rows, real_failures = run_real(args)
    failures = synthetic_failures + real_failures

    expected_synthetic = len(SYNTHETIC_PROCESSES) * len(FAMILIES)
    observed_synthetic = len(
        {(row["process"], row["family"]) for row in synthetic_rows}
    )
    complete = not failures and observed_synthetic == expected_synthetic
    synthetic_summary = (
        summarize_rows(synthetic_rows, args.seed, args.interval_resamples)
        if complete
        else None
    )
    real_summary = (
        summarize_rows(real_rows, args.seed + 1, args.interval_resamples)
        if real_rows and not real_failures
        else None
    )
    decisions = apply_decision_rules(synthetic_summary) if synthetic_summary else None

    config = vars(args) | {
        "plan_path": str(PLAN_PATH.relative_to(Path(__file__).parents[1])),
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "rules": list(RULES),
        "synthetic_processes": list(SYNTHETIC_PROCESSES),
        "real_datasets": list(REAL_DATASETS),
        "families": list(FAMILIES),
    }
    evidence = {
        "complete": complete,
        "config": config,
        "failures": failures,
        "synthetic_summary": synthetic_summary,
        "real_repeated_split_summary": real_summary,
        "decisions": decisions,
    }
    (output / "results.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False)
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    _write_csv(synthetic_rows + real_rows, output / "dataset_rows.csv")
    summary_rows = []
    if synthetic_summary:
        summary_rows.extend(_flatten_summaries(synthetic_summary, "synthetic"))
    if real_summary:
        summary_rows.extend(_flatten_summaries(real_summary, "real_repeated_split"))
    _write_csv(summary_rows, output / "summary.csv")

    print(json.dumps(decisions, indent=2))
    print(f"created {output / 'results.json'}")
    if failures:
        raise RuntimeError(f"{len(failures)} cells failed; see results.json.")


if __name__ == "__main__":
    main()
