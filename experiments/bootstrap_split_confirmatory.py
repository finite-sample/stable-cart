"""Confirmatory validation of BootstrapSplitTree against tuned, pruned CART."""

import argparse
import csv
import hashlib
import json
import platform
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    make_friedman1,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from experiments.bootstrap_split_tree import BootstrapSplitTree
from experiments.representative_confirmatory import (
    _stable_seed,
    all_pair_disagreements,
    fast_all_pair_mean,
    mean_mcse,
    symmetric_effect,
)

PLAN_PATH = (
    Path(__file__).parents[1]
    / "experiments/designs/BOOTSTRAP_SPLIT_CONFIRMATORY_PLAN.md"
)
ARMS = ("pruned_cart", "bootstrap_split")
SYNTHETIC_PROCESSES = (
    "classification_unique",
    "classification_redundant",
    "classification_interaction",
    "regression_unique_step",
    "regression_redundant",
    "regression_friedman1",
)
REAL_DATASETS = ("breast_cancer", "digits_binary", "diabetes")
FROZEN_SETTINGS = {
    "synthetic_datasets": 20,
    "real_splits": 16,
    "n_development": 600,
    "n_test": 1_000,
    "validation_refits": 16,
    "test_refits": 20,
    "interval_resamples": 20_000,
    "seed": 20_260_818,
    "output": "results/bootstrap_split_confirmatory",
    "skip_real": False,
}


def _task(process: str) -> str:
    return "classification" if process.startswith("classification") else "regression"


def generate_synthetic(
    process: str,
    seed: int,
    n_development: int = 600,
    n_test: int = 1_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate one frozen synthetic development/test dataset."""
    n_samples = n_development + n_test
    rng = np.random.default_rng(seed)
    task = _task(process)
    if process == "classification_unique":
        X = rng.normal(size=(n_samples, 10))
        probability = 1 / (1 + np.exp(-3 * X[:, 0]))
        y = rng.binomial(1, probability)
    elif process == "classification_redundant":
        latent = rng.normal(size=n_samples)
        X = np.column_stack(
            [
                latent + 0.25 * rng.normal(size=n_samples),
                latent + 0.25 * rng.normal(size=n_samples),
                rng.normal(size=(n_samples, 8)),
            ]
        )
        probability = 1 / (1 + np.exp(-2 * latent))
        y = rng.binomial(1, probability)
    elif process == "classification_interaction":
        X = rng.normal(size=(n_samples, 10))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        flip = rng.random(n_samples) < 0.08
        y[flip] = 1 - y[flip]
    elif process == "regression_unique_step":
        X = rng.normal(size=(n_samples, 10))
        y = 4 * (X[:, 0] > 0) + rng.normal(size=n_samples)
    elif process == "regression_redundant":
        latent = rng.normal(size=n_samples)
        X = np.column_stack(
            [
                latent + 0.25 * rng.normal(size=n_samples),
                latent + 0.25 * rng.normal(size=n_samples),
                rng.normal(size=(n_samples, 8)),
            ]
        )
        y = 3 * latent + rng.normal(size=n_samples)
    elif process == "regression_friedman1":
        X, y = make_friedman1(
            n_samples=n_samples,
            n_features=10,
            noise=1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown synthetic process: {process!r}.")
    return (
        np.asarray(X[:n_development]),
        np.asarray(X[n_development:]),
        np.asarray(y[:n_development]),
        np.asarray(y[n_development:]),
        task,
    )


def load_real(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one frozen offline real dataset."""
    if name == "breast_cancer":
        bunch = load_breast_cancer()
        return np.asarray(bunch.data), np.asarray(bunch.target), "classification"
    if name == "digits_binary":
        bunch = load_digits()
        return (
            np.asarray(bunch.data),
            (np.asarray(bunch.target) == 0).astype(int),
            "classification",
        )
    if name == "diabetes":
        bunch = load_diabetes()
        return np.asarray(bunch.data), np.asarray(bunch.target), "regression"
    raise ValueError(f"Unknown real dataset: {name!r}.")


def _tree_class(task: str):
    return DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor


def cart_alpha_grid(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    max_depth: int,
    min_samples_leaf: int,
) -> list[tuple[str, float]]:
    """Return three distinct pruning settings for one CART constraint pair."""
    path = _tree_class(task)(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=20,
        random_state=0,
    ).cost_complexity_pruning_path(X, y)
    positive = np.unique(np.asarray(path.ccp_alphas, dtype=float))
    positive = positive[positive > 0]
    if len(positive) < 2:
        raise RuntimeError(
            "CART tuning path has fewer than two distinct positive alphas for "
            f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf}."
        )
    median_target, p80_target = np.quantile(positive, [0.5, 0.8])
    median_index = int(np.argmin(np.abs(positive - median_target)))
    available = np.arange(len(positive)) != median_index
    p80_index = int(
        np.flatnonzero(available)[np.argmin(np.abs(positive[available] - p80_target))]
    )
    return [
        ("zero", 0.0),
        ("median", float(positive[median_index])),
        ("p80", float(positive[p80_index])),
    ]


def configurations(X: np.ndarray, y: np.ndarray, task: str) -> list[dict[str, Any]]:
    """Return exactly 12 frozen configurations per comparison arm."""
    configs: list[dict[str, Any]] = []
    for depth in (3, 5):
        for leaf in (5, 10):
            for alpha_slot, alpha in cart_alpha_grid(X, y, task, depth, leaf):
                config_id = f"cart:d={depth}:leaf={leaf}:alpha={alpha_slot}"
                configs.append(
                    {
                        "arm": "pruned_cart",
                        "config_id": config_id,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "min_samples_split": 20,
                        "ccp_alpha": alpha,
                    }
                )
    for depth in (3, 5):
        for leaf in (5, 10):
            for threshold in (0.0, 0.3, 0.5):
                config_id = f"split:d={depth}:leaf={leaf}:pi={threshold:g}"
                configs.append(
                    {
                        "arm": "bootstrap_split",
                        "config_id": config_id,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "min_samples_split": 20,
                        "consensus_threshold": threshold,
                        "leaf_shrinkage": 0.0,
                    }
                )
    counts = {arm: sum(config["arm"] == arm for config in configs) for arm in ARMS}
    if counts != {"pruned_cart": 12, "bootstrap_split": 12}:
        raise RuntimeError(f"Internal error: unequal tuning budget {counts}.")
    effective = {
        (
            config["arm"],
            config["max_depth"],
            config["min_samples_leaf"],
            config.get("min_samples_split", 20),
            config.get("ccp_alpha"),
            config.get("consensus_threshold"),
            config.get("leaf_shrinkage"),
        )
        for config in configs
    }
    if len(effective) != 24:
        raise RuntimeError("Tuning grid contains duplicate effective configurations.")
    return configs


def make_estimator(config: dict[str, Any], task: str, seed: int) -> Any:
    """Build one estimator from a serialized frozen configuration."""
    if config["arm"] == "pruned_cart":
        return _tree_class(task)(
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            min_samples_split=config["min_samples_split"],
            ccp_alpha=config["ccp_alpha"],
            random_state=seed,
        )
    return BootstrapSplitTree(
        task=task,
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        min_samples_split=config["min_samples_split"],
        n_consensus=16,
        consensus_threshold=config["consensus_threshold"],
        leaf_shrinkage=config["leaf_shrinkage"],
        max_candidates=40,
        random_state=seed,
    )


def _bootstrap_indices(
    y: np.ndarray, task: str, rng: np.random.Generator
) -> np.ndarray:
    del task
    return rng.integers(0, len(y), len(y))


def _aligned_binary_probabilities(
    estimator: Any, X: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    aligned = np.zeros((len(X), len(classes)))
    positions = {label: index for index, label in enumerate(classes)}
    for source, label in enumerate(estimator.classes_):
        aligned[:, positions[label]] = probabilities[:, source]
    return aligned


def _score(y: np.ndarray, prediction: np.ndarray, task: str) -> float:
    metric = accuracy_score if task == "classification" else r2_score
    return float(metric(y, prediction))


def _primary_pair_values(predictions: np.ndarray, task: str) -> np.ndarray:
    if task == "classification":
        return all_pair_disagreements(predictions)
    values = []
    for left in range(len(predictions) - 1):
        for right in range(left + 1, len(predictions)):
            values.append(float(np.mean((predictions[left] - predictions[right]) ** 2)))
    return np.asarray(values)


def _disjoint_primary(predictions: np.ndarray, task: str) -> np.ndarray:
    if len(predictions) % 2:
        raise ValueError("The number of refits must be even.")
    left = predictions[0::2]
    right = predictions[1::2]
    if task == "classification":
        return np.mean(left != right, axis=1)
    return np.mean((left - right) ** 2, axis=1)


def _root_feature(estimator: Any) -> int:
    if isinstance(estimator, BootstrapSplitTree):
        return (
            -1 if estimator.tree_["type"] == "leaf" else int(estimator.tree_["feature"])
        )
    feature = int(estimator.tree_.feature[0])
    return feature if feature >= 0 else -1


def _leaf_count(estimator: Any) -> int:
    if isinstance(estimator, BootstrapSplitTree):
        return estimator.get_n_leaves()
    return int(estimator.get_n_leaves())


def evaluate_configurations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    task: str,
    n_refits: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Evaluate every configuration on paired inner bootstrap samples."""
    configs = configurations(X_train, y_train, task)
    bootstrap_samples = [
        _bootstrap_indices(
            y_train,
            task,
            np.random.default_rng(_stable_seed(seed, "validation-sample", index)),
        )
        for index in range(n_refits)
    ]
    classes = np.unique(y_train) if task == "classification" else None
    rows = []
    by_id = {}
    for config in configs:
        predictions = []
        probabilities = []
        scores = []
        fit_seconds = []
        for refit, indices in enumerate(bootstrap_samples):
            estimator = make_estimator(
                config, task, _stable_seed(seed, "validation-model", refit)
            )
            started = time.perf_counter()
            estimator.fit(X_train[indices], y_train[indices])
            fit_seconds.append(time.perf_counter() - started)
            prediction = np.asarray(estimator.predict(X_validation))
            predictions.append(prediction)
            scores.append(_score(y_validation, prediction, task))
            if task == "classification":
                probabilities.append(
                    _aligned_binary_probabilities(
                        estimator, X_validation, np.asarray(classes)
                    )
                )
        prediction_array = np.stack(predictions)
        row = {
            "arm": config["arm"],
            "config_id": config["config_id"],
            "config_json": json.dumps(config, sort_keys=True),
            "validation_score": float(np.mean(scores)),
            "validation_instability": float(
                _primary_pair_values(prediction_array, task).mean()
            ),
            "validation_probability_instability": (
                fast_all_pair_mean(np.stack(probabilities)) if probabilities else None
            ),
            "mean_fit_seconds": float(np.mean(fit_seconds)),
        }
        rows.append(row)
        by_id[config["config_id"]] = config
    return rows, by_id


def select_configurations(
    rows: list[dict[str, Any]], task: str
) -> tuple[dict[str, dict[str, Any]], float]:
    """Select both arms using the frozen common CART-derived score floor."""
    tolerance = 0.01 if task == "classification" else 0.02
    cart_best = max(
        row["validation_score"] for row in rows if row["arm"] == "pruned_cart"
    )
    score_floor = cart_best - tolerance
    selected = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        eligible = [row for row in arm_rows if row["validation_score"] >= score_floor]
        if eligible:
            choice = min(
                eligible,
                key=lambda row: (
                    row["validation_instability"],
                    -row["validation_score"],
                    row["config_id"],
                ),
            )
        else:
            choice = min(
                arm_rows,
                key=lambda row: (-row["validation_score"], row["config_id"]),
            )
        selected[arm] = choice | {"eligible": bool(eligible)}
    return selected, score_floor


def evaluate_selected(
    selected: dict[str, dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    X_development: np.ndarray,
    y_development: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    n_refits: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate selected procedures on paired full-development refits."""
    bootstrap_samples = [
        _bootstrap_indices(
            y_development,
            task,
            np.random.default_rng(_stable_seed(seed, "test-sample", index)),
        )
        for index in range(n_refits)
    ]
    classes = np.unique(y_development) if task == "classification" else None
    arm_results = {}
    disjoint = {}
    for arm in ARMS:
        config = by_id[selected[arm]["config_id"]]
        predictions = []
        probabilities = []
        scores = []
        root_features = []
        leaf_counts = []
        fit_seconds = []
        for refit, indices in enumerate(bootstrap_samples):
            estimator = make_estimator(
                config, task, _stable_seed(seed, "test-model", refit)
            )
            started = time.perf_counter()
            estimator.fit(X_development[indices], y_development[indices])
            fit_seconds.append(time.perf_counter() - started)
            prediction = np.asarray(estimator.predict(X_test))
            predictions.append(prediction)
            scores.append(_score(y_test, prediction, task))
            root_features.append(_root_feature(estimator))
            leaf_counts.append(_leaf_count(estimator))
            if task == "classification":
                probabilities.append(
                    _aligned_binary_probabilities(
                        estimator, X_test, np.asarray(classes)
                    )
                )
        prediction_array = np.stack(predictions)
        primary_pairs = _primary_pair_values(prediction_array, task)
        disjoint[arm] = _disjoint_primary(prediction_array, task)
        arm_results[arm] = {
            "instability": float(primary_pairs.mean()),
            "score": float(np.mean(scores)),
            "probability_instability": (
                fast_all_pair_mean(np.stack(probabilities)) if probabilities else None
            ),
            "root_feature_disagreement": float(
                all_pair_disagreements(np.asarray(root_features)[:, None]).mean()
            ),
            "mean_leaf_count": float(np.mean(leaf_counts)),
            "mean_fit_seconds": float(np.mean(fit_seconds)),
        }

    cart = arm_results["pruned_cart"]
    split = arm_results["bootstrap_split"]
    paired_delta = disjoint["bootstrap_split"] - disjoint["pruned_cart"]
    result: dict[str, float] = {}
    for arm, values in arm_results.items():
        for metric, value in values.items():
            if value is not None:
                result[f"{arm}_{metric}"] = float(value)
    result["instability_effect_percent"] = symmetric_effect(
        split["instability"], cart["instability"]
    )
    result["instability_delta"] = float(split["instability"] - cart["instability"])
    normalization = 1.0 if task == "classification" else float(np.var(y_test))
    result["normalized_instability_delta"] = float(
        result["instability_delta"] / normalization
    )
    result["instability_delta_mcse"] = mean_mcse(paired_delta)
    result["score_delta"] = float(split["score"] - cart["score"])
    result["root_disagreement_effect_percent"] = symmetric_effect(
        split["root_feature_disagreement"], cart["root_feature_disagreement"]
    )
    result["fit_time_ratio"] = float(
        split["mean_fit_seconds"] / cart["mean_fit_seconds"]
    )
    if task == "classification":
        result["probability_instability_effect_percent"] = symmetric_effect(
            split["probability_instability"], cart["probability_instability"]
        )
    return result


def run_one_dataset(
    X_development: np.ndarray,
    X_test: np.ndarray,
    y_development: np.ndarray,
    y_test: np.ndarray,
    task: str,
    validation_refits: int,
    test_refits: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune on development data and evaluate once on untouched test data."""
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=1 / 3,
        random_state=_stable_seed(seed, "development-split"),
        stratify=y_development if task == "classification" else None,
    )
    validation_rows, by_id = evaluate_configurations(
        np.asarray(X_train),
        np.asarray(y_train),
        np.asarray(X_validation),
        np.asarray(y_validation),
        task,
        validation_refits,
        seed,
    )
    selected, score_floor = select_configurations(validation_rows, task)
    results = evaluate_selected(
        selected,
        by_id,
        X_development,
        y_development,
        X_test,
        y_test,
        task,
        test_refits,
        seed,
    )
    results.update(
        {
            "score_floor": score_floor,
            "cart_config": selected["pruned_cart"]["config_id"],
            "bootstrap_split_config": selected["bootstrap_split"]["config_id"],
            "bootstrap_split_eligible": selected["bootstrap_split"]["eligible"],
            "cart_validation_score": selected["pruned_cart"]["validation_score"],
            "bootstrap_split_validation_score": selected["bootstrap_split"][
                "validation_score"
            ],
            "cart_validation_instability": selected["pruned_cart"][
                "validation_instability"
            ],
            "bootstrap_split_validation_instability": selected["bootstrap_split"][
                "validation_instability"
            ],
        }
    )
    return results, validation_rows


def _mean_summary(
    values: Sequence[float], seed: int, n_resamples: int
) -> dict[str, float]:
    """Summarize independent units with 95% and Bonferroni 97.5% intervals."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(n_resamples, len(array)))
    means = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "ci_98_low": float(np.quantile(means, 0.0125)),
        "ci_98_high": float(np.quantile(means, 0.9875)),
        "ci_family_low": float(np.quantile(means, 1 / 120)),
        "ci_family_high": float(np.quantile(means, 119 / 120)),
        "n_units": len(array),
    }


def _stratified_summary(
    groups: Sequence[Sequence[float]], seed: int, n_resamples: int
) -> dict[str, float]:
    """Equally weight declared cells while resampling units within each cell."""
    rng = np.random.default_rng(seed)
    cell_draws = []
    observed = []
    for values in groups:
        array = np.asarray(values, dtype=float)
        observed.append(float(array.mean()))
        draws = rng.integers(0, len(array), size=(n_resamples, len(array)))
        cell_draws.append(array[draws].mean(axis=1))
    means = np.mean(cell_draws, axis=0)
    return {
        "mean": float(np.mean(observed)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "ci_98_low": float(np.quantile(means, 0.0125)),
        "ci_98_high": float(np.quantile(means, 0.9875)),
        "ci_family_low": float(np.quantile(means, 1 / 120)),
        "ci_family_high": float(np.quantile(means, 119 / 120)),
        "n_cells": len(groups),
        "n_units": int(sum(len(group) for group in groups)),
    }


def summarize_rows(
    rows: list[dict[str, Any]], seed: int, n_resamples: int
) -> dict[str, Any]:
    """Summarize cells, tasks, score guardrails, mechanism, and runtime."""
    cells = {}
    process_names = sorted({row["process"] for row in rows})
    for process in process_names:
        selected = [row for row in rows if row["process"] == process]
        cell_seed = _stable_seed(seed, process, "cell-summary")
        cells[process] = {
            "task": selected[0]["task"],
            "instability_effect_percent": _mean_summary(
                [row["instability_effect_percent"] for row in selected],
                cell_seed,
                n_resamples,
            ),
            "score_delta": _mean_summary(
                [row["score_delta"] for row in selected],
                cell_seed + 1,
                n_resamples,
            ),
            "normalized_instability_delta": _mean_summary(
                [row["normalized_instability_delta"] for row in selected],
                cell_seed + 5,
                n_resamples,
            ),
            "root_disagreement_effect_percent": _mean_summary(
                [row["root_disagreement_effect_percent"] for row in selected],
                cell_seed + 2,
                n_resamples,
            ),
            "fit_time_ratio": _mean_summary(
                [row["fit_time_ratio"] for row in selected],
                cell_seed + 3,
                n_resamples,
            ),
            "eligibility_rate": float(
                np.mean([row["bootstrap_split_eligible"] for row in selected])
            ),
            "n_units": len(selected),
        }
        if selected[0]["task"] == "classification":
            cells[process]["probability_instability_effect_percent"] = _mean_summary(
                [row["probability_instability_effect_percent"] for row in selected],
                cell_seed + 4,
                n_resamples,
            )

    def grouped(metric: str, selected_processes: list[str], summary_seed: int):
        return _stratified_summary(
            [
                [row[metric] for row in rows if row["process"] == process]
                for process in selected_processes
            ],
            summary_seed,
            n_resamples,
        )

    tasks = {}
    for task in ("classification", "regression"):
        selected_processes = [
            process for process in process_names if cells[process]["task"] == task
        ]
        tasks[task] = {
            "instability_effect_percent": grouped(
                "instability_effect_percent",
                selected_processes,
                _stable_seed(seed, task, "effect-summary"),
            ),
            "score_delta": grouped(
                "score_delta",
                selected_processes,
                _stable_seed(seed, task, "score-summary"),
            ),
            "normalized_instability_delta": grouped(
                "normalized_instability_delta",
                selected_processes,
                _stable_seed(seed, task, "absolute-effect-summary"),
            ),
            "fit_time_ratio": grouped(
                "fit_time_ratio",
                selected_processes,
                _stable_seed(seed, task, "time-summary"),
            ),
        }
    grand = grouped(
        "instability_effect_percent",
        process_names,
        _stable_seed(seed, "grand-summary"),
    )
    grand_absolute = grouped(
        "normalized_instability_delta",
        process_names,
        _stable_seed(seed, "grand-absolute-summary"),
    )
    return {
        "grand": grand,
        "grand_normalized_instability_delta": grand_absolute,
        "tasks": tasks,
        "cells": cells,
    }


def apply_decision_rules(
    summary: dict[str, Any], failures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the frozen broad and task-specific rules."""
    cells = summary["cells"]

    def decide(task: str | None) -> dict[str, Any]:
        relevant = [
            values
            for values in cells.values()
            if task is None or values["task"] == task
        ]
        effect = (
            summary["grand"]
            if task is None
            else summary["tasks"][task]["instability_effect_percent"]
        )
        absolute_effect = (
            summary["grand_normalized_instability_delta"]
            if task is None
            else summary["tasks"][task]["normalized_instability_delta"]
        )
        score_tasks = ("classification", "regression") if task is None else (task,)
        score_checks = []
        for score_task in score_tasks:
            threshold = -0.01 if score_task == "classification" else -0.02
            score_checks.append(
                summary["tasks"][score_task]["score_delta"]["ci_family_low"]
                >= threshold
            )
        cell_score_checks = []
        for cell in relevant:
            threshold = -0.01 if cell["task"] == "classification" else -0.02
            cell_score_checks.append(cell["score_delta"]["ci_low"] >= threshold)
        negative_cells = sum(
            cell["instability_effect_percent"]["mean"] < 0 for cell in relevant
        )
        required_negative = 4 if task is None else 2
        relevant_failures = [
            failure
            for failure in failures
            if task is None or _task(failure["process"]) == task
        ]
        checks = {
            "effect_at_least_five_percent": effect["mean"] <= -5,
            "effect_interval_below_zero": effect["ci_family_high"] < 0,
            "absolute_effect_at_least_threshold": absolute_effect["mean"] <= -0.002,
            "absolute_effect_interval_below_zero": absolute_effect["ci_family_high"]
            < 0,
            "enough_negative_cells": negative_cells >= required_negative,
            "no_cell_above_ten_percent": max(
                cell["instability_effect_percent"]["mean"] for cell in relevant
            )
            <= 10,
            "eligibility_at_least_ninety_percent": min(
                cell["eligibility_rate"] for cell in relevant
            )
            >= 0.9,
            "performance_guardrails_pass": all(score_checks),
            "cell_performance_guardrails_pass": all(cell_score_checks),
            "no_fit_failures": not relevant_failures,
        }
        return {
            "supported": all(checks.values()),
            "checks": checks,
            "negative_cells": negative_cells,
            "required_negative_cells": required_negative,
        }

    broad = decide(None)
    classification = decide("classification")
    regression = decide("regression")
    if broad["supported"]:
        recommendation = "keep_general_experimental"
    elif classification["supported"] or regression["supported"]:
        recommendation = "keep_task_specific_experimental"
    else:
        recommendation = "remove_from_package"
    return {
        "broad": broad,
        "classification": classification,
        "regression": regression,
        "package_recommendation": recommendation,
    }


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summary(summary: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = [
        {
            "source": source,
            "scope": "grand",
            "process": "all",
            "task": "all",
            "metric": "instability_effect_percent",
            **summary["grand"],
        },
        {
            "source": source,
            "scope": "grand",
            "process": "all",
            "task": "all",
            "metric": "normalized_instability_delta",
            **summary["grand_normalized_instability_delta"],
        },
    ]
    for task, values in summary["tasks"].items():
        for metric, result in values.items():
            rows.append(
                {
                    "source": source,
                    "scope": "task",
                    "process": "all",
                    "task": task,
                    "metric": metric,
                    **result,
                }
            )
    for process, values in summary["cells"].items():
        for metric, result in values.items():
            if isinstance(result, dict):
                rows.append(
                    {
                        "source": source,
                        "scope": "cell",
                        "process": process,
                        "task": values["task"],
                        "metric": metric,
                        **result,
                    }
                )
    return rows


def _read_summary_inputs(path: Path) -> list[dict[str, Any]]:
    """Read the minimal typed fields needed to regenerate summaries."""
    numeric = {
        "instability_effect_percent",
        "normalized_instability_delta",
        "score_delta",
        "root_disagreement_effect_percent",
        "fit_time_ratio",
        "probability_instability_effect_percent",
    }
    rows = []
    with path.open(newline="") as handle:
        for stored in csv.DictReader(handle):
            row: dict[str, Any] = {
                "source": stored["source"],
                "process": stored["process"],
                "task": stored["task"],
                "bootstrap_split_eligible": stored["bootstrap_split_eligible"]
                == "True",
            }
            for key in numeric:
                if stored.get(key):
                    row[key] = float(stored[key])
            rows.append(row)
    return rows


def _validate_artifacts(
    output: Path,
    evidence: dict[str, Any],
    synthetic_failures: list[dict[str, Any]],
    seed: int,
    n_resamples: int,
) -> None:
    """Fail unless stored JSON/CSV artifacts regenerate the reported result."""
    if json.loads((output / "results.json").read_text()) != evidence:
        raise RuntimeError("Stored results JSON failed its round-trip check.")
    if json.loads((output / "config.json").read_text()) != evidence["config"]:
        raise RuntimeError("Stored config JSON disagrees with results JSON.")

    raw_rows = _read_summary_inputs(output / "dataset_rows.csv")
    synthetic_rows = [row for row in raw_rows if row["source"] == "synthetic"]
    regenerated = summarize_rows(synthetic_rows, seed, n_resamples)
    if regenerated != evidence["synthetic_summary"]:
        raise RuntimeError("Synthetic summary does not regenerate from raw CSV rows.")
    regenerated_decisions = apply_decision_rules(regenerated, synthetic_failures)
    if regenerated_decisions != evidence["decisions"]:
        raise RuntimeError("Decisions do not regenerate from raw CSV rows.")

    expected_flat = _flatten_summary(regenerated, "synthetic")
    real_summary = evidence["real_repeated_split_summary"]
    if real_summary:
        real_rows = [row for row in raw_rows if row["source"] == "real_repeated_split"]
        regenerated_real = summarize_rows(real_rows, seed + 1, n_resamples)
        if regenerated_real != real_summary:
            raise RuntimeError(
                "Real-data summary does not regenerate from raw CSV rows."
            )
        expected_flat.extend(_flatten_summary(real_summary, "real_repeated_split"))
    with (output / "summary.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        actual_flat = list(reader)
        fields = list(reader.fieldnames or [])
    normalized_expected = [
        {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in fields
        }
        for row in expected_flat
    ]
    if actual_flat != normalized_expected:
        raise RuntimeError("Flat summary CSV disagrees with JSON summaries.")


def _study_unit(job):
    source, process, unit_index, args = job
    seed_source = "synthetic" if source == "synthetic" else "real"
    unit_seed = _stable_seed(args.seed, seed_source, process, unit_index)
    identity = {
        "source": source,
        "process": process,
        "unit_index": unit_index,
        "unit_seed": unit_seed,
    }
    try:
        if source == "synthetic":
            X_dev, X_test, y_dev, y_test, task = generate_synthetic(
                process, unit_seed, args.n_development, args.n_test
            )
        else:
            X, y, task = load_real(process)
            X_dev, X_test, y_dev, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=unit_seed,
                stratify=y if task == "classification" else None,
            )
        result, tuning = run_one_dataset(
            np.asarray(X_dev),
            np.asarray(X_test),
            np.asarray(y_dev),
            np.asarray(y_test),
            task,
            args.validation_refits,
            args.test_refits,
            unit_seed,
        )
        identity["task"] = task
        return {**identity, **result}, [{**identity, **row} for row in tuning], None
    except Exception as error:
        return (
            None,
            [],
            {**identity, "error_type": type(error).__name__, "error": str(error)},
        )


def _run_units(source, processes, count, args):
    jobs = [
        (source, process, index, args)
        for process in processes
        for index in range(count)
    ]
    dataset_rows, validation_rows, failures = [], [], []
    workers = getattr(args, "workers", 1)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        outputs = (
            executor.map(_study_unit, jobs) if executor else map(_study_unit, jobs)
        )
        for done, (row, tuning, failure) in enumerate(outputs, start=1):
            if row is not None:
                dataset_rows.append(row)
                validation_rows.extend(tuning)
            if failure is not None:
                failures.append(failure)
            print(f"{source}: {done}/{len(jobs)} units", flush=True)
    finally:
        if executor:
            executor.shutdown()
    return dataset_rows, validation_rows, failures


def run_synthetic(args: argparse.Namespace):
    return _run_units("synthetic", SYNTHETIC_PROCESSES, args.synthetic_datasets, args)


def run_real(args: argparse.Namespace):
    return _run_units("real_repeated_split", REAL_DATASETS, args.real_splits, args)


def preflight_checks() -> dict[str, bool]:
    """Execute the frozen falsification checks before any study outcome."""
    predictions = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]])
    instability = float(_primary_pair_values(predictions, "classification").mean())
    if symmetric_effect(instability, instability) != 0:
        raise RuntimeError("A/A symmetric effect check failed.")
    paired_delta = _disjoint_primary(predictions, "classification") - _disjoint_primary(
        predictions, "classification"
    )
    if not np.array_equal(paired_delta, np.zeros_like(paired_delta)):
        raise RuntimeError("A/A paired-delta check failed.")

    regression_predictions = np.array([[0.0, 2.0], [2.0, 4.0], [1.0, 5.0]])
    slow_pairs = _primary_pair_values(regression_predictions, "regression")
    if not np.allclose(slow_pairs, [4.0, 5.0, 1.0]):
        raise RuntimeError("Regression pair-distance check failed.")
    if not np.isclose(fast_all_pair_mean(regression_predictions), slow_pairs.mean()):
        raise RuntimeError("Fast all-pairs identity check failed.")

    renamed = np.where(predictions == 0, "control", "treated")
    if not np.array_equal(
        _primary_pair_values(predictions, "classification"),
        _primary_pair_values(renamed, "classification"),
    ):
        raise RuntimeError("Class-renaming invariance check failed.")

    for process_index, process in enumerate(SYNTHETIC_PROCESSES):
        X, _, y, _, task = generate_synthetic(
            process,
            seed=_stable_seed(314_159, "preflight", process_index),
            n_development=400,
            n_test=40,
        )
        configs = configurations(X, y, task)
        arm_counts = {
            arm: sum(config["arm"] == arm for config in configs) for arm in ARMS
        }
        if arm_counts != {"pruned_cart": 12, "bootstrap_split": 12}:
            raise RuntimeError(f"Effective-grid check failed for {process}.")

    selection_rows = [
        {
            "arm": "pruned_cart",
            "config_id": "cart-best",
            "validation_score": 0.90,
            "validation_instability": 0.20,
        },
        {
            "arm": "bootstrap_split",
            "config_id": "split-best",
            "validation_score": 0.87,
            "validation_instability": 0.01,
        },
    ]
    selected, _ = select_configurations(selection_rows, "classification")
    if selected["bootstrap_split"]["eligible"]:
        raise RuntimeError("Ineligible-arm retention check failed.")
    if selected["bootstrap_split"]["config_id"] != "split-best":
        raise RuntimeError("Ineligible-arm selection check failed.")

    return {
        "a_a": True,
        "a_a_paired_delta": True,
        "fast_vs_slow_pair": True,
        "class_renaming": True,
        "equal_effective_grid": True,
        "ineligible_arm_retained": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-datasets", type=int, default=20)
    parser.add_argument("--real-splits", type=int, default=16)
    parser.add_argument("--n-development", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=1_000)
    parser.add_argument("--validation-refits", type=int, default=16)
    parser.add_argument("--test-refits", type=int, default=20)
    parser.add_argument("--interval-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_818)
    parser.add_argument("--output", default="results/bootstrap_split_confirmatory")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def validate_confirmatory_settings(args: argparse.Namespace, output: Path) -> None:
    """Reject any nominal confirmatory run that differs from the frozen plan."""
    if not args.confirmatory:
        return
    mismatches = {
        key: (getattr(args, key), expected)
        for key, expected in FROZEN_SETTINGS.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise ValueError(f"Confirmatory settings differ from frozen plan: {mismatches}")
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite confirmatory output directory: {output}"
        )


def main() -> None:
    """Run the frozen study and write auditable evidence."""
    args = parse_args()
    if args.synthetic_datasets < 2:
        raise ValueError("synthetic_datasets must be at least 2.")
    if args.validation_refits < 4 or args.validation_refits % 2:
        raise ValueError("validation_refits must be an even integer of at least 4.")
    if args.test_refits < 4 or args.test_refits % 2:
        raise ValueError("test_refits must be an even integer of at least 4.")

    preflight = preflight_checks()
    output = Path(args.output)
    validate_confirmatory_settings(args, output)
    output.mkdir(parents=True, exist_ok=True)
    synthetic_rows, synthetic_tuning, synthetic_failures = run_synthetic(args)
    real_rows: list[dict[str, Any]] = []
    real_tuning: list[dict[str, Any]] = []
    real_failures: list[dict[str, Any]] = []
    if not args.skip_real:
        real_rows, real_tuning, real_failures = run_real(args)
    failures = synthetic_failures + real_failures
    expected_synthetic = len(SYNTHETIC_PROCESSES) * args.synthetic_datasets
    complete = len(synthetic_rows) == expected_synthetic and not failures
    synthetic_summary = (
        summarize_rows(synthetic_rows, args.seed, args.interval_resamples)
        if len(synthetic_rows) == expected_synthetic
        else None
    )
    real_summary = (
        summarize_rows(real_rows, args.seed + 1, args.interval_resamples)
        if real_rows and not real_failures
        else None
    )
    decisions = (
        apply_decision_rules(synthetic_summary, synthetic_failures)
        if synthetic_summary
        else None
    )
    config = vars(args) | {
        "plan_path": str(PLAN_PATH.relative_to(Path(__file__).parents[1])),
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "synthetic_processes": list(SYNTHETIC_PROCESSES),
        "real_datasets": list(REAL_DATASETS),
        "arms": list(ARMS),
        "configurations_per_arm": 12,
        "preflight_checks": preflight,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "estimator_sha256": hashlib.sha256(
            (
                Path(__file__).parents[1] / "experiments/bootstrap_split_tree.py"
            ).read_bytes()
        ).hexdigest(),
        "lockfile_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "uv.lock").read_bytes()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    evidence = {
        "complete": complete,
        "config": config,
        "failures": failures,
        "synthetic_summary": synthetic_summary,
        "real_repeated_split_summary": real_summary,
        "decisions": decisions,
        "artifact_validation": {
            "status": "pending" if synthetic_summary else "not_run"
        },
    }
    (output / "results.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False)
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    _write_csv(synthetic_rows + real_rows, output / "dataset_rows.csv")
    _write_csv(synthetic_tuning + real_tuning, output / "validation_rows.csv")
    summary_rows = []
    if synthetic_summary:
        summary_rows.extend(_flatten_summary(synthetic_summary, "synthetic"))
    if real_summary:
        summary_rows.extend(_flatten_summary(real_summary, "real_repeated_split"))
    _write_csv(summary_rows, output / "summary.csv")
    if synthetic_summary:
        _validate_artifacts(
            output,
            evidence,
            synthetic_failures,
            args.seed,
            args.interval_resamples,
        )
        evidence["artifact_validation"] = {
            "status": "passed",
            "json_roundtrip": True,
            "config_agreement": True,
            "summary_regeneration": True,
            "decision_regeneration": True,
            "flat_summary_agreement": True,
        }
        (output / "results.json").write_text(
            json.dumps(evidence, indent=2, allow_nan=False)
        )
        if json.loads((output / "results.json").read_text()) != evidence:
            raise RuntimeError("Final results JSON failed its round-trip check.")
    print(json.dumps(decisions, indent=2))
    print(f"created {output / 'results.json'}")
    if failures:
        raise RuntimeError(f"{len(failures)} fits failed; see results.json.")


if __name__ == "__main__":
    main()
