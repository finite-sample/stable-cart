"""Frozen screen of the repaired tree candidates against pruned CART."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from experiments.repaired_trees import (
    BootstrapVariancePenalizedTree,
    RobustPrefixHonestTree,
)

PROCESSES = (
    "classification_logistic",
    "classification_xor",
    "classification_multiclass",
    "regression_step",
    "regression_redundant",
    "regression_friedman1",
)
ARMS = ("pruned_cart", "variance_penalized", "robust_prefix")
SETTINGS = {
    "datasets_per_process": 6,
    "n_development": 400,
    "n_test": 600,
    "validation_refits": 6,
    "test_refits": 8,
    "node_bootstraps": 16,
    "max_candidates_per_feature": 16,
    "interval_resamples": 10_000,
    "seed": 20_260_817,
    "output": "results/repaired_trees_screen",
}


def stable_seed(*parts: Any) -> int:
    """Create a deterministic 32-bit seed from JSON-serializable parts."""
    import hashlib

    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def task_for(process: str) -> str:
    """Return the prediction task encoded by a process name."""
    return "classification" if process.startswith("classification") else "regression"


def generate(
    process: str, seed: int, n_development: int, n_test: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate one development/test unit from a frozen process."""
    n = n_development + n_test
    rng = np.random.default_rng(seed)
    task = task_for(process)
    if process == "classification_logistic":
        X = rng.normal(size=(n, 8))
        probability = 1.0 / (1.0 + np.exp(-3.0 * X[:, 0]))
        y = rng.binomial(1, probability)
    elif process == "classification_xor":
        X = rng.normal(size=(n, 8))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        flips = rng.random(n) < 0.1
        y[flips] = 1 - y[flips]
    elif process == "classification_multiclass":
        X = rng.normal(size=(n, 8))
        logits = np.column_stack(
            (2.2 * X[:, 0], -1.7 * X[:, 0] + 1.4 * X[:, 1], -1.3 * X[:, 1])
        )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        uniforms = rng.random(n)
        y = (uniforms[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    elif process == "regression_step":
        X = rng.normal(size=(n, 8))
        y = 4.0 * (X[:, 0] > 0) + rng.normal(size=n)
    elif process == "regression_redundant":
        latent = rng.normal(size=n)
        X = np.column_stack(
            (
                latent + 0.25 * rng.normal(size=n),
                latent + 0.25 * rng.normal(size=n),
                rng.normal(size=(n, 6)),
            )
        )
        y = 3.0 * latent + rng.normal(size=n)
    elif process == "regression_friedman1":
        X, y = make_friedman1(
            n_samples=n,
            n_features=8,
            noise=1.0,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown process: {process}")
    return (
        X[:n_development],
        X[n_development:],
        y[:n_development],
        y[n_development:],
        task,
    )


def tree_class(task: str) -> type[DecisionTreeClassifier] | type[DecisionTreeRegressor]:
    """Return the scikit-learn CART class for a task."""
    return DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor


def median_alpha(
    X: np.ndarray, y: np.ndarray, task: str, depth: int, leaf: int
) -> float:
    """Return the median distinct positive alpha from CART's pruning path."""
    estimator = tree_class(task)(
        max_depth=depth,
        min_samples_leaf=leaf,
        min_samples_split=30,
        random_state=0,
    )
    path = estimator.cost_complexity_pruning_path(X, y)
    positive = np.unique(np.asarray(path.ccp_alphas, dtype=float))
    positive = positive[positive > 0]
    return float(np.median(positive)) if len(positive) else 0.0


def configurations(X: np.ndarray, y: np.ndarray, task: str) -> list[dict[str, Any]]:
    """Build the frozen eight-configuration grid for each primary arm."""
    configs: list[dict[str, Any]] = []
    for depth in (3, 5):
        for leaf in (5, 15):
            for alpha_name, alpha in (
                ("zero", 0.0),
                ("median", median_alpha(X, y, task, depth, leaf)),
            ):
                configs.append(
                    {
                        "arm": "pruned_cart",
                        "config_id": f"cart:d={depth}:leaf={leaf}:alpha={alpha_name}",
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "ccp_alpha": alpha,
                    }
                )
            for penalty in (1.0, 8.0):
                configs.append(
                    {
                        "arm": "variance_penalized",
                        "config_id": f"variance:d={depth}:leaf={leaf}:lambda={penalty:g}",
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "variance_penalty": penalty,
                    }
                )
            for threshold in (0.0, 0.2):
                configs.append(
                    {
                        "arm": "robust_prefix",
                        "config_id": f"prefix:d={depth}:leaf={leaf}:support={threshold:g}",
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "consensus_threshold": threshold,
                    }
                )
    counts = {arm: sum(config["arm"] == arm for config in configs) for arm in ARMS}
    if counts != dict.fromkeys(ARMS, 8):
        raise RuntimeError(f"unequal tuning budget: {counts}")
    return configs


def make_estimator(
    config: dict[str, Any], task: str, seed: int, *, ablation: bool = False
) -> Any:
    """Instantiate a primary configuration or its named-mechanism ablation."""
    common = {
        "max_depth": config["max_depth"],
        "min_samples_leaf": config["min_samples_leaf"],
        "min_samples_split": 30,
        "random_state": seed,
    }
    if config["arm"] == "pruned_cart":
        return tree_class(task)(ccp_alpha=config["ccp_alpha"], **common)
    research_common = common | {
        "task": task,
        "n_bootstrap": SETTINGS["node_bootstraps"],
        "max_candidates_per_feature": SETTINGS["max_candidates_per_feature"],
    }
    if config["arm"] == "variance_penalized":
        return BootstrapVariancePenalizedTree(
            variance_penalty=0.0 if ablation else config["variance_penalty"],
            **research_common,
        )
    return RobustPrefixHonestTree(
        prefix_levels=0 if ablation else 1,
        consensus_threshold=config["consensus_threshold"],
        estimation_fraction=0.5,
        **research_common,
    )


def score(y: np.ndarray, prediction: np.ndarray, task: str) -> float:
    """Score a prediction using the frozen task metric."""
    metric = accuracy_score if task == "classification" else r2_score
    return float(metric(y, prediction))


def instability(predictions: np.ndarray, task: str, y: np.ndarray) -> float:
    """Calculate the frozen all-pairs instability measure."""
    values = []
    for left in range(len(predictions) - 1):
        for right in range(left + 1, len(predictions)):
            if task == "classification":
                values.append(float(np.mean(predictions[left] != predictions[right])))
            else:
                denominator = float(np.var(y))
                values.append(
                    float(
                        np.mean((predictions[left] - predictions[right]) ** 2)
                        / denominator
                    )
                )
    return float(np.mean(values))


def evaluate_config(
    config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    task: str,
    samples: list[np.ndarray],
    seed: int,
    *,
    ablation: bool = False,
) -> dict[str, float]:
    """Fit one configuration on paired resamples and evaluate it."""
    predictions = []
    scores = []
    times = []
    for refit, indices in enumerate(samples):
        estimator = make_estimator(
            config,
            task,
            stable_seed(seed, config["config_id"], "model", refit),
            ablation=ablation,
        )
        started = time.perf_counter()
        estimator.fit(X_train[indices], y_train[indices])
        times.append(time.perf_counter() - started)
        prediction = np.asarray(estimator.predict(X_evaluation))
        predictions.append(prediction)
        scores.append(score(y_evaluation, prediction, task))
    prediction_array = np.stack(predictions)
    return {
        "score": float(np.mean(scores)),
        "instability": instability(prediction_array, task, y_evaluation),
        "mean_fit_seconds": float(np.mean(times)),
    }


def paired_samples(n: int, count: int, seed: int, stage: str) -> list[np.ndarray]:
    """Generate the paired row-bootstrap samples for one stage."""
    return [
        np.random.default_rng(stable_seed(seed, stage, refit)).integers(0, n, n)
        for refit in range(count)
    ]


def select(
    rows: list[dict[str, Any]], task: str
) -> tuple[dict[str, dict[str, Any]], float]:
    """Select one configuration per arm under the common CART score floor."""
    tolerance = 0.01 if task == "classification" else 0.02
    floor = max(row["score"] for row in rows if row["arm"] == "pruned_cart") - tolerance
    selected = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        eligible = [row for row in arm_rows if row["score"] >= floor]
        if eligible:
            choice = min(
                eligible,
                key=lambda row: (row["instability"], -row["score"], row["config_id"]),
            )
        else:
            choice = min(arm_rows, key=lambda row: (-row["score"], row["config_id"]))
        selected[arm] = choice | {"eligible": bool(eligible)}
    return selected, floor


def run_unit(
    process: str, unit: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune and evaluate every arm for one independently generated dataset."""
    dataset_seed = stable_seed(seed, process, unit, "dataset")
    X_development, X_test, y_development, y_test, task = generate(
        process,
        dataset_seed,
        SETTINGS["n_development"],
        SETTINGS["n_test"],
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=1 / 3,
        random_state=stable_seed(seed, process, unit, "split"),
        stratify=y_development if task == "classification" else None,
    )
    validation_samples = paired_samples(
        len(X_train), SETTINGS["validation_refits"], dataset_seed, "validation"
    )
    configs = configurations(X_train, y_train, task)
    validation_rows = []
    by_id = {}
    for config in configs:
        result = evaluate_config(
            config,
            X_train,
            y_train,
            X_validation,
            y_validation,
            task,
            validation_samples,
            stable_seed(dataset_seed, "validation"),
        )
        validation_rows.append(config | result)
        by_id[config["config_id"]] = config
    selected, floor = select(validation_rows, task)

    test_samples = paired_samples(
        len(X_development), SETTINGS["test_refits"], dataset_seed, "test"
    )
    test_results = {}
    for arm in ARMS:
        config = by_id[selected[arm]["config_id"]]
        test_results[arm] = evaluate_config(
            config,
            X_development,
            y_development,
            X_test,
            y_test,
            task,
            test_samples,
            stable_seed(dataset_seed, "test"),
        )
    ablations = {}
    for arm in ("variance_penalized", "robust_prefix"):
        config = by_id[selected[arm]["config_id"]]
        ablations[arm] = evaluate_config(
            config,
            X_development,
            y_development,
            X_test,
            y_test,
            task,
            test_samples,
            stable_seed(dataset_seed, "test"),
            ablation=True,
        )

    cart = test_results["pruned_cart"]
    row: dict[str, Any] = {
        "process": process,
        "unit": unit,
        "dataset_seed": dataset_seed,
        "task": task,
        "score_floor": floor,
    }
    for arm in ARMS:
        values = test_results[arm]
        row[f"{arm}_config"] = selected[arm]["config_id"]
        row[f"{arm}_eligible"] = selected[arm]["eligible"]
        for metric, value in values.items():
            row[f"{arm}_{metric}"] = value
        if arm != "pruned_cart":
            row[f"{arm}_instability_delta"] = (
                values["instability"] - cart["instability"]
            )
            denominator = values["instability"] + cart["instability"]
            row[f"{arm}_instability_effect_percent"] = (
                0.0
                if denominator == 0
                else 200.0 * (values["instability"] - cart["instability"]) / denominator
            )
            row[f"{arm}_score_delta"] = values["score"] - cart["score"]
            row[f"{arm}_fit_time_ratio"] = (
                values["mean_fit_seconds"] / cart["mean_fit_seconds"]
            )
            row[f"{arm}_ablation_instability"] = ablations[arm]["instability"]
            row[f"{arm}_mechanism_delta"] = (
                values["instability"] - ablations[arm]["instability"]
            )
    return row, validation_rows


def interval(values: list[float], seed: int) -> dict[str, float]:
    """Return a dataset-bootstrap mean and 95% interval."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(array), size=(SETTINGS["interval_resamples"], len(array))
    )
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "n": len(array),
    }


def stratified_interval(groups, seed):
    """Keep process weights fixed while resampling datasets within processes."""
    rng = np.random.default_rng(seed)
    bootstrap = []
    for values in groups:
        array = np.asarray(values, dtype=float)
        indices = rng.integers(
            len(array), size=(SETTINGS["interval_resamples"], len(array))
        )
        bootstrap.append(array[indices].mean(axis=1))
    means = np.mean(bootstrap, axis=0)
    return {
        "mean": float(np.mean([np.mean(group) for group in groups])),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "n": sum(len(group) for group in groups),
    }


def summarize(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Summarize task and process effects and apply the frozen screen."""
    summary: dict[str, Any] = {"processes": {}, "tasks": {}, "decisions": {}}
    for process in PROCESSES:
        selected = [row for row in rows if row["process"] == process]
        summary["processes"][process] = {}
        for arm in ("variance_penalized", "robust_prefix"):
            summary["processes"][process][arm] = {
                metric: interval(
                    [row[f"{arm}_{metric}"] for row in selected],
                    stable_seed(seed, process, arm, metric),
                )
                for metric in (
                    "instability_delta",
                    "instability_effect_percent",
                    "score_delta",
                    "mechanism_delta",
                    "fit_time_ratio",
                )
            }
    for task in ("classification", "regression"):
        selected = [row for row in rows if row["task"] == task]
        summary["tasks"][task] = {}
        for arm in ("variance_penalized", "robust_prefix"):
            metrics = {
                metric: stratified_interval(
                    [
                        [
                            row[f"{arm}_{metric}"]
                            for row in selected
                            if row["process"] == process
                        ]
                        for process in PROCESSES
                        if task_for(process) == task
                    ],
                    stable_seed(seed, task, arm, metric),
                )
                for metric in (
                    "instability_delta",
                    "instability_effect_percent",
                    "score_delta",
                    "mechanism_delta",
                    "fit_time_ratio",
                )
            }
            process_means = [
                summary["processes"][process][arm]["instability_delta"]["mean"]
                for process in PROCESSES
                if task_for(process) == task
            ]
            tolerance = 0.01 if task == "classification" else 0.02
            checks = {
                "negative_mean_instability": metrics["instability_delta"]["mean"] < 0,
                "two_of_three_processes_negative": sum(
                    value < 0 for value in process_means
                )
                >= 2,
                "score_within_tolerance": metrics["score_delta"]["mean"] >= -tolerance,
                "mechanism_no_worse_than_ablation": metrics["mechanism_delta"]["mean"]
                <= 0,
            }
            summary["tasks"][task][arm] = metrics | {"checks": checks}
            summary["decisions"][f"{task}:{arm}"] = {
                "advance": all(checks.values()),
                "checks": checks,
            }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a rectangular CSV file."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path) -> dict[str, Any]:
    """Execute the frozen screen and write reconstructable artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    validation_rows = []
    for process in PROCESSES:
        for unit in range(SETTINGS["datasets_per_process"]):
            print(
                f"{process} {unit + 1}/{SETTINGS['datasets_per_process']}", flush=True
            )
            row, validation = run_unit(process, unit, SETTINGS["seed"])
            rows.append(row)
            validation_rows.extend(
                {"process": process, "unit": unit} | item for item in validation
            )
    result = {
        "settings": SETTINGS,
        "processes": PROCESSES,
        "rows": rows,
        "summary": summarize(rows, SETTINGS["seed"]),
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output / "dataset_rows.csv", rows)
    write_csv(output / "validation_rows.csv", validation_rows)
    return result


def main() -> None:
    """Run from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(SETTINGS["output"]))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["summary"]["decisions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
