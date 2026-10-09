"""Run the frozen persistent-trunk falsification screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, log_loss, r2_score

from experiments.persistent_trunk import (
    PersistentTrunkTree,
    RefreshedLeafTree,
    aligned_probabilities,
    tree_factory,
)

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "experiments/designs/PERSISTENT_TRUNK_PLAN.md"
PROCESSES = ("binary", "multiclass", "regression")
REGIMES = ("stable", "moving")
ARMS = ("refit_cart", "frozen_tree", "refreshed_leaves", "persistent_trunk")
SETTINGS = {
    "units_per_cell": 24,
    "n_reference": 600,
    "n_update": 400,
    "validation_updates": 8,
    "final_updates": 16,
    "n_gate": 1_000,
    "n_test": 1_000,
    "max_depth": 4,
    "trunk_depth": 1,
    "subtree_depth": 3,
    "min_samples_leaf": 10,
    "gate_routing_threshold": 0.90,
    "gate_required_updates": 6,
    "interval_resamples": 20_000,
    "seed": 20_260_817,
    "output": "results/persistent_trunk_screen",
}
ROW_FIELDS = (
    "process",
    "regime",
    "unit",
    "reference_root_feature",
    "reference_root_threshold",
    "gate_mean_routing_agreement",
    "gate_pass_count",
    "gate_accept",
    "refit_cart_score",
    "refit_cart_instability",
    "refit_cart_probability_instability",
    "refit_cart_log_loss",
    "frozen_tree_score",
    "frozen_tree_instability",
    "frozen_tree_probability_instability",
    "frozen_tree_log_loss",
    "refreshed_leaves_score",
    "refreshed_leaves_instability",
    "refreshed_leaves_probability_instability",
    "refreshed_leaves_log_loss",
    "persistent_trunk_score",
    "persistent_trunk_instability",
    "persistent_trunk_probability_instability",
    "persistent_trunk_log_loss",
    "instability_effect_vs_cart",
    "score_delta_vs_cart",
    "score_delta_vs_frozen",
    "score_delta_vs_refreshed",
    "useful",
)


def stable_seed(*parts: Any) -> int:
    """Derive the frozen 32-bit seed from JSON-serializable parts."""
    payload = json.dumps((SETTINGS["seed"], *parts), separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def task_for(process: str) -> str:
    """Return the task for a frozen process."""
    return "regression" if process == "regression" else "classification"


def generate(
    process: str,
    *,
    root_feature: int,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one sample from an exact frozen process."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 8))
    root_sign = 2.0 * (X[:, root_feature] > 0) - 1.0
    descendant = np.where(
        X[:, root_feature] > 0,
        2.0 * (X[:, 1] > 0) - 1.0,
        2.0 * (X[:, 2] > 0) - 1.0,
    )
    if process == "binary":
        eta = 2.5 * root_sign + 1.25 * descendant
        probability = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, probability)
    elif process == "multiclass":
        logits = np.column_stack(
            (
                -2.4 * root_sign + 1.1 * (2.0 * (X[:, 1] > 0) - 1.0),
                2.4 * root_sign + 1.1 * (2.0 * (X[:, 2] > 0) - 1.0),
                1.3 * (2.0 * (X[:, 4] > 0) - 1.0),
            )
        )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        uniforms = rng.random(n_samples)
        y = (uniforms[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    elif process == "regression":
        y = 4.0 * root_sign + 2.0 * descendant + rng.normal(scale=1.5, size=n_samples)
    else:
        raise ValueError(f"unknown process: {process}")
    return X, y


def root_route(tree: Any, X: np.ndarray) -> np.ndarray:
    """Return the binary partition induced by a fitted CART root."""
    feature = int(tree.tree_.feature[0])
    if feature < 0:
        return np.zeros(len(X), dtype=bool)
    return X[:, feature] <= float(tree.tree_.threshold[0])


def routing_agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Compare binary partitions while allowing a left/right label reversal."""
    direct = float(np.mean(left == right))
    return max(direct, 1.0 - direct)


def all_pairs_instability(
    predictions: np.ndarray,
    *,
    task: str,
    y_test: np.ndarray,
) -> float:
    """Calculate the frozen all-pairs prediction distance."""
    distances = []
    denominator = float(np.var(y_test)) if task == "regression" else 1.0
    if task == "regression" and denominator <= np.finfo(float).eps:
        raise ValueError("regression test outcome must have positive variance")
    for left, right in combinations(range(len(predictions)), 2):
        if task == "classification":
            distances.append(float(np.mean(predictions[left] != predictions[right])))
        else:
            difference = predictions[left] - predictions[right]
            distances.append(float(np.mean(difference**2) / denominator))
    return float(np.mean(distances))


def probability_instability(probabilities: np.ndarray) -> float:
    """Calculate all-pairs mean squared probability-vector distance."""
    distances = [
        float(
            np.mean(np.sum((probabilities[left] - probabilities[right]) ** 2, axis=1))
        )
        for left, right in combinations(range(len(probabilities)), 2)
    ]
    return float(np.mean(distances))


def symmetric_effect(candidate: float, baseline: float) -> float:
    """Return the frozen bounded symmetric relative difference."""
    if candidate <= 1e-12 and baseline <= 1e-12:
        return 0.0
    denominator = candidate + baseline
    if denominator <= 1e-12:
        raise ValueError("symmetric effect has a near-zero denominator")
    return 2.0 * (candidate - baseline) / denominator


def _score(y: np.ndarray, prediction: np.ndarray, task: str) -> float:
    metric = accuracy_score if task == "classification" else r2_score
    return float(metric(y, prediction))


def evaluate_unit(process: str, regime: str, unit: int) -> dict[str, Any]:
    """Evaluate all workflows on one independently seeded problem."""
    task = task_for(process)
    update_root = 0 if regime == "stable" else 3
    X_reference, y_reference = generate(
        process,
        root_feature=0,
        n_samples=SETTINGS["n_reference"],
        seed=stable_seed(process, regime, unit, "reference", 0),
    )
    X_gate, _ = generate(
        process,
        root_feature=update_root,
        n_samples=SETTINGS["n_gate"],
        seed=stable_seed(process, regime, unit, "gate", 0),
    )
    X_test, y_test = generate(
        process,
        root_feature=update_root,
        n_samples=SETTINGS["n_test"],
        seed=stable_seed(process, regime, unit, "test", 0),
    )
    reference_seed = stable_seed(process, regime, unit, "reference_tree", 0)
    trunk = tree_factory(
        task,
        max_depth=SETTINGS["trunk_depth"],
        min_samples_leaf=SETTINGS["min_samples_leaf"],
        random_state=reference_seed,
    ).fit(X_reference, y_reference)
    frozen_tree = tree_factory(
        task,
        max_depth=SETTINGS["max_depth"],
        min_samples_leaf=SETTINGS["min_samples_leaf"],
        random_state=reference_seed,
    ).fit(X_reference, y_reference)
    classes = np.asarray(trunk.classes_) if task == "classification" else np.empty(0)
    frozen_route = root_route(trunk, X_gate)
    gate_agreements = []
    for replicate in range(SETTINGS["validation_updates"]):
        X_update, y_update = generate(
            process,
            root_feature=update_root,
            n_samples=SETTINGS["n_update"],
            seed=stable_seed(process, regime, unit, "validation", replicate),
        )
        validation_tree = tree_factory(
            task,
            max_depth=SETTINGS["max_depth"],
            min_samples_leaf=SETTINGS["min_samples_leaf"],
            random_state=stable_seed(
                process, regime, unit, "validation_tree", replicate
            ),
        ).fit(X_update, y_update)
        gate_agreements.append(
            routing_agreement(frozen_route, root_route(validation_tree, X_gate))
        )
    gate_pass_count = sum(
        value >= SETTINGS["gate_routing_threshold"] for value in gate_agreements
    )

    predictions: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    probabilities: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    losses: dict[str, list[float]] = {arm: [] for arm in ARMS}
    frozen_prediction = frozen_tree.predict(X_test)
    frozen_probability = (
        aligned_probabilities(frozen_tree, X_test, classes)
        if task == "classification"
        else None
    )
    for replicate in range(SETTINGS["final_updates"]):
        X_update, y_update = generate(
            process,
            root_feature=update_root,
            n_samples=SETTINGS["n_update"],
            seed=stable_seed(process, regime, unit, "final", replicate),
        )
        fit_seed = stable_seed(process, regime, unit, "final_tree", replicate)
        refit = tree_factory(
            task,
            max_depth=SETTINGS["max_depth"],
            min_samples_leaf=SETTINGS["min_samples_leaf"],
            random_state=fit_seed,
        ).fit(X_update, y_update)
        refreshed = RefreshedLeafTree(frozen_tree, task=task).fit(X_update, y_update)
        persistent = PersistentTrunkTree(
            trunk,
            task=task,
            subtree_depth=SETTINGS["subtree_depth"],
            min_samples_leaf=SETTINGS["min_samples_leaf"],
            random_state=fit_seed,
        ).fit(X_update, y_update)
        fitted = {
            "refit_cart": refit,
            "refreshed_leaves": refreshed,
            "persistent_trunk": persistent,
        }
        for arm in ARMS:
            prediction = (
                frozen_prediction
                if arm == "frozen_tree"
                else fitted[arm].predict(X_test)
            )
            predictions[arm].append(np.asarray(prediction))
            scores[arm].append(_score(y_test, prediction, task))
            if task == "classification":
                probability = (
                    frozen_probability
                    if arm == "frozen_tree"
                    else (
                        aligned_probabilities(fitted[arm], X_test, classes)
                        if arm == "refit_cart"
                        else fitted[arm].predict_proba(X_test)
                    )
                )
                probabilities[arm].append(np.asarray(probability))
                losses[arm].append(float(log_loss(y_test, probability, labels=classes)))

    row: dict[str, Any] = {
        "process": process,
        "regime": regime,
        "unit": unit,
        "reference_root_feature": int(trunk.tree_.feature[0]),
        "reference_root_threshold": float(trunk.tree_.threshold[0]),
        "gate_mean_routing_agreement": float(np.mean(gate_agreements)),
        "gate_pass_count": gate_pass_count,
        "gate_accept": gate_pass_count >= SETTINGS["gate_required_updates"],
    }
    for arm in ARMS:
        arm_predictions = np.asarray(predictions[arm])
        row[f"{arm}_score"] = float(np.mean(scores[arm]))
        row[f"{arm}_instability"] = all_pairs_instability(
            arm_predictions, task=task, y_test=y_test
        )
        if task == "classification":
            row[f"{arm}_probability_instability"] = probability_instability(
                np.asarray(probabilities[arm])
            )
            row[f"{arm}_log_loss"] = float(np.mean(losses[arm]))
        else:
            row[f"{arm}_probability_instability"] = None
            row[f"{arm}_log_loss"] = None
    row["instability_effect_vs_cart"] = symmetric_effect(
        row["persistent_trunk_instability"], row["refit_cart_instability"]
    )
    row["score_delta_vs_cart"] = row["persistent_trunk_score"] - row["refit_cart_score"]
    row["score_delta_vs_frozen"] = (
        row["persistent_trunk_score"] - row["frozen_tree_score"]
    )
    row["score_delta_vs_refreshed"] = (
        row["persistent_trunk_score"] - row["refreshed_leaves_score"]
    )
    tolerance = 0.02 if task == "regression" else 0.01
    row["useful"] = bool(
        row["persistent_trunk_instability"] <= row["refit_cart_instability"]
        and row["score_delta_vs_cart"] >= -tolerance
    )
    return row


def percentile_interval(
    values: np.ndarray, *, seed: int, n_resamples: int | None = None
) -> dict[str, float]:
    """Return a deterministic independent-unit percentile interval."""
    values = np.asarray(values, dtype=float)
    count = SETTINGS["interval_resamples"] if n_resamples is None else n_resamples
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(np.mean(values)), "low": float(low), "high": float(high)}


def stratified_interval(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    """Return an equal-process interval resampling units within process."""
    rng = np.random.default_rng(stable_seed("summary", field))
    samples = []
    observed = []
    for process in PROCESSES:
        values = np.asarray(
            [row[field] for row in rows if row["process"] == process], dtype=float
        )
        observed.append(float(np.mean(values)))
        indices = rng.integers(
            0,
            len(values),
            size=(SETTINGS["interval_resamples"], len(values)),
        )
        samples.append(values[indices].mean(axis=1))
    distribution = np.mean(samples, axis=0)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {"mean": float(np.mean(observed)), "low": float(low), "high": float(high)}


def wilson_lower(successes: int, total: int) -> float:
    """Return the two-sided 95% Wilson interval's lower bound."""
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = proportion + z**2 / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
    )
    return float((center - radius) / denominator)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct every reported result and the frozen advance decision."""
    cells = []
    for process in PROCESSES:
        for regime in REGIMES:
            selected = [
                row
                for row in rows
                if row["process"] == process and row["regime"] == regime
            ]
            gate_successes = sum(
                row["gate_accept"] if regime == "stable" else not row["gate_accept"]
                for row in selected
            )
            fields = (
                "instability_effect_vs_cart",
                "score_delta_vs_cart",
                "score_delta_vs_frozen",
                "score_delta_vs_refreshed",
            )
            intervals = {
                field: percentile_interval(
                    np.asarray([row[field] for row in selected]),
                    seed=stable_seed("cell", process, regime, field),
                )
                for field in fields
            }
            cells.append(
                {
                    "process": process,
                    "regime": regime,
                    "n_units": len(selected),
                    "useful_count": sum(row["useful"] for row in selected),
                    "gate_success_count": gate_successes,
                    "gate_success_wilson_low": wilson_lower(
                        gate_successes, len(selected)
                    ),
                    "intervals": intervals,
                    "arm_means": {
                        arm: {
                            "score": float(
                                np.mean([row[f"{arm}_score"] for row in selected])
                            ),
                            "instability": float(
                                np.mean([row[f"{arm}_instability"] for row in selected])
                            ),
                            "probability_instability": (
                                float(
                                    np.mean(
                                        [
                                            row[f"{arm}_probability_instability"]
                                            for row in selected
                                        ]
                                    )
                                )
                                if process != "regression"
                                else None
                            ),
                            "log_loss": (
                                float(
                                    np.mean(
                                        [row[f"{arm}_log_loss"] for row in selected]
                                    )
                                )
                                if process != "regression"
                                else None
                            ),
                        }
                        for arm in ARMS
                    },
                }
            )
    stable_rows = [row for row in rows if row["regime"] == "stable"]
    stable_cells = [cell for cell in cells if cell["regime"] == "stable"]
    moving_cells = [cell for cell in cells if cell["regime"] == "moving"]
    stable_effect = stratified_interval(stable_rows, "instability_effect_vs_cart")
    decisions = {
        "stable_instability": stable_effect["high"] < 0.0,
        "stable_score": all(
            cell["intervals"]["score_delta_vs_cart"]["low"]
            > (-0.02 if cell["process"] == "regression" else -0.01)
            for cell in stable_cells
        ),
        "stable_useful_rate": sum(row["useful"] for row in stable_rows) >= 54,
        "beats_constrained_baselines": all(
            cell["intervals"]["score_delta_vs_frozen"]["mean"] > 0.0
            and cell["intervals"]["score_delta_vs_refreshed"]["mean"] > 0.0
            for cell in stable_cells
        ),
        "gate": all(
            cell["gate_success_count"] >= 20 and cell["gate_success_wilson_low"] > 0.65
            for cell in (*stable_cells, *moving_cells)
        ),
    }
    decisions["advance"] = all(decisions.values())
    return {
        "stable_aggregate_instability_effect": stable_effect,
        "stable_useful_count": sum(row["useful"] for row in stable_rows),
        "stable_unit_count": len(stable_rows),
        "cells": cells,
        "decisions": decisions,
    }


def write_rows(rows: list[dict[str, Any]], destination: Path) -> None:
    """Write the flat auditable unit rows."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    """Run the full frozen screen and persist its evidence."""
    started = time.monotonic()
    rows = []
    for process in PROCESSES:
        for regime in REGIMES:
            for unit in range(SETTINGS["units_per_cell"]):
                rows.append(evaluate_unit(process, regime, unit))
            print(f"completed {process}/{regime}")
    result = {
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "settings": SETTINGS,
        "processes": PROCESSES,
        "regimes": REGIMES,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "rows": rows,
        "summary": summarize(rows),
        "elapsed_seconds": time.monotonic() - started,
    }
    output = ROOT / SETTINGS["output"]
    output.mkdir(parents=True, exist_ok=True)
    write_rows(rows, output / "unit_rows.csv")
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(outcome["summary"]["decisions"], indent=2, sort_keys=True))
