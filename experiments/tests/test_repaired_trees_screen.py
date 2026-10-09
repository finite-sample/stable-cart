import json
from pathlib import Path

import numpy as np

from experiments.repaired_trees_screen import (
    ARMS,
    PROCESSES,
    SETTINGS,
    configurations,
    generate,
    instability,
    select,
    summarize,
)


def test_screen_generators_are_reproducible_and_include_multiclass():
    first = generate("classification_multiclass", 7, 40, 60)
    second = generate("classification_multiclass", 7, 40, 60)

    for left, right in zip(first[:4], second[:4], strict=True):
        assert np.array_equal(left, right)
    assert first[0].shape == (40, 8)
    assert first[1].shape == (60, 8)
    assert first[4] == "classification"
    assert len(np.unique(np.concatenate((first[2], first[3])))) == 3


def test_screen_has_equal_nonduplicate_tuning_budgets():
    X_train, _, y_train, _, task = generate("regression_step", 9, 80, 20)
    configs = configurations(X_train, y_train, task)

    assert {
        arm: sum(row["arm"] == arm for row in configs) for arm in ARMS
    } == dict.fromkeys(ARMS, 8)
    assert len({row["config_id"] for row in configs}) == 24


def test_instability_has_an_exact_small_reference():
    predictions = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0]])
    expected = np.mean([1 / 4, 3 / 4, 2 / 4])
    assert instability(predictions, "classification", predictions[0]) == expected

    regression = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
    variance = np.var(np.array([0.0, 2.0]))
    expected_regression = np.mean([1.0 / variance, 4.0 / variance, 1.0 / variance])
    assert (
        instability(regression, "regression", np.array([0.0, 2.0]))
        == expected_regression
    )


def test_selection_uses_common_cart_floor_and_marks_ineligible():
    rows = [
        {"arm": "pruned_cart", "config_id": "c1", "score": 0.8, "instability": 0.3},
        {"arm": "pruned_cart", "config_id": "c2", "score": 0.79, "instability": 0.2},
        {
            "arm": "variance_penalized",
            "config_id": "v1",
            "score": 0.79,
            "instability": 0.1,
        },
        {
            "arm": "variance_penalized",
            "config_id": "v2",
            "score": 0.75,
            "instability": 0.0,
        },
        {"arm": "robust_prefix", "config_id": "r1", "score": 0.78, "instability": 0.1},
        {"arm": "robust_prefix", "config_id": "r2", "score": 0.77, "instability": 0.0},
    ]
    selected, floor = select(rows, "classification")

    assert floor == 0.79
    assert selected["pruned_cart"]["config_id"] == "c2"
    assert selected["variance_penalized"]["config_id"] == "v1"
    assert selected["variance_penalized"]["eligible"] is True
    assert selected["robust_prefix"]["config_id"] == "r1"
    assert selected["robust_prefix"]["eligible"] is False


def test_frozen_screen_artifact_is_complete_and_reconstructs_summary():
    path = Path("results/repaired_trees_screen/results.json")
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["settings"] == SETTINGS
    assert tuple(result["processes"]) == PROCESSES
    assert len(result["rows"]) == len(PROCESSES) * SETTINGS["datasets_per_process"]
    assert {(row["process"], row["unit"]) for row in result["rows"]} == {
        (process, unit)
        for process in PROCESSES
        for unit in range(SETTINGS["datasets_per_process"])
    }
    reconstructed = summarize(result["rows"], SETTINGS["seed"])
    assert reconstructed == result["summary"]
    assert not any(
        decision["advance"] for decision in result["summary"]["decisions"].values()
    )


def test_task_interval_holds_process_mix_fixed():
    from experiments.repaired_trees_screen import stratified_interval

    result = stratified_interval([[0.0] * 6, [10.0] * 6, [20.0] * 6], seed=1)
    assert result == {"mean": 10.0, "low": 10.0, "high": 10.0, "n": 18}
