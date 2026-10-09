"""Falsification tests for the frozen BootstrapSplitTree comparison."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.bootstrap_split_confirmatory import (
    ARMS,
    FROZEN_SETTINGS,
    SYNTHETIC_PROCESSES,
    _primary_pair_values,
    _read_summary_inputs,
    apply_decision_rules,
    configurations,
    generate_synthetic,
    preflight_checks,
    select_configurations,
    summarize_rows,
    validate_confirmatory_settings,
)
from experiments.representative_confirmatory import symmetric_effect

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tuning_budgets_are_equal(task):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 5))
    y = (
        rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
        if task == "classification"
        else X[:, 0] + rng.normal(scale=0.5, size=len(X))
    )

    configs = configurations(X, y, task)

    assert {arm: sum(row["arm"] == arm for row in configs) for arm in ARMS} == {
        "pruned_cart": 12,
        "bootstrap_split": 12,
    }
    assert len({row["config_id"] for row in configs}) == 24
    assert all(
        row["leaf_shrinkage"] == 0 for row in configs if row["arm"] == "bootstrap_split"
    )
    assert (
        len(
            {
                (
                    row["arm"],
                    row["max_depth"],
                    row["min_samples_leaf"],
                    row["min_samples_split"],
                    row.get("ccp_alpha"),
                    row.get("consensus_threshold"),
                    row.get("leaf_shrinkage"),
                )
                for row in configs
            }
        )
        == 24
    )


def test_selection_uses_cart_floor_and_retains_ineligible_split_arm():
    rows = [
        {
            "arm": "pruned_cart",
            "config_id": "cart-best-score",
            "validation_score": 0.90,
            "validation_instability": 0.20,
        },
        {
            "arm": "pruned_cart",
            "config_id": "cart-stable",
            "validation_score": 0.895,
            "validation_instability": 0.10,
        },
        {
            "arm": "bootstrap_split",
            "config_id": "split-best-score",
            "validation_score": 0.88,
            "validation_instability": 0.05,
        },
        {
            "arm": "bootstrap_split",
            "config_id": "split-stable",
            "validation_score": 0.87,
            "validation_instability": 0.01,
        },
    ]

    selected, score_floor = select_configurations(rows, "classification")

    assert score_floor == pytest.approx(0.89)
    assert selected["pruned_cart"]["config_id"] == "cart-stable"
    assert selected["pruned_cart"]["eligible"] is True
    assert selected["bootstrap_split"]["config_id"] == "split-best-score"
    assert selected["bootstrap_split"]["eligible"] is False


def test_primary_classification_metric_is_label_invariant():
    numeric = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    renamed = np.where(numeric == 0, "control", "treated")

    assert np.array_equal(
        _primary_pair_values(numeric, "classification"),
        _primary_pair_values(renamed, "classification"),
    )


def test_primary_regression_metric_matches_slow_values():
    predictions = np.array([[0.0, 2.0], [2.0, 4.0], [1.0, 5.0]])

    assert np.allclose(
        _primary_pair_values(predictions, "regression"),
        [4.0, 5.0, 1.0],
    )


def test_a_a_effect_is_exactly_zero():
    predictions = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]])
    instability = _primary_pair_values(predictions, "classification").mean()

    assert symmetric_effect(instability, instability) == 0


def test_runner_preflight_checks_pass():
    assert all(preflight_checks().values())


def test_confirmatory_mode_rejects_changed_settings(tmp_path):
    values = FROZEN_SETTINGS | {"synthetic_datasets": 2, "confirmatory": True}

    with pytest.raises(ValueError, match="synthetic_datasets"):
        validate_confirmatory_settings(SimpleNamespace(**values), tmp_path / "new")


def test_confirmatory_mode_refuses_existing_output(tmp_path):
    values = FROZEN_SETTINGS | {"confirmatory": True}
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        validate_confirmatory_settings(SimpleNamespace(**values), existing)


def _passing_decision_summary():
    def statistic(mean, low, high):
        return {
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "ci_98_low": low,
            "ci_98_high": high,
            "ci_family_low": low,
            "ci_family_high": high,
            "n_units": 20,
        }

    cells = {}
    for task in ("classification", "regression"):
        for index in range(3):
            cells[f"{task}_{index}"] = {
                "task": task,
                "instability_effect_percent": statistic(-10, -12, -8),
                "score_delta": statistic(0, 0, 0),
                "eligibility_rate": 1.0,
            }
    task_summary = {
        task: {
            "instability_effect_percent": statistic(-10, -12, -8),
            "normalized_instability_delta": statistic(-0.01, -0.02, -0.005),
            "score_delta": statistic(0, 0, 0),
        }
        for task in ("classification", "regression")
    }
    return {
        "grand": statistic(-10, -12, -8),
        "grand_normalized_instability_delta": statistic(-0.01, -0.02, -0.005),
        "tasks": task_summary,
        "cells": cells,
    }


def test_task_decision_ignores_failures_in_the_other_task():
    decision = apply_decision_rules(
        _passing_decision_summary(),
        [{"process": "regression_unique_step", "error": "deliberate"}],
    )

    assert decision["classification"]["supported"] is True
    assert decision["regression"]["supported"] is False
    assert decision["broad"]["supported"] is False


def test_cell_score_harm_vetoes_otherwise_passing_task():
    summary = _passing_decision_summary()
    summary["cells"]["classification_0"]["score_delta"]["ci_low"] = -0.011

    decision = apply_decision_rules(summary, [])

    assert decision["classification"]["supported"] is False
    assert (
        decision["classification"]["checks"]["cell_performance_guardrails_pass"]
        is False
    )


def test_task_effect_uses_familywise_interval():
    summary = _passing_decision_summary()
    summary["tasks"]["classification"]["instability_effect_percent"][
        "ci_family_high"
    ] = 0.1

    decision = apply_decision_rules(summary, [])

    assert decision["classification"]["supported"] is False
    assert decision["classification"]["checks"]["effect_interval_below_zero"] is False


def test_tiny_absolute_change_cannot_pass_on_relative_effect_alone():
    summary = _passing_decision_summary()
    summary["tasks"]["classification"]["normalized_instability_delta"] = {
        "mean": -1e-12,
        "ci_low": -2e-12,
        "ci_high": -0.5e-12,
        "ci_98_low": -2e-12,
        "ci_98_high": -0.5e-12,
        "ci_family_low": -2e-12,
        "ci_family_high": -0.5e-12,
        "n_units": 20,
    }

    decision = apply_decision_rules(summary, [])

    assert decision["classification"]["supported"] is False
    assert (
        decision["classification"]["checks"]["absolute_effect_at_least_threshold"]
        is False
    )


def test_committed_run_matches_frozen_sources_and_dimensions():
    output = ROOT / "results/bootstrap_split_confirmatory"
    evidence = json.loads((output / "results.json").read_text())
    config = evidence["config"]

    assert evidence["complete"] is True
    assert evidence["failures"] == []
    assert evidence["artifact_validation"]["status"] == "passed"
    assert evidence["decisions"]["package_recommendation"] == "remove_from_package"
    assert (
        config["plan_sha256"]
        == hashlib.sha256(
            (
                ROOT / "experiments/designs/BOOTSTRAP_SPLIT_CONFIRMATORY_PLAN.md"
            ).read_bytes()
        ).hexdigest()
    )
    rows = _read_summary_inputs(output / "dataset_rows.csv")
    assert sum(row["source"] == "synthetic" for row in rows) == 6 * 20
    assert sum(row["source"] == "real_repeated_split" for row in rows) == 3 * 16


def test_committed_summaries_and_decisions_regenerate_from_raw_rows():
    output = ROOT / "results/bootstrap_split_confirmatory"
    evidence = json.loads((output / "results.json").read_text())
    config = evidence["config"]
    rows = _read_summary_inputs(output / "dataset_rows.csv")
    synthetic_rows = [row for row in rows if row["source"] == "synthetic"]
    real_rows = [row for row in rows if row["source"] == "real_repeated_split"]

    synthetic = summarize_rows(
        synthetic_rows, config["seed"], config["interval_resamples"]
    )
    real = summarize_rows(real_rows, config["seed"] + 1, config["interval_resamples"])

    assert synthetic == evidence["synthetic_summary"]
    assert real == evidence["real_repeated_split_summary"]
    assert apply_decision_rules(synthetic, []) == evidence["decisions"]


@pytest.mark.parametrize("process", SYNTHETIC_PROCESSES)
def test_synthetic_units_have_declared_shape_and_finite_values(process):
    X_development, X_test, y_development, y_test, task = generate_synthetic(
        process, seed=71, n_development=120, n_test=80
    )

    assert X_development.shape == (120, 10)
    assert X_test.shape == (80, 10)
    assert y_development.shape == (120,)
    assert y_test.shape == (80,)
    assert np.isfinite(X_development).all()
    assert np.isfinite(X_test).all()
    assert np.isfinite(y_development).all()
    assert np.isfinite(y_test).all()
    if task == "classification":
        assert np.array_equal(np.unique(y_development), [0, 1])
        assert np.array_equal(np.unique(y_test), [0, 1])
