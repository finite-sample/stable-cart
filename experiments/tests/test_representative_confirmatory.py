"""Falsification tests for the prospective representative-selection study."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from experiments.representative_confirmatory import (
    all_pair_squared_distances,
    apply_decision_rules,
    fast_all_pair_mean,
    run_dataset,
    summarize_rows,
    symmetric_effect,
)

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("shape", [(6, 9), (6, 9, 4)])
def test_fast_all_pair_mean_matches_slow_pair_loop(shape):
    predictions = np.random.default_rng(7).normal(size=shape)

    observed = fast_all_pair_mean(predictions)
    expected = all_pair_squared_distances(predictions).mean()

    assert observed == pytest.approx(expected)


def test_probability_distance_is_invariant_to_class_column_order():
    predictions = np.random.default_rng(9).dirichlet(np.ones(4), size=(8, 13))

    original = all_pair_squared_distances(predictions)
    relabeled = all_pair_squared_distances(predictions[:, :, [2, 0, 3, 1]])

    assert np.allclose(original, relabeled)


def test_one_candidate_makes_all_single_model_rules_identical():
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        random_state=11,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12, stratify=y
    )

    result = run_dataset(
        X_train,
        X_test,
        y_train,
        y_test,
        task="classification",
        family="tree",
        n_outer=4,
        n_candidates=1,
        seed=13,
    )

    for rule in ("random", "best_validation", "ensemble"):
        assert result[f"{rule}_instability"] == pytest.approx(
            result["representative_instability"]
        )
        assert result[f"{rule}_score"] == pytest.approx(result["representative_score"])
    assert result["instability_effect_vs_best_percent"] == pytest.approx(0)
    assert result["score_delta_vs_best"] == pytest.approx(0)


def test_symmetric_effect_has_expected_limits_and_direction():
    assert symmetric_effect(0, 0) == 0
    assert symmetric_effect(0, 2) == -200
    assert symmetric_effect(2, 0) == 200
    assert symmetric_effect(1, 3) == -100


def _read_evidence_rows() -> list[dict]:
    numeric_exceptions = {"source", "process", "family", "task"}
    destination = ROOT / "results/representative_confirmatory/dataset_rows.csv"
    with destination.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key: value if key in numeric_exceptions else float(value)
            for key, value in row.items()
            if value != ""
        }
        for row in rows
    ]


def test_committed_confirmatory_summary_recomputes_from_dataset_rows():
    evidence = json.loads(
        (ROOT / "results/representative_confirmatory/results.json").read_text()
    )
    config = evidence["config"]
    rows = _read_evidence_rows()
    synthetic_rows = [row for row in rows if row["source"] == "synthetic"]
    real_rows = [row for row in rows if row["source"] == "real_repeated_split"]

    synthetic = summarize_rows(
        synthetic_rows, config["seed"], config["interval_resamples"]
    )
    real = summarize_rows(real_rows, config["seed"] + 1, config["interval_resamples"])

    assert synthetic == evidence["synthetic_summary"]
    assert real == evidence["real_repeated_split_summary"]
    assert apply_decision_rules(synthetic) == evidence["decisions"]


def test_committed_run_matches_frozen_plan_and_declared_dimensions():
    evidence = json.loads(
        (ROOT / "results/representative_confirmatory/results.json").read_text()
    )
    config = evidence["config"]
    plan_digest = hashlib.sha256(
        (ROOT / "experiments/designs/REPRESENTATIVE_CONFIRMATORY_PLAN.md").read_bytes()
    ).hexdigest()

    assert evidence["complete"] is True
    assert evidence["failures"] == []
    assert config["plan_sha256"] == plan_digest
    assert config["synthetic_datasets"] == 24
    assert config["real_splits"] == 20
    assert config["n_outer"] == 20
    assert config["n_candidates"] == 12
    assert config["interval_resamples"] == 20_000

    rows = _read_evidence_rows()
    synthetic = [row for row in rows if row["source"] == "synthetic"]
    real = [row for row in rows if row["source"] == "real_repeated_split"]
    assert len(synthetic) == 6 * 24
    assert len(real) == 3 * 20


def test_committed_summary_csv_matches_json_cells():
    evidence = json.loads(
        (ROOT / "results/representative_confirmatory/results.json").read_text()
    )
    with (ROOT / "results/representative_confirmatory/summary.csv").open(
        newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == (6 + 3) * 3
    for row in rows:
        summary_name = (
            "synthetic_summary"
            if row["source"] == "synthetic"
            else "real_repeated_split_summary"
        )
        expected = evidence[summary_name]["cells"][row["scope"]][row["metric"]]
        for field in ("mean", "ci_low", "ci_high"):
            assert float(row[field]) == expected[field]
        assert int(row["n_units"]) == expected["n_units"]
