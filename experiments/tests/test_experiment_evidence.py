"""Tests for the calculations used by research evidence drivers."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.bootstrap_split_eval import matched_comparison
from experiments.selection_rules_experiment import (
    pair_values,
    paired_comparison,
    write_summary_csv,
)

ROOT = Path(__file__).parents[2]


def test_pair_values_use_disjoint_independent_replicates():
    regression = [
        np.array([0.0, 1.0]),
        np.array([4.0, 3.0]),
        np.array([1.0, 1.0]),
        np.array([2.0, 5.0]),
    ]
    classification = [
        np.array([0, 1, 1]),
        np.array([1, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 0, 0]),
    ]

    assert np.allclose(pair_values(regression, "regression"), [10.0, 8.5])
    assert np.allclose(pair_values(classification, "classification"), [2 / 3, 2 / 3])


def test_paired_comparison_uses_within_pool_differences():
    predictions = {
        "baseline": [
            np.array([0.0]),
            np.array([0.0]),
            np.array([2.0]),
            np.array([2.0]),
        ],
        "method": [
            np.array([0.0]),
            np.array([1.0]),
            np.array([2.0]),
            np.array([4.0]),
        ],
    }
    scores = {"baseline": [0.1, 0.2, 0.3, 0.4], "method": [0.2, 0.3, 0.4, 0.5]}

    result = paired_comparison(predictions, scores, "regression", "method", "baseline")

    assert result["instability_delta"] == pytest.approx(2.5)
    assert result["instability_delta_mcse"] == pytest.approx(1.5)
    assert result["score_delta"] == pytest.approx(0.1)
    assert result["score_delta_mcse"] == pytest.approx(0.0)


def test_matched_comparison_selects_eligible_rows_and_reports_paired_delta():
    rows = [
        {
            "arm": "pruned_cart",
            "config": "cart-low-score",
            "instability": 1.0,
            "instability_mcse": 0.1,
            "pair_samples": [1.0, 1.0],
            "accuracy": 0.7,
        },
        {
            "arm": "pruned_cart",
            "config": "cart-eligible",
            "instability": 3.0,
            "instability_mcse": 0.2,
            "pair_samples": [2.0, 4.0],
            "accuracy": 0.9,
        },
        {
            "arm": "bootstrap_split",
            "config": "split-eligible",
            "instability": 2.0,
            "instability_mcse": 0.3,
            "pair_samples": [1.0, 3.0],
            "accuracy": 1.0,
        },
    ]

    result = matched_comparison(rows, accuracy_floor=0.8)

    assert result is not None
    assert result["pruned_cart"]["config"] == "cart-eligible"
    assert result["bootstrap_split"]["config"] == "split-eligible"
    assert result["paired_instability_delta"] == pytest.approx(-1.0)
    assert result["paired_instability_delta_mcse"] == pytest.approx(0.0)
    assert result["paired_instability_change_percent"] == pytest.approx(-100 / 3)


def test_selection_summary_csv_contains_paired_evidence(tmp_path):
    comparison = {
        "instability_delta": -0.1,
        "instability_delta_mcse": 0.02,
        "instability_relative_change_percent": -10.0,
        "score_delta": 0.01,
        "score_delta_mcse": 0.003,
    }
    results = {
        "classification_easy": {
            "representative": {
                "pairwise_instability": 0.2,
                "pairwise_mcse": 0.01,
                "score": 0.9,
                "score_mcse": 0.004,
                "vs_random": comparison,
                "vs_best_validation": comparison,
            }
        }
    }
    destination = tmp_path / "summary.csv"

    write_summary_csv(results, destination)

    with destination.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["rule"] == "representative"
    assert float(row["instability_delta_vs_random"]) == pytest.approx(-0.1)


def test_committed_bootstrap_summary_recomputes_from_raw_rows():
    evidence = json.loads((ROOT / "results/bootstrap_split_eval/eval.json").read_text())
    with (ROOT / "results/bootstrap_split_eval/matched.csv").open(newline="") as handle:
        csv_rows = {row["dataset"]: row for row in csv.DictReader(handle)}

    for dataset, result in evidence.items():
        recomputed = matched_comparison(result["rows"], accuracy_floor=0.95)
        assert recomputed is not None
        saved = result["matched"]
        assert recomputed["target_accuracy"] == pytest.approx(saved["target_accuracy"])
        if saved.get("paired_instability_delta") is None:
            assert csv_rows[dataset]["paired_instability_delta"] == ""
            continue
        for field in (
            "paired_instability_delta",
            "paired_instability_delta_mcse",
            "paired_instability_change_percent",
        ):
            assert recomputed[field] == pytest.approx(saved[field])
            assert float(csv_rows[dataset][field]) == pytest.approx(saved[field])


def test_committed_selection_csv_recomputes_from_json(tmp_path):
    evidence = json.loads((ROOT / "results/selection_rules/results.json").read_text())
    recomputed = tmp_path / "summary.csv"
    write_summary_csv(evidence, recomputed)

    with recomputed.open(newline="") as handle:
        expected = list(csv.DictReader(handle))
    with (ROOT / "results/selection_rules/summary.csv").open(newline="") as handle:
        observed = list(csv.DictReader(handle))

    assert observed == expected
