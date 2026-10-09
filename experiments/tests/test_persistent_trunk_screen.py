"""Tests for the frozen persistent-trunk screen."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.persistent_trunk_screen import (
    evaluate_unit,
    generate,
    percentile_interval,
    routing_agreement,
    summarize,
    symmetric_effect,
    wilson_lower,
    write_rows,
)

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("process", ["binary", "multiclass", "regression"])
def test_generators_are_reproducible_and_have_expected_targets(process):
    first = generate(process, root_feature=0, n_samples=100, seed=4)
    second = generate(process, root_feature=0, n_samples=100, seed=4)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    if process == "multiclass":
        assert set(np.unique(first[1])) == {0, 1, 2}
    elif process == "binary":
        assert set(np.unique(first[1])) == {0, 1}
    else:
        assert np.issubdtype(first[1].dtype, np.floating)


def test_routing_agreement_allows_side_reversal():
    route = np.array([True, True, False, False])

    assert routing_agreement(route, route) == 1.0
    assert routing_agreement(route, ~route) == 1.0
    assert routing_agreement(route, np.array([True, False, True, False])) == 0.5


def test_symmetric_effect_has_declared_zero_rule():
    assert symmetric_effect(0.0, 0.0) == 0.0
    assert symmetric_effect(1.0, 3.0) == pytest.approx(-1.0)


def test_wilson_gate_requires_more_than_the_bare_count_threshold():
    assert wilson_lower(20, 24) < 0.65
    assert wilson_lower(21, 24) > 0.65


def test_percentile_interval_has_reasonable_gaussian_coverage():
    rng = np.random.default_rng(12)
    covered = 0
    simulations = 200
    for simulation in range(simulations):
        values = rng.normal(size=24)
        interval = percentile_interval(values, seed=simulation, n_resamples=1_000)
        covered += interval["low"] <= 0.0 <= interval["high"]

    assert 0.88 <= covered / simulations <= 0.98


def test_one_unit_exercises_all_arms_and_multiclass_probabilities():
    row = evaluate_unit("multiclass", "stable", 0)

    assert row["process"] == "multiclass"
    assert row["regime"] == "stable"
    assert 0 <= row["gate_pass_count"] <= 8
    assert np.isfinite(row["instability_effect_vs_cart"])
    for arm in (
        "refit_cart",
        "frozen_tree",
        "refreshed_leaves",
        "persistent_trunk",
    ):
        assert row[f"{arm}_probability_instability"] >= 0.0
        assert row[f"{arm}_log_loss"] >= 0.0


def test_committed_evidence_reconstructs_from_unit_rows(tmp_path):
    evidence = json.loads(
        (ROOT / "results/persistent_trunk_screen/results.json").read_text()
    )
    rows = evidence["rows"]

    assert (
        evidence["plan_sha256"]
        == hashlib.sha256(
            (ROOT / "experiments/designs/PERSISTENT_TRUNK_PLAN.md").read_bytes()
        ).hexdigest()
    )
    assert len(rows) == 144
    assert len({(row["process"], row["regime"], row["unit"]) for row in rows}) == 144
    assert summarize(rows) == evidence["summary"]
    regenerated = tmp_path / "unit_rows.csv"
    write_rows(rows, regenerated)
    assert (
        regenerated.read_text()
        == (ROOT / "results/persistent_trunk_screen/unit_rows.csv").read_text()
    )


def test_committed_result_records_the_frozen_mechanism_and_negative_verdict():
    evidence = json.loads(
        (ROOT / "results/persistent_trunk_screen/results.json").read_text()
    )
    rows = evidence["rows"]
    stable_cells = [
        cell for cell in evidence["summary"]["cells"] if cell["regime"] == "stable"
    ]

    assert {row["reference_root_feature"] for row in rows} == {0}
    assert all(row["frozen_tree_instability"] == 0.0 for row in rows)
    assert all(row["gate_accept"] == (row["regime"] == "stable") for row in rows)
    assert all(
        cell["arm_means"]["refreshed_leaves"]["score"]
        > cell["arm_means"]["persistent_trunk"]["score"]
        and cell["arm_means"]["refreshed_leaves"]["instability"]
        < cell["arm_means"]["persistent_trunk"]["instability"]
        for cell in stable_cells
    )
    assert evidence["summary"]["decisions"] == {
        "advance": False,
        "beats_constrained_baselines": False,
        "gate": True,
        "stable_instability": True,
        "stable_score": True,
        "stable_useful_rate": True,
    }
