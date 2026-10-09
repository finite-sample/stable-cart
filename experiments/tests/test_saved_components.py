"""Reconstruct reported quantities from retained simulation predictions."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.solver_comparison import summarize_predictions
from experiments.variance_budget import summarize_components

ROOT = Path(__file__).parents[2] / "results"


def test_variance_components_reconstruct_from_outer_structure_statistics():
    directory = ROOT / "variance_budget"
    rows = json.loads((directory / "budget.json").read_text())
    with np.load(directory / rows[0]["statistics_file"]) as raw:
        stored = {key: raw[key] for key in raw.files}
    for row in rows:
        statistics = {
            key: values[row["statistics_index"]] for key, values in stored.items()
        }
        summary = summarize_components(statistics, row["n_leaf_samples"])
        for key, value in summary.items():
            assert row[key] == pytest.approx(value, abs=1e-12)


def test_solver_summaries_reconstruct_from_paired_predictions():
    for name, filename in (
        ("margin_study", "margin.json"),
        ("optimal_tree_premise", "results.json"),
    ):
        directory = ROOT / name
        evidence = json.loads((directory / filename).read_text())
        rows = (
            evidence["rows"]
            if name == "margin_study"
            else evidence["datasets"].values()
        )
        for row in rows:
            with np.load(directory / row["predictions_file"]) as raw:
                predictions = {key: raw[key] for key in raw.files if key != "labels"}
                summary = summarize_predictions(
                    predictions, raw["labels"], seed=evidence["config"].get("seed", 1)
                )
            for arm, metrics in summary.items():
                for metric, value in metrics.items():
                    np.testing.assert_allclose(
                        value, row["summary"][arm][metric], rtol=1e-12, atol=1e-12
                    )
