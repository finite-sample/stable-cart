"""Analytic checks for the nested variance estimator and its resampling unit."""

import numpy as np
import pytest

from experiments.variance_budget import prediction_statistics, summarize_components


def statistics(predictions):
    means = predictions.mean(axis=1)
    return prediction_statistics(means, predictions.var(axis=1, ddof=1), means, 1)


def test_components_match_nested_anova_and_allow_negative_estimates():
    inner = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=float)[:, :, None]
    result = summarize_components(statistics(inner), 2)
    assert result["instability_leaf"] == pytest.approx(2)
    assert result["instability_structure"] == pytest.approx(-1)
    assert result["instability_total"] == pytest.approx(1)


def test_replicating_evaluation_points_does_not_shrink_uncertainty():
    inner = np.random.default_rng(19).normal(size=(15, 8, 10))
    first = summarize_components(statistics(inner), 8)
    repeated = summarize_components(statistics(np.tile(inner, (1, 1, 20))), 8)
    for key in first:
        assert repeated[key] == pytest.approx(first[key], abs=1e-12)


def test_random_effects_reference_recovers_known_components():
    rng = np.random.default_rng(80)
    estimates = []
    for _ in range(400):
        structure = rng.normal(scale=2, size=(40, 1, 1))
        leaf = rng.normal(scale=3, size=(40, 6, 1))
        result = summarize_components(statistics(structure + leaf), 6, resamples=20)
        estimates.append([result["instability_leaf"], result["instability_structure"]])
    estimates = np.asarray(estimates)
    error = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))
    assert np.all(np.abs(estimates.mean(axis=0) - [9, 4]) < 4 * error)


def test_target_offset_preserves_components_and_uncertainty():
    inner = np.random.default_rng(1).normal(size=(20, 8, 10))
    original = summarize_components(statistics(inner), 8)
    shifted = summarize_components(statistics(inner + 1e8), 8)
    for key in original:
        assert shifted[key] == pytest.approx(original[key], abs=1e-7)
