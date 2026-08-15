"""Tests for bootstrap_consensus_split.

Two defects motivated these. The loop that converts votes into candidates
unpacked ``for (feature_idx, threshold), votes in ...``, rebinding the function's
``threshold`` parameter — the requested consensus level — to the split's cut
point, so the acceptance test compared a support fraction against a feature
value. And votes were tallied per candidate rather than per replicate, so several
candidates from one bootstrap sample could vote for the same binned split and
push the support above 1.
"""

import numpy as np
import pytest

from stable_cart.stability_utils import bootstrap_consensus_split


@pytest.fixture
def separable():
    """One feature with an unambiguous split at 19.5."""
    X = np.arange(40, dtype=float).reshape(-1, 1)
    return X, (X[:, 0] > 19).astype(int)


@pytest.mark.parametrize("max_bins", [1, 2, 4])
def test_consensus_support_is_a_fraction(max_bins):
    """Support is a share of bootstrap replicates, so it cannot exceed 1.

    Coarse binning is the exposing case: many candidate thresholds from a single
    replicate collapse onto one binned key, and counting each as its own vote
    once produced a support of 14.4 at ``max_bins=1``.
    """
    X = np.arange(200, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 100).astype(int)

    best, candidates = bootstrap_consensus_split(
        X,
        y,
        n_samples=10,
        max_candidates=30,
        threshold=0.0,
        enable_quantile_binning=True,
        max_bins=max_bins,
        random_state=0,
    )

    for candidate in candidates:
        assert 0.0 <= candidate.consensus_support <= 1.0, (
            f"support {candidate.consensus_support} outside [0, 1] at max_bins={max_bins}"
        )
    if best is not None:
        assert 0.0 <= best.consensus_support <= 1.0


def test_requested_threshold_is_honoured(separable):
    """A candidate below the requested consensus level must be rejected."""
    X, y = separable

    permissive, _ = bootstrap_consensus_split(
        X, y, n_samples=30, threshold=0.0, random_state=0
    )
    strict, _ = bootstrap_consensus_split(
        X, y, n_samples=30, threshold=0.99, random_state=0
    )

    assert permissive is not None, "threshold=0 should accept something"
    if strict is not None:
        assert strict.consensus_support >= 0.99, (
            f"accepted support {strict.consensus_support} under threshold 0.99"
        )


def test_threshold_is_not_confused_with_the_split_value(separable):
    """The bug: acceptance compared support against the split's cut point.

    Shifting the feature by a large constant moves every split value without
    changing the problem, so acceptance must not move with it.
    """
    X, y = separable

    near, _ = bootstrap_consensus_split(
        X, y, n_samples=30, threshold=0.5, random_state=0
    )
    far, _ = bootstrap_consensus_split(
        X + 1000.0, y, n_samples=30, threshold=0.5, random_state=0
    )

    assert (near is None) == (far is None), (
        "translating the feature changed whether a consensus split was accepted"
    )


def test_higher_threshold_never_accepts_more(separable):
    """Acceptance must be monotone in the requested consensus level."""
    X, y = separable

    accepted = []
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        best, _ = bootstrap_consensus_split(
            X, y, n_samples=24, threshold=level, random_state=1
        )
        accepted.append(best is not None)

    for stricter, looser in zip(accepted[1:], accepted[:-1], strict=True):
        assert looser or not stricter, (
            "a stricter threshold accepted where a looser one did not"
        )


def test_returned_threshold_is_a_split_value_not_the_consensus_level(separable):
    """The candidate's threshold must be a cut point in the feature's range.

    Under the shadowing bug the two names were the same variable, so this held by
    accident; renaming one without the other would have silently returned the
    consensus level (a probability) as the split value.
    """
    X, y = separable
    lo, hi = float(X[:, 0].min()), float(X[:, 0].max())

    best, candidates = bootstrap_consensus_split(
        X, y, n_samples=30, threshold=0.2, random_state=0
    )

    for candidate in [*candidates, best]:
        if candidate is None:
            continue
        assert lo <= candidate.threshold <= hi, (
            f"threshold {candidate.threshold} outside the feature range [{lo}, {hi}]"
        )


def test_returns_nothing_on_tiny_input():
    """Too few rows for a meaningful vote is a clean no-answer, not a crash."""
    best, candidates = bootstrap_consensus_split(
        np.arange(6, dtype=float).reshape(-1, 1),
        np.array([0, 0, 1, 1, 0, 1]),
        random_state=0,
    )

    assert best is None
    assert candidates == []
