"""Documented parameters must actually reach the split strategy.

`_create_split_strategy` used to return `HybridStrategy(focus, task, random_state)`
whenever `split_strategy` was None — which is always, for every exported estimator —
discarding every feature switch and numeric setting. Twenty-plus constructor
parameters per estimator were inert as a result, and nothing failed. These tests
are the check that was missing.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from stable_cart import BaseStableTree, RobustPrefixHonestTree
from stable_cart.split_strategies import (
    ConsensusStrategy,
    LookaheadStrategy,
    ObliqueStrategy,
    VariancePenalizedStrategy,
)


def flatten(strategy):
    """Collect every strategy instance in a (possibly composed) strategy tree."""
    found = [strategy]
    for attr in ("strategies", "base_strategy", "fallback_strategy", "strategy"):
        child = getattr(strategy, attr, None)
        if child is None:
            continue
        for item in child if isinstance(child, list) else [child]:
            found.extend(flatten(item))
    return found


def built(**params):
    """Build the strategy an estimator would use, without fitting."""
    return flatten(BaseStableTree(task="regression", **params)._create_split_strategy())


def contains(strategy, cls):
    """Whether a strategy of the given type appears anywhere in the tree."""
    return any(isinstance(s, cls) for s in strategy)


@pytest.mark.parametrize(
    ("flag", "cls"),
    [
        ("enable_oblique_splits", ObliqueStrategy),
        ("enable_lookahead", LookaheadStrategy),
        ("enable_prefix_consensus", ConsensusStrategy),
        ("enable_explicit_variance_penalty", VariancePenalizedStrategy),
    ],
)
def test_feature_flag_reaches_the_strategy(flag, cls):
    """Turning a documented feature on must put its strategy in the graph."""
    assert contains(built(**{flag: True}), cls), (
        f"{flag}=True did not add {cls.__name__}"
    )
    assert not contains(built(**{flag: False}), cls), (
        f"{flag}=False still added {cls.__name__}"
    )


def test_numeric_consensus_settings_are_forwarded():
    """Numbers must arrive at the strategy, not just the on/off switch."""
    strategies = built(
        enable_prefix_consensus=True, consensus_samples=7, consensus_threshold=0.9
    )
    consensus = next(s for s in strategies if isinstance(s, ConsensusStrategy))

    assert consensus.consensus_samples == 7
    assert consensus.consensus_threshold == 0.9


def test_consensus_is_confined_to_the_prefix():
    """A *prefix* method must stop being a consensus method below its prefix.

    Until 2.0 ``prefix_levels`` was stored and never read: the depth reached
    ``ConsensusStrategy.find_best_split`` and was discarded, so consensus ran at
    every node. ``RobustPrefixHonestTree.top_levels`` — the parameter the class
    is named after — could not change a prediction.
    """
    strategies = built(enable_prefix_consensus=True, prefix_levels=2)
    consensus = next(s for s in strategies if isinstance(s, ConsensusStrategy))

    assert consensus.prefix_levels == 2

    X, y = make_regression(
        n_samples=400, n_features=8, n_informative=5, noise=1.0, random_state=0
    )
    deep = consensus.find_best_split(X, y, depth=5)
    fallback = consensus.fallback_strategy.find_best_split(X, y, depth=5)

    # Below the prefix the fallback decides. Comparing against the fallback
    # rather than against the shallow call is what makes this a test of the
    # gate: the two strategies are free to agree on any particular dataset.
    assert deep is not None
    assert fallback is not None
    assert (deep.feature_idx, deep.threshold) == (
        fallback.feature_idx,
        fallback.threshold,
    )


def test_top_levels_changes_predictions():
    """The end-to-end version of the same claim, on the estimator itself."""
    X, y = make_regression(
        n_samples=500, n_features=10, n_informative=6, noise=2.0, random_state=0
    )

    def fit(top_levels):
        return (
            RobustPrefixHonestTree(
                task="regression", max_depth=6, top_levels=top_levels, random_state=42
            )
            .fit(X, y)
            .predict(X)
        )

    assert not np.array_equal(fit(2), fit(4))


def test_numeric_lookahead_settings_are_forwarded():
    """Same for the lookahead numbers."""
    strategies = built(enable_lookahead=True, lookahead_depth=3, beam_width=5)
    lookahead = next(s for s in strategies if isinstance(s, LookaheadStrategy))

    assert lookahead.lookahead_depth == 3
    assert lookahead.beam_width == 5


def test_variance_penalty_weight_is_forwarded():
    """A penalty weight that never arrives is the C1 defect in miniature."""
    strategies = built(
        enable_explicit_variance_penalty=True, variance_penalty_weight=3.5
    )
    penalized = next(s for s in strategies if isinstance(s, VariancePenalizedStrategy))

    assert penalized.variance_penalty_weight == 3.5


def _fit_predict(**kwargs):
    """Fit on a fixed regression problem and predict on it."""
    X, y = make_regression(
        n_samples=400, n_features=8, n_informative=5, noise=1.0, random_state=0
    )
    common = {
        "task": "regression",
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
    return BaseStableTree(**common, **kwargs).fit(X, y).predict(X)


def test_a_documented_flag_changes_predictions():
    """Structural wiring is necessary but not sufficient — it must also bite.

    The consensus level has to be one the data can actually clear; see the
    companion test for what happens at the documented default.
    """
    off = _fit_predict(enable_prefix_consensus=False)
    on = _fit_predict(
        enable_prefix_consensus=True, consensus_samples=8, consensus_threshold=0.1
    )

    assert not np.array_equal(off, on), (
        "enable_prefix_consensus is wired into the strategy but changes nothing"
    )


def test_consensus_declines_when_no_split_commands_a_majority():
    """At the documented default of 0.5, consensus correctly finds nothing here.

    Not a wiring failure: no candidate split is chosen by half the bootstrap
    replicates on this data, so the consensus strategy declines and falls back.
    Before the shadowing fix this branch was unreachable, because acceptance
    compared the support fraction against the split's cut point instead of
    against the requested level.
    """
    off = _fit_predict(enable_prefix_consensus=False)
    at_default = _fit_predict(enable_prefix_consensus=True, consensus_samples=8)

    assert np.array_equal(off, at_default)
