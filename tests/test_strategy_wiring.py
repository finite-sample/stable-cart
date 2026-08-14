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

from stable_cart import BaseStableTree
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


def test_a_documented_flag_changes_predictions():
    """Structural wiring is necessary but not sufficient — it must also bite."""
    X, y = make_regression(
        n_samples=400, n_features=8, n_informative=5, noise=1.0, random_state=0
    )
    common = {
        "task": "regression",
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    }

    off = BaseStableTree(enable_prefix_consensus=False, **common).fit(X, y).predict(X)
    on = (
        BaseStableTree(enable_prefix_consensus=True, consensus_samples=8, **common)
        .fit(X, y)
        .predict(X)
    )

    assert not np.array_equal(off, on), (
        "enable_prefix_consensus is wired into the strategy but changes nothing"
    )
