"""Data-generating processes with known ground truth.

Each DGP exposes a sampler and an oracle. The oracle records the true split
features as **equivalence classes**, not indices: where two features carry the
same information, a tree that tests either one has recovered the same rule, and
scoring it as a miss would penalise representational non-identifiability rather
than a modelling failure. That distinction is what makes a fidelity measure
meaningful on the aliased DGP.

Roles, from `pap.md`:

  A tree_separated  positive control — everything should stabilise as n grows
  B aliased         representational non-identifiability
  C smooth          approximation ambiguity, no true tree
  D noise_only      placebo — y is independent of X, so any method scoring
                    "stable" here is degenerate rather than good
  E correlated      realistic version of B
  F margin_sweep    controlled margin between the best and second-best root
                    split, the mechanism behind the greedy-vs-exact hypothesis
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["DGP", "make_dgp", "DGP_NAMES"]

DGP_NAMES = (
    "tree_separated",
    "aliased",
    "smooth",
    "noise_only",
    "correlated",
    "margin_sweep",
)


@dataclass
class DGP:
    """A sampler plus the ground truth needed to score fidelity.

    Attributes
    ----------
    name
        Identifier.
    sample
        ``(n, rng) -> (X, y)``.
    true_split_classes
        One frozenset per true split; a fit recovers that split if it tests any
        feature in the set. Empty when there is no true tree.
    n_features
        Width of X.
    signal_var
        Variance of the noiseless signal, for reporting.
    """

    name: str
    sample: Callable[[int, np.random.Generator], tuple[NDArray, NDArray]]
    true_split_classes: tuple[frozenset[int], ...] = field(default_factory=tuple)
    n_features: int = 5
    signal_var: float | None = None

    def recovery(self, tested_features: set[int]) -> float:
        """
        Fraction of true splits recovered, counting equivalent features as hits.

        Parameters
        ----------
        tested_features
            Feature indices the fitted tree actually tests.

        Returns
        -------
        float
            In [0, 1]; 1.0 when every true split has a representative tested.
            Returns NaN when the DGP has no true tree, where recovery is undefined.
        """
        if not self.true_split_classes:
            return float("nan")
        hits = sum(1 for cls in self.true_split_classes if tested_features & set(cls))
        return hits / len(self.true_split_classes)


def _tree_separated(sigma: float, n_features: int):
    """True depth-2 tree with widely separated, unambiguous splits."""

    def sample(n, rng):
        X = rng.normal(size=(n, n_features))
        y = np.where(
            X[:, 0] > 0.0,
            np.where(X[:, 1] > 0.0, 6.0, 3.0),
            np.where(X[:, 2] > 0.0, -3.0, -6.0),
        )
        return X, y + sigma * rng.normal(size=n)

    return sample


def _aliased(sigma: float, n_features: int):
    """Same rule, but feature 1 is an exact copy of feature 0."""

    def sample(n, rng):
        X = rng.normal(size=(n, n_features))
        X[:, 1] = X[:, 0]
        y = np.where(X[:, 0] > 0.0, 3.0, -3.0)
        return X, y + sigma * rng.normal(size=n)

    return sample


def _smooth(sigma: float, n_features: int):
    """No true tree: a smooth additive function a tree can only approximate."""

    def sample(n, rng):
        X = rng.normal(size=(n, n_features))
        y = X[:, 0] + 0.7 * X[:, 1] + 0.4 * X[:, 2]
        return X, y + sigma * rng.normal(size=n)

    return sample


def _noise_only(sigma: float, n_features: int):
    """Placebo: y carries no information about X."""

    def sample(n, rng):
        X = rng.normal(size=(n, n_features))
        return X, sigma * rng.normal(size=n)

    return sample


def _correlated(sigma: float, n_features: int, rho: float = 0.9):
    """Feature 1 is correlated with feature 0 at rho, not identical to it."""

    def sample(n, rng):
        X = rng.normal(size=(n, n_features))
        X[:, 1] = rho * X[:, 0] + np.sqrt(1 - rho**2) * X[:, 1]
        y = np.where(X[:, 0] > 0.0, 3.0, -3.0)
        return X, y + sigma * rng.normal(size=n)

    return sample


def _margin_sweep(sigma: float, n_features: int, delta: float = 0.5):
    """Two candidate root splits whose usefulness differs by ``delta``.

    ``delta=0`` makes features 0 and 1 exactly as good, so the root choice is a
    coin flip; ``delta=1`` leaves only feature 0 predictive.
    """

    def sample(n, rng):
        X = rng.normal(size=(n, n_features))
        y = 3.0 * np.sign(X[:, 0]) + 3.0 * (1.0 - delta) * np.sign(X[:, 1])
        return X, y + sigma * rng.normal(size=n)

    return sample


def make_dgp(
    name: str, sigma: float = 1.0, n_features: int = 5, delta: float = 0.5
) -> DGP:
    """
    Build a named DGP at a given noise level.

    Parameters
    ----------
    name
        One of DGP_NAMES.
    sigma
        Standard deviation of the additive noise.
    n_features
        Width of X.
    delta
        Only used by ``margin_sweep``: separation between the best and
        second-best root split, in [0, 1].

    Returns
    -------
    DGP
        Sampler plus ground truth.

    Raises
    ------
    ValueError
        If the name is not recognised.
    """
    if name == "tree_separated":
        return DGP(
            name,
            _tree_separated(sigma, n_features),
            (frozenset({0}), frozenset({1}), frozenset({2})),
            n_features,
            signal_var=20.25,
        )
    if name == "aliased":
        # Features 0 and 1 are interchangeable, so testing either recovers the rule.
        return DGP(
            name,
            _aliased(sigma, n_features),
            (frozenset({0, 1}),),
            n_features,
            signal_var=9.0,
        )
    if name == "smooth":
        return DGP(name, _smooth(sigma, n_features), (), n_features, signal_var=1.65)
    if name == "noise_only":
        return DGP(name, _noise_only(sigma, n_features), (), n_features, signal_var=0.0)
    if name == "correlated":
        return DGP(
            name,
            _correlated(sigma, n_features),
            (frozenset({0}),),
            n_features,
            signal_var=9.0,
        )
    if name == "margin_sweep":
        classes = (
            (frozenset({0}),) if delta >= 1.0 else (frozenset({0}), frozenset({1}))
        )
        return DGP(
            name,
            _margin_sweep(sigma, n_features, delta),
            classes,
            n_features,
            signal_var=9.0 + 9.0 * (1.0 - delta) ** 2,
        )
    raise ValueError(f"unknown DGP: {name!r}. Known: {DGP_NAMES}")
