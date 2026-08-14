"""Map the accuracy-stability frontier for a model family on your own data.

Choosing a model is not a single decision between two estimators; it is a choice
of where to sit on a tradeoff. This module sweeps a parameter grid, measures how
much predictions move when the training data is perturbed, and returns the
Pareto set — the configurations for which no other configuration is both more
accurate and more stable — so the exchange rate is visible before anyone commits.

It is deliberately model-agnostic. A user can put a plain
``DecisionTreeRegressor(ccp_alpha=...)`` and a :class:`~stable_cart.StableTree`
on the same axes, and the honest answer may well be that pruning wins.

The instability measures follow the protocol of Riley and Collins, *Stability of
clinical prediction models developed using statistical or machine learning
methods*, Biometrical Journal 65(8), 2023 — refit the whole model-building step
on bootstrap resamples and compare each resampled model's predictions with the
original model's, for the same individuals. Their headline measure is the mean
absolute prediction error (MAPE); it is reported here alongside the variance of
predictions across resamples. The R package ``pminternal`` implements the same
protocol; this is the equivalent for scikit-learn estimators.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import ParameterGrid, train_test_split

__all__ = ["stability_frontier", "pareto_front"]


def _score(pred, y_true, task):
    """R² for continuous outcomes, accuracy for categorical ones."""
    if task == "categorical":
        return float(np.mean(pred == y_true))
    ss_res = float(np.sum((y_true - pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def pareto_front(points: list[dict]) -> list[dict]:
    """
    Keep the configurations no other configuration beats on both axes.

    Parameters
    ----------
    points
        Dicts carrying ``accuracy`` (higher is better) and ``instability``
        (lower is better).

    Returns
    -------
    list[dict]
        The non-dominated subset, ordered by accuracy descending.
    """
    front = []
    for point in points:
        dominated = any(
            other["accuracy"] >= point["accuracy"]
            and other["instability"] <= point["instability"]
            and (
                other["accuracy"] > point["accuracy"]
                or other["instability"] < point["instability"]
            )
            for other in points
        )
        if not dominated:
            front.append(point)
    return sorted(front, key=lambda p: -p["accuracy"])


def stability_frontier(
    estimator_factory: Callable[..., Any],
    param_grid: dict | list[dict],
    X: NDArray[Any],
    y: NDArray[Any],
    task: str = "continuous",
    n_bootstrap: int = 20,
    test_size: float = 0.3,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Sweep a parameter grid and return the accuracy-stability tradeoff.

    Parameters
    ----------
    estimator_factory
        Callable taking the grid's keyword arguments and returning a fresh,
        unfitted estimator — e.g. ``lambda **kw: DecisionTreeRegressor(**kw)``.
    param_grid
        Grid in scikit-learn's ``ParameterGrid`` form.
    X, y
        The data. It is split once into a fitting part and a held-out part; every
        configuration is scored on the same held-out rows.
    task
        ``'continuous'`` or ``'categorical'``.
    n_bootstrap
        Resamples per configuration. Riley and Collins use many more for a final
        report; 20 is enough to rank configurations, and the cost is linear.
    test_size
        Fraction held out for evaluation.
    random_state
        Seed. The **same** resampled index sets are reused for every
        configuration, so the comparison is paired and cheaper.

    Returns
    -------
    dict[str, Any]
        ``points`` — every configuration with ``accuracy``, ``instability``
        (variance of predictions across resamples), ``mape`` (Riley and Collins's
        mean absolute prediction error) and ``params``;
        ``frontier`` — the non-dominated subset;
        ``n_fits`` and ``seconds`` — what the answer cost.

    Raises
    ------
    ValueError
        If ``task`` is not recognised or ``n_bootstrap`` is below 2.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> result = stability_frontier(
    ...     lambda **kw: DecisionTreeRegressor(**kw),
    ...     {"ccp_alpha": [0.0, 0.01, 0.1]},
    ...     X, y, task="continuous",
    ... )
    >>> [(p["accuracy"], p["instability"]) for p in result["frontier"]]
    """
    if task not in ("continuous", "categorical"):
        raise ValueError("task must be 'continuous' or 'categorical'")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")

    started = time.perf_counter()
    stratify = y if task == "categorical" else None
    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # One set of resamples, shared by every configuration: this is what makes the
    # comparison paired rather than merely parallel, and it halves the noise.
    rng = np.random.default_rng(random_state)
    n = len(X_fit)
    resamples = [rng.integers(0, n, n) for _ in range(n_bootstrap)]

    points, n_fits = [], 0
    for params in ParameterGrid(param_grid):
        original = estimator_factory(**params).fit(X_fit, y_fit)
        base_pred = np.asarray(original.predict(X_eval), dtype=float)
        n_fits += 1

        boot_preds = []
        for idx in resamples:
            if task == "categorical" and len(np.unique(y_fit[idx])) < 2:
                continue
            model = estimator_factory(**params).fit(X_fit[idx], y_fit[idx])
            boot_preds.append(np.asarray(model.predict(X_eval), dtype=float))
            n_fits += 1

        if len(boot_preds) < 2:
            continue
        boot = np.array(boot_preds)

        points.append(
            {
                "params": dict(params),
                "accuracy": _score(original.predict(X_eval), y_eval, task),
                "instability": float(np.mean(np.var(boot, axis=0))),
                # Riley and Collins's MAPE: how far a resampled model's prediction
                # for an individual sits from the original model's, on average.
                "mape": float(np.mean(np.abs(boot - base_pred))),
            }
        )

    return {
        "points": points,
        "frontier": pareto_front(points),
        "n_fits": n_fits,
        "seconds": time.perf_counter() - started,
    }
