"""Map the validation-score/instability frontier for a model family.

Choosing a model is not a single decision between two estimators; it is a choice
of where to sit on a tradeoff. This module sweeps a parameter grid, measures how
much predictions move when the training data is perturbed, and returns the
Pareto set — the configurations for which no other configuration is both more
accurate and more stable — so the exchange rate is visible before anyone commits.

It is deliberately model-agnostic. A user can put a plain
``DecisionTreeRegressor(ccp_alpha=...)`` and a regularized linear model on the
same axes without either model family receiving special treatment.

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
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils.validation import column_or_1d

from .evaluation import bootstrap_predictions

__all__ = ["stability_frontier", "pareto_front"]


def _n_rows(data: Any) -> int:
    """Return the first dimension for supported sklearn containers."""
    return int(data.shape[0]) if hasattr(data, "shape") else len(data)


def _score(pred, y_true, task):
    """R² for continuous outcomes, accuracy for categorical ones."""
    if task == "categorical":
        return float(accuracy_score(y_true, pred))
    return float(r2_score(y_true, pred))


def pareto_front(
    points: list[dict], *, instability_key: str = "instability"
) -> list[dict]:
    """
    Keep the configurations no other configuration beats on both axes.

    Parameters
    ----------
    points
        Dicts carrying ``score`` (higher is better) and ``instability``
        (lower is better).
    instability_key
        Key containing the quantity to minimize.

    Returns
    -------
    list[dict]
        The non-dominated subset, ordered by accuracy descending.
    """
    front = []
    for point in points:
        dominated = any(
            other["score"] >= point["score"]
            and other[instability_key] <= point[instability_key]
            and (
                other["score"] > point["score"]
                or other[instability_key] < point[instability_key]
            )
            for other in points
        )
        if not dominated:
            front.append(point)

    # Different parameter combinations often land on exactly the same point — a
    # knob that does nothing on this data. Keeping both would overstate how many
    # distinct operating points a family actually offers.
    deduped, seen = [], set()
    for point in sorted(front, key=lambda p: -p["score"]):
        key = (
            round(point["score"], 12),
            round(point[instability_key], 12),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(point)
    return deduped


def stability_frontier(
    estimator_factory: Callable[..., Any],
    param_grid: dict | list[dict],
    X: Any,
    y: Any,
    task: str = "continuous",
    n_bootstrap: int = 20,
    test_size: float = 0.3,
    random_state: int | None = None,
    *,
    X_eval: Any = None,
    y_eval: Any = None,
    prediction_method: str = "predict",
    instability_metric: str = "pairwise",
) -> dict[str, Any]:
    """
    Sweep a parameter grid and return the validation-score/instability tradeoff.

    Parameters
    ----------
    estimator_factory
        Callable taking the grid's keyword arguments and returning a fresh,
        unfitted estimator — e.g. ``lambda **kw: DecisionTreeRegressor(**kw)``.
    param_grid
        Grid in scikit-learn's ``ParameterGrid`` form.
    X
        Feature matrix used to fit the models. When ``X_eval`` and ``y_eval`` are
        omitted, it is split once into fitting and validation parts.
    y
        Training targets, or targets to split alongside ``X``.
    task
        ``'continuous'`` or ``'categorical'``.
    n_bootstrap
        Resamples per configuration. The returned Monte Carlo standard error is
        the guide to whether this is enough; 20 is only a quick diagnostic.
    test_size
        Fraction held out for evaluation.
    random_state
        Seed for resampling and the internal validation split. The **same**
        resampled index sets are reused for every configuration, so the data
        comparison is paired. Estimator randomness remains under
        ``estimator_factory``.
    X_eval
        Optional explicit validation features. Supply with ``y_eval``.
    y_eval
        Optional explicit validation targets. Supply with ``X_eval``. The score is
        a model-selection score, not a final test-set performance estimate.
    prediction_method
        ``'predict'`` or, for classification, ``'predict_proba'``. See
        :func:`~stable_cart.bootstrap_predictions`.
    instability_metric
        Quantity minimized on the frontier: ``'pairwise'`` compares two
        independently refitted models; ``'mape'`` compares each refit with the
        model fitted on all training data.

    Returns
    -------
    dict[str, Any]
        ``points`` — every configuration with ``score`` (validation accuracy or
        R²), the selected ``instability``, its Monte Carlo standard error,
        ``pairwise``, ``mape``, resampling counts, and ``params``;
        ``frontier`` — the non-dominated subset;
        ``n_fits`` and ``seconds`` — what the answer cost.

    Raises
    ------
    ValueError
        If an argument is invalid.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from stable_cart import stability_frontier
    >>> X, y = make_regression(n_samples=300, n_features=5, noise=5.0, random_state=0)
    >>> result = stability_frontier(
    ...     lambda **kw: DecisionTreeRegressor(random_state=0, **kw),
    ...     {"max_depth": [2, 5, 8]},
    ...     X, y, task="continuous", n_bootstrap=8, random_state=0,
    ... )
    >>> len(result["points"]), len(result["frontier"]) <= len(result["points"])
    (3, True)
    """
    if task not in ("continuous", "categorical"):
        raise ValueError("task must be 'continuous' or 'categorical'")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if instability_metric not in ("pairwise", "mape"):
        raise ValueError("instability_metric must be 'pairwise' or 'mape'")
    if (X_eval is None) != (y_eval is None):
        raise ValueError("X_eval and y_eval must be supplied together")

    started = time.perf_counter()
    y_array = column_or_1d(y)
    if X_eval is None:
        stratify = y_array if task == "categorical" else None
        X_fit, X_validation, y_fit, y_validation = train_test_split(
            X,
            y_array,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        evaluation_source = "internal_validation_split"
    else:
        X_fit, y_fit = X, y_array
        X_validation, y_validation = X_eval, y_eval
        evaluation_source = "user_supplied_validation_data"

    y_fit = column_or_1d(y_fit)
    y_validation = column_or_1d(y_validation)
    n_validation = _n_rows(X_validation)
    if n_validation != len(y_validation):
        raise ValueError("X_eval and y_eval must contain the same number of rows")

    points, n_fits = [], 0
    for params in ParameterGrid(param_grid):
        raw = bootstrap_predictions(
            lambda params=params: estimator_factory(**params),
            X_fit,
            y_fit,
            X_validation,
            task=task,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            prediction_method=prediction_method,
        )
        n_fits += raw["n_fit_attempts"]

        pairwise = float(np.mean(raw["pairwise"]))
        mape = float(np.mean(raw["mape_per_point"]))
        selected = {"pairwise": pairwise, "mape": mape}[instability_metric]
        standard_error = raw[f"{instability_metric}_standard_error"]

        points.append(
            {
                "params": dict(params),
                "score": _score(raw["original_labels"], y_validation, task),
                "instability": selected,
                "instability_standard_error": standard_error,
                "pairwise": pairwise,
                "pairwise_standard_error": raw["pairwise_standard_error"],
                "mape": mape,
                "mape_standard_error": raw["mape_standard_error"],
                "n_resample_attempts": raw["n_resample_attempts"],
                "n_rejected_resamples": raw["n_rejected_resamples"],
            }
        )

    return {
        "points": points,
        "frontier": pareto_front(points),
        "n_fits": n_fits,
        "seconds": time.perf_counter() - started,
        "task": task,
        "score_name": "accuracy" if task == "categorical" else "r2",
        "instability_metric": instability_metric,
        "prediction_method": prediction_method,
        "evaluation_source": evaluation_source,
    }
