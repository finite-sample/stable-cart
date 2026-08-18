"""Resampling-based prediction-instability audits."""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import resample


def _standard_error(values: NDArray[np.floating]) -> float:
    """Monte Carlo standard error of a mean across independent draws."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _pairwise_standard_error(predictions: NDArray[Any], *, numeric: bool) -> float:
    """Jackknife SE of the mean over every unordered pair of refits."""
    predictions = np.asarray(predictions)
    n_draws = predictions.shape[0]
    if n_draws < 3:
        return float("nan")

    n_eval = predictions.shape[1]
    if numeric:
        flat = predictions.astype(float).reshape(n_draws, -1)
        # Squared distances are translation invariant. Centering before the
        # norm identity avoids catastrophic cancellation when predictions have
        # a large common level and comparatively small between-refit movement.
        flat = flat - flat[0]
        squared_norms = np.sum(flat**2, axis=1)
        row_sums = (
            n_draws * squared_norms
            + np.sum(squared_norms)
            - 2.0 * (flat @ np.sum(flat, axis=0))
        ) / n_eval
    else:
        row_sums = np.zeros(n_draws, dtype=float)
        for column in predictions.T:
            _values, inverse, counts = np.unique(
                column, return_inverse=True, return_counts=True
            )
            row_sums += n_draws - counts[inverse]
        row_sums /= n_eval

    total = 0.5 * float(np.sum(row_sums))
    leave_one_out_pairs = (n_draws - 1) * (n_draws - 2) / 2
    leave_one_out = (total - row_sums) / leave_one_out_pairs
    variance = (
        (n_draws - 1)
        / n_draws
        * float(np.sum((leave_one_out - np.mean(leave_one_out)) ** 2))
    )
    return float(np.sqrt(max(variance, 0.0)))


def _aligned_probabilities(model: Any, X: NDArray[Any], classes: NDArray[Any]):
    """Return probability columns in the reference class order."""
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError(
            "prediction_method='predict_proba' needs every refitted estimator "
            "to expose predict_proba and classes_."
        )
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = np.asarray(model.classes_)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(model_classes):
        raise ValueError("predict_proba columns do not match the estimator's classes_.")

    aligned = np.zeros((probabilities.shape[0], len(classes)), dtype=float)
    for source_column, label in enumerate(model_classes):
        target = np.flatnonzero(classes == label)
        if len(target) != 1:
            raise ValueError(
                f"A refitted estimator exposed unknown or duplicate class {label!r}."
            )
        aligned[:, target[0]] = probabilities[:, source_column]
    return aligned


def bootstrap_predictions(
    model_factory: Callable[[], Any],
    X_train: Any,
    y_train: Any,
    X_eval: Any,
    task: str = "continuous",
    n_bootstrap: int = 20,
    random_state: int | None = None,
    prediction_method: str = "predict",
) -> dict[str, Any]:
    """
    Refit a model on bootstrap resamples and return every prediction it made.

    This is the raw material of the Riley and Collins (2023) instability
    protocol: the model-building step is repeated on resamples of the training
    data, and each refitted model predicts for the *same* individuals. The
    summaries in :func:`bootstrap_instability` and the plots in
    :mod:`stable_cart.stability_plots` are both computed from what this returns,
    so a user who wants a different summary does not have to refit anything.

    Parameters
    ----------
    model_factory
        Zero-argument callable returning a fresh, unfitted estimator.
    X_train
        Training features to resample.
    y_train
        Training targets to resample.
    X_eval
        Fixed evaluation points, identical across resamples.
    task
        'continuous' for regression, 'categorical' for classification.
    n_bootstrap
        Number of bootstrap resamples.
    random_state
        Seed for the bootstrap samples. Randomness inside the estimator remains
        under ``model_factory``; set estimator seeds there when the audit should
        isolate sampling variation or be exactly reproducible.
    prediction_method
        ``'predict'`` measures movement in predicted values or class labels.
        For classification, ``'predict_proba'`` measures movement in the full
        probability vector, with columns aligned by the estimator's ``classes_``
        attribute. Probability movement is summarized by squared Euclidean
        distance for pairwise instability and mean absolute difference for MAPE.

    Returns
    -------
    dict[str, Any]
        ``original`` — predictions of the model fitted on the full training data,
        shape (n_eval,) for values or labels and (n_eval, n_classes) for class
        probabilities;
        ``original_labels`` — class labels or numeric predictions from the
        original fit, used to score a frontier;
        ``bootstrap`` — predictions of each resampled model, shape
        (n_bootstrap, n_eval) or (n_bootstrap, n_eval, n_classes);
        ``per_point`` — the instability statistic for each evaluation point
        (variance across resamples for 'continuous', disagreement with the modal
        prediction for 'categorical');
        ``pairwise`` — the expected disagreement between two *independently*
        resampled models at each point. For 'continuous' this is the mean squared
        difference; for categorical labels, the probability that two resampled
        models disagree; for class probabilities, squared Euclidean distance.
        It is computed from every unordered pair, not an arbitrary subset;
        ``mape_per_point`` — mean absolute prediction error against the original
        model, per evaluation point;
        ``mape_standard_error`` and ``pairwise_standard_error`` — Monte Carlo
        standard errors for the two aggregate means. The pairwise error uses the
        delete-one jackknife for the all-pairs U-statistic;
        ``n_fit_attempts`` — the original fit plus every accepted bootstrap fit;
        ``n_resample_attempts`` — all bootstrap draws, including rejected ones;
        ``n_rejected_resamples`` — one-class classification draws that were
        redrawn because common classifiers are undefined on them;
        ``task`` — echoed back so downstream code need not be told again.

    Raises
    ------
    ValueError
        If an argument is invalid or probability columns cannot be aligned.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from stable_cart import bootstrap_predictions
    >>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    >>> out = bootstrap_predictions(
    ...     lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
    ...     X[:150], y[:150], X[150:], n_bootstrap=10, random_state=0,
    ... )
    >>> out["bootstrap"].shape  # one row per resample, one column per eval point
    (10, 50)
    """
    if task not in ("continuous", "categorical"):
        raise ValueError("task must be 'categorical' or 'continuous'.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")
    if prediction_method not in ("predict", "predict_proba"):
        raise ValueError("prediction_method must be 'predict' or 'predict_proba'.")
    if prediction_method == "predict_proba" and task != "categorical":
        raise ValueError("predict_proba is only meaningful for categorical outcomes.")

    y_array = np.asarray(y_train)
    rng = np.random.default_rng(random_state)
    n_train = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
    if n_train == 0:
        raise ValueError("X_train and y_train must not be empty.")
    if len(y_array) != n_train:
        raise ValueError("X_train and y_train must contain the same number of rows.")
    n_eval = X_eval.shape[0] if hasattr(X_eval, "shape") else len(X_eval)
    if n_eval == 0:
        raise ValueError("X_eval must not be empty.")
    if y_array.ndim != 1:
        raise ValueError("y_train must be one-dimensional.")

    original_model = model_factory().fit(X_train, y_train)
    classes = (
        np.asarray(getattr(original_model, "classes_", np.unique(y_array)))
        if task == "categorical"
        else None
    )
    if prediction_method == "predict_proba":
        if not hasattr(original_model, "predict_proba") or not hasattr(
            original_model, "classes_"
        ):
            raise ValueError(
                "prediction_method='predict_proba' needs an estimator with "
                "predict_proba and classes_."
            )
        assert classes is not None
        original = _aligned_probabilities(original_model, X_eval, classes)
    else:
        original = np.asarray(original_model.predict(X_eval))

    predictions = []
    n_fit_attempts = 1
    n_resample_attempts = 0
    n_rejected_resamples = 0
    max_attempts = n_bootstrap + max(100, 10 * n_bootstrap)
    while len(predictions) < n_bootstrap:
        if n_resample_attempts >= max_attempts:
            raise ValueError(
                "Could not obtain the requested number of valid bootstrap fits. "
                "Too many classification resamples contained only one class."
            )
        # sklearn's public resample API preserves pandas and sparse containers.
        # This is the ordinary pairs bootstrap: class prevalence is allowed to
        # vary, because freezing it can materially understate instability.
        seed = int(rng.integers(np.iinfo(np.int32).max))
        resampled = cast(
            list[Any],
            resample(
                X_train,
                y_train,
                replace=True,
                n_samples=n_train,
                random_state=seed,
            ),
        )
        X_resampled, y_resampled = resampled
        n_resample_attempts += 1
        # A one-class draw is part of an unconditional pairs bootstrap, but many
        # standard classifiers are mathematically undefined on it. Use one
        # estimator-independent policy so every configuration in a frontier is
        # evaluated on the same conditional bootstrap distribution.
        if task == "categorical" and len(np.unique(y_resampled)) == 1:
            n_rejected_resamples += 1
            continue
        model = model_factory()
        n_fit_attempts += 1
        model.fit(X_resampled, y_resampled)
        if prediction_method == "predict_proba":
            assert classes is not None
            predictions.append(_aligned_probabilities(model, X_eval, classes))
        else:
            predictions.append(model.predict(X_eval))

    preds = np.asarray(predictions)
    if preds.ndim < 2:
        preds = np.atleast_2d(preds)

    if prediction_method == "predict_proba":
        numeric = preds.astype(float)
        original_numeric = original.astype(float)
        per_point = np.sum(np.var(numeric, axis=0, ddof=1), axis=1)
        pairwise = 2.0 * per_point
        mape_per_point = np.mean(
            np.mean(np.abs(numeric - original_numeric), axis=2), axis=0
        )
        mape_samples = np.mean(np.abs(numeric - original_numeric), axis=(1, 2))
        metric = "probability_vector"
    elif task == "continuous":
        numeric = preds.astype(float)
        original_numeric = original.astype(float)
        per_point = np.var(numeric, axis=0, ddof=1)
        mape_per_point = np.mean(np.abs(numeric - original_numeric), axis=0)
        pairwise = 2.0 * per_point
        mape_samples = np.mean(np.abs(numeric - original_numeric), axis=1)
        metric = "numeric_prediction"
    else:
        n_eval = len(original)
        per_point = np.empty(n_eval, dtype=float)
        pairwise = np.empty(n_eval, dtype=float)
        n_draws = preds.shape[0]
        for j in range(n_eval):
            values, counts = np.unique(preds[:, j], return_counts=True)
            modal = values[np.argmax(counts)]
            per_point[j] = float(np.mean(preds[:, j] != modal))
            agreement = np.sum(counts * (counts - 1)) / (n_draws * (n_draws - 1))
            pairwise[j] = 1.0 - agreement
        # For labels, "absolute error" is disagreement with the original model.
        mape_per_point = np.mean(preds != original, axis=0).astype(float)
        mape_samples = np.mean(preds != original, axis=1).astype(float)
        metric = "class_label"

    return {
        "original": original,
        "original_labels": np.asarray(original_model.predict(X_eval)),
        "bootstrap": preds,
        "per_point": per_point,
        "mape_per_point": mape_per_point,
        "pairwise": pairwise,
        "task": task,
        "prediction_method": prediction_method,
        "metric": metric,
        "classes": classes,
        "n_fit_attempts": n_fit_attempts,
        "n_resample_attempts": n_resample_attempts,
        "n_rejected_resamples": n_rejected_resamples,
        "mape_standard_error": _standard_error(mape_samples),
        "pairwise_standard_error": _pairwise_standard_error(
            preds,
            numeric=prediction_method == "predict_proba" or task == "continuous",
        ),
    }


def bootstrap_instability(
    model_factory: Callable[[], Any],
    X_train: Any,
    y_train: Any,
    X_eval: Any,
    task: str = "continuous",
    n_bootstrap: int = 20,
    random_state: int | None = None,
    prediction_method: str = "predict",
) -> dict[str, float | int]:
    """
    Measure how much a model's predictions move when the training data is perturbed.

    This is the quantity "prediction stability" usually refers to: refit the model
    on bootstrap resamples of the training data and measure the spread of its
    predictions **for the same evaluation point**. Lower is better.

    A model that ignores its training data scores a perfect zero here, so always
    read instability next to an appropriate performance measure on separate
    validation or test data.

    Parameters
    ----------
    model_factory
        Zero-argument callable returning a fresh, unfitted estimator.
    X_train
        Training features to resample.
    y_train
        Training targets to resample.
    X_eval
        Fixed evaluation points. These must not change between resamples;
        comparing predictions across different points measures nothing.
    task
        'continuous' for regression, 'categorical' for classification.
    n_bootstrap
        Number of bootstrap resamples.
    random_state
        Seed for the bootstrap samples. Estimator randomness remains under
        ``model_factory``.
    prediction_method
        Prediction representation to compare; see
        :func:`bootstrap_predictions`.

    Returns
    -------
    dict[str, float | int]
        ``instability_mean``, ``instability_p90`` and ``instability_max`` over the
        evaluation points. For 'continuous' the per-point statistic is the
        variance of predictions; for categorical labels it is the fraction of
        resamples disagreeing with that point's modal prediction. ``mape`` is
        Riley and Collins's mean absolute prediction error against the model
        fitted on the full training data. ``pairwise_mean`` compares two
        independently refitted models. Monte Carlo standard errors accompany
        both aggregate comparison measures. Fit, draw, and one-class rejection
        counts expose the classification bootstrap's conditioning.

    Raises ``ValueError`` (from :func:`bootstrap_predictions`) if task is not
    'continuous' or 'categorical', or n_bootstrap is below 2.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from stable_cart import bootstrap_instability
    >>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    >>> result = bootstrap_instability(
    ...     lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
    ...     X[:150], y[:150], X[150:], n_bootstrap=10, random_state=0,
    ... )
    >>> 0.0 <= result["mape_standard_error"] < result["mape"]
    True
    """
    raw = bootstrap_predictions(
        model_factory,
        X_train,
        y_train,
        X_eval,
        task=task,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        prediction_method=prediction_method,
    )
    per_point = raw["per_point"]

    return {
        "instability_mean": float(np.mean(per_point)),
        "instability_p90": float(np.percentile(per_point, 90)),
        "instability_max": float(np.max(per_point)),
        "mape": float(np.mean(raw["mape_per_point"])),
        "mape_standard_error": raw["mape_standard_error"],
        "pairwise_mean": float(np.mean(raw["pairwise"])),
        "pairwise_standard_error": raw["pairwise_standard_error"],
        "n_fit_attempts": int(raw["n_fit_attempts"]),
        "n_resample_attempts": int(raw["n_resample_attempts"]),
        "n_rejected_resamples": int(raw["n_rejected_resamples"]),
    }
