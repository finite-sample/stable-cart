"""Evaluation utilities for model performance and prediction stability.

Evaluation utilities for assessing both model performance and prediction stability.

This module provides functions to:
1. Measure prediction stability across multiple models (how consistent are predictions?)
2. Evaluate predictive performance across standard metrics (accuracy, RMSE, etc.)

These functions are designed to work with collections of fitted sklearn-compatible models
and are useful for comparing different tree algorithms, ensemble methods, or parameter settings.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# -------------------------------
# Prediction Stability (OOS)
# -------------------------------
def prediction_stability(
    models: dict, X_oos: NDArray[np.floating], task: str = "categorical"
) -> dict[str, float]:
    """
    Measure how consistent model predictions are across models on the SAME OOS data.

    This metric quantifies prediction stability by measuring how much models agree
    with each other on the same out-of-sample data. Lower values indicate more
    stable/consistent predictions.

    Parameters
    ----------
    models
        Mapping of model name -> fitted model (must have .predict() method).
        Requires at least 2 models.
    X_oos
        Out-of-sample feature matrix to evaluate on.
    task
        Type of prediction task.

    Returns
    -------
    dict[str, float]
        Stability score for each model.

        For 'categorical':
            Average pairwise DISAGREEMENT rate per model (range: 0-1).
            Lower is better (more stable). 0 = perfect agreement with all other models.

        For 'continuous':
            RMSE of each model's predictions vs the ensemble mean.
            Lower is better (more stable). 0 = identical to ensemble mean.

    Raises
    ------
    ValueError
        If fewer than 2 models provided, or if task is not 'categorical' or 'continuous'.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> models = {
    ...     'tree1': DecisionTreeClassifier(random_state=1).fit(X_train, y_train),
    ...     'tree2': DecisionTreeClassifier(random_state=2).fit(X_train, y_train),
    ... }
    >>> stability = prediction_stability(models, X_test, task='categorical')
    >>> print(stability)  # Lower values = more stable predictions
    {'tree1': 0.15, 'tree2': 0.15}

    Notes
    -----
    - Stability is measured relative to other models in the collection
    - For categorical tasks, uses pairwise agreement rates
    - For continuous tasks, uses RMSE to ensemble mean as stability proxy
    - This metric is complementary to predictive accuracy - a model can be
      accurate but unstable, or stable but inaccurate
    """
    names = list(models.keys())
    K = len(names)

    if K < 2:
        raise ValueError("Need at least 2 models to assess stability.")

    match task:
        case "categorical":
            # --- CATEGORICAL: pairwise disagreement (1 - agreement rate) ---
            preds = np.column_stack([models[n].predict(X_oos) for n in names])  # (n, K)

            # Ensure numeric label space for comparisons
            if not np.issubdtype(preds.dtype, np.number):
                # Map labels to integers consistently
                _unique, inv = np.unique(preds, return_inverse=True)
                preds = inv.reshape(preds.shape)

            # Compute pairwise agreement matrix A[k,j] = mean(pred_k == pred_j)
            agree = np.ones((K, K), dtype=float)
            for k in range(K):
                for j in range(k + 1, K):
                    agreement_rate = float(np.mean(preds[:, k] == preds[:, j]))
                    agree[k, j] = agree[j, k] = agreement_rate

            # Per-model disagreement = average disagreement over pairs involving the model
            scores = {}
            for k, name in enumerate(names):
                # Exclude self-comparison
                other_agreements = [agree[k, j] for j in range(K) if j != k]
                avg_disagreement = float(np.mean([1.0 - a for a in other_agreements]))
                scores[name] = avg_disagreement
            return scores

        case "continuous":
            # --- CONTINUOUS: RMSE to ensemble mean ---
            preds = np.column_stack([models[n].predict(X_oos) for n in names])  # (n, K)
            mean_pred = np.mean(preds, axis=1)  # Ensemble mean per sample

            scores = {}
            for k, name in enumerate(names):
                deviation = mean_pred - preds[:, k]
                rmse = float(np.sqrt(np.mean(np.square(deviation))))
                scores[name] = rmse  # Lower = more stable
            return scores

        case _:
            raise ValueError("task must be 'categorical' or 'continuous'.")


# -------------------------------
# Model Performance Evaluation
# -------------------------------
def evaluate_models(
    models: dict,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    task: str = "categorical",
) -> dict[str, dict[str, float]]:
    """
    Evaluate predictive performance of multiple models using standard metrics.

    Computes task-appropriate performance metrics for each model. For classification,
    includes accuracy and AUC (if predict_proba available). For regression, includes
    MAE, RMSE, and R².

    Parameters
    ----------
    models
        Model name -> fitted model mapping. Models must have .predict() method.
    X
        Feature matrix for evaluation.
    y
        Ground-truth labels (classification) or targets (regression).
    task
        Type of prediction task.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dictionary: {model_name: {metric_name: value}}

        For 'categorical':
            - 'acc': Classification accuracy (0-1)
            - 'auc': ROC AUC score (0-1, if predict_proba available)
                    For binary: standard AUC
                    For multi-class: one-vs-rest macro AUC

        For 'continuous':
            - 'mae': Mean Absolute Error (lower is better)
            - 'rmse': Root Mean Squared Error (lower is better)
            - 'r2': R² coefficient of determination (-∞ to 1, higher is better)

    Raises
    ------
    ValueError
        If task is not 'categorical' or 'continuous'.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = make_regression(n_samples=100, random_state=42)
    >>> models = {
    ...     'shallow': DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y),
    ...     'deep': DecisionTreeRegressor(max_depth=10, random_state=42).fit(X, y),
    ... }
    >>> performance = evaluate_models(models, X, y, task='continuous')
    >>> print(performance['shallow'])
    {'mae': 12.3, 'rmse': 15.7, 'r2': 0.85}

    Notes
    -----
    - AUC computation gracefully handles cases where predict_proba is not available
    - For multi-class classification, uses one-vs-rest strategy for AUC
    - All metrics use standard sklearn implementations
    - Consider using separate train/test sets to avoid overfitting bias
    """
    results: dict[str, dict[str, float]] = {}

    match task:
        case "categorical":
            y_unique = np.unique(y)
            is_binary = len(y_unique) == 2

            for name, mdl in models.items():
                y_hat = mdl.predict(X)
                try:
                    acc = float(accuracy_score(y, y_hat))
                except ValueError:
                    # Handle cases where predictions contain NaN or invalid values
                    acc = np.nan
                entry = {"acc": acc}

                # Compute AUC if model supports probability predictions
                if hasattr(mdl, "predict_proba"):
                    try:
                        # Direct method call - type checker will verify compatibility
                        proba = mdl.predict_proba(X)  # type: ignore[attr-defined]
                        if is_binary:
                            auc = float(roc_auc_score(y, proba[:, 1]))
                        else:
                            # One-vs-rest macro AUC for multi-class
                            Yb = label_binarize(y, classes=y_unique)
                            auc = float(
                                roc_auc_score(
                                    Yb, proba, average="macro", multi_class="ovr"
                                )
                            )
                        entry["auc"] = auc
                    except Exception:  # noqa: S110
                        # AUC is undefined for a single-class fold; the metric is
                        # optional, so the entry is simply left without it
                        pass

                results[name] = entry

        case "continuous":
            for name, mdl in models.items():
                y_pred = mdl.predict(X)
                try:
                    mae = float(mean_absolute_error(y, y_pred))
                    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
                    r2 = float(r2_score(y, y_pred))
                except ValueError:
                    # Handle cases where predictions contain NaN or invalid values
                    mae = np.nan
                    rmse = np.nan
                    r2 = np.nan
                results[name] = {"mae": mae, "rmse": rmse, "r2": r2}

        case _:
            raise ValueError("task must be 'categorical' or 'continuous'.")

    return results


# -------------------------------
# Instability under resampling
# -------------------------------
def bootstrap_predictions(
    model_factory: Callable[[], Any],
    X_train: NDArray[np.floating],
    y_train: NDArray[Any],
    X_eval: NDArray[np.floating],
    task: str = "continuous",
    n_bootstrap: int = 20,
    random_state: int | None = None,
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
        Seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        ``original`` — predictions of the model fitted on the full training data,
        shape (n_eval,);
        ``bootstrap`` — predictions of each resampled model, shape
        (n_bootstrap, n_eval);
        ``per_point`` — the instability statistic for each evaluation point
        (variance across resamples for 'continuous', disagreement with the modal
        prediction for 'categorical');
        ``mape_per_point`` — mean absolute prediction error against the original
        model, per evaluation point;
        ``task`` — echoed back so downstream code need not be told again.

    Raises
    ------
    ValueError
        If task is not 'continuous' or 'categorical', or n_bootstrap < 2.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> out = bootstrap_predictions(
    ...     lambda: DecisionTreeRegressor(max_depth=6),
    ...     X_train, y_train, X_test,
    ... )
    >>> out["bootstrap"].shape
    """
    if task not in ("continuous", "categorical"):
        raise ValueError("task must be 'categorical' or 'continuous'.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    rng = np.random.default_rng(random_state)
    n_train = X_train.shape[0]

    original = np.asarray(model_factory().fit(X_train, y_train).predict(X_eval))

    predictions = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_train, n_train)
        if task == "categorical" and len(np.unique(y_train[idx])) < 2:
            idx = np.arange(n_train)
        model = model_factory()
        model.fit(X_train[idx], y_train[idx])
        predictions.append(model.predict(X_eval))

    preds = np.atleast_2d(np.asarray(predictions))

    if task == "continuous":
        per_point = np.var(preds.astype(float), axis=0)
        mape_per_point = np.mean(
            np.abs(preds.astype(float) - original.astype(float)), axis=0
        )
    else:
        per_point = np.empty(preds.shape[1], dtype=float)
        for j in range(preds.shape[1]):
            values, counts = np.unique(preds[:, j], return_counts=True)
            modal = values[np.argmax(counts)]
            per_point[j] = float(np.mean(preds[:, j] != modal))
        # For labels, "absolute error" is disagreement with the original model.
        mape_per_point = np.mean(preds != original, axis=0).astype(float)

    return {
        "original": original,
        "bootstrap": preds,
        "per_point": per_point,
        "mape_per_point": mape_per_point,
        "task": task,
    }


def bootstrap_instability(
    model_factory: Callable[[], Any],
    X_train: NDArray[np.floating],
    y_train: NDArray[Any],
    X_eval: NDArray[np.floating],
    task: str = "continuous",
    n_bootstrap: int = 20,
    random_state: int | None = None,
) -> dict[str, float]:
    """
    Measure how much a model's predictions move when the training data is perturbed.

    This is the quantity "prediction stability" usually refers to: refit the model
    on bootstrap resamples of the training data and measure the spread of its
    predictions **for the same evaluation point**. Lower is better.

    It is a different question from :func:`prediction_stability`, which measures
    whether a *set* of already-fitted models agree with each other. A model can
    agree with its peers and still be wildly unstable under resampling, and a
    model that ignores its training data entirely scores a perfect zero here — so
    always read this next to an accuracy measure.

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
        Seed for reproducibility.

    Returns
    -------
    dict[str, float]
        ``instability_mean``, ``instability_p90`` and ``instability_max`` over the
        evaluation points. For 'continuous' the per-point statistic is the
        variance of predictions; for 'categorical' it is the fraction of
        resamples disagreeing with that point's modal prediction. ``mape`` is
        Riley and Collins's mean absolute prediction error against the model
        fitted on the full training data.

    Raises ``ValueError`` (from :func:`bootstrap_predictions`) if task is not
    'continuous' or 'categorical', or n_bootstrap is below 2.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> bootstrap_instability(
    ...     lambda: DecisionTreeRegressor(max_depth=6),
    ...     X_train, y_train, X_test, task="continuous",
    ... )
    """
    raw = bootstrap_predictions(
        model_factory,
        X_train,
        y_train,
        X_eval,
        task=task,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    per_point = raw["per_point"]

    return {
        "instability_mean": float(np.mean(per_point)),
        "instability_p90": float(np.percentile(per_point, 90)),
        "instability_max": float(np.max(per_point)),
        "mape": float(np.mean(raw["mape_per_point"])),
    }
