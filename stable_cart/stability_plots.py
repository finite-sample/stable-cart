"""Three plots for prediction stability, and nothing else.

A single number for instability hides the two things a user actually needs to
know: *how much* the predictions move, and *for whom*. A model can have a
respectable average and still be a coin flip for the tenth of cases that matter
most. These plots put both on the page.

They follow Riley and Collins, *Stability of clinical prediction models
developed using statistical or machine learning methods*, Biometrical Journal
65(8), 2023, and implement the protocol for scikit-learn-compatible fitting
procedures.

Matplotlib is an optional dependency::

    pip install "stable-cart[plots]"

Each function takes an ``ax`` and returns it, so the plots compose into a figure
the caller controls rather than dictating one.
"""

from typing import Any

import numpy as np

from .frontier import pareto_front

__all__ = [
    "plot_prediction_instability",
    "plot_mape_by_prediction",
    "plot_stability_frontier",
]


def _require_matplotlib():
    """Import matplotlib, or explain how to get it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Plotting needs matplotlib. Install it with: "
            'pip install "stable-cart[plots]"'
        ) from exc
    return plt


def _display_predictions(
    result: dict[str, Any], class_label: Any
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None]:
    """Select the one-dimensional predictions a scatter plot can display."""
    original = np.asarray(result["original"])
    bootstrap = np.asarray(result["bootstrap"])
    if result.get("metric") != "probability_vector":
        if result.get("task") != "categorical":
            return original.astype(float), bootstrap.astype(float), "prediction", None
        supplied_classes = result.get("classes")
        classes = (
            np.asarray(supplied_classes)
            if supplied_classes is not None
            else np.unique(np.concatenate([original.ravel(), bootstrap.ravel()]))
        )
        encoded_original = np.full(original.shape, np.nan, dtype=float)
        encoded_bootstrap = np.full(bootstrap.shape, np.nan, dtype=float)
        for index, label in enumerate(classes):
            encoded_original[original == label] = index
            encoded_bootstrap[bootstrap == label] = index
        if np.any(np.isnan(encoded_original)) or np.any(np.isnan(encoded_bootstrap)):
            raise ValueError("result contains a class absent from result['classes']")
        return encoded_original, encoded_bootstrap, "class label", classes

    if class_label is None:
        raise ValueError(
            "class_label is required to plot probability vectors; the audit "
            "metrics still use the full vector."
        )
    classes = np.asarray(result.get("classes"))
    matches = np.flatnonzero(classes == class_label)
    if len(matches) != 1:
        raise ValueError(
            "class_label must identify exactly one class in result['classes']"
        )
    column = int(matches[0])
    return (
        original[:, column].astype(float),
        bootstrap[:, :, column].astype(float),
        f"predicted probability for class {class_label!r}",
        None,
    )


def plot_prediction_instability(
    result: dict[str, Any],
    ax: Any = None,
    max_points: int = 400,
    random_state: int | None = 0,
    band: bool = True,
    n_bins: int = 25,
    class_label: Any = None,
) -> Any:
    """
    Draw the instability plot: original prediction against resampled predictions.

    One row of the training data is one individual. The x-axis is what the model
    fitted on the full data predicts for them; the y-axis is what each model
    fitted on a bootstrap resample predicts for the *same* individual. A perfectly
    stable procedure puts every point on the diagonal. The vertical spread at a
    given x is the honest answer to "how much would this prediction have differed
    if the data had come out slightly differently".

    Parameters
    ----------
    result
        Output of :func:`~stable_cart.bootstrap_predictions`.
    ax
        Axes to draw on. A new figure is created when omitted.
    max_points
        Cap on the number of *individuals* scattered — each contributes one dot
        per resample, so a few hundred is already tens of thousands of dots.
        Beyond that the cloud saturates into a slab and stops showing density,
        so a random subset is drawn instead.
    random_state
        Seed for that subset.
    band
        Overlay the 5th-95th percentile of resampled predictions, binned along
        the x-axis. This is the part that survives overplotting, and it is what
        makes the width of the cloud readable rather than merely visible.
    n_bins
        Number of equal-count bins for that band.
    class_label
        Class whose probability to put on the axes when ``result`` contains
        probability vectors. Required for probability audits. This affects only
        the display; the audit statistics use the full probability vector.

    Returns
    -------
    Any
        The axes, for further customization.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from stable_cart import bootstrap_predictions, plot_prediction_instability
    >>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    >>> raw = bootstrap_predictions(
    ...     lambda: DecisionTreeRegressor(max_depth=5, random_state=0),
    ...     X[:150], y[:150], X[150:], n_bootstrap=10, random_state=0,
    ... )
    >>> type(plot_prediction_instability(raw)).__name__
    'Axes'
    """
    plt = _require_matplotlib()
    ax = ax or plt.subplots(figsize=(5.5, 5.0))[1]

    original, boot, prediction_label, tick_labels = _display_predictions(
        result, class_label
    )

    columns = np.arange(len(original))
    if len(columns) > max_points:
        rng = np.random.default_rng(random_state)
        columns = np.sort(rng.choice(columns, size=max_points, replace=False))

    x = np.repeat(original[columns], boot.shape[0])
    y = boot[:, columns].T.ravel()

    # Enough dots to see the shape, faint enough that the middle is not a slab.
    alpha = float(np.clip(3000.0 / max(len(x), 1), 0.02, 0.35))
    ax.scatter(x, y, s=4, alpha=alpha, edgecolors="none", color="#1f77b4")

    if band and tick_labels is None and len(original) > n_bins:
        order = np.argsort(original)
        groups = [g for g in np.array_split(order, n_bins) if len(g)]
        centers = np.array([np.mean(original[g]) for g in groups])
        lower = np.array([np.percentile(boot[:, g], 5) for g in groups])
        upper = np.array([np.percentile(boot[:, g], 95) for g in groups])
        ax.plot(centers, lower, color="#1f77b4", lw=1.4)
        ax.plot(centers, upper, color="#1f77b4", lw=1.4, label="5th-95th percentile")

    limits = [
        min(float(np.min(x)), float(np.min(y))),
        max(float(np.max(x)), float(np.max(y))),
    ]
    ax.plot(limits, limits, color="#d62728", lw=1.4, ls="--", label="perfect stability")

    ax.set_xlabel(f"full-data {prediction_label}")
    ax.set_ylabel(f"resampled {prediction_label}")
    if tick_labels is not None:
        positions = np.arange(len(tick_labels))
        labels = [str(label) for label in tick_labels]
        ax.set_xticks(positions, labels)
        ax.set_yticks(positions, labels)
    ax.set_title(f"Prediction instability ({boot.shape[0]} resamples)")
    ax.legend(loc="upper left", frameon=False)
    return ax


def plot_mape_by_prediction(
    result: dict[str, Any],
    ax: Any = None,
    n_bins: int = 20,
    class_label: Any = None,
) -> Any:
    """
    Show instability as a function of predicted value: *who* the model is unsure about.

    Averaged over everyone, instability is a single number that hides its own
    distribution. Binned against the original prediction it answers the question
    a user actually has — whether the movement is spread evenly or concentrated
    in the range where decisions get made.

    Parameters
    ----------
    result
        Output of :func:`~stable_cart.bootstrap_predictions`.
    ax
        Axes to draw on. A new figure is created when omitted.
    n_bins
        Number of equal-count bins along the predicted-value axis. Equal-count
        rather than equal-width, so a sparse tail cannot produce a bin of two
        points and a dramatic-looking mean. A tree predicts one value per leaf,
        so when there are fewer distinct predictions than bins the distinct
        values are used directly — otherwise one leaf is split across two bins
        and the difference between them is noise drawn as signal.
    class_label
        Class whose original-fit probability defines the horizontal axis when
        ``result`` contains probability vectors. Required for probability
        audits. The vertical statistic still measures the full vector.

    Returns
    -------
    Any
        The axes.

    Raises
    ------
    ValueError
        If ``n_bins`` is below 2, or probability vectors are supplied without
        one valid ``class_label``.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    plt = _require_matplotlib()
    ax = ax or plt.subplots(figsize=(6.0, 4.0))[1]

    original, _boot, prediction_label, tick_labels = _display_predictions(
        result, class_label
    )
    mape = np.asarray(result["mape_per_point"], dtype=float)

    distinct = np.unique(original)
    if len(distinct) <= n_bins:
        groups = [np.flatnonzero(original == value) for value in distinct]
    else:
        order = np.argsort(original)
        groups = np.array_split(order, min(n_bins, len(order)))
    centers = np.array([np.mean(original[g]) for g in groups if len(g)])
    heights = np.array([np.mean(mape[g]) for g in groups if len(g)])
    spread = np.array([np.percentile(mape[g], 90) for g in groups if len(g)])

    ax.fill_between(
        centers, heights, spread, alpha=0.2, color="#1f77b4", label="90th pct"
    )
    ax.plot(centers, heights, marker="o", ms=4, color="#1f77b4", label="mean")
    ax.axhline(
        float(np.mean(mape)), color="#7f7f7f", lw=1.0, ls=":", label="overall mean"
    )

    if result.get("metric") == "probability_vector":
        label = "mean absolute probability-vector difference"
    elif result.get("task") == "categorical":
        label = "disagreement with the original model"
    else:
        label = "mean absolute prediction error"
    ax.set_xlabel(f"full-data {prediction_label}")
    ax.set_ylabel(label)
    if tick_labels is not None:
        ax.set_xticks(
            np.arange(len(tick_labels)), [str(value) for value in tick_labels]
        )
    ax.set_title("Where the model is unreliable")
    ax.legend(frameon=False)
    return ax


def plot_stability_frontier(
    results: dict[str, dict[str, Any]],
    ax: Any = None,
    annotate: bool = True,
    metric: str = "instability",
) -> Any:
    """
    Plot one or more model families on the validation-score/stability plane.

    The point of putting families on shared axes is that the answer is often
    "pruning wins", and a plot that cannot show that is advocacy rather than
    measurement. Filled markers joined by a line are each family's Pareto set;
    hollow markers are the configurations it dominates.

    Parameters
    ----------
    results
        Mapping of family name to the output of
        :func:`~stable_cart.stability_frontier`.
    ax
        Axes to draw on. A new figure is created when omitted.
    annotate
        Label each frontier point with its parameters. Turn off when the grid is
        large enough that the labels collide.
    metric
        ``'instability'`` (the quantity selected when constructing the frontier)
        or ``'mape'`` (Riley and Collins's mean absolute prediction error).

    Returns
    -------
    Any
        The axes.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the two supported keys.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from stable_cart import plot_stability_frontier, stability_frontier
    >>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    >>> cart = stability_frontier(
    ...     lambda **kw: DecisionTreeRegressor(random_state=0, **kw),
    ...     {"max_depth": [2, 5]}, X, y, n_bootstrap=8, random_state=0,
    ... )
    >>> type(plot_stability_frontier({"CART": cart})).__name__
    'Axes'
    """
    if metric not in ("instability", "mape"):
        raise ValueError("metric must be 'instability' or 'mape'")

    plt = _require_matplotlib()
    ax = ax or plt.subplots(figsize=(6.5, 4.5))[1]

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    for index, (name, result) in enumerate(results.items()):
        plot_color = palette[index % len(palette)]
        plotted_frontier = pareto_front(result["points"], instability_key=metric)
        front = {id(p) for p in plotted_frontier}

        dominated = [p for p in result["points"] if id(p) not in front]
        if dominated:
            ax.scatter(
                [p[metric] for p in dominated],
                [p["score"] for p in dominated],
                s=28,
                facecolors="none",
                edgecolors=plot_color,
                alpha=0.5,
            )

        ordered = sorted(plotted_frontier, key=lambda p: p[metric])
        ax.plot(
            [p[metric] for p in ordered],
            [p["score"] for p in ordered],
            marker="o",
            ms=6,
            color=plot_color,
            label=name,
        )
        if annotate:
            for point in ordered:
                text = ", ".join(f"{k}={v}" for k, v in point["params"].items())
                ax.annotate(
                    text,
                    (point[metric], point["score"]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7,
                    color=plot_color,
                )

    ax.set_xlabel(
        "prediction instability"
        if metric == "instability"
        else "mean absolute prediction error"
    )
    score_names = {result.get("score_name", "score") for result in results.values()}
    score_label = score_names.pop() if len(score_names) == 1 else "score"
    ax.set_ylabel(f"validation {score_label}")
    ax.set_title("Validation score against instability — up and left is better")
    ax.legend(frameon=False)
    return ax
