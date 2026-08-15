"""Three plots for prediction stability, and nothing else.

A single number for instability hides the two things a user actually needs to
know: *how much* the predictions move, and *for whom*. A model can have a
respectable average and still be a coin flip for the tenth of cases that matter
most. These plots put both on the page.

They follow Riley and Collins, *Stability of clinical prediction models
developed using statistical or machine learning methods*, Biometrical Journal
65(8), 2023 — the same protocol implemented by the R package ``pminternal``,
which has had no scikit-learn equivalent.

Matplotlib is an optional dependency::

    pip install "stable-cart[plots]"

Each function takes an ``ax`` and returns it, so the plots compose into a figure
the caller controls rather than dictating one.
"""

from typing import Any

import numpy as np

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


def plot_prediction_instability(
    result: dict[str, Any],
    ax: Any = None,
    max_points: int = 400,
    random_state: int | None = 0,
    band: bool = True,
    n_bins: int = 25,
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

    Returns
    -------
    Any
        The axes, for further customisation.

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

    original = np.asarray(result["original"], dtype=float)
    boot = np.asarray(result["bootstrap"], dtype=float)

    columns = np.arange(len(original))
    if len(columns) > max_points:
        rng = np.random.default_rng(random_state)
        columns = np.sort(rng.choice(columns, size=max_points, replace=False))

    x = np.repeat(original[columns], boot.shape[0])
    y = boot[:, columns].T.ravel()

    # Enough dots to see the shape, faint enough that the middle is not a slab.
    alpha = float(np.clip(3000.0 / max(len(x), 1), 0.02, 0.35))
    ax.scatter(x, y, s=4, alpha=alpha, edgecolors="none", color="#1f77b4")

    if band and len(original) > n_bins:
        order = np.argsort(original)
        groups = [g for g in np.array_split(order, n_bins) if len(g)]
        centres = np.array([np.mean(original[g]) for g in groups])
        lower = np.array([np.percentile(boot[:, g], 5) for g in groups])
        upper = np.array([np.percentile(boot[:, g], 95) for g in groups])
        ax.plot(centres, lower, color="#1f77b4", lw=1.4)
        ax.plot(centres, upper, color="#1f77b4", lw=1.4, label="5th-95th percentile")

    limits = [
        min(float(np.min(x)), float(np.min(y))),
        max(float(np.max(x)), float(np.max(y))),
    ]
    ax.plot(limits, limits, color="#d62728", lw=1.4, ls="--", label="perfect stability")

    ax.set_xlabel("prediction from the model fitted on all the data")
    ax.set_ylabel("prediction from a model fitted on a resample")
    ax.set_title(f"Prediction instability ({boot.shape[0]} resamples)")
    ax.legend(loc="upper left", frameon=False)
    return ax


def plot_mape_by_prediction(
    result: dict[str, Any], ax: Any = None, n_bins: int = 20
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

    Returns
    -------
    Any
        The axes.

    Raises
    ------
    ValueError
        If ``n_bins`` is below 2.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    plt = _require_matplotlib()
    ax = ax or plt.subplots(figsize=(6.0, 4.0))[1]

    original = np.asarray(result["original"], dtype=float)
    mape = np.asarray(result["mape_per_point"], dtype=float)

    distinct = np.unique(original)
    if len(distinct) <= n_bins:
        groups = [np.flatnonzero(original == value) for value in distinct]
    else:
        order = np.argsort(original)
        groups = np.array_split(order, min(n_bins, len(order)))
    centres = np.array([np.mean(original[g]) for g in groups if len(g)])
    heights = np.array([np.mean(mape[g]) for g in groups if len(g)])
    spread = np.array([np.percentile(mape[g], 90) for g in groups if len(g)])

    ax.fill_between(
        centres, heights, spread, alpha=0.2, color="#1f77b4", label="90th pct"
    )
    ax.plot(centres, heights, marker="o", ms=4, color="#1f77b4", label="mean")
    ax.axhline(
        float(np.mean(mape)), color="#7f7f7f", lw=1.0, ls=":", label="overall mean"
    )

    label = (
        "disagreement with the original model"
        if result.get("task") == "categorical"
        else "mean absolute prediction error"
    )
    ax.set_xlabel("prediction from the model fitted on all the data")
    ax.set_ylabel(label)
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
    Plot one or more model families on the accuracy-stability plane.

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
        ``'instability'`` (variance across resamples) or ``'mape'`` (Riley and
        Collins's mean absolute prediction error).

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
        colour = palette[index % len(palette)]
        front = {id(p) for p in result["frontier"]}

        dominated = [p for p in result["points"] if id(p) not in front]
        if dominated:
            ax.scatter(
                [p[metric] for p in dominated],
                [p["accuracy"] for p in dominated],
                s=28,
                facecolors="none",
                edgecolors=colour,
                alpha=0.5,
            )

        ordered = sorted(result["frontier"], key=lambda p: p[metric])
        ax.plot(
            [p[metric] for p in ordered],
            [p["accuracy"] for p in ordered],
            marker="o",
            ms=6,
            color=colour,
            label=name,
        )
        if annotate:
            for point in ordered:
                text = ", ".join(f"{k}={v}" for k, v in point["params"].items())
                ax.annotate(
                    text,
                    (point[metric], point["accuracy"]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7,
                    color=colour,
                )

    ax.set_xlabel(
        "prediction variance across resamples"
        if metric == "instability"
        else "mean absolute prediction error"
    )
    ax.set_ylabel("held-out accuracy")
    ax.set_title("Accuracy against stability — up and to the left is better")
    ax.legend(frameon=False)
    return ax
