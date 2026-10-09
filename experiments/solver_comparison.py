"""Shared representation and complexity controls for CART/GOSDT comparisons."""

from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def binarizer_edges(X, n_thresholds):
    """Learn quantile cutpoints from fitting observations."""
    qs = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    return [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]


def binarize(X, edges):
    cols, origin = [], []
    for j, cuts in enumerate(edges):
        for cut in cuts:
            cols.append(X[:, j] <= cut)
            origin.append(j)
    return np.column_stack(cols), origin


def tree_features(model, origin):
    counts = Counter()

    def walk(node):
        feature = getattr(node, "feature", None)
        if feature is not None:
            counts[origin[int(feature)]] += 1
            walk(node.left_child)
            walk(node.right_child)

    walk(model.trees_[0].tree)
    return counts


def fit_arms(X, y, X_eval, thresholds, depth, regularization, time_limit, forest=False):
    """Fit on a common sample; record certificates and realized complexity.

    This compares algorithms with different objectives, not search alone.
    GOSDT minimizes penalized misclassification; CART uses greedy Gini splits.
    """
    from gosdt import GOSDTClassifier

    edges = binarizer_edges(X, thresholds)
    Xb, origin = binarize(X, edges)
    Xe, _ = binarize(X_eval, edges)
    classes, counts = np.unique(y, return_counts=True)
    majority = classes[np.argmax(counts)]
    constant_error = 1 - counts.max() / len(y)
    if constant_error <= regularization:
        leaves = 1
        features = {"gosdt": Counter()}
        solver_prediction = np.full(len(X_eval), majority)
        solver_accuracy = float(1 - constant_error)
        objective = float(constant_error + regularization)
        certificate = {
            "status": "analytical_constant_bound",
            "lower_bound": objective,
            "upper_bound": objective,
            "certified": True,
            "seconds": 0.0,
            "leaf_budget": 1,
            "cutpoints": [cuts.tolist() for cuts in edges],
        }
    else:
        solver = GOSDTClassifier(
            regularization=regularization,
            depth_budget=depth,
            time_limit=time_limit,
            verbose=False,
            allow_small_reg=True,
        ).fit(Xb, y)
        result = solver.result_
        features = {"gosdt": tree_features(solver, origin)}
        leaves = sum(features["gosdt"].values()) + 1
        bounds_closed = np.isclose(
            result.lowerbound, result.upperbound, rtol=0, atol=1e-8
        )
        certificate = {
            "status": str(result.status),
            "lower_bound": float(result.lowerbound),
            "upper_bound": float(result.upperbound),
            "certified": bool(
                bounds_closed and str(result.status) == "Status.CONVERGED"
            ),
            "seconds": float(result.time),
            "leaf_budget": leaves,
            "cutpoints": [cuts.tolist() for cuts in edges],
        }
        solver_prediction = np.asarray(solver.predict(Xe))
        solver_accuracy = float(np.mean(solver.predict(Xb) == y))
    # A one-leaf solution must remain a constant predictor in every single-tree arm.
    complexity = (
        {"max_leaf_nodes": leaves} if leaves > 1 else {"min_samples_split": len(y) + 1}
    )
    models = {
        "cart_raw": DecisionTreeClassifier(
            max_depth=depth, random_state=0, **complexity
        ),
        "cart_binary": DecisionTreeClassifier(
            max_depth=depth, random_state=0, **complexity
        ),
    }
    if forest:
        models["forest"] = RandomForestClassifier(
            n_estimators=50, max_depth=depth, random_state=0, **complexity
        )
    predictions = {"gosdt": solver_prediction}
    metadata = {
        "gosdt": {
            "leaves": leaves,
            "train_accuracy": solver_accuracy,
        }
    }
    for name, model in models.items():
        fit_X, test_X = (X, X_eval) if name == "cart_raw" else (Xb, Xe)
        model.fit(fit_X, y)
        predictions[name] = model.predict(test_X)
        if name != "forest":
            features[name] = Counter(
                (int(f) if name == "cart_raw" else origin[int(f)])
                for f in model.tree_.feature
                if f >= 0
            )
            metadata[name] = {
                "leaves": int(model.get_n_leaves()),
                "depth": int(model.get_depth()),
                "train_accuracy": float(np.mean(model.predict(fit_X) == y)),
            }
    return predictions, features, metadata, certificate


def summarize_predictions(predictions, y, seed=0, resamples=2000):
    """Paired whole-fit bootstrap, conditional on the evaluation sample."""
    arrays = {name: np.asarray(values) for name, values in predictions.items()}
    size = len(next(iter(arrays.values())))
    if size < 2:
        raise ValueError("At least two independent fits are required.")
    draws = np.random.default_rng(seed).integers(size, size=(resamples, size))
    summaries = {}
    for name, arr in arrays.items():
        accuracy = np.mean(arr == y, axis=1)
        # Binary disagreement across all distinct pairs of fits.
        sums = arr.sum(axis=0)
        instability = np.mean(2 * sums * (size - sums) / (size * (size - 1)))
        boot = []
        for indices in draws:
            counts = arr[indices].sum(axis=0)
            boot.append(np.mean(2 * counts * (size - counts) / (size * (size - 1))))
        summaries[name] = {
            "accuracy": float(accuracy.mean()),
            "pair_disagreement": float(instability),
            "accuracy_ci": np.quantile(
                accuracy[draws].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "disagreement_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    delta = np.mean(arrays["gosdt"] == y, axis=1) - np.mean(
        arrays["cart_binary"] == y, axis=1
    )
    summaries["gosdt_minus_cart_binary_accuracy_pp"] = {
        "mean": float(delta.mean() * 100),
        "ci": (np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]) * 100).tolist(),
    }
    return summaries
