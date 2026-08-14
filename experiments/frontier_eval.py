"""Every estimator on one set of axes: which one do I reach for?

The package ships five estimators plus plain CART, and a user is entitled to a
straight answer about which to use. This sweeps each one's own parameters on
each dataset, pools the results, and reports who contributes points to the joint
accuracy-stability frontier — the configurations no other configuration beats on
both axes at once.

It is not a ship/no-ship bar. A family that owns no frontier point on a dataset
is not broken; it is dominated *there*, and where the curves cross is the part a
user needs to see.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

from benchmark_datasets import ALL_DATASETS, load_dataset  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

from stable_cart import (  # noqa: E402
    BootstrapVariancePenalizedTree,
    CentroidTree,
    LessGreedyHybridTree,
    RobustPrefixHonestTree,
    StableTree,
    pareto_front,
    stability_frontier,
)


def arms(task, max_depth):
    """The estimators to sweep, each with the knobs it actually documents.

    Every grid is the parameters that estimator exposes for stability, plus the
    same depth in every case — a family given a deeper tree than the others
    would look better on accuracy and worse on stability for no reason a user
    could act on.
    """
    tree_cls = DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
    shared = {"max_depth": max_depth, "min_samples_leaf": 20, "random_state": 0}

    return {
        "cart": (
            lambda **kw: tree_cls(**shared, **kw),
            {"ccp_alpha": [0.0, 0.001, 0.01, 0.1, 1.0]},
        ),
        "stable": (
            lambda **kw: StableTree(task=task, n_consensus=12, **shared, **kw),
            {"consensus_threshold": [0.0, 0.3, 0.6], "leaf_shrinkage": [0.0, 5.0]},
        ),
        "less_greedy": (
            lambda **kw: LessGreedyHybridTree(task=task, **shared, **kw),
            {"leaf_smoothing": [0.0, 5.0], "enable_oblique_splits": [False, True]},
        ),
        "bootstrap_var": (
            lambda **kw: BootstrapVariancePenalizedTree(task=task, **shared, **kw),
            {"variance_penalty": [0.0, 1.0, 5.0], "leaf_smoothing": [0.0, 5.0]},
        ),
        "robust_prefix": (
            # This one derives min_samples_split from min_samples_leaf and takes
            # no min_samples_split of its own.
            lambda **kw: RobustPrefixHonestTree(
                task=task,
                max_depth=max_depth,
                min_samples_leaf=20,
                random_state=0,
                **kw,
            ),
            {"top_levels": [1, 2], "smoothing": [1.0, 10.0]},
        ),
        "centroid": (
            lambda **kw: CentroidTree(
                task="regression" if task == "regression" else "classification",
                base_params={"max_depth": max_depth, "min_samples_leaf": 20},
                random_state=0,
                **kw,
            ),
            {"n_candidates": [5, 20], "proximity_metric": ["rmse", "correlation"]}
            if task == "regression"
            else {
                "n_candidates": [5, 20],
                "proximity_metric": ["rmse", "disagreement"],
            },
        ),
    }


def main():
    """Sweep every arm per dataset and report who owns the frontier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=12)
    parser.add_argument("--cap", type=int, default=1200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="drop configurations that cannot beat predicting the mean/majority class",
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=list(ALL_DATASETS))
    parser.add_argument("--output", type=str, default="results/frontier_eval")
    args = parser.parse_args()

    names = list(arms("regression", args.max_depth))
    header = f"{'dataset':22s} {'front':>6s} " + " ".join(f"{n:>13s}" for n in names)
    print(header)
    print("-" * len(header))

    out_rows: dict[str, dict] = {}
    tally = dict.fromkeys(names, 0)
    for name in args.datasets:
        X_train, _X_test, y_train, _y_test, task = load_dataset(name, random_state=42)
        X, y = X_train[: args.cap], y_train[: args.cap]
        metric = "continuous" if task == "regression" else "categorical"
        common = {
            "X": X,
            "y": y,
            "task": metric,
            "n_bootstrap": args.n_bootstrap,
            "random_state": 42,
        }

        points, cost = [], {"n_fits": 0, "seconds": 0.0}
        for arm, (factory, grid) in arms(task, args.max_depth).items():
            try:
                result = stability_frontier(factory, grid, **common)
            except Exception as exc:
                print(f"{name:22s} {arm} failed: {type(exc).__name__}: {exc}")
                continue
            for point in result["points"]:
                point["arm"] = arm
            points.extend(result["points"])
            cost["n_fits"] += result["n_fits"]
            cost["seconds"] += result["seconds"]

        # A Pareto frontier always contains the stable-and-useless corner: a
        # model that predicts near-constantly is never dominated, because nothing
        # is more stable. Frontier ownership only means something among usable
        # models, so drop anything that cannot beat predicting the mean (or the
        # majority class).
        floor = args.min_accuracy
        if metric == "categorical":
            floor = max(floor, float(np.mean(y == np.bincount(y.astype(int)).argmax())))
        usable = [p for p in points if p["accuracy"] > floor]
        joint = pareto_front(usable)
        if not joint:
            print(f"{name:22s} nothing clears accuracy {floor:.3f}")
            continue

        counts = {arm: sum(p["arm"] == arm for p in joint) for arm in names}
        for arm, count in counts.items():
            tally[arm] += count > 0
        out_rows[name] = {"joint_frontier": joint, **cost}
        cells = " ".join(f"{counts[arm]:13d}" for arm in names)
        print(f"{name:22s} {len(joint):6d} {cells}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "frontiers.json").write_text(
        json.dumps(out_rows, indent=2, default=str)
    )

    print()
    print("Datasets on which each estimator contributes at least one frontier point:")
    for arm, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {arm:16s} {count} / {len(out_rows)}")
    print(
        "\nMore than one arm on a frontier means the curves cross -- neither"
        "\ndominates, and the operating point is the user's choice."
    )


if __name__ == "__main__":
    main()
