"""Both families on one set of axes: where does each win?

Not a ship/no-ship bar. The question is where the accuracy-stability curves lie
relative to each other and where they cross, because that is what a user needs to
choose an operating point.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

from benchmark_datasets import ALL_DATASETS, load_dataset  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

from stable_cart import StableTree, pareto_front, stability_frontier  # noqa: E402


def main():
    """Sweep both families per dataset and report who owns the frontier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=12)
    parser.add_argument("--cap", type=int, default=1200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--datasets", type=str, nargs="+", default=list(ALL_DATASETS))
    parser.add_argument("--output", type=str, default="results/frontier_eval")
    args = parser.parse_args()

    header = f"{'dataset':22s} {'frontier points':>16s} {'cart':>6s} {'stable':>7s} {'owner':>10s} {'fits':>6s} {'secs':>7s}"
    print(header)
    print("-" * len(header))

    out_rows, tally = {}, {"cart": 0, "stable": 0, "shared": 0}
    for name in args.datasets:
        X_train, X_test, y_train, y_test, task = load_dataset(name, random_state=42)
        X = X_train[: args.cap]
        y = y_train[: args.cap]
        metric = "continuous" if task == "regression" else "categorical"
        tree_cls = (
            DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
        )

        common = dict(
            X=X, y=y, task=metric, n_bootstrap=args.n_bootstrap, random_state=42
        )
        try:
            cart = stability_frontier(
                lambda **kw: tree_cls(
                    max_depth=args.max_depth, min_samples_leaf=20, random_state=0, **kw
                ),
                {"ccp_alpha": [0.0, 0.001, 0.01, 0.1, 1.0]},
                **common,
            )
            stable = stability_frontier(
                lambda **kw: StableTree(
                    task=task,
                    max_depth=args.max_depth,
                    min_samples_leaf=20,
                    n_consensus=12,
                    random_state=0,
                    **kw,
                ),
                {"consensus_threshold": [0.0, 0.3, 0.6], "leaf_shrinkage": [0.0, 5.0]},
                **common,
            )
        except Exception as exc:
            print(f"{name:22s} failed: {type(exc).__name__}: {exc}")
            continue

        for p in cart["points"]:
            p["family"] = "cart"
        for p in stable["points"]:
            p["family"] = "stable"
        joint = pareto_front(cart["points"] + stable["points"])
        families = {p["family"] for p in joint}
        owner = "shared" if len(families) > 1 else families.pop()
        tally[owner] += 1

        n_cart = sum(p["family"] == "cart" for p in joint)
        n_stable = sum(p["family"] == "stable" for p in joint)
        fits = cart["n_fits"] + stable["n_fits"]
        secs = cart["seconds"] + stable["seconds"]
        out_rows[name] = {"joint_frontier": joint, "n_fits": fits, "seconds": secs}
        print(
            f"{name:22s} {len(joint):16d} {n_cart:6d} {n_stable:7d} {owner:>10s} {fits:6d} {secs:7.1f}"
        )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "frontiers.json").write_text(
        json.dumps(out_rows, indent=2, default=str)
    )

    print()
    print("Who appears on the joint accuracy-stability frontier:")
    for key, count in tally.items():
        print(f"  {key:8s} {count}")
    print(
        "\n'shared' means the curves cross -- neither dominates, and the choice is the user's."
    )


if __name__ == "__main__":
    main()
