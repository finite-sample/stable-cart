"""Does StableTree beat a tuned, pruned CART at matched accuracy?

The acceptance bar was fixed before the estimator was written: ship only if
StableTree reaches lower bootstrap prediction instability than a
cost-complexity-pruned CART **at matched accuracy** on at least 8 of the 14
benchmark datasets. A default-parameter CART is not the baseline — the strongest
honest baseline is, because pruning is the mechanism already known to work.

Matching accuracy is what makes the comparison mean anything. A more regularised
model looks more stable for free, so both methods are swept over their own
regularisation path and compared at a common accuracy target:

    target      = accuracy_floor x the best accuracy any configuration reaches
    comparison  = the lowest instability each method achieves while clearing it

A random forest is reported as the unreachable ceiling; it is not a tree and is
not part of the bar.

Instability and accuracy are always printed together: a configuration that looks
stable at chance accuracy is degenerate, not good.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

from benchmark_datasets import ALL_DATASETS, load_dataset  # noqa: E402

from stable_cart import StableTree, bootstrap_instability  # noqa: E402

N_CCP = 6  # alphas sampled from each dataset's own cost-complexity path
CONSENSUS_GRID = [0.0, 0.2, 0.3, 0.4, 0.6]
SHRINKAGE_GRID = [0.0, 5.0]


def _score(pred, y_true, task):
    """R² for regression, accuracy for classification."""
    if task == "classification":
        return float(np.mean(pred == y_true))
    ss_res = float(np.sum((y_true - pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def ccp_alphas(X, y, task, max_depth):
    """Alphas spread across this dataset's own cost-complexity pruning path.

    A fixed grid of absolute alphas is meaningless across datasets, because the
    scale depends on the outcome's variance; using the path makes the pruned-CART
    baseline genuinely tuned rather than nominally so.
    """
    tree_cls = DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
    path = tree_cls(
        max_depth=max_depth, min_samples_leaf=20, random_state=0
    ).cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas)
    alphas = alphas[alphas >= 0]
    if len(alphas) <= N_CCP:
        return list(alphas)
    return list(alphas[np.linspace(0, len(alphas) - 1, N_CCP).astype(int)])


def configurations(task, max_depth, n_consensus, alphas):
    """Every configuration of every arm, as (arm, label, factory)."""
    tree_cls = DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
    forest_cls = (
        RandomForestRegressor if task == "regression" else RandomForestClassifier
    )

    configs = []
    for alpha in alphas:
        configs.append(
            (
                "pruned_cart",
                f"ccp={alpha}",
                lambda a=alpha: tree_cls(
                    max_depth=max_depth,
                    min_samples_leaf=20,
                    ccp_alpha=a,
                    random_state=0,
                ),
            )
        )
    for level in CONSENSUS_GRID:
        for shrink in SHRINKAGE_GRID:
            configs.append(
                (
                    "stable_tree",
                    f"pi={level},shrink={shrink}",
                    lambda level=level, shrink=shrink: StableTree(
                        task=task,
                        max_depth=max_depth,
                        min_samples_leaf=20,
                        n_consensus=n_consensus,
                        consensus_threshold=level,
                        leaf_shrinkage=shrink,
                        random_state=0,
                    ),
                )
            )
    configs.append(
        (
            "random_forest",
            "n=50",
            lambda: forest_cls(n_estimators=50, max_depth=max_depth, random_state=0),
        )
    )
    return configs


def evaluate_dataset(name, max_depth, n_consensus, n_bootstrap, cap, seed):
    """Score every configuration on one dataset."""
    X_train, X_test, y_train, y_test, task = load_dataset(name, random_state=seed)
    if len(X_train) > cap:
        X_train, y_train = X_train[:cap], y_train[:cap]

    metric_task = "continuous" if task == "regression" else "categorical"
    alphas = ccp_alphas(X_train, y_train, task, max_depth)
    rows = []
    for arm, label, factory in configurations(task, max_depth, n_consensus, alphas):
        try:
            instability = bootstrap_instability(
                factory,
                X_train,
                y_train,
                X_test,
                task=metric_task,
                n_bootstrap=n_bootstrap,
                random_state=seed,
            )["instability_mean"]
            accuracy = _score(
                factory().fit(X_train, y_train).predict(X_test), y_test, task
            )
        except Exception as exc:  # record and continue the sweep
            rows.append(
                {"arm": arm, "config": label, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        rows.append(
            {
                "arm": arm,
                "config": label,
                "instability": instability,
                "accuracy": accuracy,
            }
        )
    return task, rows


def matched_comparison(rows, accuracy_floor):
    """Lowest instability each arm reaches while clearing the accuracy target."""
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return None

    # The target is set by the single-tree arms only. Letting the random forest
    # set it makes the bar unreachable for any tree and voids the comparison.
    trees = [r for r in ok if r["arm"] in ("pruned_cart", "stable_tree")]
    if not trees:
        return None
    best_accuracy = max(r["accuracy"] for r in trees)
    target = accuracy_floor * best_accuracy if best_accuracy > 0 else best_accuracy

    out = {"target_accuracy": target, "best_accuracy": best_accuracy}
    for arm in ("pruned_cart", "stable_tree", "random_forest"):
        eligible = [r for r in ok if r["arm"] == arm and r["accuracy"] >= target]
        if eligible:
            best = min(eligible, key=lambda r: r["instability"])
            out[arm] = {
                "instability": best["instability"],
                "accuracy": best["accuracy"],
                "config": best["config"],
            }
        else:
            out[arm] = None
    return out


def main():
    """Run the sweep on every dataset and report against the acceptance bar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-consensus", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=15)
    parser.add_argument("--cap", type=int, default=1500)
    parser.add_argument("--accuracy-floor", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", type=str, nargs="+", default=list(ALL_DATASETS))
    parser.add_argument("--output", type=str, default="results/stable_tree_eval")
    args = parser.parse_args()

    header = (
        f"{'dataset':22s} {'cart inst':>10s} {'stable inst':>12s} {'rf inst':>10s} "
        f"{'gain':>8s} {'cart acc':>9s} {'stable acc':>11s} {'winner':>8s}"
    )
    print(header)
    print("-" * len(header))

    all_rows, wins, comparable = {}, 0, 0
    for name in args.datasets:
        task, rows = evaluate_dataset(
            name,
            args.max_depth,
            args.n_consensus,
            args.n_bootstrap,
            args.cap,
            args.seed,
        )
        comparison = matched_comparison(rows, args.accuracy_floor)
        all_rows[name] = {"task": task, "rows": rows, "matched": comparison}

        if not comparison or not comparison["pruned_cart"]:
            print(f"{name:22s} {'baseline produced no usable configuration':>60s}")
            continue

        if not comparison["stable_tree"]:
            # StableTree could not reach the accuracy target on this dataset.
            # That is a loss, not an exclusion: skipping it would let the
            # estimator dodge every dataset it cannot compete on.
            comparable += 1
            print(
                f"{name:22s} {comparison['pruned_cart']['instability']:10.5g} "
                f"{'-- below accuracy target --':>36s} {'cart':>8s}"
            )
            continue

        cart = comparison["pruned_cart"]
        stable = comparison["stable_tree"]
        forest = comparison["random_forest"]
        gain = 100 * (cart["instability"] - stable["instability"]) / cart["instability"]
        won = stable["instability"] < cart["instability"]
        wins += int(won)
        comparable += 1
        print(
            f"{name:22s} {cart['instability']:10.5g} {stable['instability']:12.5g} "
            f"{(forest['instability'] if forest else float('nan')):10.5g} {gain:7.1f}% "
            f"{cart['accuracy']:9.3f} {stable['accuracy']:11.3f} "
            f"{'stable' if won else 'cart':>8s}"
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(all_rows, indent=2, default=str))

    print()
    print("=" * 72)
    print("ACCEPTANCE BAR: StableTree beats tuned pruned CART on >= 8 of 14 datasets")
    print("=" * 72)
    print(f"  won {wins} of {comparable} comparable datasets")
    print(f"  verdict: {'SHIP' if wins >= 8 else 'DO NOT SHIP as an estimator'}")


if __name__ == "__main__":
    main()
