"""Does BootstrapSplitTree beat tuned, pruned CART at a matched predictive score?

A default-parameter CART is not the baseline. Each dataset supplies CART's own
cost-complexity path, and the comparison is made only among configurations that
reach a shared score target.

Matching accuracy is what makes the comparison mean anything. A more regularized
model looks more stable for free, so both methods are swept over their own
regularization path and compared at a common accuracy target:

    target      = accuracy_floor times the best accuracy any configuration reaches
    comparison  = the lowest instability each method achieves while clearing it

Instability and accuracy are always printed together: a configuration that looks
stable at chance accuracy is degenerate, not good.
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

from experiments.benchmark_datasets import load_dataset  # noqa: E402
from experiments.bootstrap_split_tree import BootstrapSplitTree  # noqa: E402
from stable_cart import bootstrap_predictions  # noqa: E402

N_CCP = 6  # alphas sampled from each dataset's own cost-complexity path
CONSENSUS_GRID = [0.0, 0.2, 0.3, 0.4, 0.6]
SHRINKAGE_GRID = [0.0, 5.0]
VERIFICATION_DATASETS = [
    "friedman1",
    "friedman2",
    "quadrant_interaction",
    "heteroscedastic",
    "diabetes",
    "breast_cancer",
    "digits_binary",
]


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
                    "bootstrap_split",
                    f"pi={level},shrink={shrink}",
                    lambda level=level, shrink=shrink: BootstrapSplitTree(
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
    return configs


def pair_values(predictions, task):
    """Return distances for disjoint pairs of bootstrap refits."""
    half = len(predictions) // 2
    left = predictions[:half]
    right = predictions[half : 2 * half]
    if task == "classification":
        return np.mean(left != right, axis=1)
    return np.mean((left.astype(float) - right.astype(float)) ** 2, axis=1)


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
            audit = bootstrap_predictions(
                factory,
                X_train,
                y_train,
                X_test,
                task=metric_task,
                n_bootstrap=n_bootstrap,
                random_state=seed,
            )
            instability = float(np.mean(audit["pairwise"]))
            instability_mcse = audit["pairwise_standard_error"]
            paired = pair_values(audit["bootstrap"], task)
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
                "instability_mcse": instability_mcse,
                "pair_samples": paired.tolist(),
                "accuracy": accuracy,
            }
        )
    return task, rows


def matched_comparison(rows, accuracy_floor):
    """Lowest instability each arm reaches while clearing the accuracy target."""
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return None

    trees = [r for r in ok if r["arm"] in ("pruned_cart", "bootstrap_split")]
    if not trees:
        return None
    best_accuracy = max(r["accuracy"] for r in trees)
    target = accuracy_floor * best_accuracy if best_accuracy > 0 else best_accuracy

    out = {"target_accuracy": target, "best_accuracy": best_accuracy}
    for arm in ("pruned_cart", "bootstrap_split"):
        eligible = [r for r in ok if r["arm"] == arm and r["accuracy"] >= target]
        if eligible:
            best = min(eligible, key=lambda r: r["instability"])
            out[arm] = {
                "instability": best["instability"],
                "instability_mcse": best["instability_mcse"],
                "pair_samples": best["pair_samples"],
                "accuracy": best["accuracy"],
                "config": best["config"],
            }
        else:
            out[arm] = None
    out["eligibility"] = (
        "both"
        if out["pruned_cart"] and out["bootstrap_split"]
        else "pruned_cart_only"
        if out["pruned_cart"]
        else "bootstrap_split_only"
    )
    if out["pruned_cart"] is not None and out["bootstrap_split"] is not None:
        cart_pairs = np.asarray(out["pruned_cart"]["pair_samples"])
        split_pairs = np.asarray(out["bootstrap_split"]["pair_samples"])
        deltas = split_pairs - cart_pairs
        point_delta = (
            out["bootstrap_split"]["instability"] - out["pruned_cart"]["instability"]
        )
        out["paired_instability_delta"] = float(point_delta)
        out["paired_instability_delta_mcse"] = float(
            deltas.std(ddof=1) / np.sqrt(len(deltas))
        )
        out["paired_instability_change_percent"] = float(
            100 * point_delta / out["pruned_cart"]["instability"]
            if out["pruned_cart"]["instability"] != 0
            else float("nan")
        )
    return out


def write_matched_csv(results, destination):
    """Write matched results in a directly inspectable format."""
    fields = [
        "dataset",
        "task",
        "eligibility",
        "cart_instability",
        "cart_mcse",
        "cart_score",
        "cart_config",
        "bootstrap_split_instability",
        "bootstrap_split_mcse",
        "bootstrap_split_score",
        "bootstrap_split_config",
        "paired_instability_delta",
        "paired_instability_delta_mcse",
        "paired_instability_change_percent",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, result in results.items():
            matched = result["matched"]
            cart = matched["pruned_cart"] if matched else None
            split = matched["bootstrap_split"] if matched else None
            writer.writerow(
                {
                    "dataset": dataset,
                    "task": result["task"],
                    "eligibility": matched["eligibility"]
                    if matched
                    else "no_usable_fits",
                    "cart_instability": cart["instability"] if cart else None,
                    "cart_mcse": cart["instability_mcse"] if cart else None,
                    "cart_score": cart["accuracy"] if cart else None,
                    "cart_config": cart["config"] if cart else None,
                    "bootstrap_split_instability": (
                        split["instability"] if split else None
                    ),
                    "bootstrap_split_mcse": (
                        split["instability_mcse"] if split else None
                    ),
                    "bootstrap_split_score": split["accuracy"] if split else None,
                    "bootstrap_split_config": split["config"] if split else None,
                    "paired_instability_delta": (
                        matched.get("paired_instability_delta") if matched else None
                    ),
                    "paired_instability_delta_mcse": (
                        matched.get("paired_instability_delta_mcse")
                        if matched
                        else None
                    ),
                    "paired_instability_change_percent": (
                        matched.get("paired_instability_change_percent")
                        if matched
                        else None
                    ),
                }
            )


def main():
    """Run the matched-score sweep and report the descriptive comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-consensus", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=40)
    parser.add_argument("--cap", type=int, default=1000)
    parser.add_argument("--accuracy-floor", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=VERIFICATION_DATASETS
    )
    parser.add_argument("--output", type=str, default="results/bootstrap_split_eval")
    args = parser.parse_args()

    header = (
        f"{'dataset':22s} {'cart pair':>10s} {'split pair':>12s} "
        f"{'change':>8s} {'paired MCSE':>12s} {'cart score':>10s} "
        f"{'split score':>11s} {'winner':>8s}"
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

        if not comparison:
            print(f"{name:22s} no usable fits")
            continue
        if not comparison["pruned_cart"]:
            comparable += 1
            wins += 1
            print(
                f"{name:22s} pruned CART below score target; bootstrap split eligible"
            )
            continue

        if not comparison["bootstrap_split"]:
            # BootstrapSplitTree could not reach the accuracy target on this dataset.
            # That is a loss, not an exclusion: skipping it would let the
            # estimator dodge every dataset it cannot compete on.
            comparable += 1
            print(
                f"{name:22s} {comparison['pruned_cart']['instability']:10.5g} "
                f"{'-- below accuracy target --':>36s} {'cart':>8s}"
            )
            continue

        cart = comparison["pruned_cart"]
        stable = comparison["bootstrap_split"]
        change = comparison["paired_instability_change_percent"]
        won = stable["instability"] < cart["instability"]
        wins += int(won)
        comparable += 1
        print(
            f"{name:22s} {cart['instability']:10.5g} {stable['instability']:12.5g} "
            f"{change:7.1f}% {comparison['paired_instability_delta_mcse']:12.3g} "
            f"{cart['accuracy']:9.3f} {stable['accuracy']:11.3f} "
            f"{'split' if won else 'cart':>8s}"
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(all_rows, indent=2, default=str))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    write_matched_csv(all_rows, out / "matched.csv")

    print()
    print("=" * 72)
    print("MATCHED-SCORE FALSIFICATION SUMMARY")
    print("=" * 72)
    print(f"  won {wins} of {comparable} comparable datasets")
    print("  This count is descriptive; inspect each score-instability pair.")


if __name__ == "__main__":
    main()
