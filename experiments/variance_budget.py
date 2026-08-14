"""The variance budget of a decision tree.

Splits prediction instability into the part caused by the tree choosing a
different *structure* on a new sample and the part caused by re-estimating
*leaf* values within a fixed structure. These call for different fixes — better
split selection versus shrinkage — so knowing which dominates decides what can
possibly work.

Method — a nested design, so the two parts are a genuine decomposition. Fitting
one reference structure and re-estimating its leaves does **not** decompose
anything: the reference comes from a different sample than the refits, the two
quantities are not nested, and the "share" can exceed 100% (observed: 122.7%).

Instead, apply the law of total variance by conditioning on the structure. Draw
S independent samples and fit a structure to each; for every structure draw L
further independent samples and re-estimate only its leaf values:

    within  = E_structure[ Var_leafsample( prediction ) ]      the leaf part
    between = Var_structure[ E_leafsample( prediction ) ]      the structure part
    total   = within + between                                 exactly

so the leaf share lies in [0, 1] by construction. The ordinary refit variance —
where structure and leaves come from the *same* sample — is reported alongside as
``instability_refit``; it differs from ``total`` by the dependence between the
structure and the leaf estimates, and that difference is itself informative.

Everything is divided by Var(y): raw variance is in y-units² and is not
comparable across DGPs or noise levels.

Every reported quantity carries a Monte Carlo standard error (Morris, White &
Crowther, Stat Med 2019). Accuracy and fidelity are reported beside every
stability number, because a model that ignores its training data is perfectly
stable and useless.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

from dgps import DGP_NAMES, make_dgp  # noqa: E402


def _leaf_means(tree, X, y):
    """Re-estimate each leaf's value from a fresh sample."""
    leaves = tree.apply(X)
    return {int(leaf): float(y[leaves == leaf].mean()) for leaf in np.unique(leaves)}


def _predict_with_means(tree, X_eval_leaves, means, fallback):
    """Predict by routing to the reference structure and using supplied means."""
    return np.array([means.get(int(leaf), fallback) for leaf in X_eval_leaves])


def _mc_se(values):
    """Monte Carlo standard error of a mean over independent replicates."""
    values = np.asarray(values, dtype=float)
    return (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else float("nan")
    )


def budget(dgp, n, max_depth, min_samples_leaf, n_structures, n_leaf_samples, seed=0):
    """
    Decompose instability into structure and leaf components for one regime.

    Uses the law of total variance conditioning on the fitted structure, so the
    two components sum to the total exactly and the share is in [0, 1].

    Parameters
    ----------
    dgp
        A DGP from ``dgps.make_dgp``.
    n
        Training sample size.
    max_depth, min_samples_leaf
        Tree complexity controls.
    n_structures
        Independently fitted structures (the outer Monte Carlo loop).
    n_leaf_samples
        Independent leaf re-estimations per structure (the inner loop).
    seed
        Base seed.

    Returns
    -------
    dict
        Normalised within/between/total instabilities with Monte Carlo standard
        errors, the leaf share, the ordinary refit instability for reference,
        accuracy, fidelity and mean leaf count.
    """
    rng = np.random.default_rng(seed)

    X_eval, y_eval = dgp.sample(4000, np.random.default_rng(seed + 10_000))
    var_y = float(np.var(y_eval))
    ss_tot = float(np.sum((y_eval - np.mean(y_eval)) ** 2))

    per_structure_means = []
    per_structure_within = []
    refit_preds, scores, recoveries, n_leaves = [], [], [], []

    for _ in range(n_structures):
        X, y = dgp.sample(n, rng)
        tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0
        ).fit(X, y)

        # The ordinary refit prediction: structure and leaves from the same sample.
        pred = tree.predict(X_eval)
        refit_preds.append(pred)
        ss_res = float(np.sum((y_eval - pred) ** 2))
        scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        recoveries.append(dgp.recovery({int(f) for f in tree.tree_.feature if f >= 0}))
        n_leaves.append(int(tree.get_n_leaves()))

        # Hold this structure fixed and re-estimate its leaves from fresh samples.
        eval_leaves = tree.apply(X_eval)
        inner = []
        for _ in range(n_leaf_samples):
            Xl, yl = dgp.sample(n, rng)
            means = _leaf_means(tree, Xl, yl)
            inner.append(
                _predict_with_means(tree, eval_leaves, means, float(np.mean(yl)))
            )
        inner = np.array(inner)
        per_structure_within.append(np.var(inner, axis=0))
        per_structure_means.append(np.mean(inner, axis=0))

    within_per_point = np.mean(np.array(per_structure_within), axis=0) / var_y
    between_per_point = np.var(np.array(per_structure_means), axis=0) / var_y
    total_per_point = within_per_point + between_per_point
    refit_per_point = np.var(np.array(refit_preds), axis=0) / var_y

    within = float(np.mean(within_per_point))
    between = float(np.mean(between_per_point))
    total = within + between

    return {
        "dgp": dgp.name,
        "n": n,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_structures": n_structures,
        "n_leaf_samples": n_leaf_samples,
        "var_y": var_y,
        "instability_total": total,
        "instability_total_mcse": _mc_se(total_per_point),
        "instability_leaf": within,
        "instability_leaf_mcse": _mc_se(within_per_point),
        "instability_structure": between,
        "instability_structure_mcse": _mc_se(between_per_point),
        "instability_refit": float(np.mean(refit_per_point)),
        "leaf_share": within / total if total > 0 else float("nan"),
        "r2_mean": float(np.mean(scores)),
        "r2_mcse": _mc_se(scores),
        "recovery_mean": float(np.nanmean(recoveries)) if recoveries else float("nan"),
        "n_leaves_mean": float(np.mean(n_leaves)),
    }


def main():
    """Sweep the regime grid and report the budget with Monte Carlo errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-structures", type=int, default=25)
    parser.add_argument("--n-leaf-samples", type=int, default=10)
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=[0.1, 1.0, 3.0, 10.0]
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 1000, 4000])
    parser.add_argument("--leaf-sizes", type=int, nargs="+", default=[5, 20, 100])
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--dgps", type=str, nargs="+", default=list(DGP_NAMES))
    parser.add_argument("--output", type=str, default="results/variance_budget")
    args = parser.parse_args()

    rows = []
    header = (
        f"{'dgp':16s} {'sigma':>6s} {'n':>6s} {'leaf':>5s} "
        f"{'total':>10s} {'leaf':>10s} {'struct':>10s} {'share':>7s} {'R2':>7s} {'recov':>6s}"
    )
    print(header)
    print("-" * len(header))

    for name in args.dgps:
        for sigma in args.sigmas:
            dgp = make_dgp(name, sigma=sigma)
            for n in args.sizes:
                for leaf_size in args.leaf_sizes:
                    row = budget(
                        dgp,
                        n,
                        args.max_depth,
                        leaf_size,
                        args.n_structures,
                        args.n_leaf_samples,
                    )
                    row["sigma"] = sigma
                    rows.append(row)
                    print(
                        f"{name:16s} {sigma:6.1f} {n:6d} {leaf_size:5d} "
                        f"{row['instability_total']:10.4g} {row['instability_leaf']:10.4g} "
                        f"{row['instability_structure']:10.4g} "
                        f"{row['leaf_share']:6.1%} {row['r2_mean']:7.3f} "
                        f"{row['recovery_mean']:6.2f}"
                    )
        print()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "budget.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out / 'budget.json'}  ({len(rows)} regimes)")

    shares = [r["leaf_share"] for r in rows if np.isfinite(r["leaf_share"])]
    if shares:
        print()
        print("H1b — is the leaf share regime-dependent?")
        print(f"  min {min(shares):.1%}   max {max(shares):.1%}")
        print(f"  settings with share > 40%: {sum(s > 0.4 for s in shares)}")
        print(f"  settings with share <  5%: {sum(s < 0.05 for s in shares)}")


if __name__ == "__main__":
    main()
