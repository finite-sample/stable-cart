"""Variance components for trees whose leaf values use an independent sample.

For a fitted partition S and an independent leaf sample L, total prediction
variance is E_S Var_L(f | S) + Var_S E_L(f | S). This describes an honest-leaf
estimator, not ordinary CART, which estimates splits and leaves on the same data.
Finite inner replication adds W/L to the variance of the estimated conditional
means; subtract it to estimate the structure component. Negative estimates are
retained. Standard errors resample independent outer structures, conditional on
the common evaluation sample. An empty leaf uses the fresh sample's global mean.
All variances are divided by its Var(y).
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

from experiments.dgps import DGP_NAMES, make_dgp  # noqa: E402


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


def prediction_statistics(means, within, refits, var_y):
    """Store sufficient statistics without cancellation from a target offset."""
    means = np.asarray(means)
    refits = np.asarray(refits)
    means = means - means.mean(axis=0)
    refits = refits - refits.mean(axis=0)
    return {
        "mean_gram": (means @ means.T / means.shape[1] / var_y).tolist(),
        "within": (np.mean(within, axis=1) / var_y).tolist(),
        "refit_gram": (refits @ refits.T / refits.shape[1] / var_y).tolist(),
    }


def summarize_components(statistics, n_leaf_samples, seed=0, resamples=2000):
    """Reconstruct integrated components and outer-block bootstrap errors."""
    gram = np.asarray(statistics["mean_gram"])
    refit = np.asarray(statistics["refit_gram"])
    within = np.asarray(statistics["within"])
    size = len(within)
    if size < 2 or n_leaf_samples < 2:
        raise ValueError("At least two structures and two leaf samples are required.")

    def estimate(weights):
        leaf = weights @ within

        def variance(matrix):
            return (
                (
                    weights @ np.diag(matrix)
                    - np.einsum("bi,ij,bj->b", weights, matrix, weights)
                )
                * size
                / (size - 1)
            )

        structure = variance(gram) - leaf / n_leaf_samples
        return np.column_stack((leaf, structure, leaf + structure, variance(refit)))

    point = estimate(np.full((1, size), 1 / size))[0]
    counts = np.random.default_rng(seed + 20000).multinomial(
        size, np.full(size, 1 / size), size=resamples
    )
    errors = estimate(counts / size).std(axis=0, ddof=1)
    result = {}
    for name, value, error in zip(
        ("leaf", "structure", "total", "refit"), point, errors, strict=True
    ):
        result[f"instability_{name}"] = float(value)
        result[f"instability_{name}_mcse"] = float(error)
    return result


def budget(dgp, n, max_depth, min_samples_leaf, n_structures, n_leaf_samples, seed=0):
    """Estimate honest-leaf variance components and ordinary CART variance."""
    if n_structures < 2 or n_leaf_samples < 2:
        raise ValueError("At least two structures and two leaf samples are required.")
    rng = np.random.default_rng(seed)

    X_eval, y_eval = dgp.sample(4000, np.random.default_rng(seed + 10_000))
    var_y = float(np.var(y_eval))
    if var_y <= 0 or not np.isfinite(var_y):
        raise ValueError("Evaluation target variance must be positive and finite.")
    ss_tot = float(np.sum((y_eval - np.mean(y_eval)) ** 2))

    per_structure_means = []
    per_structure_within = []
    refit_preds, scores, recoveries, n_leaves = [], [], [], []
    honest_scores = []

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
        honest_scores.append(
            float(np.mean(1 - ((inner - y_eval) ** 2).sum(axis=1) / ss_tot))
        )
        per_structure_within.append(np.var(inner, axis=0, ddof=1))
        per_structure_means.append(np.mean(inner, axis=0))

    means = np.asarray(per_structure_means)
    refits = np.asarray(refit_preds)
    statistics = prediction_statistics(means, per_structure_within, refits, var_y)
    components = summarize_components(statistics, n_leaf_samples, seed=seed)
    within = components["instability_leaf"]
    total = components["instability_total"]

    return {
        "dgp": dgp.name,
        "n": n,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_structures": n_structures,
        "n_leaf_samples": n_leaf_samples,
        "var_y": var_y,
        **components,
        "outer_structure_statistics": statistics,
        "leaf_share": within / total if total > 0 else float("nan"),
        "honest_r2_mean": float(np.mean(honest_scores)),
        "honest_r2_mcse": _mc_se(honest_scores),
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
    statistics = [row.pop("outer_structure_statistics") for row in rows]
    np.savez_compressed(
        out / "components.npz",
        **{
            key: np.asarray([item[key] for item in statistics]) for key in statistics[0]
        },
    )
    for index, row in enumerate(rows):
        row["statistics_file"] = "components.npz"
        row["statistics_index"] = index
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
    (out / "budget.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(f"Wrote {out / 'budget.json'}  ({len(rows)} regimes)")

    shares = [r["leaf_share"] for r in rows if r["leaf_share"] is not None]
    if shares:
        print()
        print("H1b — is the leaf share regime-dependent?")
        print(f"  min {min(shares):.1%}   max {max(shares):.1%}")
        print(f"  settings with share > 40%: {sum(s > 0.4 for s in shares)}")
        print(f"  settings with share <  5%: {sum(s < 0.05 for s in shares)}")


if __name__ == "__main__":
    main()
