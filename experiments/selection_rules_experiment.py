"""Does selecting the ensemble-centroid tree actually buy stability?

The shipped comparison (`centroid_experiment.py`) pits `CentroidTree` against a
single fixed-seed CART, so any benefit of *selecting from N candidates at all* is
credited to centroid-proximity. This driver removes that confound: one candidate
pool is fit per replicate and every selection rule is applied to that same pool,
so the comparison is paired by construction.

Rules, from do-nothing to upper bound:

- ``random``    take an arbitrary candidate; the baseline that must be beaten
- ``centroid``  closest to the ensemble mean prediction (what CentroidTree does)
- ``medoid``    minimum mean distance to the other candidates (Banerjee et al.,
                *Identifying representative trees from ensembles*, Stat Med 2012)
- ``best_val``  best validation score; controls for "selection per se"
- ``ensemble``  the bagged mean; not interpretable, bounds achievable stability

Instability is measured the only way that means anything: a fixed test set, models
refit on bootstrap resamples of the training data, and the spread of predictions
*at each fixed test point*.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

RULES = ["random", "centroid", "medoid", "best_val", "ensemble"]


def fit_candidate_pool(
    X_tr, y_tr, task, n_candidates, rng, max_depth, diversity="bagging"
):
    """
    Fit one pool of candidate trees plus the validation split they are scored on.

    ``diversity='seed'`` reproduces what ``CentroidTree`` does today: identical
    training data for every candidate, varying only ``random_state``. Because an
    sklearn tree is deterministic given (data, params) — ``random_state`` only
    breaks ties between exactly-equal splits — that pool collapses to N copies of
    one tree whenever the splits are unambiguous. Measured pool spread under
    'seed' is ~1e-14, i.e. floating-point noise.

    ``diversity='bagging'`` is the fix, and matches how the representative-tree
    literature generates its ensemble: bootstrap resample per candidate plus
    feature subsampling, so the pool actually spans the model-multiplicity set the
    centroid is meant to summarise.

    Parameters
    ----------
    X_tr, y_tr
        Training data for this replicate.
    task
        'regression' or 'classification'.
    n_candidates
        Pool size.
    rng
        Random generator supplying candidate seeds and resamples.
    max_depth
        Depth for every candidate.
    diversity
        'bagging' or 'seed'.

    Returns
    -------
    tuple
        (candidates, X_val, y_val) — the fitted pool and its validation split.
    """
    cls = DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
    stratify = y_tr if task == "classification" else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr,
        y_tr,
        test_size=0.2,
        random_state=int(rng.integers(1 << 30)),
        stratify=stratify,
    )

    candidates = []
    n_fit = len(X_fit)
    for _ in range(n_candidates):
        tree = cls(
            max_depth=max_depth,
            min_samples_leaf=20,
            max_features="sqrt" if diversity == "bagging" else None,
            random_state=int(rng.integers(1 << 30)),
        )
        if diversity in ("bagging", "bootstrap"):
            idx = rng.integers(0, n_fit, n_fit)
            if task == "classification" and len(np.unique(y_fit[idx])) < 2:
                idx = np.arange(n_fit)
            tree.fit(X_fit[idx], y_fit[idx])
        else:
            tree.fit(X_fit, y_fit)
        candidates.append(tree)
    return candidates, X_val, y_val


def pool_spread(candidates, X):
    """
    Mean across-pool standard deviation of predictions — the pool's diversity.

    A value near machine epsilon means every candidate is the same tree and any
    selection rule applied to the pool is a no-op.

    Parameters
    ----------
    candidates
        Fitted candidate trees.
    X
        Points to predict on.

    Returns
    -------
    float
        Mean per-point standard deviation across the pool.
    """
    preds = np.array([c.predict(X) for c in candidates], dtype=float)
    return float(preds.std(axis=0).mean())


def select(rule, candidates, X_val, y_val, task, rng):
    """
    Apply one selection rule to a fitted candidate pool.

    Parameters
    ----------
    rule
        One of RULES.
    candidates
        Fitted candidate trees.
    X_val, y_val
        Validation split used for scoring proximity.
    task
        'regression' or 'classification'.
    rng
        Random generator, used only by the ``random`` rule.

    Returns
    -------
    int | None
        Index of the selected candidate, or None for the ``ensemble`` rule.
    """
    if rule == "ensemble":
        return None
    if rule == "random":
        return int(rng.integers(len(candidates)))
    if rule == "best_val":
        return int(np.argmax([c.score(X_val, y_val) for c in candidates]))

    preds = np.array([c.predict(X_val) for c in candidates], dtype=float)
    if rule == "centroid":
        # Distance to the ensemble mean: what CentroidTree scores.
        target = preds.mean(axis=0)
        return int(np.argmin(np.sqrt(((preds - target) ** 2).mean(axis=1))))
    if rule == "medoid":
        # Mean distance to every other candidate: Banerjee's formulation.
        diff = preds[:, None, :] - preds[None, :, :]
        return int(np.argmin(np.sqrt((diff**2).mean(axis=2)).mean(axis=1)))
    raise ValueError(f"unknown rule: {rule}")


def run_dataset(
    X, y, task, n_replicates, n_candidates, max_depth, seed, diversity="bagging"
):
    """
    Measure each rule's instability and accuracy on one dataset.

    Parameters
    ----------
    X, y
        Full dataset.
    task
        'regression' or 'classification'.
    n_replicates
        Bootstrap replicates of the training set.
    n_candidates
        Candidate pool size.
    max_depth
        Depth for every candidate.
    seed
        Base seed.

    Returns
    -------
    dict
        Per-rule instability and accuracy, plus a pool-identity hash.
    """
    stratify = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=stratify
    )
    n_train = len(X_tr)

    preds = {r: [] for r in RULES}
    scores = {r: [] for r in RULES}
    spreads = []

    for b in range(n_replicates):
        rng = np.random.default_rng(seed * 100_000 + b)
        idx = rng.integers(0, n_train, n_train)
        Xb, yb = X_tr[idx], y_tr[idx]
        if task == "classification" and len(np.unique(yb)) < 2:
            continue

        # One pool, reused by every rule -> the comparison is paired.
        pool_rng = np.random.default_rng(seed * 100_000 + b)
        candidates, X_val, y_val = fit_candidate_pool(
            Xb, yb, task, n_candidates, pool_rng, max_depth, diversity
        )
        spreads.append(pool_spread(candidates, X_te))

        for rule in RULES:
            pick = select(
                rule, candidates, X_val, y_val, task, np.random.default_rng(b)
            )
            if pick is None:
                p = np.mean([c.predict(X_te) for c in candidates], axis=0)
                if task == "classification":
                    p = (p >= 0.5).astype(float)
                s = _score(p, y_te, task)
            else:
                p = candidates[pick].predict(X_te).astype(float)
                s = _score(p, y_te, task)
            preds[rule].append(p)
            scores[rule].append(s)

    out = {}
    for rule in RULES:
        arr = np.array(preds[rule])
        if task == "regression":
            instability = float(np.mean(np.var(arr, axis=0)))
        else:
            mode = (arr.mean(axis=0) >= 0.5).astype(float)
            instability = float(np.mean(arr != mode))
        out[rule] = {"instability": instability, "score": float(np.mean(scores[rule]))}
    out["_pool_spread"] = float(np.mean(spreads))
    return out


def _score(pred, y_true, task):
    """
    Score predictions: R² for regression, accuracy for classification.

    Parameters
    ----------
    pred
        Predicted values.
    y_true
        True values.
    task
        'regression' or 'classification'.

    Returns
    -------
    float
        The score.
    """
    if task == "regression":
        ss_res = float(np.sum((y_true - pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(np.mean(pred == y_true))


def main():
    """Run the selection-rule comparison across datasets and report per dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=100)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/selection_rules")
    parser.add_argument(
        "--diversity",
        choices=["bagging", "bootstrap", "seed"],
        default="bootstrap",
        help=(
            "'seed' reproduces CentroidTree's degenerate pool; 'bootstrap' resamples "
            "rows per candidate (works with any base class); 'bagging' adds feature "
            "subsampling (sklearn trees only)"
        ),
    )
    args = parser.parse_args()

    from centroid_experiment import get_datasets

    datasets = get_datasets("all")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name, (X, y, task) in datasets.items():
        print(f"\n{name} ({task}, n={len(X)})")
        res = run_dataset(
            X,
            y,
            task,
            args.n_replicates,
            args.n_candidates,
            args.max_depth,
            args.seed,
            args.diversity,
        )
        all_results[name] = res
        base = res["random"]["instability"]
        spread = res["_pool_spread"]
        flag = "  <-- DEGENERATE POOL, selection cannot matter" if spread < 1e-8 else ""
        print(f"  pool spread: {spread:.4g}{flag}")
        print(f"  {'rule':10s} {'instability':>13s} {'vs random':>11s} {'score':>8s}")
        for rule in RULES:
            inst, sc = res[rule]["instability"], res[rule]["score"]
            rel = 100 * (base - inst) / base if base > 0 else 0.0
            print(f"  {rule:10s} {inst:13.5g} {rel:10.1f}% {sc:8.3f}")

    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
