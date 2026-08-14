"""H2: how much of structure churn is the greedy heuristic's fault?

Greedy split selection and exact optimisation can only disagree when two
candidate splits are close. If the best and second-best root splits are well
separated, greedy finds the optimum by construction and there is nothing for an
exact solver to recover; as they converge, the greedy choice becomes a coin flip.
So the greedy-vs-exact gap should be **governed by the split margin**, and the
pre-specified claim is that it shrinks to nothing as the margin grows while
staying below what averaging buys at every margin.

The margin δ is a knob here: at δ=0 features 0 and 1 are equally predictive and
the root is a toss-up; at δ=1 only feature 0 carries signal.

Arms, all on the same binarized representation so the solver is the only thing
that differs:

    greedy      ordinary CART on the binarized features
    exact       GOSDT, the optimal sparse tree
    bagging     random forest — not a single tree, included to bound the gap

**Complexity is matched per draw.** GOSDT's leaf penalty makes it far sparser than
a depth-limited CART — measured at 1 split against greedy's 7 — and a sparser tree
is trivially more structurally stable. Comparing them unmatched measures sparsity,
not search quality. So each draw fits GOSDT first, counts its leaves, and gives
the greedy arm the same `max_leaf_nodes` budget.

Requires the `gosdt` package and scikit-learn < 1.7 (gosdt calls the removed
`force_all_finite`). Run it from the dedicated environment, not the project venv.
"""

import argparse
import json
import warnings
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def margin_dgp(delta, sigma=1.0, n_features=6):
    """Binary-classification DGP whose two best root splits differ by ``delta``."""

    def sample(n, rng):
        X = rng.normal(size=(n, n_features))
        logit = 2.0 * np.sign(X[:, 0]) + 2.0 * (1.0 - delta) * np.sign(X[:, 1])
        p = 1.0 / (1.0 + np.exp(-logit / sigma))
        return X, (rng.random(n) < p).astype(int)

    return sample


def binarizer_edges(X, n_thresholds):
    """Per-feature quantile cut points shared by every arm."""
    qs = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    return [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]


def binarize(X, edges):
    """Expand features into indicator columns, and record each column's source."""
    cols, origin = [], []
    for j, cuts in enumerate(edges):
        for c in cuts:
            cols.append((X[:, j] <= c).astype(int))
            origin.append(j)
    return np.column_stack(cols), origin


def gosdt_features(model, origin):
    """Original-feature multiset tested by a fitted GOSDT tree."""
    counts: Counter = Counter()

    def walk(node):
        feature = getattr(node, "feature", None)
        if feature is None:
            return
        counts[origin[int(feature)]] += 1
        walk(getattr(node, "left_child", None))
        walk(getattr(node, "right_child", None))

    walk(model.trees_[0].tree)
    return counts


def sklearn_features(tree, origin):
    """Original-feature multiset tested by a fitted sklearn tree."""
    inner = tree.tree_
    return Counter(origin[int(f)] for f in inner.feature if f >= 0)


def jaccard(a, b):
    """Multiset Jaccard distance between two feature counters."""
    if not a and not b:
        return 0.0
    union = sum((a | b).values())
    return 1.0 - sum((a & b).values()) / union if union else 0.0


def run_delta(delta, n, n_draws, n_thresholds, depth, reg, time_limit, seed):
    """Fit every arm across independent draws at one margin and score them."""
    from gosdt import GOSDTClassifier

    sample = margin_dgp(delta)
    rng = np.random.default_rng(seed)

    X_eval, y_eval = sample(1500, np.random.default_rng(seed + 777))
    edges = binarizer_edges(X_eval, n_thresholds)
    Xb_eval, origin = binarize(X_eval, edges)

    preds = {"greedy": [], "exact": [], "bagging": []}
    feats = {"greedy": [], "exact": []}
    accs = {k: [] for k in preds}
    sizes = {"greedy": [], "exact": []}

    for _ in range(n_draws):
        X, y = sample(n, rng)
        Xb, _ = binarize(X, edges)
        if len(np.unique(y)) < 2:
            continue

        # Fit the exact solver first: its sparsity sets the budget for the greedy
        # arm, so the two differ in search rather than in size.
        exact = GOSDTClassifier(
            regularization=reg, depth_budget=depth, time_limit=time_limit, verbose=False
        ).fit(Xb, y)
        exact_feats = gosdt_features(exact, origin)
        preds["exact"].append(np.asarray(exact.predict(Xb_eval)))
        feats["exact"].append(exact_feats)
        n_splits = max(1, sum(exact_feats.values()))
        sizes["exact"].append(n_splits)

        greedy = DecisionTreeClassifier(
            max_depth=depth, max_leaf_nodes=n_splits + 1, random_state=0
        ).fit(Xb, y)
        greedy_feats = sklearn_features(greedy, origin)
        preds["greedy"].append(greedy.predict(Xb_eval))
        feats["greedy"].append(greedy_feats)
        sizes["greedy"].append(sum(greedy_feats.values()))

        forest = RandomForestClassifier(
            n_estimators=50,
            max_depth=depth,
            max_leaf_nodes=n_splits + 1,
            random_state=0,
        )
        preds["bagging"].append(forest.fit(Xb, y).predict(Xb_eval))

        for arm in ("greedy", "exact", "bagging"):
            accs[arm].append(float(np.mean(preds[arm][-1] == y_eval)))

    out = {
        "delta": delta,
        "n_draws": len(preds["greedy"]),
        "splits_exact": float(np.mean(sizes["exact"])),
        "splits_greedy": float(np.mean(sizes["greedy"])),
    }
    for arm in preds:
        arr = np.array(preds[arm])
        modal = (arr.mean(axis=0) >= 0.5).astype(int)
        out[f"{arm}_instability"] = float(np.mean(arr != modal))
        out[f"{arm}_accuracy"] = float(np.mean(accs[arm]))
    for arm in feats:
        pairs = [
            jaccard(feats[arm][i], feats[arm][j])
            for i in range(len(feats[arm]))
            for j in range(i + 1, len(feats[arm]))
        ]
        out[f"{arm}_structural"] = float(np.mean(pairs)) if pairs else float("nan")

    base = out["greedy_instability"]
    out["exact_gain_pct"] = (
        100 * (base - out["exact_instability"]) / base if base else 0.0
    )
    out["bagging_gain_pct"] = (
        100 * (base - out["bagging_instability"]) / base if base else 0.0
    )
    return out


def main():
    """Sweep the margin and report the greedy-vs-exact gap against bagging's."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deltas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--n-draws", type=int, default=12)
    parser.add_argument("--n-thresholds", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=0.02)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=str, default="results/margin_study")
    args = parser.parse_args()

    header = (
        f"{'delta':>6s} {'greedy':>9s} {'exact':>9s} {'bagging':>9s} "
        f"{'exact gain':>11s} {'bag gain':>9s} {'greedy str':>11s} {'exact str':>10s} "
        f"{'splits g/e':>11s} {'acc g/e':>12s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for delta in args.deltas:
        row = run_delta(
            delta,
            args.n,
            args.n_draws,
            args.n_thresholds,
            args.depth,
            args.regularization,
            args.time_limit,
            args.seed,
        )
        rows.append(row)
        print(
            f"{delta:6.2f} {row['greedy_instability']:9.4f} {row['exact_instability']:9.4f} "
            f"{row['bagging_instability']:9.4f} {row['exact_gain_pct']:10.1f}% "
            f"{row['bagging_gain_pct']:8.1f}% {row['greedy_structural']:11.3f} "
            f"{row['exact_structural']:10.3f} "
            f"{row['splits_greedy']:5.1f}/{row['splits_exact']:<5.1f} "
            f"{row['greedy_accuracy']:.3f}/{row['exact_accuracy']:.3f}"
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "margin.json").write_text(json.dumps(rows, indent=2))

    print()
    print("H2: gap decreasing in delta, and below bagging's reduction at every delta")
    gains = [r["exact_gain_pct"] for r in rows]
    bags = [r["bagging_gain_pct"] for r in rows]
    decreasing = all(a >= b - 1e-9 for a, b in pairwise(gains))
    below = all(g < b for g, b in zip(gains, bags, strict=True))
    print(
        f"  monotone decreasing in delta: {decreasing}   ({[round(g, 1) for g in gains]})"
    )
    print(f"  below bagging at every delta: {below}   ({[round(b, 1) for b in bags]})")


if __name__ == "__main__":
    main()
