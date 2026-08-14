"""Phase 0 gate: is greedy split choice really the source of tree instability?

The root-cause claim from AUDIT.md is that ~89-98% of a tree's prediction
instability comes from *which splits get chosen* on resampled data, and that this
is a consequence of greedy, myopic split selection with arbitrary tie-breaking.

That claim is falsifiable. A globally-optimal tree solver (GOSDT) does not choose
splits greedily — it optimises the whole tree against a regularised objective — so
if the claim holds, its trees should be markedly more stable under resampling than
a greedy CART at comparable accuracy. If they are not, the root cause is wrong.

Three arms, because GOSDT requires binarized features and binarization is itself a
threshold-snapping step that could carry the effect on its own:

    greedy CART            raw features, ordinary sklearn tree
    greedy CART binarized  same tree, same binarizer as GOSDT   <- the control
    GOSDT                  optimal tree on the binarized features

Binary classification only: that is what GOSDT supports.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_classification,
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

from stable_cart import bootstrap_instability  # noqa: E402


def binary_datasets():
    """Binary-classification datasets small enough for an exact solver."""
    data = {}

    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, n_redundant=2, random_state=42
    )
    data["synth_easy"] = (X, y)

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=3,
        n_redundant=5,
        class_sep=0.5,
        flip_y=0.1,
        random_state=42,
    )
    data["synth_hard"] = (X, y)

    bc = load_breast_cancer()
    data["breast_cancer"] = (bc.data, bc.target)

    wine = load_wine()
    mask = wine.target < 2
    data["wine_binary"] = (wine.data[mask], wine.target[mask])

    digits = load_digits()
    mask = digits.target < 2
    data["digits_binary"] = (digits.data[mask], digits.target[mask])

    return data


class BinarizedCART:
    """Greedy CART fitted on GOSDT's binarized features — the control arm.

    Isolates the solver from the representation: any stability that comes from
    binarizing rather than from optimal search shows up here.
    """

    def __init__(self, max_depth=4, n_thresholds=8, random_state=0):
        self.max_depth = max_depth
        self.n_thresholds = n_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        """Binarize, then fit an ordinary greedy tree."""
        self.edges_ = _binarizer_edges(X, self.n_thresholds)
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth, random_state=self.random_state
        ).fit(_binarize(X, self.edges_), y)
        return self

    def predict(self, X):
        """Predict on the same binarized representation."""
        return self.tree_.predict(_binarize(X, self.edges_))


def _binarizer_edges(X, n_thresholds):
    """Per-feature quantile cut points, the representation both arms share."""
    qs = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    return [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]


def _binarize(X, edges):
    """Expand each feature into indicator columns at the given cut points."""
    cols = []
    for j, cuts in enumerate(edges):
        for c in cuts:
            cols.append((X[:, j] <= c).astype(int))
    return np.column_stack(cols) if cols else np.zeros((len(X), 1), dtype=int)


class GosdtTree:
    """Optimal sparse decision tree over the shared binarized representation."""

    def __init__(
        self, depth_budget=4, regularization=0.02, n_thresholds=8, time_limit=30
    ):
        self.depth_budget = depth_budget
        self.regularization = regularization
        self.n_thresholds = n_thresholds
        self.time_limit = time_limit

    def fit(self, X, y):
        """Binarize with the same grid as the control, then solve exactly."""
        from gosdt import GOSDTClassifier

        self.edges_ = _binarizer_edges(X, self.n_thresholds)
        self.model_ = GOSDTClassifier(
            regularization=self.regularization,
            depth_budget=self.depth_budget,
            time_limit=self.time_limit,
            verbose=False,
        )
        self.model_.fit(_binarize(X, self.edges_), y)
        return self

    def predict(self, X):
        """Predict with the solved tree."""
        return np.asarray(self.model_.predict(_binarize(X, self.edges_)))


def main():
    """Run the three arms on every dataset and print the gate result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-thresholds", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=0.02)
    parser.add_argument("--time-limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/optimal_tree_premise")
    args = parser.parse_args()

    results = {}
    print(
        f"{'dataset':16s} {'model':24s} {'instability':>12s} {'vs greedy':>10s} {'accuracy':>9s}"
    )
    print("-" * 78)

    for name, (X, y) in binary_datasets().items():
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        # Several of these datasets arrive sorted by class; shuffle before splitting.
        order = np.random.default_rng(0).permutation(len(X))
        X, y = X[order], y[order]
        n = int(0.7 * len(X))
        Xtr, ytr, Xev, yev = X[:n], y[:n], X[n:], y[n:]
        if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
            print(f"{name:16s} skipped (a split lost a class)")
            continue

        arms = {
            "greedy CART (raw)": lambda: DecisionTreeClassifier(
                max_depth=args.max_depth, random_state=0
            ),
            "greedy CART (binarized)": lambda: BinarizedCART(
                max_depth=args.max_depth, n_thresholds=args.n_thresholds
            ),
            "GOSDT (optimal)": lambda: GosdtTree(
                depth_budget=args.max_depth,
                regularization=args.regularization,
                n_thresholds=args.n_thresholds,
                time_limit=args.time_limit,
            ),
        }

        row = {}
        baseline = None
        for label, factory in arms.items():
            try:
                inst = bootstrap_instability(
                    factory,
                    Xtr,
                    ytr,
                    Xev,
                    task="categorical",
                    n_bootstrap=args.n_bootstrap,
                    random_state=1,
                )["instability_mean"]
                acc = float(np.mean(factory().fit(Xtr, ytr).predict(Xev) == yev))
            except Exception as exc:
                print(f"{name:16s} {label:24s} FAILED: {type(exc).__name__}: {exc}")
                continue
            if baseline is None:
                baseline = inst
            rel = 100 * (baseline - inst) / baseline if baseline > 0 else 0.0
            row[label] = {"instability": inst, "accuracy": acc}
            print(f"{name:16s} {label:24s} {inst:12.5g} {rel:9.1f}% {acc:9.3f}")
        results[name] = row
        print()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 78)
    print("GATE: is the optimal solver more stable than greedy on the same features?")
    print("=" * 78)
    wins = 0
    total = 0
    for name, row in results.items():
        if "GOSDT (optimal)" in row and "greedy CART (binarized)" in row:
            g = row["GOSDT (optimal)"]
            c = row["greedy CART (binarized)"]
            better = g["instability"] < c["instability"]
            wins += better
            total += 1
            print(
                f"  {name:16s} GOSDT {g['instability']:.5g} (acc {g['accuracy']:.3f}) vs "
                f"binarized greedy {c['instability']:.5g} (acc {c['accuracy']:.3f})"
                f"  -> {'GOSDT more stable' if better else 'no'}"
            )
    print(f"\n  GOSDT more stable on {wins}/{total} datasets")


if __name__ == "__main__":
    main()
