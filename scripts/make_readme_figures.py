"""Generate the three prediction-stability figures used in the README and docs.

Every number and picture in the README comes from a committed script; this is
that script for the figures. Run: ``uv run python scripts/make_readme_figures.py``
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from stable_cart import (
    StableTree,
    bootstrap_predictions,
    plot_mape_by_prediction,
    plot_prediction_instability,
    plot_stability_frontier,
    stability_frontier,
)

warnings.filterwarnings("ignore")


def main():
    """Draw the three figures on the diabetes data and save them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/figures")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--frontier-bootstrap", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # California housing: 20,640 rows. Large enough that a tree has hundreds of
    # distinct predicted values, which is what makes the instability cloud and
    # the MAPE curve readable rather than a handful of vertical stripes.
    X, y = fetch_california_housing(return_X_y=True)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(X), size=args.n_train * 2, replace=False)
    X_train, X_test, y_train, _y_test = train_test_split(
        X[keep], y[keep], test_size=0.5, random_state=0
    )

    raw = bootstrap_predictions(
        lambda: DecisionTreeRegressor(
            max_depth=args.depth, min_samples_leaf=10, random_state=0
        ),
        X_train,
        y_train,
        X_test,
        n_bootstrap=args.n_bootstrap,
        random_state=0,
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_prediction_instability(raw, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "instability.png", dpi=140)
    print(f"Created: {out / 'instability.png'}")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plot_mape_by_prediction(raw, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "mape_by_prediction.png", dpi=140)
    print(f"Created: {out / 'mape_by_prediction.png'}")

    # The frontier ranks configurations rather than reporting one, so it needs
    # far fewer resamples than the instability plot -- and it pays for them
    # once per configuration rather than once.
    common = {
        "X": X_train,
        "y": y_train,
        "n_bootstrap": args.frontier_bootstrap,
        "random_state": 0,
    }
    # Both families get the same complexity knob as well as their own stability
    # knob. Fixing CART's depth and sweeping only StableTree's would make the
    # baseline look worse than pruning actually is.
    depths = [3, 4, 5, 6, 8]
    frontiers = {
        "CART (pruned)": stability_frontier(
            lambda **kw: DecisionTreeRegressor(
                min_samples_leaf=10, random_state=0, **kw
            ),
            {
                "max_depth": depths,
                "ccp_alpha": [0.0, 0.001, 0.005, 0.02, 0.05, 0.1],
            },
            **common,
        ),
        "StableTree": stability_frontier(
            lambda **kw: StableTree(
                task="regression",
                min_samples_leaf=10,
                n_consensus=12,
                random_state=0,
                **kw,
            ),
            {
                "max_depth": depths,
                "consensus_threshold": [0.0, 0.3, 0.6],
                "leaf_shrinkage": [0.0, 2.0, 10.0],
            },
            **common,
        ),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plot_stability_frontier(frontiers, ax=ax, annotate=False)
    fig.tight_layout()
    fig.savefig(out / "frontier.png", dpi=140)
    print(f"Created: {out / 'frontier.png'}")

    for name, result in frontiers.items():
        best = max(result["frontier"], key=lambda p: p["accuracy"])
        print(
            f"  {name:16s} most accurate frontier point: "
            f"R2={best['accuracy']:.3f} instability={best['instability']:.1f} "
            f"mape={best['mape']:.2f} {best['params']}"
        )


if __name__ == "__main__":
    main()
