"""H3: does the measured variance budget tell you which knob to turn?

`StableTree` has two levers, and they attack different halves of the variance:
`consensus_threshold` averages the split decision (the structure component) and
`leaf_shrinkage` pulls leaf values toward the parent (the leaf component). The
variance budget measured the leaf share at 6-7% when sigma=0.1 and 76-83% when
sigma=10, so the prediction is that which lever pays follows the noise level.

If it holds, the package can *measure* a user's data and say which knob to reach
for, instead of leaving them to sweep both. If it does not, the recommendation is
not shippable and the two knobs are simply two knobs.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "experiments")

from dgps import make_dgp  # noqa: E402
from variance_budget import budget  # noqa: E402

from stable_cart import StableTree, bootstrap_instability  # noqa: E402


def main():
    """Sweep noise, measure the leaf share, and see which knob helps more."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=[0.1, 1.0, 3.0, 10.0]
    )
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--n-bootstrap", type=int, default=15)
    parser.add_argument("--dgp", type=str, default="tree_separated")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--output", type=str, default="results/knob_study")
    args = parser.parse_args()

    header = (
        f"{'sigma':>6s} {'leaf share':>11s} | {'base':>9s} {'+consensus':>11s} "
        f"{'+shrinkage':>11s} | {'cons gain':>10s} {'shrink gain':>12s} {'predicted':>10s} {'actual':>8s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for sigma in args.sigmas:
        dgp = make_dgp(args.dgp, sigma=sigma)
        share = budget(dgp, args.n, 4, 20, n_structures=12, n_leaf_samples=6)[
            "leaf_share"
        ]

        rng = np.random.default_rng(0)
        X, y = dgp.sample(args.n, rng)
        X_eval, _ = dgp.sample(1000, np.random.default_rng(999))

        y_eval_true = dgp.sample(1000, np.random.default_rng(999))[1]

        def measure(**kw):
            """Instability *and* what it cost in accuracy -- neither alone means anything."""
            factory = lambda kw=kw: StableTree(  # noqa: E731
                task="regression",
                max_depth=4,
                min_samples_leaf=20,
                n_consensus=12,
                random_state=0,
                **kw,
            )
            inst = bootstrap_instability(
                factory,
                X,
                y,
                X_eval,
                task="continuous",
                n_bootstrap=args.n_bootstrap,
                random_state=1,
            )["instability_mean"]
            pred = factory().fit(X, y).predict(X_eval)
            ss_res = float(np.sum((y_eval_true - pred) ** 2))
            ss_tot = float(np.sum((y_eval_true - np.mean(y_eval_true)) ** 2))
            return inst, (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

        base, base_r2 = measure(consensus_threshold=0.0, leaf_shrinkage=0.0)
        cons, cons_r2 = measure(consensus_threshold=0.5, leaf_shrinkage=0.0)
        shrink, shrink_r2 = measure(consensus_threshold=0.0, leaf_shrinkage=10.0)

        # Report the two movements plainly. An exchange rate divides by the
        # accuracy given up, which is meaningless when nothing is given up -- and
        # here accuracy sometimes *rises*, so a ratio would explode rather than
        # inform.
        cons_gain = 100 * (base - cons) / base if base else 0.0
        shrink_gain = 100 * (base - shrink) / base if base else 0.0
        predicted = "shrinkage" if share > 0.5 else "consensus"
        actual = "shrinkage" if shrink_gain > cons_gain else "consensus"
        rows.append(
            {
                "sigma": sigma,
                "leaf_share": share,
                "base": base,
                "consensus_gain_pct": cons_gain,
                "shrinkage_gain_pct": shrink_gain,
                "base_r2": base_r2,
                "consensus_r2": cons_r2,
                "shrinkage_r2": shrink_r2,
                "predicted": predicted,
                "actual": actual,
            }
        )
        print(
            f"{sigma:6.1f} {share:10.1%} | {base:9.4g} {cons:11.4g} {shrink:11.4g} | "
            f"{cons_gain:9.1f}% {shrink_gain:11.1f}% {predicted:>10s} {actual:>8s}"
            + f"   r2 {base_r2:+.3f} -> {cons_r2:+.3f} / {shrink_r2:+.3f}"
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "knobs.json").write_text(json.dumps(rows, indent=2))

    hits = sum(r["predicted"] == r["actual"] for r in rows)
    print()
    print(
        f"H3: the measured leaf share picked the better knob in {hits}/{len(rows)} regimes"
    )
    print(
        f"    verdict: {'supported' if hits == len(rows) else 'NOT supported -- do not ship recommend_knob'}"
    )


if __name__ == "__main__":
    main()
