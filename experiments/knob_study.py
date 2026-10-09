"""Compare support-based stopping with parent-mean leaf shrinkage.

Both arms already aggregate twelve bootstrap split votes. Raising the support
threshold can stop growth; it does not switch bootstrap averaging on. The honest
variance budget is a candidate diagnostic for this comparison, not an identified
mediator or a validated rule for choosing a setting. Results are exploratory and
conditional on one training dataset per regime.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from experiments.bootstrap_split_tree import BootstrapSplitTree  # noqa: E402
from experiments.dgps import make_dgp  # noqa: E402
from experiments.variance_budget import budget  # noqa: E402
from stable_cart import bootstrap_instability  # noqa: E402


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

        def measure(_X=X, _y=y, _Xe=X_eval, _ye=y_eval_true, **kw):
            """Instability *and* what it cost in accuracy -- neither alone means anything."""
            factory = lambda kw=kw: BootstrapSplitTree(  # noqa: E731
                task="regression",
                max_depth=4,
                min_samples_leaf=20,
                n_consensus=12,
                random_state=0,
                **kw,
            )
            inst = bootstrap_instability(
                factory,
                _X,
                _y,
                _Xe,
                task="continuous",
                n_bootstrap=args.n_bootstrap,
                random_state=1,
            )["instability_mean"]
            pred = factory().fit(_X, _y).predict(_Xe)
            ss_res = float(np.sum((_ye - pred) ** 2))
            ss_tot = float(np.sum((_ye - np.mean(_ye)) ** 2))
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
