"""Change a signal coefficient and compare CART, GOSDT and a forest reference.

Delta weakens the second signal. It is not a literal impurity-gap intervention:
Bayes error and useful tree complexity can change too. A separated best root
split does not guarantee a globally optimal tree. Cutpoints use fitting data.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.solver_comparison import fit_arms, summarize_predictions


def margin_dgp(delta, sigma=1.0, n_features=6):
    def sample(n, rng):
        X = rng.normal(size=(n, n_features))
        logit = 2 * np.sign(X[:, 0]) + 2 * (1 - delta) * np.sign(X[:, 1])
        return X, (rng.random(n) < 1 / (1 + np.exp(-logit / sigma))).astype(int)

    return sample


def run_delta(delta, n, n_draws, n_thresholds, depth, reg, time_limit, seed):
    sample = margin_dgp(delta)
    rng = np.random.default_rng(seed)
    X_eval, y_eval = sample(1500, np.random.default_rng(seed + 777))
    predictions = {}
    fits = []
    for index in range(n_draws):
        X, y = sample(n, rng)
        pred, _, metadata, certificate = fit_arms(
            X, y, X_eval, n_thresholds, depth, reg, time_limit, forest=True
        )
        for name, values in pred.items():
            predictions.setdefault(name, []).append(values.tolist())
        fits.append({"draw": index, "solver": certificate, "complexity": metadata})
    return {
        "delta": delta,
        "n_draws": n_draws,
        "fits": fits,
        "predictions": predictions,
        "evaluation_labels": y_eval.tolist(),
        "summary": summarize_predictions(predictions, y_eval, seed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deltas", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1]
    )
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--n-draws", type=int, default=12)
    parser.add_argument("--n-thresholds", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=0.02)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="results/margin_study")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
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
        filename = f"predictions_{len(rows):02d}.npz"
        np.savez_compressed(
            out / filename,
            labels=row.pop("evaluation_labels"),
            **row.pop("predictions"),
        )
        row["predictions_file"] = filename
        rows.append(row)
        (out / "margin.json").write_text(
            json.dumps({"config": vars(args), "rows": rows}, indent=2)
        )
        print(delta, row["summary"], flush=True)


if __name__ == "__main__":
    main()
