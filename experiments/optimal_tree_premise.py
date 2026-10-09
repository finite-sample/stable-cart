"""Paired bootstrap comparisons on fixed binary-classification datasets.

CART on raw and training-binarized features shares GOSDT's realized leaf budget.
This probes representation and algorithm differences; Gini and penalized
misclassification remain different objectives. Resampling intervals are
conditional on each fixed development/test split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split

from experiments.solver_comparison import fit_arms, summarize_predictions


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-thresholds", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=0.02)
    parser.add_argument("--time-limit", type=int, default=30)
    parser.add_argument("--output", default="results/optimal_tree_premise")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {"config": vars(args), "datasets": {}}
    for name, (X, y) in binary_datasets().items():
        Xtr, Xev, ytr, yev = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        rng = np.random.default_rng(1)
        predictions, fits = {}, []
        for index in range(args.n_bootstrap):
            indices = rng.integers(len(ytr), size=len(ytr))
            pred, _, metadata, certificate = fit_arms(
                Xtr[indices],
                ytr[indices],
                Xev,
                args.n_thresholds,
                args.max_depth,
                args.regularization,
                args.time_limit,
            )
            for arm, values in pred.items():
                predictions.setdefault(arm, []).append(values.tolist())
            fits.append({"draw": index, "solver": certificate, "complexity": metadata})
        row = {
            "fits": fits,
            "predictions": predictions,
            "evaluation_labels": yev.tolist(),
            "summary": summarize_predictions(predictions, yev, seed=1),
        }
        filename = f"{name}_predictions.npz"
        np.savez_compressed(
            output / filename,
            labels=row.pop("evaluation_labels"),
            **row.pop("predictions"),
        )
        row["predictions_file"] = filename
        results["datasets"][name] = row
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(name, row["summary"], flush=True)


if __name__ == "__main__":
    main()
