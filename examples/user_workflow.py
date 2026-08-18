"""Run a complete installed-package workflow on built-in real datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import stable_cart
from stable_cart import (
    RepresentativeEstimator,
    bootstrap_instability,
    bootstrap_predictions,
    explanation_instability,
    linear_frontier,
    linear_instability,
    path_agreement,
    plot_mape_by_prediction,
    plot_prediction_instability,
    plot_stability_frontier,
    root_agreement,
    shrinkage_coefficients,
    stability_frontier,
)

matplotlib.use("Agg")


def _regression_workflow(output: Path, *, n_bootstrap: int, n_candidates: int) -> dict:
    X, y = load_diabetes(return_X_y=True)
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.2, random_state=41
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development, y_development, test_size=0.25, random_state=42
    )

    def ridge_factory():
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    raw = bootstrap_predictions(
        ridge_factory,
        X_train,
        y_train,
        X_validation,
        task="continuous",
        n_bootstrap=n_bootstrap,
        random_state=43,
    )
    audit = bootstrap_instability(
        ridge_factory,
        X_train,
        y_train,
        X_validation,
        task="continuous",
        n_bootstrap=n_bootstrap,
        random_state=43,
    )
    instability_ax = plot_prediction_instability(raw)
    instability_ax.figure.savefig(
        output / "regression_instability.png", dpi=120, bbox_inches="tight"
    )
    mape_ax = plot_mape_by_prediction(raw)
    mape_ax.figure.savefig(output / "regression_mape.png", dpi=120, bbox_inches="tight")

    frontier = stability_frontier(
        lambda **params: DecisionTreeRegressor(random_state=44, **params),
        {
            "max_depth": [2, 4, 6],
            "min_samples_leaf": [5, 15],
        },
        X_train,
        y_train,
        X_eval=X_validation,
        y_eval=y_validation,
        task="continuous",
        n_bootstrap=n_bootstrap,
        random_state=45,
    )
    frontier_ax = plot_stability_frontier({"CART": frontier}, annotate=False)
    frontier_ax.figure.savefig(
        output / "regression_frontier.png", dpi=120, bbox_inches="tight"
    )
    best_validation_score = max(point["score"] for point in frontier["points"])
    eligible = [
        point
        for point in frontier["points"]
        if point["score"] >= best_validation_score - 0.02
    ]
    selected = min(
        eligible,
        key=lambda point: (point["instability"], -point["score"]),
    )
    selected_tree = DecisionTreeRegressor(random_state=44, **selected["params"]).fit(
        X_development, y_development
    )

    representative = RepresentativeEstimator(
        estimator=make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        task="regression",
        n_candidates=n_candidates,
        random_state=46,
    ).fit(X_development, y_development)
    ridge = ridge_factory().fit(X_development, y_development)

    rng = np.random.default_rng(47)
    refitted_trees = []
    for seed in range(max(6, n_bootstrap // 2)):
        indices = rng.integers(0, len(X_development), len(X_development))
        refitted_trees.append(
            DecisionTreeRegressor(
                max_depth=selected["params"]["max_depth"],
                min_samples_leaf=selected["params"]["min_samples_leaf"],
                random_state=seed,
            ).fit(X_development[indices], y_development[indices])
        )
    structure = explanation_instability(refitted_trees, max_depth=3)
    structure["root_agreement"] = root_agreement(refitted_trees)
    structure["path_agreement"] = path_agreement(refitted_trees, X_test)

    design_development = np.column_stack((np.ones(len(X_development)), X_development))
    design_test = np.column_stack((np.ones(len(X_test)), X_test))
    analytic = linear_instability(
        design_development,
        design_test,
        y=y_development,
        robust=True,
    )
    analytic_frontier = linear_frontier(design_development, y_development, n_points=25)
    shrunk = shrinkage_coefficients(design_development, y_development, mu=1.0)

    return {
        "n_train": len(X_train),
        "n_validation": len(X_validation),
        "n_test": len(X_test),
        "audit": audit,
        "frontier": {
            "n_configurations": len(frontier["points"]),
            "n_frontier": len(frontier["frontier"]),
            "n_fits": frontier["n_fits"],
            "selected_params": selected["params"],
            "selected_validation_score": selected["score"],
            "selected_instability": selected["instability"],
            "test_score": float(r2_score(y_test, selected_tree.predict(X_test))),
        },
        "representative": {
            "selected_index": representative.selected_index_,
            "candidate_count": len(representative.candidates_),
            "test_score": representative.score(X_test, y_test),
            "ridge_test_score": float(r2_score(y_test, ridge.predict(X_test))),
        },
        "structure": structure,
        "fixed_design": {
            "robust_variance_mean": analytic["variance_mean"],
            "robust_s1_mean": analytic["s1_mean"],
            "risk_optimal_mu": analytic_frontier["risk_optimal"]["mu"],
            "risk_optimal_risk": analytic_frontier["risk_optimal"]["risk"],
            "shrunk_coefficient_norm": float(np.linalg.norm(shrunk)),
        },
    }


def _classification_workflow(*, n_bootstrap: int, n_candidates: int) -> dict:
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=48, stratify=y
    )

    def factory():
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2_000, random_state=49)
        )

    raw = bootstrap_predictions(
        factory,
        X_train,
        y_train,
        X_test,
        task="categorical",
        prediction_method="predict_proba",
        n_bootstrap=n_bootstrap,
        random_state=50,
    )
    representative = RepresentativeEstimator(
        estimator=factory(),
        task="classification",
        n_candidates=n_candidates,
        random_state=51,
    ).fit(X_train, y_train)
    probabilities = representative.predict_proba(X_test)
    predictions = representative.predict(X_test)

    return {
        "classes": raw["classes"].tolist(),
        "bootstrap_shape": list(raw["bootstrap"].shape),
        "probabilities_sum_to_one": bool(
            np.allclose(raw["bootstrap"].sum(axis=2), 1.0)
            and np.allclose(probabilities.sum(axis=1), 1.0)
        ),
        "pairwise_probability_instability": float(np.mean(raw["pairwise"])),
        "pairwise_mcse": raw["pairwise_standard_error"],
        "representative_accuracy": float(accuracy_score(y_test, predictions)),
        "representative_selected_index": representative.selected_index_,
    }


def run_workflow(
    output: Path, *, n_bootstrap: int = 24, n_candidates: int = 12
) -> dict:
    """Run the user workflow and return its JSON-serializable summary."""
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "package": {
            "version": stable_cart.__version__,
            "path": str(Path(stable_cart.__file__).resolve()),
        },
        "regression": _regression_workflow(
            output, n_bootstrap=n_bootstrap, n_candidates=n_candidates
        ),
        "multiclass": _classification_workflow(
            n_bootstrap=n_bootstrap, n_candidates=n_candidates
        ),
    }
    expected_plots = {
        "regression_instability.png",
        "regression_mape.png",
        "regression_frontier.png",
    }
    created = {path.name for path in output.glob("*.png") if path.stat().st_size > 0}
    if created != expected_plots:
        raise RuntimeError(f"expected plots {expected_plots}, created {created}")
    if not result["multiclass"]["probabilities_sum_to_one"]:
        raise RuntimeError("multiclass probability alignment failed")
    destination = output / "workflow_summary.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    """Parse command-line arguments and run the workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=24)
    parser.add_argument("--n-candidates", type=int, default=12)
    args = parser.parse_args()
    result = run_workflow(
        args.output,
        n_bootstrap=args.n_bootstrap,
        n_candidates=args.n_candidates,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
