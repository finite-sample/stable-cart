"""End-to-end tests for stable_cart package."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from stable_cart import bootstrap_instability, bootstrap_predictions


@pytest.mark.e2e
@pytest.mark.slow
def test_regression_instability_audit_end_to_end():
    """Audit a complete regression fitting procedure."""
    X, y = make_regression(n_samples=1200, n_features=16, noise=12.0, random_state=7)
    X_train, X_eval, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=7)

    result = bootstrap_instability(
        lambda: DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=40,
            min_samples_leaf=20,
            random_state=7,
        ),
        X_train,
        y_train,
        X_eval,
        n_bootstrap=30,
        random_state=7,
    )

    assert set(result) == {
        "instability_mean",
        "instability_p90",
        "instability_max",
        "mape",
        "mape_standard_error",
        "pairwise_mean",
        "pairwise_standard_error",
        "n_fit_attempts",
        "n_resample_attempts",
        "n_rejected_resamples",
    }
    assert all(np.isfinite(value) and value >= 0 for value in result.values())


@pytest.mark.e2e
def test_multiclass_probability_audit_end_to_end():
    """Audit aligned probability vectors from a fitted pipeline."""
    X, y = make_classification(
        n_samples=800, n_features=10, n_informative=5, n_classes=3, random_state=42
    )
    X_train, X_eval, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    result = bootstrap_predictions(
        lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        X_train,
        y_train,
        X_eval,
        task="categorical",
        prediction_method="predict_proba",
        n_bootstrap=20,
        random_state=42,
    )

    assert result["bootstrap"].shape == (20, len(X_eval), 3)
    assert np.allclose(result["bootstrap"].sum(axis=2), 1.0)
    assert np.all(np.asarray(result["pairwise"]) >= 0)


@pytest.mark.e2e
def test_sklearn_ecosystem_integration():
    """Test that our models work with the sklearn ecosystem."""
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

    # Test with cross-validation
    model = DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=10, min_samples_split=20, random_state=42
    )
    scores = cross_val_score(model, X, y, cv=3, scoring="r2")
    assert len(scores) == 3
    assert all(isinstance(s, (int, float)) for s in scores)

    # Test in pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(max_depth=3, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    predictions = pipe.predict(X)
    assert predictions.shape == y.shape

    # Test with GridSearchCV
    param_grid = {"max_depth": [2, 3], "min_samples_leaf": [10, 20]}
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X, y)
    assert hasattr(grid, "best_params_")
    assert hasattr(grid, "best_estimator_")


@pytest.mark.e2e
def test_model_persistence():
    """Test that models can be pickled and unpickled."""
    import pickle

    X, y = make_regression(n_samples=100, n_features=5, random_state=42)

    model = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)

    pickled = pickle.dumps(model)
    unpickled = pickle.loads(pickled)  # noqa: S301

    assert np.allclose(model.predict(X), unpickled.predict(X))
