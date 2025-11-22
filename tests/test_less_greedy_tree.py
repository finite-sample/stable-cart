"""Unit tests for LessGreedyHybridTree and related utilities."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stable_cart.less_greedy_tree import (
    LessGreedyHybridTree,
    _ComparableFloat,
    _sse,
)

# Tolerance for floating-point comparisons
TOL = 1e-6


# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture
def small_regression_data():
    """Small regression dataset for basic tests."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    return X, y


@pytest.fixture
def regression_data():
    """Larger regression dataset for testing."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    return X, y


# -------------------------------
# Test utility functions
# -------------------------------


def test_sse():
    """Test SSE calculation utility."""
    y = np.array([1.0, 2.0, 3.0])
    assert _sse(y) == pytest.approx(2.0, abs=TOL)  # var=2/3, n*var=2

    y = np.array([5.0])
    assert _sse(y) == pytest.approx(0.0, abs=TOL)

    y = np.array([])
    assert _sse(y) == pytest.approx(0.0, abs=TOL)


def test_comparable_float():
    """Test ComparableFloat for pytest.approx compatibility."""
    cf = _ComparableFloat(1.0)
    assert cf == pytest.approx(1.0)
    assert cf > pytest.approx(0.999)
    assert cf < pytest.approx(1.001)
    assert cf >= pytest.approx(1.0)
    assert cf <= pytest.approx(1.0)


# -------------------------------
# Test LessGreedyHybridTree
# -------------------------------


def test_less_greedy_hybrid_basic_fit_predict(small_regression_data):
    """Test basic fitting and prediction."""
    X, y = small_regression_data
    model = LessGreedyHybridTree(
        task="regression",
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        split_frac=0.5,
        val_frac=0.25,
        est_frac=0.25,
        random_state=42,
    )
    model.fit(X, y)

    # Check tree was built
    assert model.tree_["type"] in ["split", "split_oblique", "leaf"]

    # Check predictions work
    preds = model.predict(X)
    assert preds.shape == y.shape

    # Check fit time was recorded
    assert model.fit_time_sec_ >= 0

    # Check splits were scanned
    assert model.splits_scanned_ >= 0


def test_less_greedy_hybrid_sklearn_compatibility(regression_data):
    """Test sklearn API compatibility."""
    X, y = regression_data

    # Test cloning
    model = LessGreedyHybridTree(task="regression", max_depth=3, random_state=42)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()

    # Test cross-validation
    scores = cross_val_score(model, X, y, cv=3, scoring="r2")
    assert len(scores) == 3
    assert all(isinstance(s, (int, float)) for s in scores)

    # Test in pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LessGreedyHybridTree(task="regression", max_depth=3, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    predictions = pipe.predict(X)
    assert predictions.shape == y.shape

    # Test GridSearchCV
    param_grid = {"max_depth": [2, 3], "leaf_smoothing": [0.0, 0.1]}
    grid = GridSearchCV(model, param_grid, cv=3, scoring="r2")
    grid.fit(X, y)
    assert hasattr(grid, "best_params_")
    assert hasattr(grid, "best_score_")


def test_less_greedy_hybrid_score_method(regression_data):
    """Test the score method returns R²."""
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LessGreedyHybridTree(task="regression", max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    # Check it's a comparable float (for pytest compatibility)
    assert isinstance(score, (_ComparableFloat, float))

    # R² should be reasonable for this easy dataset
    assert -1 <= score <= 1


def test_less_greedy_hybrid_different_depths(regression_data):
    """Test with different max_depth values."""
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for depth in [1, 3, 5]:
        model = LessGreedyHybridTree(task="regression", max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

        # Deeper trees should generally have more leaves
        leaf_count = model.count_leaves()
        assert leaf_count >= 1


def test_less_greedy_hybrid_oblique_root(regression_data):
    """Test oblique root split functionality."""
    X, y = regression_data

    # Enable oblique root
    model_oblique = LessGreedyHybridTree(
        task="regression",
        max_depth=3,
        enable_oblique_root=True,
        random_state=42,
    )
    model_oblique.fit(X, y)

    # Disable oblique root
    model_no_oblique = LessGreedyHybridTree(
        task="regression",
        max_depth=3,
        enable_oblique_root=False,
        random_state=42,
    )
    model_no_oblique.fit(X, y)

    # Both should produce valid predictions
    preds_oblique = model_oblique.predict(X)
    preds_no_oblique = model_no_oblique.predict(X)

    assert preds_oblique.shape == y.shape
    assert preds_no_oblique.shape == y.shape

    # Check if oblique root was actually used (when enabled)
    if model_oblique.oblique_info_ is not None:
        assert "alpha" in model_oblique.oblique_info_
        assert "nnz" in model_oblique.oblique_info_


def test_less_greedy_hybrid_lookahead(regression_data):
    """Test lookahead functionality."""
    X, y = regression_data

    # With lookahead
    model_lookahead = LessGreedyHybridTree(
        task="regression",
        max_depth=4,
        root_k=2,
        inner_k=1,
        min_n_for_lookahead=100,  # Low threshold to trigger lookahead
        random_state=42,
    )
    model_lookahead.fit(X, y)

    # Without lookahead (disable by high threshold)
    model_no_lookahead = LessGreedyHybridTree(
        task="regression",
        max_depth=4,
        root_k=0,
        inner_k=0,
        min_n_for_lookahead=10000,  # High threshold to disable
        random_state=42,
    )
    model_no_lookahead.fit(X, y)

    # Both should work
    assert model_lookahead.predict(X).shape == y.shape
    assert model_no_lookahead.predict(X).shape == y.shape


def test_less_greedy_hybrid_leaf_shrinkage(regression_data):
    """Test leaf shrinkage functionality."""
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # No shrinkage
    model_no_shrink = LessGreedyHybridTree(
        task="regression",
        max_depth=5,
        leaf_smoothing=0.0,
        random_state=42,
    )
    model_no_shrink.fit(X_train, y_train)

    # With shrinkage
    model_shrink = LessGreedyHybridTree(
        task="regression",
        max_depth=5,
        leaf_smoothing=10.0,
        random_state=42,
    )
    model_shrink.fit(X_train, y_train)

    # Both should produce valid predictions
    preds_no_shrink = model_no_shrink.predict(X_test)
    preds_shrink = model_shrink.predict(X_test)

    assert preds_no_shrink.shape == y_test.shape
    assert preds_shrink.shape == y_test.shape

    # Shrinkage typically reduces variance (predictions closer to mean)
    # This is a soft check - not guaranteed but likely
    assert np.std(preds_shrink) <= np.std(preds_no_shrink) * 1.5


def test_less_greedy_hybrid_count_leaves(small_regression_data):
    """Test leaf counting functionality."""
    X, y = small_regression_data

    model = LessGreedyHybridTree(
        task="regression",
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    model.fit(X, y)

    leaf_count = model.count_leaves()
    assert isinstance(leaf_count, int)
    assert leaf_count >= 1
    assert leaf_count <= 2**2  # Max leaves for depth 2


def test_less_greedy_hybrid_empty_data_error():
    """Test error handling with empty data."""
    X = np.array([]).reshape(0, 2)
    y = np.array([])

    model = LessGreedyHybridTree(task="regression", random_state=42)

    with pytest.raises(ValueError, match="X and y must contain at least one sample"):
        model.fit(X, y)


def test_less_greedy_hybrid_invalid_fractions():
    """Test error handling with invalid data fractions."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)

    model = LessGreedyHybridTree(
        task="regression",
        split_frac=0.8,
        val_frac=0.15,
        est_frac=0.1,  # Doesn't sum to 1
        random_state=42,
    )

    with pytest.raises(AssertionError, match="split_frac \\+ val_frac \\+ est_frac must sum to 1"):
        model.fit(X, y)


def test_less_greedy_hybrid_deterministic_with_random_state():
    """Test that results are deterministic with fixed random_state."""
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)

    model1 = LessGreedyHybridTree(task="regression", max_depth=4, random_state=42)
    model2 = LessGreedyHybridTree(task="regression", max_depth=4, random_state=42)

    model1.fit(X, y)
    model2.fit(X, y)

    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    # Same random state should give identical results
    assert np.allclose(pred1, pred2)


def test_less_greedy_hybrid_different_random_states():
    """Test that different random states give different results."""
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)

    model1 = LessGreedyHybridTree(task="regression", max_depth=4, random_state=42)
    model2 = LessGreedyHybridTree(task="regression", max_depth=4, random_state=99)

    model1.fit(X, y)
    model2.fit(X, y)

    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    # Different random states should likely give different results
    # (not guaranteed, but very likely)
    assert not np.allclose(pred1, pred2)


def test_less_greedy_hybrid_honest_partitioning():
    """Test that honest data partitioning is working correctly."""
    X, y = make_regression(n_samples=300, n_features=5, random_state=42)

    model = LessGreedyHybridTree(
        task="regression",
        max_depth=3,
        split_frac=0.6,
        val_frac=0.2,
        est_frac=0.2,
        random_state=42,
    )

    # Should fit without errors
    model.fit(X, y)

    # Check that splits were scanned
    assert model.splits_scanned_ > 0

    # Check predictions work
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.all(np.isfinite(preds))


def test_less_greedy_hybrid_single_feature():
    """Test with single feature."""
    X, y = make_regression(n_samples=100, n_features=1, random_state=42)

    model = LessGreedyHybridTree(
        task="regression",
        max_depth=3,
        enable_oblique_root=False,  # Disable oblique for single feature
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape


def test_less_greedy_hybrid_very_small_data():
    """Test with very small dataset."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.0, 2.0, 1.5, 2.5, 2.0])

    model = LessGreedyHybridTree(
        task="regression",
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape


def test_less_greedy_hybrid_get_params():
    """Test that get_params works correctly (sklearn requirement)."""
    model = LessGreedyHybridTree(
        task="regression",
        max_depth=5,
        min_samples_split=40,
        leaf_smoothing=1.0,
        random_state=42,
    )

    params = model.get_params()

    assert params["max_depth"] == 5
    assert params["min_samples_split"] == 40
    assert params["leaf_smoothing"] == 1.0
    assert params["random_state"] == 42


def test_less_greedy_hybrid_set_params():
    """Test that set_params works correctly (sklearn requirement)."""
    model = LessGreedyHybridTree(task="regression", max_depth=3, random_state=42)

    model.set_params(max_depth=5, min_samples_split=50)

    params = model.get_params()
    assert params["max_depth"] == 5
    assert params["min_samples_split"] == 50
