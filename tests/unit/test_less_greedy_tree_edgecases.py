import numpy as np
import pytest

from less_greedy_tree import LessGreedyHybridRegressor, GreedyCARTExact


def test_handles_constant_feature_and_constant_target():
    """Test that LessGreedyHybridRegressor handles constant features and targets."""
    X = np.ones((40, 3))
    y = np.full(40, 5.0)
    m = LessGreedyHybridRegressor(
        max_depth=2, min_samples_split=4, min_samples_leaf=2, random_state=0
    ).fit(X, y)
    yhat = m.predict(X)
    assert np.allclose(yhat, 5.0, rtol=1e-6)
    assert np.isfinite(yhat).all()
    assert m.count_leaves() == 1


def test_small_min_leaf_and_duplicate_rows():
    """Test that LessGreedyHybridRegressor handles duplicate rows with small min leaf."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(60, 5))
    X[30:] = X[:30]  # Duplicate rows
    y = rng.normal(size=60)
    m = LessGreedyHybridRegressor(
        max_depth=6, min_samples_split=4, min_samples_leaf=1, random_state=123
    ).fit(X, y)
    yhat = m.predict(X)
    assert yhat.shape == (60,)
    assert np.isfinite(yhat).all()
    assert m.count_leaves() >= 1


def test_empty_input_raises():
    """Test that empty input raises an exception."""
    m = LessGreedyHybridRegressor()
    with pytest.raises(ValueError):  # Raised by np.random.permutation
        m.fit(np.empty((0, 3)), np.empty(0))


def test_invalid_split_fractions_raises():
    """Test that invalid split fractions raise an assertion error."""
    X = np.random.normal(size=(100, 3))
    y = np.random.normal(size=100)
    m = LessGreedyHybridRegressor(split_frac=0.5, val_frac=0.5, est_frac=0.1)
    with pytest.raises(AssertionError, match="must sum to 1"):
        m.fit(X, y)


def test_collinear_features():
    """Test that LessGreedyHybridRegressor handles collinear features."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))
    X[:, 2] = X[:, 1] * 2.0  # Collinear feature
    y = rng.normal(size=50)
    m = LessGreedyHybridRegressor(
        max_depth=3, min_samples_split=5, min_samples_leaf=2, random_state=42
    ).fit(X, y)
    yhat = m.predict(X)
    assert yhat.shape == (50,)
    assert np.isfinite(yhat).all()