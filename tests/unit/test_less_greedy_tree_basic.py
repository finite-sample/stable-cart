import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

from less_greedy_tree import LessGreedyHybridRegressor, GreedyCARTExact


def _toy_regression(n=600, d=8, noise=0.3, random_state=0):
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=n, n_features=d, n_informative=d//2, noise=10*noise,
        random_state=random_state
    )
    return train_test_split(X, y, test_size=0.33, random_state=random_state)


def test_fit_predict_shapes_and_score_reasonable():
    """Test that LessGreedyHybridRegressor fits and predicts with correct shapes and reasonable R²."""
    Xtr, Xte, ytr, yte = _toy_regression()
    lgr = LessGreedyHybridRegressor(
        max_depth=5, min_samples_split=40, min_samples_leaf=20, random_state=42
    )
    lgr.fit(Xtr, ytr)
    yhat = lgr.predict(Xte)
    assert yhat.shape == yte.shape
    assert np.isfinite(yhat).all()
    dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    r2_lgr = lgr.score(Xte, yte)
    r2_dummy = dummy.score(Xte, yte)
    assert r2_lgr >= pytest.approx(r2_dummy, rel=1e-6)


def test_determinism_with_fixed_random_state():
    """Test that LessGreedyHybridRegressor is deterministic with fixed random_state."""
    Xtr, Xte, ytr, _ = _toy_regression(random_state=123)
    kwargs = dict(max_depth=4, min_samples_split=30, min_samples_leaf=10, random_state=7)
    m1 = LessGreedyHybridRegressor(**kwargs).fit(Xtr, ytr)
    m2 = LessGreedyHybridRegressor(**kwargs).fit(Xtr, ytr)
    p1 = m1.predict(Xte)
    p2 = m2.predict(Xte)
    assert np.allclose(p1, p2, rtol=1e-6)


def test_cart_baseline_runs_and_scores():
    """Test that GreedyCARTExact fits, predicts, and produces finite R²."""
    Xtr, Xte, ytr, yte = _toy_regression()
    cart = GreedyCARTExact(max_depth=5, min_samples_split=20, min_samples_leaf=10)
    cart.fit(Xtr, ytr)
    yhat = cart.predict(Xte)
    assert yhat.shape == yte.shape
    assert np.isfinite(yhat).all()
    r2 = cart.score(Xte, yte)
    assert np.isfinite(r2)


def test_hybrid_oblique_root_enabled():
    """Test that LessGreedyHybridRegressor with oblique root produces valid predictions."""
    Xtr, Xte, ytr, yte = _toy_regression(n=200, d=4)
    lgr = LessGreedyHybridRegressor(
        max_depth=3, min_samples_split=10, min_samples_leaf=5, enable_oblique_root=True, random_state=0
    )
    lgr.fit(Xtr, ytr)
    yhat = lgr.predict(Xte)
    assert yhat.shape == yte.shape
    assert np.isfinite(yhat).all()
    r2 = lgr.score(Xte, yte)
    assert np.isfinite(r2)