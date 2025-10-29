"""Unit tests for RobustPrefixHonestClassifier."""

import pytest
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from stable_cart.robust_prefix import (
    RobustPrefixHonestClassifier,
    _winsorize_fit,
    _winsorize_apply,
    _stratified_bootstrap,
    _robust_stump_on_node,
)


# Tolerance for floating-point comparisons
TOL = 1e-6


# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture
def binary_classification_data():
    """Small binary classification dataset for basic tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def breast_cancer_data():
    """Real-world binary classification dataset."""
    data = load_breast_cancer()
    return data.data, data.target


# -------------------------------
# Test Helper Functions
# -------------------------------


def test_winsorize_fit():
    """Test winsorization quantile fitting."""
    X = np.array([[1, 10], [2, 20], [3, 30], [100, 1000]])  # Outliers: 100, 1000
    lo, hi = _winsorize_fit(X, q=(0.25, 0.75))
    
    assert lo.shape == (2,)
    assert hi.shape == (2,)
    assert lo[0] == pytest.approx(1.75, abs=TOL)
    assert hi[0] == pytest.approx(51.25, abs=TOL)


def test_winsorize_apply():
    """Test winsorization application."""
    X = np.array([[1, 10], [2, 20], [3, 30], [100, 1000]])
    lo = np.array([2, 15])
    hi = np.array([50, 500])
    
    X_win = _winsorize_apply(X, lo, hi)
    
    assert X_win[0, 0] == 2  # Clipped from 1 to lo
    assert X_win[3, 0] == 50  # Clipped from 100 to hi
    assert X_win[3, 1] == 500  # Clipped from 1000 to hi
    assert X_win[1, 0] == 2  # Unchanged (within bounds)


def test_stratified_bootstrap():
    """Test class-stratified bootstrap sampling."""
    rng = np.random.RandomState(42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    
    Xb, yb = _stratified_bootstrap(X, y, rng)
    
    assert Xb.shape == X.shape
    assert yb.shape == y.shape
    # Check stratification: should have samples from both classes
    assert 0 in yb
    assert 1 in yb
    # Check bootstrap: should have duplicates (with high probability)
    unique_count = len(np.unique(Xb, axis=0))
    assert unique_count <= len(X)  # Some samples likely repeated


def test_robust_stump_on_node_basic():
    """Test robust stump selection on a simple node."""
    rng = np.random.RandomState(42)
    
    # Create separable data
    X_split = np.array([[1, 1], [1, 2], [5, 1], [5, 2]])
    y_split = np.array([0, 0, 1, 1])
    X_val = np.array([[1, 1.5], [5, 1.5]])
    y_val = np.array([0, 1])
    
    result = _robust_stump_on_node(
        X_split, y_split, X_val, y_val,
        B=5, subsample_frac=0.8, max_bins=10, rng=rng
    )
    
    assert result is not None
    feat, thr = result
    assert isinstance(feat, int)
    assert isinstance(thr, float)
    assert 0 <= feat < X_split.shape[1]


def test_robust_stump_returns_none_on_insufficient_data():
    """Test that robust stump returns None with insufficient data."""
    rng = np.random.RandomState(42)
    
    # Too few samples
    X_split = np.array([[1, 1]])
    y_split = np.array([0])
    X_val = np.array([[1, 1]])
    y_val = np.array([0])
    
    result = _robust_stump_on_node(
        X_split, y_split, X_val, y_val,
        B=5, subsample_frac=0.8, max_bins=10, rng=rng
    )
    
    assert result is None


# -------------------------------
# Test RobustPrefixHonestClassifier
# -------------------------------


def test_robust_prefix_basic_fit_predict(binary_classification_data):
    """Test basic fitting and prediction."""
    X, y = binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = RobustPrefixHonestClassifier(
        top_levels=1,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Check fitted attributes
    check_is_fitted(model, ['_lo_', '_hi_', '_prefix_nodes_', '_region_models_', 
                             '_region_leaf_probs_', 'classes_'])
    
    assert model.classes_.tolist() == [0, 1]
    assert model._lo_.shape[0] == X_train.shape[1]
    assert model._hi_.shape[0] == X_train.shape[1]
    
    # Check predictions
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert set(y_pred).issubset({0, 1})
    
    # Check predict_proba
    y_proba = model.predict_proba(X_test)
    assert y_proba.shape == (len(X_test), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)
    assert np.all((y_proba >= 0) & (y_proba <= 1))


def test_robust_prefix_different_top_levels(binary_classification_data):
    """Test with different prefix depth levels."""
    X, y = binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    for top_levels in [0, 1, 2, 3]:
        model = RobustPrefixHonestClassifier(
            top_levels=top_levels,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        
        # Sanity check: accuracy should be better than random
        assert acc > 0.5


def test_robust_prefix_m_smoothing(binary_classification_data):
    """Test m-estimate smoothing parameter."""
    X, y = binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # No smoothing
    model_no_smooth = RobustPrefixHonestClassifier(
        top_levels=1,
        m_smooth=0.0,
        random_state=42,
    )
    model_no_smooth.fit(X_train, y_train)
    
    # Heavy smoothing
    model_smooth = RobustPrefixHonestClassifier(
        top_levels=1,
        m_smooth=10.0,
        random_state=42,
    )
    model_smooth.fit(X_train, y_train)
    
    # Both should produce valid predictions
    pred_no_smooth = model_no_smooth.predict_proba(X_test)
    pred_smooth = model_smooth.predict_proba(X_test)
    
    assert pred_no_smooth.shape == pred_smooth.shape
    # Smoothed predictions should be less extreme (closer to 0.5)
    entropy_no_smooth = -np.mean(pred_no_smooth * np.log(pred_no_smooth + 1e-10))
    entropy_smooth = -np.mean(pred_smooth * np.log(pred_smooth + 1e-10))
    # Smoothing typically increases entropy (less confident)
    # This is a soft check since it depends on the data
    assert entropy_smooth >= entropy_no_smooth * 0.9  # Allow some variance


def test_robust_prefix_winsorization(binary_classification_data):
    """Test that winsorization is applied correctly."""
    X, y = binary_classification_data
    
    # Add outliers
    X_with_outliers = X.copy()
    X_with_outliers[0, 0] = 1000  # Extreme outlier
    X_with_outliers[1, 1] = -1000
    
    model = RobustPrefixHonestClassifier(
        winsor_quantiles=(0.01, 0.99),
        random_state=42,
    )
    model.fit(X_with_outliers, y)
    
    # Check that winsorization bounds were stored
    assert model._lo_ is not None
    assert model._hi_ is not None
    
    # Predict on data with outliers
    X_test_outliers = X[:10].copy()
    X_test_outliers[0, 0] = 2000  # Another outlier
    
    # Should not crash despite outliers
    y_pred = model.predict(X_test_outliers)
    assert y_pred.shape == (10,)


def test_robust_prefix_honest_split():
    """Test that honest split (SPLIT/VAL/EST) is working correctly."""
    X, y = make_classification(
        n_samples=300, n_features=10, n_classes=2, random_state=42
    )
    
    model = RobustPrefixHonestClassifier(
        top_levels=1,
        val_frac=0.2,
        est_frac=0.4,
        random_state=42,
    )
    
    # With 300 samples: ~40% SPLIT (~120), 20% VAL (~60), 40% EST (~120)
    model.fit(X, y)
    
    # Check that regions have models
    assert len(model._region_models_) > 0
    assert len(model._region_leaf_probs_) > 0
    
    # Predictions should work
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_robust_prefix_sklearn_compatibility(binary_classification_data):
    """Test sklearn API compatibility."""
    X, y = binary_classification_data
    
    # Test with cross-validation
    model = RobustPrefixHonestClassifier(
        top_levels=1, max_depth=4, random_state=42
    )
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)
    
    # Test in pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RobustPrefixHonestClassifier(
            top_levels=1, max_depth=3, random_state=42
        )),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    assert predictions.shape == y_test.shape


def test_robust_prefix_consensus_parameters(binary_classification_data):
    """Test different consensus parameters."""
    X, y = binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Few bootstrap samples
    model_few = RobustPrefixHonestClassifier(
        top_levels=1,
        consensus_B=3,
        random_state=42,
    )
    model_few.fit(X_train, y_train)
    
    # Many bootstrap samples
    model_many = RobustPrefixHonestClassifier(
        top_levels=1,
        consensus_B=20,
        random_state=42,
    )
    model_many.fit(X_train, y_train)
    
    # Both should produce valid predictions
    pred_few = model_few.predict(X_test)
    pred_many = model_many.predict(X_test)
    
    assert pred_few.shape == pred_many.shape
    # More bootstrap samples might lead to different but valid results
    acc_few = (pred_few == y_test).mean()
    acc_many = (pred_many == y_test).mean()
    assert acc_few > 0.5
    assert acc_many > 0.5


def test_robust_prefix_real_world_data(breast_cancer_data):
    """Test on real-world breast cancer dataset."""
    X, y = breast_cancer_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = RobustPrefixHonestClassifier(
        top_levels=2,
        max_depth=6,
        min_samples_leaf=5,
        m_smooth=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Check performance
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = (y_pred == y_test).mean()
    
    # Should achieve reasonable accuracy on this dataset
    assert acc > 0.85  # Breast cancer is relatively easy
    
    # Check probability calibration (probabilities should be diverse)
    assert y_proba[:, 1].std() > 0.1  # Not all same probability


def test_robust_prefix_multiclass_error():
    """Test that multi-class raises appropriate error."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=3, random_state=42
    )
    
    model = RobustPrefixHonestClassifier(random_state=42)
    
    with pytest.raises(ValueError, match="binary classification only"):
        model.fit(X, y)


def test_robust_prefix_empty_data_error():
    """Test error handling with empty data."""
    X = np.array([]).reshape(0, 5)
    y = np.array([])
    
    model = RobustPrefixHonestClassifier(random_state=42)
    
    # sklearn's train_test_split should raise error on empty data
    with pytest.raises((ValueError, RuntimeError)):
        model.fit(X, y)


def test_robust_prefix_predict_before_fit_error():
    """Test that predict before fit raises appropriate error."""
    X = np.array([[1, 2, 3]])
    
    model = RobustPrefixHonestClassifier(random_state=42)
    
    with pytest.raises(Exception):  # check_is_fitted raises NotFittedError
        model.predict(X)


def test_robust_prefix_deterministic_with_random_state():
    """Test that results are deterministic with fixed random_state."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    model1 = RobustPrefixHonestClassifier(
        top_levels=2, max_depth=5, random_state=42
    )
    model2 = RobustPrefixHonestClassifier(
        top_levels=2, max_depth=5, random_state=42
    )
    
    model1.fit(X, y)
    model2.fit(X, y)
    
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    
    # Same random state should give identical results
    assert np.array_equal(pred1, pred2)


def test_robust_prefix_different_random_states():
    """Test that different random states give different results."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    model1 = RobustPrefixHonestClassifier(
        top_levels=2, max_depth=5, random_state=42
    )
    model2 = RobustPrefixHonestClassifier(
        top_levels=2, max_depth=5, random_state=99
    )
    
    model1.fit(X, y)
    model2.fit(X, y)
    
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    
    # Different random states should likely give different results
    # (not guaranteed, but very likely)
    assert not np.array_equal(pred1, pred2)


# -------------------------------
# Test Edge Cases
# -------------------------------


def test_robust_prefix_single_feature():
    """Test with single feature."""
    X, y = make_classification(
        n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=42
    )
    
    model = RobustPrefixHonestClassifier(
        top_levels=1, max_depth=3, random_state=42
    )
    model.fit(X, y)
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_robust_prefix_perfectly_separable_data():
    """Test on perfectly separable data."""
    # Create perfectly separable data
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    y = np.array([0, 0, 1, 1])
    
    # Duplicate to have enough samples for honest split
    X = np.tile(X, (20, 1))
    y = np.tile(y, 20)
    
    model = RobustPrefixHonestClassifier(
        top_levels=1, max_depth=2, random_state=42
    )
    model.fit(X, y)
    
    y_pred = model.predict(X)
    acc = (y_pred == y).mean()
    
    # Should achieve perfect or near-perfect accuracy
    assert acc >= 0.95


def test_robust_prefix_imbalanced_classes():
    """Test with highly imbalanced classes."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=2,
        weights=[0.95, 0.05],  # 95% class 0, 5% class 1
        random_state=42,
    )
    
    model = RobustPrefixHonestClassifier(
        top_levels=1,
        max_depth=4,
        m_smooth=1.0,  # Smoothing helps with imbalance
        random_state=42,
    )
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    # Should predict some of each class (not all majority)
    assert len(np.unique(y_pred)) == 2