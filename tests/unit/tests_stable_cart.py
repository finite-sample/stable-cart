import pytest
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import label_binarize
from sklearn.dummy import DummyRegressor

from stable_cart.evaluation import prediction_stability, accuracy
from stable_cart.less_greedy_tree import GreedyCARTExact, LessGreedyHybridRegressor, _sse, _ComparableFloat

# Tolerance for floating-point comparisons
TOL = 1e-6

# -------------------------------
# Fixtures
# -------------------------------

@pytest.fixture
def small_regression_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    return X, y

@pytest.fixture
def regression_models_and_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    X_tr, X_oos, y_tr, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    models = {
        "dt1": DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr),
        "dt2": DecisionTreeRegressor(random_state=1).fit(X_tr, y_tr)
    }
    return models, X_oos, "continuous"

@pytest.fixture
def classification_models_and_data():
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
    X_tr, X_oos, y_tr, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    models = {
        "dt1": DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr),
        "dt2": DecisionTreeClassifier(random_state=1).fit(X_tr, y_tr)
    }
    return models, X_oos, "categorical"

@pytest.fixture
def regression_eval_data():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    models = {
        "dt": DecisionTreeRegressor(random_state=42).fit(X, y)
    }
    return models, X, y, "continuous"

@pytest.fixture
def classification_eval_data():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
    models = {
        "dt": DecisionTreeClassifier(random_state=42).fit(X, y)
    }
    return models, X, y, "categorical"

# -------------------------------
# Utilities
# -------------------------------

def test_sse():
    y = np.array([1.0, 2.0, 3.0])
    assert _sse(y) == pytest.approx(2.0, abs=TOL)  # var=2/3, n*var=2
    y = np.array([5.0])
    assert _sse(y) == pytest.approx(0.0, abs=TOL)
    y = np.array([])
    assert _sse(y) == pytest.approx(0.0, abs=TOL)

def test_comparable_float():
    cf = _ComparableFloat(1.0)
    assert cf == pytest.approx(1.0)
    assert cf > pytest.approx(0.999)
    assert cf < pytest.approx(1.001)
    assert cf >= pytest.approx(1.0)
    assert cf <= pytest.approx(1.0)
    assert cf != pytest.approx(1.1)

# -------------------------------
# GreedyCARTExact
# -------------------------------

def test_greedy_cart_exact_fit_predict(small_regression_data):
    X, y = small_regression_data
    model = GreedyCARTExact(max_depth=2, min_samples_split=2, min_samples_leaf=1)
    model.fit(X, y)
    assert model.tree_['type'] == 'split'
    assert 'left' in model.tree_
    assert 'right' in model.tree_
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert r2_score(y, preds) > 0.9
    assert model.fit_time_sec_ > 0
    assert model.splits_scanned_ > 0

def test_greedy_cart_exact_leaf_count(small_regression_data):
    X, y = small_regression_data
    model = GreedyCARTExact(max_depth=1)
    model.fit(X, y)
    assert model.count_leaves() == 2

def test_greedy_cart_exact_edge_cases():
    # Empty data
    with pytest.raises(ValueError):
        model = GreedyCARTExact()
        model.fit(np.array([]).reshape(0, 2), np.array([]))
    
    # Single sample
    X = np.array([[1, 2]])
    y = np.array([3])
    model = GreedyCARTExact()
    model.fit(X, y)
    assert model.predict(X) == pytest.approx([3])
    assert model.count_leaves() == 1

    # No split possible
    X = np.array([[1], [1], [1]])
    y = np.array([1, 2, 3])
    model = GreedyCARTExact(min_samples_leaf=2)
    model.fit(X, y)
    assert model.count_leaves() == 1

# -------------------------------
# LessGreedyHybridRegressor
# -------------------------------

def test_less_greedy_hybrid_fit_predict(small_regression_data):
    X, y = small_regression_data
    model = LessGreedyHybridRegressor(max_depth=2, min_samples_split=2, min_samples_leaf=1,
                                      split_frac=0.5, val_frac=0.25, est_frac=0.25,
                                      random_state=42)
    model.fit(X, y)
    assert model.tree_['type'] in ['split', 'split_oblique', 'leaf']
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.score(X, y) == pytest.approx(r2_score(y, preds))
    assert model.fit_time_sec_ > 0
    assert model.splits_scanned_ > 0

def test_less_greedy_hybrid_oblique_root():
    X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=42)
    model = LessGreedyHybridRegressor(enable_oblique_root=True, random_state=42)
    model.fit(X, y)
    if model.oblique_info_ is not None:
        assert 'alpha' in model.oblique_info_
        assert model.tree_['type'] == 'split_oblique'

def test_less_greedy_hybrid_leaf_shrinkage():
    X, y = make_regression(n_samples=50, n_features=3, random_state=42)
    model = LessGreedyHybridRegressor(leaf_shrinkage_lambda=1.0, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()

def test_less_greedy_hybrid_constant_features_targets():
    X = np.ones((40, 3))
    y = np.full(40, 5.0)
    model = LessGreedyHybridRegressor(
        max_depth=2, min_samples_split=4, min_samples_leaf=2, random_state=0,
        split_frac=0.5, val_frac=0.25, est_frac=0.25
    )
    model.fit(X, y)
    yhat = model.predict(X)
    assert np.allclose(yhat, 5.0, rtol=1e-6)
    assert np.isfinite(yhat).all()
    assert model.count_leaves() == 1

def test_less_greedy_hybrid_duplicate_rows():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(60, 5))
    X[30:] = X[:30]  # Duplicate rows
    y = rng.normal(size=60)
    model = LessGreedyHybridRegressor(
        max_depth=6, min_samples_split=4, min_samples_leaf=1, random_state=123,
        split_frac=0.5, val_frac=0.25, est_frac=0.25
    )
    model.fit(X, y)
    yhat = model.predict(X)
    assert yhat.shape == (60,)
    assert np.isfinite(yhat).all()
    assert model.count_leaves() >= 1

def test_less_greedy_hybrid_collinear_features():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))
    X[:, 2] = X[:, 1] * 2.0  # Collinear feature
    y = rng.normal(size=50)
    model = LessGreedyHybridRegressor(
        max_depth=3, min_samples_split=5, min_samples_leaf=2, random_state=42,
        split_frac=0.5, val_frac=0.25, est_frac=0.25
    )
    model.fit(X, y)
    yhat = model.predict(X)
    assert yhat.shape == (50,)
    assert np.isfinite(yhat).all()

def test_less_greedy_hybrid_edge_cases():
    # Empty input
    with pytest.raises(ValueError):
        model = LessGreedyHybridRegressor()
        model.fit(np.empty((0, 3)), np.empty(0))
    
    # Invalid split fractions
    with pytest.raises(AssertionError):
        model = LessGreedyHybridRegressor(split_frac=0.5, val_frac=0.5, est_frac=0.1)
        model.fit(np.random.rand(10, 2), np.random.rand(10))
    
    # Small data
    X = np.array([[1]])
    y = np.array([1])
    model = LessGreedyHybridRegressor(split_frac=1.0, val_frac=0.0, est_frac=0.0)
    model.fit(X, y)
    assert model.predict(X) == pytest.approx([1])

def test_less_greedy_hybrid_determinism():
    X, y = make_regression(n_samples=100, n_features=5, random_state=123)
    kwargs = dict(max_depth=4, min_samples_split=30, min_samples_leaf=10, random_state=7,
                  split_frac=0.5, val_frac=0.25, est_frac=0.25)
    m1 = LessGreedyHybridRegressor(**kwargs).fit(X, y)
    m2 = LessGreedyHybridRegressor(**kwargs).fit(X, y)
    p1 = m1.predict(X)
    p2 = m2.predict(X)
    assert np.allclose(p1, p2, rtol=1e-6)

# -------------------------------
# evaluation.py: prediction_stability
# -------------------------------

def test_prediction_stability_continuous(regression_models_and_data):
    models, X_oos, task = regression_models_and_data
    scores = prediction_stability(models, X_oos, task)
    assert len(scores) == len(models)
    assert all(isinstance(v, float) for v in scores.values())
    assert all(v >= 0 for v in scores.values())

def test_prediction_stability_categorical(classification_models_and_data):
    models, X_oos, task = classification_models_and_data
    scores = prediction_stability(models, X_oos, task)
    assert len(scores) == len(models)
    assert all(0 <= v <= 1 for v in scores.values())

def test_prediction_stability_errors():
    models = {"dt": DecisionTreeRegressor()}
    X = np.random.rand(10, 2)
    with pytest.raises(ValueError):  # <2 models
        prediction_stability(models, X, "continuous")
    with pytest.raises(ValueError):  # Invalid task
        prediction_stability({"dt1": DecisionTreeRegressor(), "dt2": DecisionTreeRegressor()}, X, "invalid")
    with pytest.raises(ValueError):  # Empty X_oos
        prediction_stability({"dt1": DecisionTreeRegressor(), "dt2": DecisionTreeRegressor()}, np.empty((0, 2)), "continuous")

# -------------------------------
# evaluation.py: accuracy
# -------------------------------

def test_accuracy_continuous(regression_eval_data):
    models, X, y, task = regression_eval_data
    results = accuracy(models, X, y, task)
    assert len(results) == len(models)
    for metrics in results.values():
        assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
        assert metrics["mae"] == pytest.approx(mean_absolute_error(y, models["dt"].predict(X)), abs=TOL)
        assert metrics["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y, models["dt"].predict(X))), abs=TOL)
        assert metrics["r2"] == pytest.approx(r2_score(y, models["dt"].predict(X)), abs=TOL)

def test_accuracy_categorical(classification_eval_data):
    models, X, y, task = classification_eval_data
    results = accuracy(models, X, y, task)
    assert len(results) == len(models)
    for metrics in results.values():
        assert "acc" in metrics
        assert metrics["acc"] == pytest.approx(accuracy_score(y, models["dt"].predict(X)), abs=TOL)
        if "auc" in metrics:
            proba = models["dt"].predict_proba(X)
            auc = roc_auc_score(label_binarize(y, classes=np.unique(y)), proba, average="macro", multi_class="ovr")
            assert metrics["auc"] == pytest.approx(auc, abs=TOL)

def test_accuracy_categorical_no_predict_proba():
    class NoProbaClassifier(DecisionTreeClassifier):
        def predict_proba(self, X):
            raise AttributeError("No predict_proba")
    
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    model = NoProbaClassifier(max_depth=3).fit(X_tr, y_tr)
    results = accuracy({"nop": model}, X_te, y_te, task="categorical")
    assert set(results["nop"].keys()) == {"acc"}
    assert 0.0 <= results["nop"]["acc"] <= 1.0

def test_accuracy_errors():
    models = {"dt": DecisionTreeRegressor()}
    X, y = np.random.rand(10, 2), np.random.rand(10)
    with pytest.raises(ValueError):
        accuracy(models, X, y, "invalid")