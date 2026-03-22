"""Tests for CentroidTree."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stable_cart import CentroidTree, LessGreedyHybridTree


class TestCentroidTreeBasics:
    """Test basic functionality of CentroidTree."""

    def test_classification_fit_predict(self):
        """Test basic classification workflow."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="classification", n_candidates=5, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(np.unique(predictions)).issubset(set(np.unique(y)))

    def test_regression_fit_predict(self):
        """Test basic regression workflow."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="regression", n_candidates=5, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert np.issubdtype(predictions.dtype, np.floating)

    def test_predict_proba_classification(self):
        """Test predict_proba for classification."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="classification", n_candidates=5, random_state=42)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_regression_error(self):
        """Test that predict_proba raises error for regression."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="regression", n_candidates=5, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="only available for classification"):
            model.predict_proba(X)

    def test_score_classification(self):
        """Test score method for classification."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = CentroidTree(task="classification", n_candidates=5, random_state=42)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        assert 0 <= score <= 1

    def test_score_regression(self):
        """Test score method for regression."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = CentroidTree(task="regression", n_candidates=5, random_state=42)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        assert isinstance(score, float)


class TestCentroidTreeSelection:
    """Test model selection behavior."""

    def test_single_candidate_equals_single_tree(self):
        """Test that n_candidates=1 is equivalent to single tree."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = CentroidTree(task="classification", n_candidates=1, random_state=42)
        model.fit(X_train, y_train)

        assert model.selected_index_ == 0
        assert len(model.candidate_scores_) == 1

    def test_selected_index_valid(self):
        """Test that selected_index is within valid range."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        n_candidates = 10

        model = CentroidTree(
            task="classification", n_candidates=n_candidates, random_state=42
        )
        model.fit(X, y)

        assert 0 <= model.selected_index_ < n_candidates
        assert len(model.candidate_scores_) == n_candidates

    def test_candidate_scores_nonnegative(self):
        """Test that all candidate scores are non-negative."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="regression", n_candidates=10, random_state=42)
        model.fit(X, y)

        assert np.all(model.candidate_scores_ >= 0)

    def test_get_selected_tree(self):
        """Test get_selected_tree method."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="classification", n_candidates=5, random_state=42)
        model.fit(X, y)

        tree = model.get_selected_tree()
        assert tree is model.selected_tree_
        assert hasattr(tree, "predict")

    def test_reproducibility(self):
        """Test that same random_state gives same results."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)

        model1 = CentroidTree(task="classification", n_candidates=10, random_state=42)
        model1.fit(X, y)

        model2 = CentroidTree(task="classification", n_candidates=10, random_state=42)
        model2.fit(X, y)

        np.testing.assert_array_equal(
            model1.candidate_scores_, model2.candidate_scores_
        )
        assert model1.selected_index_ == model2.selected_index_


class TestProximityMetrics:
    """Test different proximity metrics."""

    def test_rmse_regression(self):
        """Test RMSE metric for regression."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            task="regression",
            n_candidates=5,
            proximity_metric="rmse",
            random_state=42,
        )
        model.fit(X, y)
        assert model.candidate_scores_ is not None

    def test_mae_regression(self):
        """Test MAE metric for regression."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            task="regression",
            n_candidates=5,
            proximity_metric="mae",
            random_state=42,
        )
        model.fit(X, y)
        assert model.candidate_scores_ is not None

    def test_correlation_regression(self):
        """Test correlation metric for regression."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            task="regression",
            n_candidates=5,
            proximity_metric="correlation",
            random_state=42,
        )
        model.fit(X, y)
        assert model.candidate_scores_ is not None

    def test_disagreement_classification(self):
        """Test disagreement metric for classification."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            task="classification",
            n_candidates=5,
            proximity_metric="disagreement",
            random_state=42,
        )
        model.fit(X, y)
        assert model.candidate_scores_ is not None
        assert np.all(model.candidate_scores_ <= 1)

    def test_probability_mse_classification(self):
        """Test probability MSE metric for classification."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            task="classification",
            n_candidates=5,
            proximity_metric="probability_mse",
            random_state=42,
        )
        model.fit(X, y)
        assert model.candidate_scores_ is not None


class TestBaseTreeClass:
    """Test using custom base tree classes."""

    def test_sklearn_decision_tree_explicit(self):
        """Test explicitly passing sklearn DecisionTreeClassifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            base_tree_class=DecisionTreeClassifier,
            task="classification",
            n_candidates=5,
            random_state=42,
        )
        model.fit(X, y)
        assert isinstance(model.get_selected_tree(), DecisionTreeClassifier)

    def test_sklearn_decision_tree_regressor_explicit(self):
        """Test explicitly passing sklearn DecisionTreeRegressor."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            base_tree_class=DecisionTreeRegressor,
            task="regression",
            n_candidates=5,
            random_state=42,
        )
        model.fit(X, y)
        assert isinstance(model.get_selected_tree(), DecisionTreeRegressor)

    def test_with_base_params(self):
        """Test passing parameters to base tree."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            task="classification",
            n_candidates=5,
            base_params={"max_depth": 3},
            random_state=42,
        )
        model.fit(X, y)

        tree = model.get_selected_tree()
        assert tree.max_depth == 3

    def test_with_stable_cart_tree(self):
        """Test using LessGreedyHybridTree as base."""
        X, y = make_classification(n_samples=300, n_features=5, random_state=42)
        model = CentroidTree(
            base_tree_class=LessGreedyHybridTree,
            task="classification",
            n_candidates=3,
            base_params={"task": "classification", "max_depth": 3},
            random_state=42,
        )
        model.fit(X, y)
        assert isinstance(model.get_selected_tree(), LessGreedyHybridTree)


class TestSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_clone_compatibility(self):
        """Test that model can be cloned."""
        from sklearn.base import clone

        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="classification", n_candidates=5, random_state=42)
        model.fit(X, y)

        cloned = clone(model)
        assert cloned.n_candidates == model.n_candidates
        assert cloned.random_state == model.random_state

    def test_get_set_params(self):
        """Test get_params and set_params."""
        model = CentroidTree(task="classification", n_candidates=10, random_state=42)

        params = model.get_params()
        assert params["n_candidates"] == 10
        assert params["random_state"] == 42

        model.set_params(n_candidates=20)
        assert model.n_candidates == 20

    def test_cross_validation_compatible(self):
        """Test compatibility with cross-validation."""
        from sklearn.model_selection import cross_val_score

        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="classification", n_candidates=3, random_state=42)

        scores = cross_val_score(model, X, y, cv=3)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unfitted_predict_error(self):
        """Test that predict on unfitted model raises error."""
        from sklearn.exceptions import NotFittedError

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = CentroidTree(task="classification", n_candidates=5)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_small_dataset(self):
        """Test with small dataset."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]] * 10)
        y = np.array([0, 1, 0, 1, 0] * 10)

        model = CentroidTree(
            task="classification",
            n_candidates=3,
            validation_fraction=0.2,
            random_state=42,
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_multiclass_classification(self):
        """Test with multiclass classification."""
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42,
        )
        model = CentroidTree(task="classification", n_candidates=5, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert set(np.unique(predictions)).issubset(set(np.unique(y)))


class TestAttributes:
    """Test fitted attributes."""

    def test_classes_attribute_classification(self):
        """Test classes_ attribute for classification."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="classification", n_candidates=5, random_state=42)
        model.fit(X, y)

        assert model.classes_ is not None
        assert len(model.classes_) == 2

    def test_classes_attribute_regression(self):
        """Test classes_ is None for regression."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="regression", n_candidates=5, random_state=42)
        model.fit(X, y)

        assert model.classes_ is None

    def test_ensemble_predictions_regression(self):
        """Test ensemble_predictions_ attribute for regression."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(task="regression", n_candidates=5, random_state=42)
        model.fit(X, y)

        assert model.ensemble_predictions_ is not None
        val_size = int(len(X) * model.validation_fraction)
        assert len(model.ensemble_predictions_) == val_size

    def test_ensemble_predictions_classification_proba(self):
        """Test ensemble_predictions_ attribute for classification with proba."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = CentroidTree(
            task="classification",
            n_candidates=5,
            proximity_metric="probability_mse",
            random_state=42,
        )
        model.fit(X, y)

        assert model.ensemble_predictions_ is not None
