"""
CentroidTree: Select the tree closest to ensemble centroid.

This module implements centroid-based model selection:
train N candidate trees and select the one whose predictions most closely
resemble the ensemble mean (centroid) on a held-out validation set.

Key insight: This approach reduces prediction variance by selecting models
that are representative of the model space, rather than outliers.
"""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

ProximityMetric = Literal[
    "rmse", "mae", "correlation", "disagreement", "probability_mse"
]


class CentroidTree(BaseEstimator, ClassifierMixin, RegressorMixin):
    r"""
    Select the tree closest to ensemble centroid (mean prediction).

    This method trains `n_candidates` trees with different random seeds,
    computes the ensemble mean prediction on a validation set, and selects
    the single tree whose predictions are closest to that ensemble mean.

    The selected tree retains full interpretability (it's just a single tree)
    while having predictions more representative of the model space.

    Parameters
    ----------
    base_tree_class
        The tree class to use. Defaults to sklearn's DecisionTreeClassifier
        for classification and DecisionTreeRegressor for regression.
        Can also use stable-cart trees like LessGreedyHybridTree.
    task
        The prediction task type.
    n_candidates
        Number of candidate trees to train. More candidates = better
        approximation of ensemble but higher computational cost.
    proximity_metric
        Metric for measuring proximity to ensemble mean.
        For regression: 'rmse', 'mae', 'correlation'
        For classification: 'disagreement', 'probability_mse'
    validation_fraction
        Fraction of training data to hold out for selection.
    base_params
        Parameters to pass to the base tree class.
    bootstrap_candidates
        Fit each candidate on its own bootstrap resample of the training split.
        This is what makes the candidates differ: a decision tree is deterministic
        given (data, parameters), and ``random_state`` only breaks ties between
        exactly-equal splits, so candidates trained on identical data collapse to
        copies of one tree whenever the splits are unambiguous — leaving nothing
        for the proximity criterion to choose between. Set False to reproduce that
        behaviour.
    random_state
        Random state for reproducibility.

    Attributes
    ----------
    selected_tree\_
        The selected tree model after fitting.
    selected_index\_
        Index of the selected tree in the candidate list.
    candidate_scores\_
        Proximity scores for all candidates (lower = closer to ensemble).
    ensemble_predictions\_
        Ensemble mean predictions on validation set.
    classes\_
        Class labels for classification tasks.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from stable_cart import CentroidTree
    >>> X, y = make_classification(n_samples=500, random_state=42)
    >>> model = CentroidTree(n_candidates=20, random_state=42)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X[:5])

    Notes
    -----
    The method is based on the observation that selecting a model close to
    the ensemble mean reduces variance compared to arbitrary model selection.
    This is particularly valuable when interpretability (single tree) is
    required but stability (low variance) is also desired.
    """

    def __init__(
        self,
        base_tree_class: type | None = None,
        task: Literal["classification", "regression"] = "classification",
        n_candidates: int = 20,
        proximity_metric: Literal[
            "rmse", "mae", "correlation", "disagreement", "probability_mse"
        ] = "rmse",
        validation_fraction: float = 0.2,
        base_params: dict[str, Any] | None = None,
        bootstrap_candidates: bool = True,
        random_state: int | None = None,
    ):
        self.base_tree_class = base_tree_class
        self.task = task
        self.n_candidates = n_candidates
        self.proximity_metric = proximity_metric
        self.validation_fraction = validation_fraction
        self.base_params = base_params
        self.bootstrap_candidates = bootstrap_candidates
        self.random_state = random_state

    def fit(self, X: NDArray[np.floating], y: NDArray[Any]) -> "CentroidTree":
        """
        Fit the centroid tree.

        1. Split data into training and validation sets
        2. Train n_candidates trees with different random seeds
        3. Compute ensemble mean predictions on validation set
        4. Select tree closest to ensemble mean

        Parameters
        ----------
        X
            Training feature matrix of shape (n_samples, n_features).
        y
            Training target values of shape (n_samples,).

        Returns
        -------
        CentroidTree
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]

        rng = np.random.default_rng(self.random_state)

        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else:
            self.classes_ = None
            self.n_classes_ = None

        tree_class = self._get_tree_class()
        base_params = self.base_params or {}

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y if self.task == "classification" else None,
        )

        candidates: list[Any] = []
        val_predictions: list[NDArray[np.floating]] = []
        val_probas: list[NDArray[np.floating]] = []

        for _ in range(self.n_candidates):
            seed = int(rng.integers(0, 2**31 - 1))
            params = {**base_params, "random_state": seed}

            X_cand, y_cand = self._candidate_sample(
                np.asarray(X_train), np.asarray(y_train), rng
            )

            tree = tree_class(**params)
            tree.fit(X_cand, y_cand)
            candidates.append(tree)

            if self.task == "regression":
                preds = tree.predict(X_val)
                val_predictions.append(preds)
            else:
                preds = tree.predict(X_val)
                val_predictions.append(preds)
                if hasattr(tree, "predict_proba"):
                    probas = tree.predict_proba(X_val)
                    val_probas.append(probas)

        val_predictions_array = np.array(val_predictions)

        if self.task == "regression":
            ensemble_mean = np.mean(val_predictions_array, axis=0)
            self.ensemble_predictions_ = ensemble_mean
        else:
            if val_probas:
                probas_array = np.array(val_probas)
                ensemble_proba = np.mean(probas_array, axis=0)
                self.ensemble_predictions_ = ensemble_proba
            else:
                self.ensemble_predictions_ = None

        scores = self._compute_proximity_scores(
            candidates,
            val_predictions_array,
            val_probas,
            len(y_val),
        )
        self.candidate_scores_ = scores

        self.selected_index_ = int(np.argmin(scores))
        self.selected_tree_ = candidates[self.selected_index_]

        self._all_candidates_ = candidates
        self._n_samples_ = n_samples

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[Any]:
        """
        Predict using the selected tree.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        NDArray[Any]
            Predicted values of shape (n_samples,).
        """
        check_is_fitted(self, "selected_tree_")
        X = check_array(X, accept_sparse=False)
        return self.selected_tree_.predict(X)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities using the selected tree.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        NDArray[np.floating]
            Class probabilities of shape (n_samples, n_classes).

        Raises
        ------
        ValueError
            If task is not classification or tree doesn't support predict_proba.
        """
        check_is_fitted(self, "selected_tree_")

        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification")

        X = check_array(X, accept_sparse=False)

        if not hasattr(self.selected_tree_, "predict_proba"):
            raise ValueError("Selected tree does not support predict_proba")

        return self.selected_tree_.predict_proba(X)

    def score(
        self,
        X: NDArray[np.floating],
        y: NDArray[Any],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """
        Return accuracy (classification) or R^2 (regression).

        Parameters
        ----------
        X
            Feature matrix for evaluation.
        y
            True target values.
        sample_weight
            Sample weights (optional).

        Returns
        -------
        float
            Score value.
        """
        check_is_fitted(self, "selected_tree_")
        if sample_weight is not None:
            return self.selected_tree_.score(X, y, sample_weight=sample_weight)
        return self.selected_tree_.score(X, y)

    def get_selected_tree(self) -> Any:
        """
        Get the selected tree model for inspection.

        Returns
        -------
        Any
            The selected tree estimator.
        """
        check_is_fitted(self, "selected_tree_")
        return self.selected_tree_

    def _candidate_sample(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        rng: np.random.Generator,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Draw one candidate's training sample.

        Classification resamples within each class so that every candidate sees
        every class; otherwise a candidate could be missing a column in
        ``predict_proba`` and the proximity comparison would be ragged.

        Parameters
        ----------
        X_train
            Feature matrix of the training split.
        y_train
            Targets of the training split.
        rng
            Random generator shared with candidate seeding.

        Returns
        -------
        tuple[NDArray[np.floating], NDArray[Any]]
            The (X, y) this candidate is fitted on.
        """
        if not self.bootstrap_candidates:
            return X_train, y_train

        if self.task == "classification":
            parts = []
            for cls in np.unique(y_train):
                cls_idx = np.flatnonzero(y_train == cls)
                parts.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
            idx = np.concatenate(parts)
        else:
            idx = rng.integers(0, len(y_train), len(y_train))

        return X_train[idx], y_train[idx]

    def _get_tree_class(self) -> type:
        """Determine the tree class to use."""
        if self.base_tree_class is not None:
            return self.base_tree_class

        if self.task == "classification":
            return DecisionTreeClassifier
        else:
            return DecisionTreeRegressor

    def _compute_proximity_scores(
        self,
        candidates: list[Any],
        val_predictions: NDArray[np.floating],
        val_probas: list[NDArray[np.floating]],
        n_val_samples: int,
    ) -> NDArray[np.floating]:
        """
        Compute proximity scores for each candidate.

        Lower scores indicate closer proximity to ensemble mean.
        """
        n_candidates = len(candidates)
        scores = np.zeros(n_candidates)

        if self.task == "regression":
            ensemble_mean = np.mean(val_predictions, axis=0)

            for i in range(n_candidates):
                preds = val_predictions[i]

                if self.proximity_metric == "rmse":
                    scores[i] = np.sqrt(np.mean((preds - ensemble_mean) ** 2))
                elif self.proximity_metric == "mae":
                    scores[i] = np.mean(np.abs(preds - ensemble_mean))
                elif self.proximity_metric == "correlation":
                    if np.std(preds) > 0 and np.std(ensemble_mean) > 0:
                        corr = np.corrcoef(preds, ensemble_mean)[0, 1]
                        scores[i] = 1 - corr
                    else:
                        scores[i] = 0
                else:
                    scores[i] = np.sqrt(np.mean((preds - ensemble_mean) ** 2))

        else:
            if self.proximity_metric == "disagreement":
                mode_preds = np.zeros(n_val_samples, dtype=val_predictions.dtype)
                for j in range(n_val_samples):
                    values, counts = np.unique(
                        val_predictions[:, j], return_counts=True
                    )
                    mode_preds[j] = values[np.argmax(counts)]

                for i in range(n_candidates):
                    disagreement = np.mean(val_predictions[i] != mode_preds)
                    scores[i] = disagreement

            elif self.proximity_metric == "probability_mse" and val_probas:
                probas_array = np.array(val_probas)
                ensemble_proba = np.mean(probas_array, axis=0)

                for i in range(n_candidates):
                    mse = np.mean((val_probas[i] - ensemble_proba) ** 2)
                    scores[i] = mse

            else:
                mode_preds = np.zeros(n_val_samples, dtype=val_predictions.dtype)
                for j in range(n_val_samples):
                    values, counts = np.unique(
                        val_predictions[:, j], return_counts=True
                    )
                    mode_preds[j] = values[np.argmax(counts)]

                for i in range(n_candidates):
                    disagreement = np.mean(val_predictions[i] != mode_preds)
                    scores[i] = disagreement

        return scores
