"""Select one fitted estimator representative of a prediction pool."""

from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import ClassifierTags, RegressorTags, resample
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

from .stability_utils import check_predict_input

ProximityMetric = Literal[
    "auto", "rmse", "mae", "correlation", "disagreement", "probability_mse"
]

__all__ = ["RepresentativeEstimator"]


class RepresentativeEstimator(BaseEstimator):
    """Select one fitted estimator closest to a candidate-pool centroid.

    Candidates are fitted to independent bootstrap samples of a training split;
    classification candidates use class-stratified samples. The candidate closest
    to their mean prediction on a held-out selection split is retained. This
    produces one fitted estimator representative of that finite pool. It does not
    promise lower sampling instability than an unselected fit.

    With squared Euclidean distance, choosing the candidate closest to the mean
    is equivalent to choosing the prediction medoid: the candidate with the
    smallest average squared distance to every other candidate. This implements
    representative-model selection; it is not a new algorithm.

    Parameters
    ----------
    estimator
        Unfitted scikit-learn-compatible estimator to clone for each candidate.
        The default is the corresponding scikit-learn decision tree.
    task
        ``"classification"`` or ``"regression"``.
    n_candidates
        Number of fitted candidates.
    proximity_metric
        ``"auto"`` uses RMSE for regression and probability MSE for classifiers
        with ``predict_proba`` (otherwise label disagreement). Explicit metrics
        must be compatible with the task.
    validation_fraction
        Fraction held out from candidate fitting for candidate selection.
    bootstrap_candidates
        Whether to bootstrap the candidate-fitting split. If false, deterministic
        base estimators can produce an identical candidate pool.
    random_state
        Random seed controlling the split, bootstrap samples, and candidate seeds.
    """

    def __init__(
        self,
        estimator: Any | None = None,
        task: Literal["classification", "regression"] = "classification",
        n_candidates: int = 20,
        proximity_metric: ProximityMetric = "auto",
        validation_fraction: float = 0.2,
        bootstrap_candidates: bool = True,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.task = task
        self.n_candidates = n_candidates
        self.proximity_metric = proximity_metric
        self.validation_fraction = validation_fraction
        self.bootstrap_candidates = bootstrap_candidates
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "RepresentativeEstimator":
        """Fit the candidate pool, select its representative, and return ``self``."""
        self._validate_parameters()
        original_X = X
        is_dataframe = hasattr(X, "columns") and hasattr(X, "iloc")
        checked_X, y = cast(
            tuple[Any, Any],
            cast(Any, validate_data)(
                self,
                X=X,
                y=y,
                dtype=None if is_dataframe else "numeric",
                accept_sparse=False,
                reset=True,
            ),
        )
        X = original_X if is_dataframe else checked_X
        check_consistent_length(X, y)
        if self.task == "classification":
            target_type = type_of_target(y, input_name="y", raise_unknown=True)
            if target_type not in {"binary", "multiclass"}:
                raise ValueError(f"Unknown label type: {target_type}.")
            self.classes_ = np.unique(y)
            if len(self.classes_) < 2:
                raise ValueError("Classification requires more than one class.")

        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y if self.task == "classification" else None,
        )
        rng = np.random.default_rng(self.random_state)
        prototype = self._prototype()

        candidates: list[Any] = []
        validation_predictions: list[NDArray[Any]] = []
        validation_probabilities: list[NDArray[np.floating]] = []
        all_support_probabilities = True

        for _ in range(self.n_candidates):
            seed = int(rng.integers(0, 2**31 - 1))
            X_candidate, y_candidate = self._candidate_sample(X_train, y_train, rng)
            candidate = cast(Any, clone(prototype))
            random_states = {
                name: seed
                for name in candidate.get_params(deep=True)
                if name == "random_state" or name.endswith("__random_state")
            }
            if random_states:
                candidate.set_params(**random_states)
            candidate.fit(X_candidate, y_candidate)
            candidates.append(candidate)
            validation_predictions.append(np.asarray(candidate.predict(X_validation)))
            if self.task == "classification" and hasattr(candidate, "predict_proba"):
                validation_probabilities.append(
                    self._aligned_probabilities(candidate, X_validation)
                )
            elif self.task == "classification":
                all_support_probabilities = False

        self.validation_predictions_ = np.stack(validation_predictions)
        if self.task == "classification":
            self.candidate_performance_scores_ = np.asarray(
                [
                    accuracy_score(y_validation, prediction)
                    for prediction in self.validation_predictions_
                ]
            )
        else:
            self.candidate_performance_scores_ = np.asarray(
                [
                    r2_score(y_validation, prediction)
                    for prediction in self.validation_predictions_
                ]
            )
        if self.task == "classification" and all_support_probabilities:
            self.validation_probabilities_ = np.stack(validation_probabilities)
        else:
            self.validation_probabilities_ = None

        self.proximity_metric_ = self._resolve_metric(all_support_probabilities)
        self.candidate_scores_, self.ensemble_predictions_ = self._proximity_scores()
        self.selected_index_ = int(np.argmin(self.candidate_scores_))
        self.selected_estimator_ = candidates[self.selected_index_]
        self.candidates_ = candidates
        self.n_validation_samples_ = len(y_validation)
        return self

    def _validate_parameters(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        if (
            not isinstance(self.n_candidates, Integral)
            or isinstance(self.n_candidates, bool)
            or self.n_candidates < 1
        ):
            raise ValueError("n_candidates must be a positive integer.")
        if (
            not isinstance(self.validation_fraction, Real)
            or not 0 < self.validation_fraction < 1
        ):
            raise ValueError("validation_fraction must be strictly between 0 and 1.")
        valid_metrics = {
            "auto",
            "rmse",
            "mae",
            "correlation",
            "disagreement",
            "probability_mse",
        }
        if self.proximity_metric not in valid_metrics:
            raise ValueError(f"Unknown proximity_metric: {self.proximity_metric!r}.")
        regression_metrics = {"auto", "rmse", "mae", "correlation"}
        classification_metrics = {"auto", "disagreement", "probability_mse"}
        allowed = (
            regression_metrics if self.task == "regression" else classification_metrics
        )
        if self.proximity_metric not in allowed:
            raise ValueError(
                f"proximity_metric={self.proximity_metric!r} is not valid for {self.task}."
            )
        if not isinstance(self.bootstrap_candidates, bool):
            raise ValueError("bootstrap_candidates must be bool.")

    def _prototype(self) -> Any:
        """Return the estimator that will be cloned into the candidate pool."""
        if self.estimator is not None:
            return self.estimator
        if self.task == "classification":
            return DecisionTreeClassifier()
        return DecisionTreeRegressor()

    def _candidate_sample(
        self,
        X_train: Any,
        y_train: Any,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        if not self.bootstrap_candidates:
            return X_train, y_train
        seed = int(rng.integers(0, 2**31 - 1))
        sampled = cast(
            list[Any],
            resample(
                X_train,
                y_train,
                replace=True,
                n_samples=len(y_train),
                random_state=seed,
                stratify=y_train if self.task == "classification" else None,
            ),
        )
        return sampled[0], sampled[1]

    def _aligned_probabilities(
        self, candidate: Any, X_validation: Any
    ) -> NDArray[np.floating]:
        probabilities = np.asarray(candidate.predict_proba(X_validation), dtype=float)
        candidate_classes = getattr(candidate, "classes_", None)
        if candidate_classes is None or probabilities.ndim != 2:
            raise ValueError(
                "A candidate with predict_proba must expose classes_ and a 2D result."
            )
        aligned = np.zeros((len(X_validation), len(self.classes_)))
        class_to_column = {label: index for index, label in enumerate(self.classes_)}
        for source, label in enumerate(candidate_classes):
            if label not in class_to_column:
                raise ValueError(f"Candidate produced unknown class {label!r}.")
            aligned[:, class_to_column[label]] = probabilities[:, source]
        return aligned

    def _resolve_metric(self, all_support_probabilities: bool) -> str:
        if self.proximity_metric != "auto":
            if (
                self.proximity_metric == "probability_mse"
                and not all_support_probabilities
            ):
                raise ValueError(
                    "probability_mse requires predict_proba on every candidate."
                )
            return self.proximity_metric
        if self.task == "regression":
            return "rmse"
        return "probability_mse" if all_support_probabilities else "disagreement"

    def _proximity_scores(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[Any]]:
        if self.proximity_metric_ == "probability_mse":
            probabilities = self.validation_probabilities_
            if probabilities is None:
                raise RuntimeError(
                    "Internal error: probability metric without probabilities."
                )
            centroid = probabilities.mean(axis=0)
            scores = np.mean((probabilities - centroid) ** 2, axis=(1, 2))
            return scores, centroid

        predictions = self.validation_predictions_
        if self.proximity_metric_ == "disagreement":
            centroid = self._column_modes(predictions)
            scores = np.mean(predictions != centroid, axis=1)
            return scores, centroid

        numeric_predictions = predictions.astype(float)
        centroid = numeric_predictions.mean(axis=0)
        if self.proximity_metric_ == "rmse":
            scores = np.sqrt(np.mean((numeric_predictions - centroid) ** 2, axis=1))
        elif self.proximity_metric_ == "mae":
            scores = np.mean(np.abs(numeric_predictions - centroid), axis=1)
        else:
            scores = np.array(
                [
                    self._correlation_distance(row, centroid)
                    for row in numeric_predictions
                ]
            )
        return scores, centroid

    @staticmethod
    def _column_modes(predictions: NDArray[Any]) -> NDArray[Any]:
        modes = []
        for column in predictions.T:
            values, counts = np.unique(column, return_counts=True)
            modes.append(values[np.argmax(counts)])
        return np.asarray(modes, dtype=predictions.dtype)

    @staticmethod
    def _correlation_distance(
        predictions: NDArray[np.floating], centroid: NDArray[np.floating]
    ) -> float:
        predictions_constant = bool(np.std(predictions) == 0)
        centroid_constant = bool(np.std(centroid) == 0)
        if predictions_constant or centroid_constant:
            return 0.0 if np.allclose(predictions, centroid) else float("inf")
        return float(1.0 - np.corrcoef(predictions, centroid)[0, 1])

    def predict(self, X: Any) -> NDArray[Any]:
        """Predict with the selected candidate estimator."""
        checked = check_predict_input(self, X, "selected_estimator_")
        return np.asarray(self.selected_estimator_.predict(checked))

    def _supports_predict_proba(self) -> bool:
        """Return whether the configured or selected classifier has probabilities."""
        if self.task != "classification":
            return False
        estimator = getattr(self, "selected_estimator_", self.estimator)
        if estimator is None:
            estimator = DecisionTreeClassifier()
        return hasattr(estimator, "predict_proba")

    @available_if(_supports_predict_proba)
    def predict_proba(self, X: Any) -> NDArray[np.floating]:
        """Return aligned class probabilities from the selected classifier."""
        check_is_fitted(self, "selected_estimator_")
        checked = check_predict_input(self, X, "selected_estimator_")
        return self._aligned_probabilities(self.selected_estimator_, checked)

    def score(
        self,
        X: Any,
        y: NDArray[Any],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> float:
        """Return accuracy for classification or R-squared for regression."""
        predictions = self.predict(X)
        if self.task == "classification":
            return float(accuracy_score(y, predictions, sample_weight=sample_weight))
        return float(r2_score(y, predictions, sample_weight=sample_weight))

    def get_selected_estimator(self) -> Any:
        """Return the fitted candidate selected by the proximity rule."""
        check_is_fitted(self, "selected_estimator_")
        return self.selected_estimator_

    def __sklearn_tags__(self):
        """Expose the task-dependent estimator type to scikit-learn."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        if self.task == "classification":
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags(multi_class=True)
            tags.regressor_tags = None
        else:
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
            tags.classifier_tags = None
        return tags
