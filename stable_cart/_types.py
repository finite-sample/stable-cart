"""Type definitions and protocols for stable_cart."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from numpy.typing import NDArray


class PredictorProtocol(Protocol):
    """Protocol for sklearn-compatible models with predict method."""

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict target values for input features.

        Parameters
        ----------
        X
            Input feature matrix.

        Returns
        -------
        NDArray[Any]
            Predicted target values.
        """
        ...


class ClassifierProtocol(PredictorProtocol, Protocol):
    """Protocol for sklearn-compatible classifiers."""

    def predict_proba(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict class probabilities for input features.

        Parameters
        ----------
        X
            Input feature matrix.

        Returns
        -------
        NDArray[Any]
            Class probabilities.
        """
        ...


# Enums for better type safety
class AlgorithmFocus(Enum):
    """
    Algorithm focus strategies for tree building.

    Attributes
    ----------
    SPEED : str
        Focus on speed optimization.
    ACCURACY : str
        Focus on accuracy optimization.
    STABILITY : str
        Focus on stability optimization.
    """

    SPEED = "speed"
    ACCURACY = "accuracy"
    STABILITY = "stability"


class LeafSmoothingStrategy(Enum):
    """
    Strategies for leaf value smoothing.

    Attributes
    ----------
    M_ESTIMATE : str
        Use m-estimate smoothing.
    SHRINK_TO_PARENT : str
        Shrink leaf values toward parent.
    BETA_SMOOTHING : str
        Use beta distribution smoothing.
    """

    M_ESTIMATE = "m_estimate"
    SHRINK_TO_PARENT = "shrink_to_parent"
    BETA_SMOOTHING = "beta_smoothing"


class ObliqueRegularization(Enum):
    """
    Regularization strategies for oblique splits.

    Attributes
    ----------
    LASSO : str
        Use Lasso regularization.
    RIDGE : str
        Use Ridge regularization.
    ELASTIC_NET : str
        Use Elastic Net regularization.
    """

    LASSO = "lasso"
    RIDGE = "ridge"
    ELASTIC_NET = "elastic_net"


class StoppingStrategy(Enum):
    """
    Stopping strategies for tree building.

    Attributes
    ----------
    ONE_SE : str
        Use one standard error rule.
    VARIANCE_PENALTY : str
        Use variance penalty stopping.
    BOTH : str
        Use both stopping strategies.
    """

    ONE_SE = "one_se"
    VARIANCE_PENALTY = "variance_penalty"
    BOTH = "both"


class Task(Enum):
    """
    Prediction task types.

    Attributes
    ----------
    REGRESSION : str
        Regression task type.
    CLASSIFICATION : str
        Classification task type.
    """

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


# Dataclasses for complex return types
@dataclass
class AxisSplit:
    """
    Result of axis-aligned split selection.

    Attributes
    ----------
    split_type : str
        Split type identifier (e.g., "k0", "k1", etc.).
    feature : int
        Feature index for split.
    threshold : float
        Threshold value for split.
    """

    split_type: str  # e.g., "k0", "k1", etc.
    feature: int
    threshold: float


@dataclass
class ObliqueSplit:
    """
    Result of oblique split selection.

    Attributes
    ----------
    threshold : float
        Threshold value for oblique split.
    mean_values : NDArray[Any]
        Scaling mean values.
    scale_values : NDArray[Any]
        Scaling scale values.
    weights : NDArray[Any]
        Oblique split weights.
    """

    threshold: float
    mean_values: NDArray[Any]  # scaling mean
    scale_values: NDArray[Any]  # scaling scale
    weights: NDArray[Any]  # oblique weights


# Type aliases for better readability
ModelDict = dict[str, PredictorProtocol]
ClassifierDict = dict[str, ClassifierProtocol]
ArrayLike = NDArray[Any]
SplitResult = AxisSplit | ObliqueSplit
