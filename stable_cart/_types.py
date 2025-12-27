"""Type definitions and protocols for stable_cart."""

from typing import Any, Protocol

import numpy as np
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


# Type aliases for better readability
ModelDict = dict[str, PredictorProtocol]
ClassifierDict = dict[str, ClassifierProtocol]
ArrayLike = NDArray[Any]