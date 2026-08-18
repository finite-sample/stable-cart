"""Shared validation utilities."""

from typing import Any, cast

from sklearn.utils.validation import check_is_fitted, validate_data


def check_predict_input(estimator: Any, X: Any, fitted_attribute: str) -> Any:
    """Check fitted metadata while preserving the selected estimator's input type."""
    check_is_fitted(estimator, fitted_attribute)
    is_dataframe = hasattr(X, "columns") and hasattr(X, "iloc")
    checked = cast(Any, validate_data)(
        estimator,
        X=X,
        dtype=None if is_dataframe else "numeric",
        accept_sparse=False,
        reset=False,
    )
    return X if is_dataframe else checked
