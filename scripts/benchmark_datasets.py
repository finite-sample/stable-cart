"""
benchmark_datasets.py
=====================
Comprehensive dataset collection for benchmarking tree algorithms.

Provides both real-world and synthetic datasets for:
- Regression tasks (continuous targets)
- Classification tasks (categorical targets)

Focus on datasets that highlight prediction stability differences between algorithms.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_friedman1,
    make_friedman2,
    make_friedman3,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# SYNTHETIC REGRESSION DATASETS
# ============================================================================


def friedman1_dataset(
    n_samples: int = 3000, noise: float = 1.0, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Friedman #1: y = 10 sin(πx₁x₂) + 20(x₃-0.5)² + 10x₄ + 5x₅ + ε
    Standard nonlinear benchmark with clear feature importance hierarchy.
    """
    X, y = make_friedman1(
        n_samples=n_samples, n_features=10, noise=noise, random_state=random_state
    )
    return X, y


def friedman2_dataset(
    n_samples: int = 3000, noise: float = 1.0, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Friedman #2: y = (x₁² + (x₂x₃ - 1/(x₂x₄))²)^0.5 + ε
    Highly nonlinear with multiplicative interactions.
    """
    X, y = make_friedman2(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def friedman3_dataset(
    n_samples: int = 3000, noise: float = 1.0, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Friedman #3: y = arctan((x₂x₃ - 1/(x₂x₄))/x₁) + ε
    Bounded target with complex interactions.
    """
    X, y = make_friedman3(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def quadrant_interaction_dataset(
    n_samples: int = 3000, noise: float = 0.5, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quadrant-based interaction: ideal for tree methods.
    Clear region-specific patterns that trees can capture well.
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, 6))
    y = np.zeros(n_samples)

    # Define quadrants based on first two features
    q1 = (X[:, 0] > 0) & (X[:, 1] > 0)
    q2 = (X[:, 0] > 0) & (X[:, 1] <= 0)
    q3 = (X[:, 0] <= 0) & (X[:, 1] > 0)
    q4 = (X[:, 0] <= 0) & (X[:, 1] <= 0)

    # Quadrant-specific linear relationships
    y[q1] = 5.0 + 0.5 * X[q1, 2]
    y[q2] = -2.0 + 0.3 * X[q2, 3]
    y[q3] = 2.0 - 0.4 * X[q3, 4]
    y[q4] = -5.0 + 0.6 * X[q4, 5]

    # Add interactions and nonlinearity
    y += 0.2 * X[:, 2] * X[:, 3] + 0.1 * np.sin(2 * X[:, 4])
    y += rng.normal(0, noise, n_samples)

    return X, y


def high_dimensional_sparse_dataset(
    n_samples: int = 3000,
    n_features: int = 50,
    n_informative: int = 5,
    noise: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    High-dimensional sparse signal: many irrelevant features.
    Tests feature selection and overfitting resistance.
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))

    # Only first n_informative features are relevant
    beta = np.zeros(n_features)
    beta[:n_informative] = rng.uniform(-3, 3, n_informative)

    # Linear combination + nonlinear transforms
    y = X @ beta
    y += 2.0 * np.sin(X[:, 0]) - 1.5 * np.cos(X[:, 1])
    if n_informative >= 3:
        y += 0.5 * X[:, 2] * X[:, min(3, n_informative - 1)]

    # Add noise
    y += rng.normal(0, noise, n_samples)

    return X, y


def heteroscedastic_dataset(
    n_samples: int = 3000, noise_scale: float = 1.0, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heteroscedastic noise: variance depends on features.
    Tests robustness to non-constant noise.
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, 8))

    # Base signal
    y = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.8 * X[:, 2] * X[:, 3]
        + 0.5 * np.tanh(X[:, 4])
        - 0.3 * np.abs(X[:, 5])
    )

    # Heteroscedastic noise: variance proportional to |X[:, 0]|
    noise_std = noise_scale * (1.0 + 0.8 * np.abs(X[:, 0]))
    y += rng.normal(0, 1, n_samples) * noise_std

    return X, y


def xor_nonlinear_dataset(
    n_samples: int = 3000, n_features: int = 8, noise: float = 0.7, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    XOR pattern with additional nonlinear features.
    Tests interaction detection capability.
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))

    # Core XOR pattern
    xor_signal = np.where((X[:, 0] > 0) ^ (X[:, 1] > 0), 3.0, -3.0)

    # Additional continuous features
    y = xor_signal + 0.4 * X[:, 2] - 0.3 * X[:, 3] + 0.2 * np.sin(X[:, 4])
    y += rng.normal(0, noise, n_samples)

    return X, y


# ============================================================================
# REAL-WORLD REGRESSION DATASETS
# ============================================================================


def california_housing_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """California housing prices: medium-scale real dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def diabetes_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Diabetes progression: small medical dataset."""
    data = load_diabetes()
    X, y = data.data, data.target

    # Already standardized
    return X, y


# ============================================================================
# REAL-WORLD CLASSIFICATION DATASETS
# ============================================================================


def breast_cancer_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Breast cancer detection: binary classification."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def wine_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Wine classification: 3-class problem with continuous features."""
    data = load_wine()
    X, y = data.data, data.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def iris_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Iris classification: classic small 3-class dataset."""
    data = load_iris()
    X, y = data.data, data.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def digits_binary_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Digits 0 vs rest: binary classification from multi-class."""
    data = load_digits()
    X, y = data.data, (data.target == 0).astype(int)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def digits_multiclass_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Digits multi-class: 10-class classification."""
    data = load_digits()
    X, y = data.data, data.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# ============================================================================
# DATASET REGISTRY
# ============================================================================

REGRESSION_DATASETS: dict[str, Callable] = {
    "friedman1": friedman1_dataset,
    "friedman2": friedman2_dataset,
    "friedman3": friedman3_dataset,
    "quadrant_interaction": quadrant_interaction_dataset,
    "high_dim_sparse": high_dimensional_sparse_dataset,
    "heteroscedastic": heteroscedastic_dataset,
    "xor_nonlinear": xor_nonlinear_dataset,
    "california_housing": california_housing_dataset,
    "diabetes": diabetes_dataset,
}

CLASSIFICATION_DATASETS: dict[str, Callable] = {
    "breast_cancer": breast_cancer_dataset,
    "wine": wine_dataset,
    "iris": iris_dataset,
    "digits_binary": digits_binary_dataset,
    "digits_multiclass": digits_multiclass_dataset,
}

ALL_DATASETS = {**REGRESSION_DATASETS, **CLASSIFICATION_DATASETS}


# ============================================================================
# DATASET UTILITIES
# ============================================================================


def get_dataset_info() -> pd.DataFrame:
    """Get information about all available datasets."""
    info = []

    for name, func in REGRESSION_DATASETS.items():
        X, y = func()
        info.append(
            {
                "name": name,
                "task": "regression",
                "n_samples": len(X),
                "n_features": X.shape[1],
                "target_type": "continuous",
                "description": (
                    func.__doc__.split("\n")[1].strip()
                    if func.__doc__ and len(func.__doc__.split("\n")) > 1
                    else func.__doc__.strip() if func.__doc__ else ""
                ),
            }
        )

    for name, func in CLASSIFICATION_DATASETS.items():
        X, y = func()
        n_classes = len(np.unique(y))
        info.append(
            {
                "name": name,
                "task": "classification",
                "n_samples": len(X),
                "n_features": X.shape[1],
                "target_type": f"{n_classes}-class",
                "description": (
                    func.__doc__.split("\n")[1].strip()
                    if func.__doc__ and len(func.__doc__.split("\n")) > 1
                    else func.__doc__.strip() if func.__doc__ else ""
                ),
            }
        )

    return pd.DataFrame(info)


def load_dataset(
    name: str, test_size: float = 0.3, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load a dataset and split into train/test.

    Parameters
    ----------
    name : str
        Dataset name from ALL_DATASETS registry
    test_size : float, default=0.3
        Fraction of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Train and test splits
    task : str
        'regression' or 'classification'
    """
    if name not in ALL_DATASETS:
        available = list(ALL_DATASETS.keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available}")

    X, y = ALL_DATASETS[name](random_state=random_state)

    # Determine task type
    task = "regression" if name in REGRESSION_DATASETS else "classification"

    # Stratify for classification tasks
    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return X_train, X_test, y_train, y_test, task


def get_dataset_recommendations() -> dict[str, list]:
    """Get recommended dataset subsets for different benchmark scenarios."""
    return {
        "quick": ["friedman1", "quadrant_interaction", "breast_cancer", "iris"],
        "comprehensive": list(ALL_DATASETS.keys()),
        "regression_focus": list(REGRESSION_DATASETS.keys()),
        "classification_focus": list(CLASSIFICATION_DATASETS.keys()),
        "stability_showcase": [
            "quadrant_interaction",
            "heteroscedastic",
            "xor_nonlinear",
            "breast_cancer",
            "digits_binary",
        ],
        "real_world_only": [
            "california_housing",
            "diabetes",
            "breast_cancer",
            "wine",
            "iris",
            "digits_multiclass",
        ],
    }


if __name__ == "__main__":
    # Print dataset information
    print("Available Benchmark Datasets")
    print("=" * 50)
    info_df = get_dataset_info()
    print(info_df.to_string(index=False))

    print("\n\nDataset Recommendations")
    print("=" * 30)
    recommendations = get_dataset_recommendations()
    for scenario, datasets in recommendations.items():
        print(f"{scenario}: {', '.join(datasets)}")
