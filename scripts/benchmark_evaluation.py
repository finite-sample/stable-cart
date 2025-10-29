"""
benchmark_evaluation.py
=======================
Unified evaluation module focusing on out-of-sample prediction variance and standard metrics.

Key focus: Bootstrap prediction variance as the primary stability metric, complemented by
standard discrimination metrics (accuracy, RMSE, etc.).
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Callable, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from stable_cart import (
    LessGreedyHybridTree,
    BootstrapVariancePenalizedTree,
    RobustPrefixHonestTree,
)


# ============================================================================
# MODEL FACTORIES
# ============================================================================


def get_unified_models(task: str, random_state: int = 42) -> Dict[str, Callable]:
    """Get factory functions for unified models that support both regression and classification."""
    if task == "regression":
        return {
            "CART": lambda: DecisionTreeRegressor(
                max_depth=6, min_samples_leaf=20, random_state=random_state
            ),
            "CART_Pruned": lambda: DecisionTreeRegressor(
                max_depth=6, min_samples_leaf=20, ccp_alpha=0.01, random_state=random_state
            ),
            "RandomForest": lambda: RandomForestRegressor(
                n_estimators=100, max_depth=6, min_samples_leaf=20, random_state=random_state
            ),
            "LessGreedyHybrid": lambda: LessGreedyHybridTree(
                task="regression",
                max_depth=6,
                min_samples_split=40,
                min_samples_leaf=20,
                split_frac=0.6,
                val_frac=0.2,
                est_frac=0.2,
                enable_oblique_root=True,
                gain_margin=0.03,
                beam_topk=12,
                leaf_shrinkage_lambda=10.0,
                random_state=random_state,
            ),
            "BootstrapVariancePenalized": lambda: BootstrapVariancePenalizedTree(
                task="regression",
                max_depth=6,
                min_samples_split=40,
                min_samples_leaf=20,
                split_frac=0.6,
                val_frac=0.2,
                est_frac=0.2,
                variance_penalty=1.0,
                n_bootstrap=10,
                beam_topk=12,
                leaf_shrinkage_lambda=10.0,
                random_state=random_state,
            ),
            "RobustPrefixHonest": lambda: RobustPrefixHonestTree(
                task="regression",
                top_levels=2,
                max_depth=6,
                min_samples_leaf=20,
                val_frac=0.2,
                est_frac=0.4,
                smoothing=1.0,
                consensus_B=12,
                random_state=random_state,
            ),
        }
    else:  # classification
        return {
            "CART": lambda: DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20, random_state=random_state
            ),
            "CART_Pruned": lambda: DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20, ccp_alpha=0.01, random_state=random_state
            ),
            "RandomForest": lambda: RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=20, random_state=random_state
            ),
            "LessGreedyHybrid": lambda: LessGreedyHybridTree(
                task="classification",
                max_depth=6,
                min_samples_split=40,
                min_samples_leaf=20,
                split_frac=0.6,
                val_frac=0.2,
                est_frac=0.2,
                enable_oblique_root=True,
                gain_margin=0.03,
                beam_topk=12,
                leaf_shrinkage_lambda=1.0,  # m-estimate smoothing for classification
                random_state=random_state,
            ),
            "BootstrapVariancePenalized": lambda: BootstrapVariancePenalizedTree(
                task="classification",
                max_depth=6,
                min_samples_split=40,
                min_samples_leaf=20,
                split_frac=0.6,
                val_frac=0.2,
                est_frac=0.2,
                variance_penalty=1.0,
                n_bootstrap=10,
                beam_topk=12,
                leaf_shrinkage_lambda=1.0,  # m-estimate smoothing for classification
                random_state=random_state,
            ),
            "RobustPrefixHonest": lambda: RobustPrefixHonestTree(
                task="classification",
                top_levels=2,
                max_depth=6,
                min_samples_leaf=20,
                val_frac=0.2,
                est_frac=0.4,
                smoothing=1.0,
                consensus_B=12,
                random_state=random_state,
            ),
        }


# ============================================================================
# BOOTSTRAP PREDICTION VARIANCE (PRIMARY STABILITY METRIC)
# ============================================================================


def bootstrap_prediction_variance(
    model_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_bootstrap: int = 20,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Measure bootstrap prediction variance - primary stability metric.

    Trains multiple models on bootstrap samples of training data and measures
    prediction variance on the same test set. Lower variance = more stable.

    Parameters
    ----------
    model_factory : Callable
        Function that returns a fresh model instance
    X_train, y_train : np.ndarray
        Training data
    X_test : np.ndarray
        Test features to predict on
    n_bootstrap : int, default=20
        Number of bootstrap samples
    random_state : int, default=42
        Random seed

    Returns
    -------
    metrics : Dict[str, float]
        - 'pred_variance_mean': Mean prediction variance across test samples
        - 'pred_variance_p90': 90th percentile prediction variance
        - 'pred_variance_max': Maximum prediction variance
    """
    rng = np.random.default_rng(random_state)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    predictions = np.zeros((n_bootstrap, n_test))

    for b in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_idx = rng.integers(0, n_train, n_train)
        X_boot = X_train[bootstrap_idx]
        y_boot = y_train[bootstrap_idx]

        # Train model and predict
        model = model_factory()
        model.fit(X_boot, y_boot)

        # For classification, use probabilities if available
        if hasattr(model, "predict_proba") and len(np.unique(y_train)) == 2:
            # Binary classification - use probability of positive class
            pred = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "predict_proba") and len(np.unique(y_train)) > 2:
            # Multi-class - use max probability
            proba = model.predict_proba(X_test)
            pred = np.max(proba, axis=1)
        else:
            # Regression or classification without probabilities
            pred = model.predict(X_test)

        predictions[b] = pred

    # Compute variance across bootstrap samples for each test point
    point_variances = np.var(predictions, axis=0, ddof=1)

    return {
        "pred_variance_mean": float(np.mean(point_variances)),
        "pred_variance_p90": float(np.percentile(point_variances, 90)),
        "pred_variance_max": float(np.max(point_variances)),
    }


# ============================================================================
# STANDARD DISCRIMINATION METRICS
# ============================================================================


def evaluate_discrimination_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metrics."""
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_discrimination_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

    # Add AUC if probabilities available
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                # Multi-class
                metrics["auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            # Skip AUC if computation fails
            pass

    return metrics


# ============================================================================
# MODEL CHARACTERISTICS
# ============================================================================


def get_model_characteristics(model: Any) -> Dict[str, float]:
    """Extract model characteristics like size, training time, etc."""
    characteristics = {}

    try:
        # Tree size for tree-based models
        if hasattr(model, "tree_"):
            # sklearn DecisionTree
            characteristics["n_leaves"] = int(np.sum(model.tree_.children_left == -1))
            characteristics["tree_depth"] = int(model.tree_.max_depth)
        elif hasattr(model, "count_leaves"):
            # stable_cart models
            characteristics["n_leaves"] = int(model.count_leaves())
            characteristics["tree_depth"] = getattr(model, "max_depth", np.nan)
        elif hasattr(model, "n_estimators"):
            # Random Forest
            characteristics["n_estimators"] = int(model.n_estimators)
            # Average leaves across trees
            if hasattr(model, "estimators_"):
                avg_leaves = np.mean(
                    [np.sum(tree.tree_.children_left == -1) for tree in model.estimators_]
                )
                characteristics["avg_n_leaves"] = float(avg_leaves)

        # Training time (if recorded)
        if hasattr(model, "fit_time_sec_"):
            characteristics["fit_time_sec"] = float(model.fit_time_sec_)
    except Exception:
        # Silently handle any issues with characteristic extraction
        pass

    return characteristics


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================


def evaluate_single_model(
    model_name: str,
    model_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    n_bootstrap: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a single model.

    Returns
    -------
    results : Dict[str, Any]
        Combined stability, discrimination, and characteristic metrics
    """
    start_time = time.time()

    # Train model for discrimination evaluation
    model = model_factory()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    # Predictions for discrimination metrics
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    # 1. Bootstrap prediction variance (stability)
    stability_metrics = bootstrap_prediction_variance(
        model_factory, X_train, y_train, X_test, n_bootstrap, random_state
    )

    # 2. Discrimination metrics
    if task == "regression":
        discrimination_metrics = evaluate_discrimination_regression(y_test, y_pred)
    else:
        discrimination_metrics = evaluate_discrimination_classification(y_test, y_pred, y_proba)

    # 3. Model characteristics
    characteristics = get_model_characteristics(model)
    characteristics["fit_time_sec"] = fit_time

    # Combine all metrics
    results = {
        "model": model_name,
        **stability_metrics,
        **discrimination_metrics,
        **characteristics,
    }

    return results


def evaluate_dataset(
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    models_to_run: Optional[List[str]] = None,
    n_bootstrap: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate all models on a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    X_train, y_train, X_test, y_test : np.ndarray
        Train/test splits
    task : str
        'regression' or 'classification'
    models_to_run : List[str], optional
        Subset of models to evaluate (default: all)
    n_bootstrap : int, default=20
        Bootstrap samples for stability measurement
    random_state : int, default=42
        Random seed

    Returns
    -------
    results_df : pd.DataFrame
        Results for all models on this dataset
    """
    print(f"  Evaluating dataset: {dataset_name}")

    # Get unified model factories
    model_factories = get_unified_models(task, random_state)

    # Filter models if specified
    if models_to_run:
        model_factories = {k: v for k, v in model_factories.items() if k in models_to_run}

    results = []

    for model_name, factory in model_factories.items():
        print(f"    - {model_name:25s}", end=" ", flush=True)

        try:
            start_time = time.time()
            model_results = evaluate_single_model(
                model_name,
                factory,
                X_train,
                y_train,
                X_test,
                y_test,
                task,
                n_bootstrap,
                random_state,
            )
            eval_time = time.time() - start_time

            # Add dataset and task info
            model_results["dataset"] = dataset_name
            model_results["task"] = task
            model_results["eval_time_sec"] = eval_time

            results.append(model_results)
            print(f"✓ ({eval_time:.1f}s)")

        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            # Add failed entry with NaN values
            results.append(
                {"dataset": dataset_name, "model": model_name, "task": task, "error": str(e)}
            )

    return pd.DataFrame(results)


# ============================================================================
# CROSS-VALIDATION STABILITY (OPTIONAL)
# ============================================================================


def cross_validation_stability(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Alternative stability metric: prediction variance across CV folds.

    Less comprehensive than bootstrap but faster for initial screening.
    """
    if task == "regression":
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_predictions = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        model = model_factory()
        model.fit(X_train_cv, y_train_cv)

        if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
            pred = model.predict_proba(X_val_cv)[:, 1]
        else:
            pred = model.predict(X_val_cv)

        fold_predictions.append(pred)

    # Compute variance in predictions across folds
    # Note: This is approximate since validation sets differ
    all_preds = np.concatenate(fold_predictions)
    cv_variance = float(np.var(all_preds, ddof=1))

    return {"cv_pred_variance": cv_variance}


if __name__ == "__main__":
    # Quick test with a simple dataset
    from sklearn.datasets import make_classification

    print("Testing benchmark evaluation module...")

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = X[:350], X[350:], y[:350], y[350:]

    # Test single model evaluation
    factory = lambda: DecisionTreeClassifier(max_depth=5, random_state=42)
    results = evaluate_single_model(
        "TestCART", factory, X_train, y_train, X_test, y_test, "classification", n_bootstrap=5
    )

    print("\nSample results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
