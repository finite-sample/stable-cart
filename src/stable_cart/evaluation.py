# evaluation.py
import numpy as np
from typing import Dict

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression


# -------------------------------
# Prediction Stability (OOS)
# -------------------------------
def prediction_stability(
    models: Dict[str, object],
    X_oos: np.ndarray,
    task: str = "categorical"
) -> Dict[str, float]:
    """
    Measure how consistent model predictions are across models on the SAME OOS data.

    Parameters
    ----------
    models : dict[str, fitted_model]
        Mapping of model name -> fitted model (CARTs here).
    X_oos : np.ndarray
        Out-of-sample feature matrix.
    task : {'categorical','continuous'}
        Type of problem.

    Returns
    -------
    scores : dict[str, float]
        For 'categorical': average pairwise DISAGREEMENT per model (higher = less stable).
        For 'continuous' : RMSE of each model's predictions vs the ensemble mean
                           (lower = more stable).
    """
    names = list(models.keys())
    K = len(names)

    if K < 2:
        raise ValueError("Need at least 2 models to assess stability.")

    # --- CAT: pairwise disagreement (1 - agreement rate) ---
    if task == "categorical":
        preds = np.column_stack([models[n].predict(X_oos) for n in names])  # (n, K)
        # ensure numeric label space for comparisons
        if not np.issubdtype(preds.dtype, np.number):
            # map labels to integers consistently
            unique, inv = np.unique(preds, return_inverse=True)
            preds = inv.reshape(preds.shape)

        # pairwise agreement matrix A[k,l] = mean(pred_k == pred_l)
        n = preds.shape[0]
        agree = np.ones((K, K), dtype=float)
        for k in range(K):
            for l in range(k + 1, K):
                a = float(np.mean(preds[:, k] == preds[:, l]))
                agree[k, l] = agree[l, k] = a

        # per-model disagreement = average over pairs involving the model
        scores = {}
        for k, name in enumerate(names):
            # exclude self
            others = [agree[k, l] for l in range(K) if l != k]
            avg_disagree = float(np.mean([1.0 - a for a in others]))
            scores[name] = avg_disagree
        return scores

    # --- CONT: RMSE to ensemble mean ---
    elif task == "continuous":
        preds = np.column_stack([models[n].predict(X_oos) for n in names])  # (n, K)
        mean_pred = np.mean(preds, axis=1)  # ensemble mean per sample
        scores = {}
        for k, name in enumerate(names):
            err = mean_pred - preds[:, k]
            rmse = float(np.sqrt(np.mean(np.square(err))))
            scores[name] = rmse  # lower = more stable
        return scores

    else:
        raise ValueError("task must be 'categorical' or 'continuous'.")


# -------------------------------
# Accuracy / Performance
# -------------------------------
def accuracy(
    models: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    task: str = "categorical"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictive performance per model.

    Parameters
    ----------
    models : dict[str, fitted_model]
        Model name -> fitted model.
    X : np.ndarray
        Features for evaluation.
    y : np.ndarray
        Ground-truth labels/targets.
    task : {'categorical','continuous'}
        Type of problem.

    Returns
    -------
    metrics : dict[str, dict]
        For 'categorical': {'acc': ..., 'auc': ... (if available)} per model.
        For 'continuous': {'mae': ..., 'rmse': ..., 'r2': ...} per model.
    """
    results: Dict[str, Dict[str, float]] = {}

    if task == "categorical":
        y_unique = np.unique(y)
        is_binary = (len(y_unique) == 2)

        for name, mdl in models.items():
            y_hat = mdl.predict(X)
            acc = float(accuracy_score(y, y_hat))
            entry = {"acc": acc}

            # AUC if possible
            if hasattr(mdl, "predict_proba"):
                try:
                    proba = mdl.predict_proba(X)
                    if is_binary:
                        auc = float(roc_auc_score(y, proba[:, 1]))
                    else:
                        # One-vs-rest macro AUC
                        Yb = label_binarize(y, classes=y_unique)
                        auc = float(roc_auc_score(Yb, proba, average="macro", multi_class="ovr"))
                    entry["auc"] = auc
                except Exception:
                    pass

            results[name] = entry

    elif task == "continuous":
        for name, mdl in models.items():
            y_pred = mdl.predict(X)
            mae = float(mean_absolute_error(y, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            r2 = float(r2_score(y, y_pred))
            results[name] = {"mae": mae, "rmse": rmse, "r2": r2}

    else:
        raise ValueError("task must be 'categorical' or 'continuous'.")

    return results


# -------------------------------
# Minimal tests to prove it works
# -------------------------------
def _test_classification():
    """Tiny proof: two CART classifiers, OOS stability + accuracy."""
    X, y = make_classification(
        n_samples=800, n_features=12, n_informative=6, n_redundant=2,
        n_classes=3, random_state=0
    )
    X_tr, X_oos, y_tr, y_oos = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "CART_seed0": DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr),
        "CART_seed1": DecisionTreeClassifier(random_state=1).fit(X_tr, y_tr),
        "CART_seed2": DecisionTreeClassifier(random_state=2).fit(X_tr, y_tr),
    }

    print("\n=== Classification: OOS Prediction Stability (disagreement; lower is better) ===")
    stab = prediction_stability(models, X_oos, task="categorical")
    for k, (name, v) in enumerate(stab.items()):
        print(f"Model {k} ({name}) disagrees with peers on about {v*100:.1f}% of OOS samples.")

    print("\n=== Classification: OOS Accuracy (and AUC if available) ===")
    perf = accuracy(models, X_oos, y_oos, task="categorical")
    for k, (name, d) in enumerate(perf.items()):
        extra = f" with AUC={d['auc']:.3f}" if "auc" in d else ""
        print(f"Model {k} ({name}) got accuracy {d['acc']*100:.1f}%{extra}")


def _test_regression():
    """Tiny proof: two CART regressors, OOS stability + error metrics."""
    X, y = make_regression(
        n_samples=800, n_features=12, n_informative=6, noise=10.0, random_state=0
    )
    X_tr, X_oos, y_tr, y_oos = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "CART_R_seed0": DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr),
        "CART_R_seed1": DecisionTreeRegressor(random_state=1).fit(X_tr, y_tr),
        "CART_R_seed2": DecisionTreeRegressor(random_state=2).fit(X_tr, y_tr),
    }

    print("\n=== Regression: OOS Prediction Stability (RMSE to ensemble mean; lower is better) ===")
    stab = prediction_stability(models, X_oos, task="continuous")
    for k, (name, v) in enumerate(stab.items()):
        print(f"Model {k} ({name}) predictions differ from ensemble mean by RMSE={v:.2f}")

    print("\n=== Regression: OOS Error Metrics ===")
    perf = accuracy(models, X_oos, y_oos, task="continuous")
    for k, (name, d) in enumerate(perf.items()):
        print(f"Model {k} ({name}): MAE={d['mae']:.2f}, RMSE={d['rmse']:.2f}, R²={d['r2']:.3f}")


def test():
    """Run both quick demos to verify evaluation works."""
    _test_classification()
    _test_regression()


if __name__ == "__main__":
    test()
