import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Project modules
import evalutation as ev  # file name intentionally "evalutation.py"
from less_greedy_tree import LessGreedyHybridRegressor, GreedyCARTExact


@pytest.mark.e2e
@pytest.mark.slow
def test_regression_end_to_end(tmp_path):
    """
    Train two models on synthetic regression data, predict, evaluate, and
    write artifacts: predictions.csv, metrics.json, stability.json, plot.png
    """
    rng = np.random.default_rng(7)
    X, y = make_regression(n_samples=1200, n_features=16, noise=12.0, random_state=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7)

    models = {
        "less_greedy": LessGreedyHybridRegressor(
            max_depth=6, min_samples_split=40, min_samples_leaf=20, random_state=7
        ),
        "cart": GreedyCARTExact(max_depth=6, min_samples_split=40, min_samples_leaf=20),
        "dtr": DecisionTreeRegressor(max_depth=8, random_state=7),
    }
    for m in models.values():
        m.fit(Xtr, ytr)

    # Predict + persist predictions
    preds = {name: m.predict(Xte) for name, m in models.items()}
    df_pred = pd.DataFrame({"y_true": yte, **{f"pred_{k}": v for k, v in preds.items()}})
    pred_path = tmp_path / "predictions.csv"
    df_pred.to_csv(pred_path, index=False)

    # Accuracy-style metrics for continuous task
    perf = ev.accuracy(models, Xte, yte, task="continuous")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(perf, indent=2))

    # Stability (RMSE to ensemble mean)
    stab = ev.prediction_stability(models, Xte, task="continuous")
    stab_path = tmp_path / "stability.json"
    stab_path.write_text(json.dumps(stab, indent=2))

    # Quick (optional) plot artifact without relying on seaborn
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(yte, preds["less_greedy"], s=6, alpha=0.7)
        ax.set_xlabel("y_true")
        ax.set_ylabel("pred_less_greedy")
        ax.set_title("Regression: y_true vs prediction")
        plot_path = tmp_path / "scatter.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        assert plot_path.exists() and plot_path.stat().st_size > 0
    except Exception:  # plotting is optional; don't fail the whole E2E
        pass

    # Assertions – artifacts exist and contain sane values
    assert pred_path.exists() and pred_path.stat().st_size > 0
    assert metrics_path.exists() and metrics_path.stat().st_size > 0
    assert stab_path.exists() and stab_path.stat().st_size > 0

    # Sanity: RMSE/MAE should be non-negative finite; R2 should be finite
    for name, d in perf.items():
        assert np.isfinite(d["rmse"]) and d["rmse"] >= 0
        assert np.isfinite(d["mae"]) and d["mae"] >= 0
        assert np.isfinite(d["r2"])

    # Sanity: stability scores non-negative finite
    for v in stab.values():
        assert np.isfinite(v) and v >= 0.0


@pytest.mark.e2e
@pytest.mark.slow
def test_classification_end_to_end(tmp_path):
    """
    Classification flow with accuracy & AUC plus disagreement stability.
    """
    X, y = make_classification(
        n_samples=1400, n_features=20, n_informative=8, class_sep=1.2, random_state=11
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=11)

    models = {
        "tree_d3": DecisionTreeClassifier(max_depth=3, random_state=1).fit(Xtr, ytr),
        "tree_d5": DecisionTreeClassifier(max_depth=5, random_state=2).fit(Xtr, ytr),
    }

    # Predictions CSV
    preds = {name: m.predict(Xte) for name, m in models.items()}
    df = pd.DataFrame({"y_true": yte, **{f"pred_{k}": v for k, v in preds.items()}})
    out = tmp_path / "cls_predictions.csv"
    df.to_csv(out, index=False)

    # Accuracy dict (expects accuracy+auc keys)
    acc = ev.accuracy(models, Xte, yte, task="categorical")
    acc_path = tmp_path / "cls_metrics.json"
    acc_path.write_text(json.dumps(acc, indent=2))

    # Stability (pairwise disagreement vs ensemble mode/mean)
    stab = ev.prediction_stability(models, Xte, task="categorical")
    stab_path = tmp_path / "cls_stability.json"
    stab_path.write_text(json.dumps(stab, indent=2))

    assert out.exists() and out.stat().st_size > 0
    assert acc_path.exists() and acc_path.stat().st_size > 0
    assert stab_path.exists() and stab_path.stat().st_size > 0

    # Sanity: accuracy in [0,1], auc in [0,1]
    for d in acc.values():
        assert 0.0 <= d["accuracy"] <= 1.0
        if "auc" in d and d["auc"] is not None:
            assert 0.0 <= d["auc"] <= 1.0

    # Stability disagreement ∈ [0,1]
    for v in stab.values():
        assert 0.0 <= v <= 1.0