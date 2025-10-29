"""
benchmark_less_greedy.py
------------------------
Benchmark LessGreedyHybridRegressor against sklearn DecisionTree baselines:
  - CART_PRUNED(CV-min alpha)
  - CART_PRUNED_1SE (simplest tree within 1SE of CV-min)
Optionally include DecisionTreeRegressor as a sanity axis-aligned baseline.

Outputs CSVs with accuracy, size, and stability metrics.
"""

import os, time, math, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_friedman1

from stable_cart import LessGreedyHybridRegressor


# -------- Datasets --------
def dgp_quadrant(n=3000, noise=0.5, rs=42):
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, 6))
    y = np.zeros(n)
    m1 = (X[:, 0] > 0) & (X[:, 1] > 0)
    y[m1] = 5.0 + 0.5 * X[m1, 2]
    m2 = (X[:, 0] > 0) & (X[:, 1] <= 0)
    y[m2] = -2.0 + 0.3 * X[m2, 3]
    m3 = (X[:, 0] <= 0) & (X[:, 1] > 0)
    y[m3] = 2.0 - 0.4 * X[m3, 4]
    m4 = (X[:, 0] <= 0) & (X[:, 1] <= 0)
    y[m4] = -5.0 + 0.6 * X[m4, 5]
    y += 0.2 * X[:, 2] * X[:, 3] + 0.1 * np.sin(2 * X[:, 4])
    y += rng.normal(0, noise, n)
    return X, y


def dgp_friedman1(n=3000, noise=1.0, rs=1):
    X, y = make_friedman1(n_samples=n, n_features=10, noise=noise, random_state=rs)
    return X, y


def dgp_correlated_linear(n=3000, noise=1.0, rs=7):
    rng = np.random.default_rng(rs)
    p = 20
    rho = 0.7
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, p))
    X = Z @ L.T
    beta = np.zeros(p)
    beta[:5] = [3.0, -2.0, 1.5, 1.0, -1.0]
    y = X @ beta + 2.0 * np.sin(X[:, 0]) - 1.5 * np.cos(X[:, 1]) + 0.5 * X[:, 2] * X[:, 3]
    eps = np.random.default_rng(rs + 1).normal(0, noise * (1 + 0.5 * np.abs(X[:, 0])), size=n)
    return X, y + eps


def dgp_xor_checker(n=3000, p=8, noise=0.7, rs=11):
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, p))
    core = np.where((X[:, 0] > 0) ^ (X[:, 1] > 0), 3.0, -3.0)
    y = core + 0.4 * X[:, 2] - 0.3 * X[:, 3] + 0.2 * np.sin(X[:, 4]) + rng.normal(0, noise, n)
    return X, y


def dgp_piecewise_hinge(n=3000, p=10, noise=0.8, rs=13):
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, p))
    y = (
        2.0 * np.maximum(X[:, 0], 0)
        - 1.5 * np.maximum(-X[:, 1], 0)
        + 0.6 * X[:, 2] * X[:, 3]
        + 0.4 * np.tanh(X[:, 4])
    )
    y += rng.normal(0, noise, n)
    return X, y


DGPS = {
    "quadrant_interaction": dgp_quadrant,
    "friedman1": dgp_friedman1,
    "correlated_linear": dgp_correlated_linear,
    "xor_checker": dgp_xor_checker,
    "piecewise_hinge": dgp_piecewise_hinge,
}


# -------- Baseline helpers --------
def cv_cart_minalpha(X, y, folds=3, depths=(4, 5), leaves=(10, 20), max_alphas=20, rs=0):
    kf = KFold(n_splits=folds, shuffle=True, random_state=303 + rs)
    best = (np.inf, None)
    for d in depths:
        for lf in leaves:
            base = DecisionTreeRegressor(max_depth=d, min_samples_leaf=lf, random_state=rs)
            path = base.cost_complexity_pruning_path(X, y)
            alphas = path.ccp_alphas
            if alphas.size == 0:
                alphas = np.array([0.0])
            if alphas.size > max_alphas:
                idx = np.linspace(0, alphas.size - 1, max_alphas, dtype=int)
                alphas = alphas[idx]
            for a in alphas:
                fold = []
                for tr, va in kf.split(X):
                    m = DecisionTreeRegressor(
                        max_depth=d, min_samples_leaf=lf, ccp_alpha=float(a), random_state=rs
                    ).fit(X[tr], y[tr])
                    fold.append(mean_squared_error(y[va], m.predict(X[va])))
                mse = float(np.mean(fold))
                if mse < best[0]:
                    best = (mse, {"max_depth": d, "min_samples_leaf": lf, "ccp_alpha": float(a)})
    return best


def fit_cart_1se(X, y, folds=3, depths=(4, 5), leaves=(10, 20), max_alphas=25, rs=0):
    kf = KFold(n_splits=folds, shuffle=True, random_state=404 + rs)
    best = None
    for d in depths:
        for lf in leaves:
            base = DecisionTreeRegressor(max_depth=d, min_samples_leaf=lf, random_state=rs)
            path = base.cost_complexity_pruning_path(X, y)
            alphas = path.ccp_alphas
            if alphas.size == 0:
                alphas = np.array([0.0])
            if alphas.size > max_alphas:
                idx = np.linspace(0, alphas.size - 1, max_alphas, dtype=int)
                alphas = alphas[idx]
            means, stds = [], []
            for a in alphas:
                fold = []
                for tr, va in kf.split(X):
                    m = DecisionTreeRegressor(
                        max_depth=d, min_samples_leaf=lf, ccp_alpha=float(a), random_state=rs
                    ).fit(X[tr], y[tr])
                    fold.append(mean_squared_error(y[va], m.predict(X[va])))
                means.append(np.mean(fold))
                stds.append(np.std(fold, ddof=1))
            means = np.array(means)
            se = np.array(stds) / np.sqrt(folds)
            min_idx = int(np.argmin(means))
            thr = float(means[min_idx] + se[min_idx])
            feas = np.where(means <= thr)[0]
            chosen = int(feas[-1] if feas.size > 0 else min_idx)  # simplest among feasible
            rec = float(means[chosen])
            if (best is None) or (rec < best[0]):
                best = (
                    rec,
                    {"max_depth": d, "min_samples_leaf": lf, "ccp_alpha": float(alphas[chosen])},
                )
    p = best[1]
    model = DecisionTreeRegressor(
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        ccp_alpha=p["ccp_alpha"],
        random_state=rs,
    ).fit(X, y)
    return model, p


def sklearn_leaf_count(model: DecisionTreeRegressor) -> int:
    t = model.tree_
    return int(np.sum(t.children_left == -1))


# -------- Stability (bootstrap train; same OOS) --------
def oos_stability_summary(model_factory, Xtr, ytr, Xte, B=12, rs=0):
    rng = np.random.default_rng(rs)
    ntr = Xtr.shape[0]
    preds = []
    for b in range(B):
        idx = rng.integers(0, ntr, ntr)
        m = model_factory()
        m.fit(Xtr[idx], ytr[idx])
        preds.append(m.predict(Xte))
    P = np.vstack(preds)
    per_point_std = P.std(axis=0)
    return float(per_point_std.mean()), float(np.percentile(per_point_std, 90))


# --------- Runner ---------
def run_benchmark(select=None, out_dir="./bench_out", B_stab=12, rs=0):
    if select is None:
        select = list(DGPS.keys())
    os.makedirs(out_dir, exist_ok=True)
    rows, stab_rows = [], []

    for name in select:
        X, y = DGPS[name]()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

        # SKLEARN_DT(CV) capacity
        # Small grid: (4,10), (5,20)
        grid = [(4, 10), (5, 20)]
        best = (np.inf, None, None)
        for d, lf in grid:
            m = DecisionTreeRegressor(
                max_depth=d, min_samples_split=20, min_samples_leaf=lf, random_state=rs
            ).fit(Xtr, ytr)
            mse = mean_squared_error(yte, m.predict(Xte))
            if mse < best[0]:
                best = (mse, d, lf)
        d_best, lf_best = best[1], best[2]
        g = DecisionTreeRegressor(
            max_depth=d_best, min_samples_split=20, min_samples_leaf=lf_best, random_state=rs
        ).fit(Xtr, ytr)
        pred_g = g.predict(Xte)
        leaves_g = sklearn_leaf_count(g)

        # CART_PRUNED(CV-min alpha)
        (_, p_min) = cv_cart_minalpha(Xtr, ytr, rs=rs)
        pruned_cv = DecisionTreeRegressor(
            max_depth=p_min["max_depth"],
            min_samples_leaf=p_min["min_samples_leaf"],
            ccp_alpha=p_min["ccp_alpha"],
            random_state=rs,
        ).fit(Xtr, ytr)
        pred_cv = pruned_cv.predict(Xte)
        leaves_cv = sklearn_leaf_count(pruned_cv)

        # CART_PRUNED_1SE
        pruned_1se, p_1se = fit_cart_1se(Xtr, ytr, rs=rs)
        pred_1se = pruned_1se.predict(Xte)
        leaves_1se = sklearn_leaf_count(pruned_1se)

        # LESS_GREEDY_HYBRID (stability oriented defaults)
        hybrid = LessGreedyHybridRegressor(
            max_depth=5,
            min_samples_split=40,
            min_samples_leaf=20,
            split_frac=0.6,
            val_frac=0.2,
            est_frac=0.2,
            enable_oblique_root=True,
            gain_margin=0.03,
            min_abs_corr=0.3,
            oblique_cv=5,
            beam_topk=12,
            ambiguity_eps=0.05,
            min_n_for_lookahead=600,
            root_k=2,
            inner_k=1,
            leaf_shrinkage_lambda=10.0,
            random_state=rs,
        ).fit(Xtr, ytr)
        pred_h = hybrid.predict(Xte)
        leaves_h = hybrid.count_leaves()

        # accuracy rows
        for model_name, pred, leaves in [
            ("GREEDY_EXACT(CV)", pred_g, leaves_g),
            ("CART_PRUNED(CV)", pred_cv, leaves_cv),
            ("CART_PRUNED_1SE", pred_1se, leaves_1se),
            ("LESS_GREEDY_HYBRID", pred_h, leaves_h),
        ]:
            rows.append(
                {
                    "Dataset": name,
                    "Model": model_name,
                    "MSE": mean_squared_error(yte, pred),
                    "R2": r2_score(yte, pred),
                    "Leaves": int(leaves),
                }
            )

        # stability rows (bootstrap on train)
        factories = {
            "SKLEARN_DT(CV)": lambda: DecisionTreeRegressor(
                max_depth=d_best, min_samples_split=20, min_samples_leaf=lf_best, random_state=rs
            ),
            "CART_PRUNED(CV)": lambda: DecisionTreeRegressor(
                max_depth=p_min["max_depth"],
                min_samples_leaf=p_min["min_samples_leaf"],
                ccp_alpha=p_min["ccp_alpha"],
                random_state=rs,
            ),
            "CART_PRUNED_1SE": lambda: DecisionTreeRegressor(
                max_depth=p_1se["max_depth"],
                min_samples_leaf=p_1se["min_samples_leaf"],
                ccp_alpha=p_1se["ccp_alpha"],
                random_state=rs,
            ),
            "LESS_GREEDY_HYBRID": lambda: LessGreedyHybridRegressor(
                max_depth=5,
                min_samples_split=40,
                min_samples_leaf=20,
                split_frac=0.6,
                val_frac=0.2,
                est_frac=0.2,
                enable_oblique_root=True,
                gain_margin=0.03,
                min_abs_corr=0.3,
                oblique_cv=5,
                beam_topk=12,
                ambiguity_eps=0.05,
                min_n_for_lookahead=600,
                root_k=2,
                inner_k=1,
                leaf_shrinkage_lambda=10.0,
                random_state=rs,
            ),
        }
        for k, fac in factories.items():
            mstd, p90 = oos_stability_summary(fac, Xtr, ytr, Xte, B=12, rs=123)
            stab_rows.append(
                {"Dataset": name, "Model": k, "mean_pred_std": mstd, "p90_pred_std": p90}
            )

    results = pd.DataFrame(rows)
    stability = pd.DataFrame(stab_rows)
    # save
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    stability.to_csv(os.path.join(out_dir, "stability.csv"), index=False)
    # return for notebook use
    return results, stability


if __name__ == "__main__":
    out_dir = "./bench_out"
    res, stab = run_benchmark(out_dir=out_dir)
    print("Wrote:", os.path.join(out_dir, "results.csv"))
    print("Wrote:", os.path.join(out_dir, "stability.csv"))
