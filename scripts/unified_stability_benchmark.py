"""
unified_stability_benchmark.py
==============================
Comprehensive benchmark comparing tree algorithms on BOTH predictive accuracy
and out-of-sample prediction stability across multiple synthetic datasets.

Metrics:
- Accuracy: MSE, R2
- Stability: Bootstrap prediction variance (lower = more stable)
- Efficiency: Tree size (leaves), fit time

Models compared:
1. CART (sklearn baseline with CV-tuned depth)
2. CART_PRUNED (cost-complexity pruning, CV-optimal alpha)
3. CART_PRUNED_1SE (1-SE rule for simpler tree)
4. LessGreedyHybrid (honest splits + lookahead + oblique root)
5. BootstrapVariancePenalized (explicitly penalizes variance)

Usage:
    python unified_stability_benchmark.py --output bench_results
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Callable, Tuple
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_friedman1

from stable_cart import LessGreedyHybridRegressor, BootstrapVariancePenalizedRegressor


# ============================================================================
# SYNTHETIC DATA GENERATORS
# ============================================================================

def dgp_quadrant_interaction(n=3000, noise=0.5, rs=42) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrant-based interaction with continuous features."""
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, 6))
    y = np.zeros(n)
    
    # Quadrants
    m1 = (X[:, 0] > 0) & (X[:, 1] > 0)
    m2 = (X[:, 0] > 0) & (X[:, 1] <= 0)
    m3 = (X[:, 0] <= 0) & (X[:, 1] > 0)
    m4 = (X[:, 0] <= 0) & (X[:, 1] <= 0)
    
    y[m1] = 5.0 + 0.5 * X[m1, 2]
    y[m2] = -2.0 + 0.3 * X[m2, 3]
    y[m3] = 2.0 - 0.4 * X[m3, 4]
    y[m4] = -5.0 + 0.6 * X[m4, 5]
    
    # Add interactions and noise
    y += 0.2 * X[:, 2] * X[:, 3] + 0.1 * np.sin(2 * X[:, 4])
    y += rng.normal(0, noise, n)
    return X, y


def dgp_friedman1(n=3000, noise=1.0, rs=1) -> Tuple[np.ndarray, np.ndarray]:
    """Friedman #1 benchmark: nonlinear additive function."""
    X, y = make_friedman1(n_samples=n, n_features=10, noise=noise, random_state=rs)
    return X, y


def dgp_correlated_linear(n=3000, noise=1.0, rs=7) -> Tuple[np.ndarray, np.ndarray]:
    """Correlated features with nonlinear components."""
    rng = np.random.default_rng(rs)
    p = 20
    rho = 0.7
    
    # Create correlation matrix
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, p))
    X = Z @ L.T
    
    # Sparse linear + nonlinear
    beta = np.zeros(p)
    beta[:5] = [3.0, -2.0, 1.5, 1.0, -1.0]
    y = X @ beta + 2.0 * np.sin(X[:, 0]) - 1.5 * np.cos(X[:, 1]) + 0.5 * X[:, 2] * X[:, 3]
    
    # Heteroscedastic noise
    eps = np.random.default_rng(rs + 1).normal(0, noise * (1 + 0.5 * np.abs(X[:, 0])), size=n)
    return X, y + eps


def dgp_xor_checker(n=3000, p=8, noise=0.7, rs=11) -> Tuple[np.ndarray, np.ndarray]:
    """XOR pattern with additional features."""
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, p))
    
    # XOR core
    core = np.where((X[:, 0] > 0) ^ (X[:, 1] > 0), 3.0, -3.0)
    y = core + 0.4 * X[:, 2] - 0.3 * X[:, 3] + 0.2 * np.sin(X[:, 4])
    y += rng.normal(0, noise, n)
    return X, y


def dgp_piecewise_hinge(n=3000, p=10, noise=0.8, rs=13) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise linear with hinge functions."""
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, p))
    
    y = (2.0 * np.maximum(X[:, 0], 0) - 
         1.5 * np.maximum(-X[:, 1], 0) + 
         0.6 * X[:, 2] * X[:, 3] + 
         0.4 * np.tanh(X[:, 4]))
    y += rng.normal(0, noise, n)
    return X, y


DATASETS = {
    "quadrant_interaction": dgp_quadrant_interaction,
    "friedman1": dgp_friedman1,
    "correlated_linear": dgp_correlated_linear,
    "xor_checker": dgp_xor_checker,
    "piecewise_hinge": dgp_piecewise_hinge,
}


# ============================================================================
# BASELINE MODEL UTILITIES
# ============================================================================

def cv_tune_cart_depth(X, y, folds=3, depths=(3, 4, 5, 6), 
                        min_samples_leaf=20, rs=0) -> Dict:
    """CV-tune max_depth for sklearn DecisionTreeRegressor."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=303 + rs)
    best = (np.inf, None)
    
    for d in depths:
        fold_scores = []
        for tr, va in kf.split(X):
            m = DecisionTreeRegressor(
                max_depth=d, 
                min_samples_leaf=min_samples_leaf, 
                random_state=rs
            ).fit(X[tr], y[tr])
            fold_scores.append(mean_squared_error(y[va], m.predict(X[va])))
        
        mse = float(np.mean(fold_scores))
        if mse < best[0]:
            best = (mse, {'max_depth': d, 'min_samples_leaf': min_samples_leaf})
    
    return best[1]


def cv_cart_minalpha(X, y, folds=3, depths=(4, 5), leaves=(10, 20), 
                      max_alphas=20, rs=0) -> Dict:
    """CV-optimal cost-complexity pruning alpha."""
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
                fold_scores = []
                for tr, va in kf.split(X):
                    m = DecisionTreeRegressor(
                        max_depth=d, 
                        min_samples_leaf=lf, 
                        ccp_alpha=float(a), 
                        random_state=rs
                    ).fit(X[tr], y[tr])
                    fold_scores.append(mean_squared_error(y[va], m.predict(X[va])))
                
                mse = float(np.mean(fold_scores))
                if mse < best[0]:
                    best = (mse, {
                        'max_depth': d,
                        'min_samples_leaf': lf,
                        'ccp_alpha': float(a)
                    })
    
    return best[1]


def fit_cart_1se(X, y, folds=3, depths=(4, 5), leaves=(10, 20), 
                 max_alphas=25, rs=0) -> Tuple[DecisionTreeRegressor, Dict]:
    """Fit CART with 1-SE rule for simpler tree."""
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
                fold_scores = []
                for tr, va in kf.split(X):
                    m = DecisionTreeRegressor(
                        max_depth=d, 
                        min_samples_leaf=lf, 
                        ccp_alpha=float(a), 
                        random_state=rs
                    ).fit(X[tr], y[tr])
                    fold_scores.append(mean_squared_error(y[va], m.predict(X[va])))
                means.append(np.mean(fold_scores))
                stds.append(np.std(fold_scores, ddof=1))
            
            means = np.array(means)
            se = np.array(stds) / np.sqrt(folds)
            min_idx = int(np.argmin(means))
            thr = float(means[min_idx] + se[min_idx])
            feas = np.where(means <= thr)[0]
            chosen = int(feas[-1] if feas.size > 0 else min_idx)
            
            rec = float(means[chosen])
            if (best is None) or (rec < best[0]):
                best = (rec, {
                    'max_depth': d,
                    'min_samples_leaf': lf,
                    'ccp_alpha': float(alphas[chosen])
                })
    
    params = best[1]
    model = DecisionTreeRegressor(
        max_depth=params['max_depth'],
        min_samples_leaf=params['min_samples_leaf'],
        ccp_alpha=params['ccp_alpha'],
        random_state=rs
    ).fit(X, y)
    
    return model, params


def sklearn_leaf_count(model: DecisionTreeRegressor) -> int:
    """Count leaves in sklearn tree."""
    t = model.tree_
    return int(np.sum(t.children_left == -1))


# ============================================================================
# STABILITY MEASUREMENT
# ============================================================================

def oos_stability_bootstrap(model_factory, Xtr, ytr, Xte, B=12, rs=0):
    """
    Bootstrap on training data, measure prediction variance on same test set.
    
    Returns:
        mean_std: Mean of per-point prediction std dev (lower = more stable)
        p90_std: 90th percentile of per-point std dev
    """
    rng = np.random.default_rng(rs)
    ntr = Xtr.shape[0]
    preds = []
    
    for b in range(B):
        idx = rng.integers(0, ntr, ntr)  # Bootstrap indices
        m = model_factory()
        m.fit(Xtr[idx], ytr[idx])
        preds.append(m.predict(Xte))
    
    P = np.vstack(preds)  # Shape: (B, n_test)
    per_point_std = P.std(axis=0, ddof=1)
    
    return float(per_point_std.mean()), float(np.percentile(per_point_std, 90))


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_benchmark(
    select_datasets=None,
    output_dir="./bench_results",
    B_stability=12,
    random_seed=0
):
    """
    Run comprehensive benchmark comparing tree algorithms.
    
    Args:
        select_datasets: List of dataset names to run (None = all)
        output_dir: Directory to save results
        B_stability: Number of bootstrap samples for stability measurement
        random_seed: Random seed for reproducibility
    """
    if select_datasets is None:
        select_datasets = list(DATASETS.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    
    accuracy_rows = []
    stability_rows = []
    
    print(f"\n{'='*70}")
    print(f"STABLE TREE BENCHMARK")
    print(f"{'='*70}")
    print(f"Datasets: {', '.join(select_datasets)}")
    print(f"Stability bootstraps: {B_stability}")
    print(f"Random seed: {random_seed}\n")
    
    for dataset_name in select_datasets:
        print(f"\n[{dataset_name}]")
        print("-" * 70)
        
        # Generate data
        X, y = DATASETS[dataset_name]()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # ---- Model 1: CART (CV-tuned depth) ----
        print("  • CART (CV-tuned)...", end=" ", flush=True)
        t0 = time.time()
        params_cart = cv_tune_cart_depth(
            Xtr, ytr, 
            depths=(3, 4, 5, 6), 
            min_samples_leaf=20, 
            rs=random_seed
        )
        cart = DecisionTreeRegressor(**params_cart, random_state=random_seed)
        cart.fit(Xtr, ytr)
        fit_time_cart = time.time() - t0
        
        pred_cart = cart.predict(Xte)
        leaves_cart = sklearn_leaf_count(cart)
        print(f"done ({fit_time_cart:.2f}s, {leaves_cart} leaves)")
        
        # ---- Model 2: CART_PRUNED (CV-optimal alpha) ----
        print("  • CART_PRUNED (CV-alpha)...", end=" ", flush=True)
        t0 = time.time()
        params_pruned = cv_cart_minalpha(Xtr, ytr, rs=random_seed)
        cart_pruned = DecisionTreeRegressor(**params_pruned, random_state=random_seed)
        cart_pruned.fit(Xtr, ytr)
        fit_time_pruned = time.time() - t0
        
        pred_pruned = cart_pruned.predict(Xte)
        leaves_pruned = sklearn_leaf_count(cart_pruned)
        print(f"done ({fit_time_pruned:.2f}s, {leaves_pruned} leaves)")
        
        # ---- Model 3: CART_PRUNED_1SE ----
        print("  • CART_PRUNED_1SE...", end=" ", flush=True)
        t0 = time.time()
        cart_1se, params_1se = fit_cart_1se(Xtr, ytr, rs=random_seed)
        fit_time_1se = time.time() - t0
        
        pred_1se = cart_1se.predict(Xte)
        leaves_1se = sklearn_leaf_count(cart_1se)
        print(f"done ({fit_time_1se:.2f}s, {leaves_1se} leaves)")
        
        # ---- Model 4: LessGreedyHybrid ----
        print("  • LessGreedyHybrid...", end=" ", flush=True)
        t0 = time.time()
        less_greedy = LessGreedyHybridRegressor(
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
            random_state=random_seed
        )
        less_greedy.fit(Xtr, ytr)
        fit_time_lg = less_greedy.fit_time_sec_
        
        pred_lg = less_greedy.predict(Xte)
        leaves_lg = less_greedy.count_leaves()
        print(f"done ({fit_time_lg:.2f}s, {leaves_lg} leaves)")
        
        # ---- Model 5: BootstrapVariancePenalized ----
        print("  • BootstrapVariancePenalized...", end=" ", flush=True)
        t0 = time.time()
        bv_penalized = BootstrapVariancePenalizedRegressor(
            max_depth=5,
            min_samples_split=40,
            min_samples_leaf=20,
            split_frac=0.6,
            val_frac=0.2,
            est_frac=0.2,
            variance_penalty=1.0,
            n_bootstrap=10,
            bootstrap_max_depth=2,
            enable_oblique_root=False,  # Disable for speed
            beam_topk=12,
            ambiguity_eps=0.05,
            min_n_for_lookahead=600,
            root_k=2,
            inner_k=1,
            leaf_shrinkage_lambda=10.0,
            random_state=random_seed
        )
        bv_penalized.fit(Xtr, ytr)
        fit_time_bv = bv_penalized.fit_time_sec_
        
        pred_bv = bv_penalized.predict(Xte)
        leaves_bv = bv_penalized.count_leaves()
        print(f"done ({fit_time_bv:.2f}s, {leaves_bv} leaves)")
        
        # ---- Collect accuracy metrics ----
        models_eval = [
            ('CART', pred_cart, leaves_cart, fit_time_cart),
            ('CART_PRUNED', pred_pruned, leaves_pruned, fit_time_pruned),
            ('CART_PRUNED_1SE', pred_1se, leaves_1se, fit_time_1se),
            ('LessGreedyHybrid', pred_lg, leaves_lg, fit_time_lg),
            ('BootstrapVariancePenalized', pred_bv, leaves_bv, fit_time_bv),
        ]
        
        for model_name, pred, leaves, fit_t in models_eval:
            mse = mean_squared_error(yte, pred)
            r2 = r2_score(yte, pred)
            
            accuracy_rows.append({
                'dataset': dataset_name,
                'model': model_name,
                'mse': mse,
                'r2': r2,
                'leaves': int(leaves),
                'fit_time_sec': fit_t
            })
        
        # ---- Measure stability (bootstrap on train, same test) ----
        print("\n  Measuring stability...")
        
        factories = {
            'CART': lambda: DecisionTreeRegressor(**params_cart, random_state=random_seed),
            'CART_PRUNED': lambda: DecisionTreeRegressor(**params_pruned, random_state=random_seed),
            'CART_PRUNED_1SE': lambda: DecisionTreeRegressor(**params_1se, random_state=random_seed),
            'LessGreedyHybrid': lambda: LessGreedyHybridRegressor(
                max_depth=5, min_samples_split=40, min_samples_leaf=20,
                split_frac=0.6, val_frac=0.2, est_frac=0.2,
                enable_oblique_root=True, gain_margin=0.03, min_abs_corr=0.3,
                oblique_cv=5, beam_topk=12, ambiguity_eps=0.05,
                min_n_for_lookahead=600, root_k=2, inner_k=1,
                leaf_shrinkage_lambda=10.0, random_state=random_seed
            ),
            'BootstrapVariancePenalized': lambda: BootstrapVariancePenalizedRegressor(
                max_depth=5, min_samples_split=40, min_samples_leaf=20,
                split_frac=0.6, val_frac=0.2, est_frac=0.2,
                variance_penalty=1.0, n_bootstrap=10, bootstrap_max_depth=2,
                enable_oblique_root=False, beam_topk=12, ambiguity_eps=0.05,
                min_n_for_lookahead=600, root_k=2, inner_k=1,
                leaf_shrinkage_lambda=10.0, random_state=random_seed
            ),
        }
        
        for model_name, factory in factories.items():
            mean_std, p90_std = oos_stability_bootstrap(
                factory, Xtr, ytr, Xte, 
                B=B_stability, 
                rs=123
            )
            
            stability_rows.append({
                'dataset': dataset_name,
                'model': model_name,
                'mean_pred_std': mean_std,
                'p90_pred_std': p90_std
            })
            print(f"    - {model_name:30s}: mean_std={mean_std:.4f}, p90_std={p90_std:.4f}")
    
    # ---- Save results ----
    df_accuracy = pd.DataFrame(accuracy_rows)
    df_stability = pd.DataFrame(stability_rows)
    
    acc_path = os.path.join(output_dir, "accuracy_metrics.csv")
    stab_path = os.path.join(output_dir, "stability_metrics.csv")
    
    df_accuracy.to_csv(acc_path, index=False)
    df_stability.to_csv(stab_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved:")
    print(f"  • {acc_path}")
    print(f"  • {stab_path}")
    print(f"{'='*70}\n")
    
    # ---- Print summary ----
    print("\nACCURACY SUMMARY (mean across datasets):")
    print(df_accuracy.groupby('model')[['mse', 'r2', 'leaves']].mean().round(4))
    
    print("\nSTABILITY SUMMARY (mean across datasets, lower = better):")
    print(df_stability.groupby('model')[['mean_pred_std', 'p90_pred_std']].mean().round(4))
    
    # Compute stability improvement over CART baseline
    print("\nSTABILITY IMPROVEMENT vs CART (% reduction in variance):")
    for dataset_name in select_datasets:
        cart_var = df_stability[(df_stability.dataset == dataset_name) & 
                                 (df_stability.model == 'CART')]['mean_pred_std'].values[0]
        
        for model_name in ['CART_PRUNED', 'CART_PRUNED_1SE', 'LessGreedyHybrid', 
                           'BootstrapVariancePenalized']:
            model_var = df_stability[(df_stability.dataset == dataset_name) & 
                                      (df_stability.model == model_name)]['mean_pred_std'].values[0]
            improvement = (1 - model_var / cart_var) * 100
            print(f"  {dataset_name:25s} | {model_name:30s}: {improvement:+6.1f}%")
    
    return df_accuracy, df_stability


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark stable tree algorithms on accuracy and stability"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=None,
        help="Datasets to run (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./bench_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=12,
        help="Number of bootstrap samples for stability measurement"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        select_datasets=args.datasets,
        output_dir=args.output,
        B_stability=args.bootstrap_samples,
        random_seed=args.seed
    )