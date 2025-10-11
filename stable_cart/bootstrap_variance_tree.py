"""
bootstrap_variance_tree.py
---------------------------
A variant of LessGreedyHybridRegressor that explicitly penalizes bootstrap
prediction variance on the validation set during split selection.

This encourages the tree to make splits that lead to more stable predictions
across different bootstrap samples of the training data.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Dict, Any, Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

from stable_cart.less_greedy_tree import _sse


class BootstrapVariancePenalizedRegressor(BaseEstimator, RegressorMixin):
    """
    A tree regressor that penalizes bootstrap prediction variance during split selection.

    This extends the LessGreedyHybridRegressor by adding a bootstrap variance penalty
    term to the split evaluation criterion. For each candidate split, we:
    1. Generate B bootstrap samples from the training data
    2. Fit a simple model to each bootstrap sample
    3. Compute prediction variance on the validation set
    4. Add this variance as a penalty to the validation SSE

    Parameters
    ----------
    variance_penalty : float, default=1.0
        Weight for the bootstrap variance penalty term.
        Higher values encourage more stable splits.

    n_bootstrap : int, default=10
        Number of bootstrap samples to use for variance estimation.
        More samples give better estimates but increase computation.

    bootstrap_max_depth : int, default=2
        Maximum depth for bootstrap trees used in variance estimation.
        Shallow trees are faster and often sufficient for stability assessment.

    Other parameters are inherited from LessGreedyHybridRegressor.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=40,
        min_samples_leaf=20,
        split_frac=0.6,
        val_frac=0.2,
        est_frac=0.2,
        # Bootstrap variance penalty parameters
        variance_penalty=1.0,
        n_bootstrap=10,
        bootstrap_max_depth=2,
        # oblique root
        enable_oblique_root=True,
        gain_margin=0.03,
        min_abs_corr=0.3,
        oblique_cv=5,
        # lookahead (axis-only)
        beam_topk=12,
        ambiguity_eps=0.05,
        min_n_for_lookahead=600,
        root_k=2,
        inner_k=1,
        # leaf shrinkage
        leaf_shrinkage_lambda=0.0,
        random_state=0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.split_frac = split_frac
        self.val_frac = val_frac
        self.est_frac = est_frac

        # Bootstrap variance penalty
        self.variance_penalty = variance_penalty
        self.n_bootstrap = n_bootstrap
        self.bootstrap_max_depth = bootstrap_max_depth

        self.enable_oblique_root = enable_oblique_root
        self.gain_margin = gain_margin
        self.min_abs_corr = min_abs_corr
        self.oblique_cv = oblique_cv
        self.beam_topk = beam_topk
        self.ambiguity_eps = ambiguity_eps
        self.min_n_for_lookahead = min_n_for_lookahead
        self.root_k = root_k
        self.inner_k = inner_k
        self.leaf_shrinkage_lambda = leaf_shrinkage_lambda
        self.random_state = random_state

        # Learned attributes
        self.tree_: Dict[str, Any] = {}
        self.oblique_info_: Optional[Dict[str, Any]] = None
        self.fit_time_sec_: float = 0.0
        self.splits_scanned_: int = 0
        self.bootstrap_evaluations_: int = 0

    def _compute_bootstrap_variance(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        Xv: np.ndarray,
        feat: int,
        thr: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute the variance of predictions on validation set across bootstrap samples.

        Parameters
        ----------
        Xs : array-like of shape (n_split, n_features)
            Split subset features
        ys : array-like of shape (n_split,)
            Split subset targets
        Xv : array-like of shape (n_val, n_features)
            Validation subset features
        feat : int
            Feature index for the split
        thr : float
            Threshold value for the split
        rng : numpy random generator
            Random number generator

        Returns
        -------
        variance : float
            Mean variance of predictions across validation samples
        """
        n_samples = Xs.shape[0]
        n_val = Xv.shape[0]

        # Store predictions from each bootstrap sample
        bootstrap_preds = np.zeros((self.n_bootstrap, n_val))

        for b in range(self.n_bootstrap):
            # Generate bootstrap sample
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            Xs_boot = Xs[boot_idx]
            ys_boot = ys[boot_idx]

            # Apply the split to bootstrap data
            mask_boot = Xs_boot[:, feat] <= thr

            # Skip if split creates empty children
            if (
                mask_boot.sum() < self.min_samples_leaf
                or (n_samples - mask_boot.sum()) < self.min_samples_leaf
            ):
                # Use parent mean as fallback
                bootstrap_preds[b, :] = ys_boot.mean()
                continue

            # Fit simple models to each child (could be more sophisticated)
            # For now, we'll use a shallow tree
            mask_val = Xv[:, feat] <= thr

            # Left child predictions
            if mask_val.sum() > 0:
                # Simple approach: fit small tree on left child
                left_tree = SimpleTree(max_depth=self.bootstrap_max_depth)
                left_tree.fit(Xs_boot[mask_boot], ys_boot[mask_boot])
                bootstrap_preds[b, mask_val] = left_tree.predict(Xv[mask_val])

            # Right child predictions
            if (~mask_val).sum() > 0:
                # Simple approach: fit small tree on right child
                right_tree = SimpleTree(max_depth=self.bootstrap_max_depth)
                right_tree.fit(Xs_boot[~mask_boot], ys_boot[~mask_boot])
                bootstrap_preds[b, ~mask_val] = right_tree.predict(Xv[~mask_val])

        # Compute variance of predictions for each validation sample
        pred_variance = np.var(bootstrap_preds, axis=0)

        # Return mean variance across validation samples
        self.bootstrap_evaluations_ += 1
        return float(np.mean(pred_variance))

    def _val_score_with_variance_penalty(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        Xv: np.ndarray,
        yv: np.ndarray,
        feat: int,
        thr: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute validation score with bootstrap variance penalty.

        Score = validation_SSE + variance_penalty * bootstrap_variance
        """
        # Standard validation SSE
        mask_val = Xv[:, feat] <= thr
        val_sse = _sse(yv[mask_val]) + _sse(yv[~mask_val])

        # Bootstrap variance penalty
        if self.variance_penalty > 0 and self.n_bootstrap > 0:
            boot_var = self._compute_bootstrap_variance(Xs, ys, Xv, feat, thr, rng)
            penalty = self.variance_penalty * boot_var
        else:
            penalty = 0.0

        return val_sse + penalty

    def _children_sse_vec(self, xs, ys, min_leaf):
        """Vectorized computation of children SSE along sorted feature."""
        n = ys.size
        if n < 2 * min_leaf:
            return np.array([]), np.array([], dtype=bool)

        ps1 = np.cumsum(ys, dtype=np.float64)
        ps2 = np.cumsum(ys * ys, dtype=np.float64)
        tot1 = ps1[-1]
        tot2 = ps2[-1]
        idx = np.arange(n - 1)
        valid = xs[:-1] != xs[1:]
        nL = idx + 1
        nR = n - nL
        valid &= (nL >= min_leaf) & (nR >= min_leaf)
        sumL = ps1[idx]
        sumL2 = ps2[idx]
        sumR = tot1 - sumL
        sumR2 = tot2 - sumL2

        # Avoid division by zero
        sseL = np.where(nL > 0, sumL2 - (sumL * sumL) / nL, np.inf)
        sseR = np.where(nR > 0, sumR2 - (sumR * sumR) / nR, np.inf)

        self.splits_scanned_ += int(valid.sum())
        return sseL + sseR, valid

    def _topk_axis_candidates(self, Xs, ys, topk):
        """Find top-k axis-aligned split candidates."""
        parent_sse = _sse(ys)
        gains = []
        p = Xs.shape[1]

        for j in range(p):
            order = np.argsort(Xs[:, j], kind="mergesort")
            xs = Xs[order, j]
            ys_ord = ys[order]
            children_sse, valid = self._children_sse_vec(xs, ys_ord, self.min_samples_leaf)

            if not valid.any():
                continue

            thr = 0.5 * (xs[:-1] + xs[1:])
            idx = np.where(valid)[0]
            g = parent_sse - children_sse[idx]

            for i, gi in zip(idx, g):
                gains.append((float(gi), int(j), float(thr[i])))

        if not gains:
            return []

        gains.sort(key=lambda t: t[0], reverse=True)
        return gains[:topk]

    def _build(self, Xs, ys, Xv, yv, Xe, ye, depth, parent_mean_est, rng):
        """Recursively build tree with bootstrap variance penalty."""
        n_split = ys.size
        n_val = yv.size

        # Stopping conditions
        if depth >= self.max_depth or n_split < self.min_samples_split:
            mu_leaf = float(ye.mean()) if ye.size > 0 else float(ys.mean())
            lam = self.leaf_shrinkage_lambda
            mu = (
                ((ye.size * mu_leaf + lam * parent_mean_est) / (ye.size + lam))
                if lam > 0
                else mu_leaf
            )
            return {
                "type": "leaf",
                "value": mu,
                "n_split": int(n_split),
                "n_val": int(n_val),
                "n_est": int(ye.size),
            }

        # Get axis-aligned candidates
        cand_axis = self._topk_axis_candidates(Xs, ys, self.beam_topk)

        if not cand_axis:
            # No valid splits, return leaf
            mu_leaf = float(ye.mean()) if ye.size > 0 else float(ys.mean())
            lam = self.leaf_shrinkage_lambda
            mu = (
                ((ye.size * mu_leaf + lam * parent_mean_est) / (ye.size + lam))
                if lam > 0
                else mu_leaf
            )
            return {
                "type": "leaf",
                "value": mu,
                "n_split": int(n_split),
                "n_val": int(n_val),
                "n_est": int(ye.size),
            }

        # Evaluate splits with variance penalty
        best_score = np.inf
        best_split = None

        for gain, feat, thr in cand_axis[:5]:  # Limit evaluation for efficiency
            # Check if split is valid
            mask_s = Xs[:, feat] <= thr
            if (
                mask_s.sum() < self.min_samples_leaf
                or (n_split - mask_s.sum()) < self.min_samples_leaf
            ):
                continue

            # Compute score with variance penalty
            score = self._val_score_with_variance_penalty(Xs, ys, Xv, yv, feat, thr, rng)

            if score < best_score:
                best_score = score
                best_split = (feat, thr)

        if best_split is None:
            # No valid split found
            mu_leaf = float(ye.mean()) if ye.size > 0 else float(ys.mean())
            lam = self.leaf_shrinkage_lambda
            mu = (
                ((ye.size * mu_leaf + lam * parent_mean_est) / (ye.size + lam))
                if lam > 0
                else mu_leaf
            )
            return {
                "type": "leaf",
                "value": mu,
                "n_split": int(n_split),
                "n_val": int(n_val),
                "n_est": int(ye.size),
            }

        # Make the split
        feat, thr = best_split
        mask_s = Xs[:, feat] <= thr
        mask_v = Xv[:, feat] <= thr
        mask_e = Xe[:, feat] <= thr if Xe.size > 0 else np.array([], dtype=bool)

        node = {
            "type": "split",
            "f": int(feat),
            "t": float(thr),
            "n_split": int(n_split),
            "n_val": int(n_val),
            "n_est": int(ye.size),
        }

        # Recursively build children
        node["left"] = self._build(
            Xs[mask_s],
            ys[mask_s],
            Xv[mask_v],
            yv[mask_v],
            Xe[mask_e] if Xe.size > 0 else Xe,
            ye[mask_e] if Xe.size > 0 else ye,
            depth + 1,
            parent_mean_est=float(ye.mean()) if ye.size > 0 else float(ys.mean()),
            rng=rng,
        )

        node["right"] = self._build(
            Xs[~mask_s],
            ys[~mask_s],
            Xv[~mask_v],
            yv[~mask_v],
            Xe[~mask_e] if Xe.size > 0 else Xe,
            ye[~mask_e] if Xe.size > 0 else ye,
            depth + 1,
            parent_mean_est=float(ye.mean()) if ye.size > 0 else float(ys.mean()),
            rng=rng,
        )

        return node

    def fit(self, X, y):
        """Fit the bootstrap variance penalized tree."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must contain at least one sample.")

        assert 0 < self.split_frac < 1 and 0 < self.val_frac < 1 and 0 < self.est_frac < 1
        assert (
            abs((self.split_frac + self.val_frac + self.est_frac) - 1.0) < 1e-8
        ), "split_frac + val_frac + est_frac must sum to 1"

        t0 = time.time()
        rng = np.random.default_rng(self.random_state)

        # Split data into SPLIT/VAL/EST
        n = X.shape[0]
        idx = rng.permutation(n)
        n_split = int(self.split_frac * n)
        n_val = int(self.val_frac * n)

        iS = idx[:n_split]
        iV = idx[n_split : n_split + n_val]
        iE = idx[n_split + n_val :]

        Xs, ys = X[iS], y[iS]
        Xv, yv = X[iV], y[iV]
        Xe, ye = X[iE], y[iE]

        self.splits_scanned_ = 0
        self.bootstrap_evaluations_ = 0

        parent_mean_est = float(ye.mean()) if ye.size > 0 else float(ys.mean())
        self.tree_ = self._build(
            Xs, ys, Xv, yv, Xe, ye, depth=0, parent_mean_est=parent_mean_est, rng=rng
        )

        self.fit_time_sec_ = time.time() - t0
        return self

    def _predict_one(self, x, node):
        """Predict for a single sample."""
        if node["type"] == "leaf":
            return node["value"]

        if x[node["f"]] <= node["t"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        """Predict for multiple samples."""
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def score(self, X, y):
        """Compute R² score."""
        y = np.asarray(y)
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def count_leaves(self) -> int:
        """Count the number of leaves in the tree."""

        def _count(node):
            if node["type"] == "leaf":
                return 1
            return _count(node["left"]) + _count(node["right"])

        return _count(self.tree_)


class SimpleTree:
    """
    A very simple decision tree for bootstrap variance estimation.
    This is much faster than using full sklearn trees.
    """

    def __init__(self, max_depth=2, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def _build(self, X, y, depth):
        n = len(y)

        # Stopping conditions
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return {"type": "leaf", "value": float(np.mean(y))}

        # Find best split (simplified)
        best_gain = 0
        best_split = None
        parent_sse = np.var(y) * n

        for j in range(X.shape[1]):
            # Sort by feature
            order = np.argsort(X[:, j])
            Xs = X[order, j]
            ys = y[order]

            # Try each split point
            for i in range(self.min_samples_leaf, n - self.min_samples_leaf):
                if Xs[i - 1] == Xs[i]:
                    continue

                # Compute SSE reduction
                left_sse = np.var(ys[:i]) * i if i > 0 else 0
                right_sse = np.var(ys[i:]) * (n - i) if i < n else 0
                gain = parent_sse - (left_sse + right_sse)

                if gain > best_gain:
                    best_gain = gain
                    best_split = (j, 0.5 * (Xs[i - 1] + Xs[i]))

        if best_split is None:
            return {"type": "leaf", "value": float(np.mean(y))}

        # Make split
        feat, thr = best_split
        mask = X[:, feat] <= thr

        return {
            "type": "split",
            "feat": feat,
            "thr": thr,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        """Fit the simple tree."""
        self.tree_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        """Predict for single sample."""
        if node["type"] == "leaf":
            return node["value"]

        if x[node["feat"]] <= node["thr"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        """Predict for multiple samples."""
        return np.array([self._predict_one(x, self.tree_) for x in X])
