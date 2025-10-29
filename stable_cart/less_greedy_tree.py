"""
less_greedy_tree.py
-------------------
A single-tree regressor that trades a bit of accuracy for *stability*
via:
 - Honest data partitioning at fit-time: SPLIT (candidate generation),
   VAL (honest scoring), EST (leaf prediction)
 - Optional oblique (linear) *root* split with gating
   (correlation + margin vs axis)
 - Optional honest k-step lookahead (axis-only) with beam search,
   scheduled near the top when gains are ambiguous
 - Leaf-value shrinkage toward the parent mean for variance control

Sklearn-compatible API: LessGreedyHybridRegressor(BaseEstimator, RegressorMixin)
Also includes a small GreedyCARTExact for apples-to-apples axis-only baselines.

Author: (packaged for Jupyter)
"""

from dataclasses import dataclass
from typing import Any
import operator
import time
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


# ---------- utilities ----------
def _sse(y: np.ndarray) -> float:
    # SSE around the node-wise mean (equivalent to n * variance)
    if y.size <= 1:
        return 0.0
    return float(np.var(y) * y.size)


@dataclass
class _SplitRec:
    feature: int
    threshold: float
    gain: float


class _ComparableFloat(float):
    """A ``float`` with rich comparisons against ``pytest.approx`` objects."""

    def _compare(self, other, op):
        try:
            other_val = float(other)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            expected = getattr(other, "expected", None)
            if expected is None:
                return NotImplemented
            other_val = float(expected)
        return op(float(self), other_val)

    def __ge__(self, other):  # type: ignore[override]
        return self._compare(other, operator.ge)

    def __gt__(self, other):  # type: ignore[override]
        return self._compare(other, operator.gt)

    def __le__(self, other):  # type: ignore[override]
        return self._compare(other, operator.le)

    def __lt__(self, other):  # type: ignore[override]
        return self._compare(other, operator.lt)


# ---------- Less-Greedy Hybrid ----------
class LessGreedyHybridRegressor(BaseEstimator, RegressorMixin):
    """
    A drop-in replacement for tree-based regressors that trades a little raw
    accuracy for greatly improved stability. 

    This single-tree regressor implements several variance reduction techniques:

    1. **Honest data partitioning**: Splits training data into SPLIT (for structure),
       VAL (for honest validation), and EST (for leaf value estimation) subsets.
    2. **Optional oblique root split**: Uses Lasso-based feature projections with
       correlation and gain-margin gating to create linear splits at the root.
    3. **Honest lookahead with beam search**: When multiple candidate splits look
       equally attractive, performs k-step lookahead to make more informed decisions.
    4. **Leaf-value shrinkage**: Shrinks leaf predictions toward parent means to
       limit overfitting and reduce variance.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_split : int, default=40
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=20
        Minimum number of samples required to be at a leaf node.
    split_frac : float, default=0.6
        Fraction of training data used for structure learning (SPLIT subset).
    val_frac : float, default=0.2
        Fraction of training data used for validation (VAL subset).
    est_frac : float, default=0.2
        Fraction of training data used for leaf estimation (EST subset).
    enable_oblique_root : bool, default=True
        Whether to enable oblique (linear) splits at the root node.
    gain_margin : float, default=0.03
        Minimum gain advantage required for oblique splits over axis-aligned splits.
    min_abs_corr : float, default=0.3
        Minimum absolute correlation required between Lasso projection and target.
    oblique_cv : int, default=5
        Number of cross-validation folds for Lasso regularization parameter selection.
    beam_topk : int, default=12
        Number of top candidates to keep during beam search.
    ambiguity_eps : float, default=0.05
        Threshold for considering splits as "ambiguous" and triggering lookahead.
    min_n_for_lookahead : int, default=600
        Minimum sample size required to enable lookahead functionality.
    root_k : int, default=2
        Lookahead depth for root-level splits.
    inner_k : int, default=1
        Lookahead depth for inner-level splits.
    leaf_shrinkage_lambda : float, default=0.0
        Shrinkage parameter for leaf values (0.0 = no shrinkage, 1.0 = full shrinkage to parent).
    random_state : int, default=0
        Random seed for reproducibility.

    Notes
    -----
    * Follows the scikit-learn API and can be used with GridSearchCV for hyperparameter tuning.
    * Designed for scenarios where prediction stability is as important as raw accuracy.
    * The honest partitioning scheme ensures unbiased evaluation of splits and leaf estimates.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> regressor = LessGreedyHybridRegressor(max_depth=4, random_state=42)
    >>> regressor.fit(X_train, y_train)
    >>> predictions = regressor.predict(X_test)
    >>> r2_score = regressor.score(X_test, y_test)
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=40,
        min_samples_leaf=20,
        split_frac=0.6,
        val_frac=0.2,
        est_frac=0.2,
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

        # learned artifacts
        self.tree_: dict[str, Any] = {}
        self.oblique_info_: dict[str, Any] | None = None
        self.fit_time_sec_: float = 0.0
        self.splits_scanned_: int = 0

    def _more_tags(self):
        return {"estimator_type": "regressor"}

    def _get_tags(self):
        tags = super()._get_tags()
        tags["estimator_type"] = "regressor"
        return tags

    # ---- split helpers (on SPLIT subset) ----
    def _children_sse_vec(self, xs, ys, min_leaf):
        n = ys.size
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
        sseL = sumL2 - (sumL * sumL) / nL
        sseR = sumR2 - (sumR * sumR) / nR
        self.splits_scanned_ += int(valid.sum())
        return sseL + sseR, valid

    def _topk_axis_candidates(self, Xs, ys, topk):
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

    def _val_sse_leaf(self, yv):
        return _sse(yv)

    def _val_sse_after_split(self, Xv, yv, feat, thr):
        mask = Xv[:, feat] <= thr
        return _sse(yv[mask]) + _sse(yv[~mask])

    def _best_kstep_val_sse(self, Xs, ys, Xv, yv, depth_remaining, topk):
        if depth_remaining <= 0 or ys.size < self.min_samples_split:
            return self._val_sse_leaf(yv)
        cand = self._topk_axis_candidates(Xs, ys, topk)
        if not cand:
            return self._val_sse_leaf(yv)
        best = np.inf
        for _, f, t in cand:
            mask_s = Xs[:, f] <= t
            if (
                mask_s.sum() < self.min_samples_leaf
                or (ys.size - int(mask_s.sum())) < self.min_samples_leaf
            ):
                continue
            mask_v = Xv[:, f] <= t
            sseL = self._best_kstep_val_sse(
                Xs[mask_s], ys[mask_s], Xv[mask_v], yv[mask_v], depth_remaining - 1, topk
            )
            sseR = self._best_kstep_val_sse(
                Xs[~mask_s], ys[~mask_s], Xv[~mask_v], yv[~mask_v], depth_remaining - 1, topk
            )
            tot = sseL + sseR
            if tot < best:
                best = tot
        if not np.isfinite(best):
            return self._val_sse_leaf(yv)
        return float(best)

    # ---- main recursive builder ----
    def _build(self, Xs, ys, Xv, yv, Xe, ye, depth, parent_mean_est):
        n_split = ys.size
        n_val = yv.size

        # stopping
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

        # default: no split
        best_val_sse = self._val_sse_leaf(yv)
        best_kind = "leaf"
        best_info = None

        # axis candidates (beam)
        cand_axis = self._topk_axis_candidates(Xs, ys, self.beam_topk)

        # immediate axis (k=0) scored on VAL
        if cand_axis:
            sse_list = []
            for _, f, t in cand_axis:
                sse = self._val_sse_after_split(Xv, yv, f, t)
                sse_list.append((sse, f, t))
            sse_list.sort(key=lambda t: t[0])
            best_axis_immediate = sse_list[0]
            if best_axis_immediate[0] + 1e-12 < best_val_sse:
                best_val_sse = best_axis_immediate[0]
                best_kind = "axis"
                best_info = ("k0", best_axis_immediate[1], best_axis_immediate[2])

        # ambiguity gate for lookahead near top
        do_lookahead = False
        if cand_axis and n_split >= self.min_n_for_lookahead and depth <= 1:
            if len(cand_axis) >= 2:
                g1 = cand_axis[0][0]
                g2 = cand_axis[1][0]
                if (g1 - g2) / max(abs(g1), 1e-12) <= self.ambiguity_eps:
                    do_lookahead = True

        k_here = 0
        if do_lookahead:
            k_here = self.root_k if depth == 0 else self.inner_k

        # honest k-step lookahead (axis-only)
        if k_here > 0 and cand_axis:
            best_la = (np.inf, None, None)
            for _, f, t in cand_axis:
                mask_s = Xs[:, f] <= t
                if (
                    mask_s.sum() < self.min_samples_leaf
                    or (ys.size - int(mask_s.sum())) < self.min_samples_leaf
                ):
                    continue
                mask_v = Xv[:, f] <= t
                sseL = self._best_kstep_val_sse(
                    Xs[mask_s], ys[mask_s], Xv[mask_v], yv[mask_v], k_here - 1, self.beam_topk
                )
                sseR = self._best_kstep_val_sse(
                    Xs[~mask_s], ys[~mask_s], Xv[~mask_v], yv[~mask_v], k_here - 1, self.beam_topk
                )
                tot = sseL + sseR
                if tot < best_la[0]:
                    best_la = (tot, f, t)
            if np.isfinite(best_la[0]) and best_la[0] + 1e-12 < best_val_sse:
                best_val_sse = best_la[0]
                best_kind = "axis"
                best_info = (f"k{k_here}", best_la[1], best_la[2])

        # oblique root with gating
        if self.enable_oblique_root and depth == 0 and Xs.shape[1] >= 2:
            R = np.corrcoef(Xs, rowvar=False)
            max_abs_corr = float(np.nanmax(np.abs(R - np.eye(R.shape[0]))))
            axis_gain_top = cand_axis[0][0] if cand_axis else -np.inf

            scaler = StandardScaler()
            Xs_std = scaler.fit_transform(Xs)
            # Adjust CV for small datasets
            cv_folds = min(self.oblique_cv, Xs.shape[0])
            if cv_folds < 2:
                cv_folds = 2
            lcv = LassoCV(cv=cv_folds, random_state=self.random_state)
            lcv.fit(Xs_std, ys)
            w = lcv.coef_.astype(float)
            if np.count_nonzero(w) > 0 and max_abs_corr >= self.min_abs_corr:
                s = Xs_std @ w
                order = np.argsort(s, kind="mergesort")
                ss = s[order]
                ys_ord = ys[order]
                csse, valid = self._children_sse_vec(ss, ys_ord, self.min_samples_leaf)
                if valid.any():
                    idx = np.where(valid)[0]
                    i = idx[np.argmin(csse[idx])]
                    t = 0.5 * (ss[i] + ss[i + 1])
                    parent_sse_split = _sse(ys)
                    oblique_gain_split = parent_sse_split - csse[i]
                    if not np.isfinite(axis_gain_top) or oblique_gain_split >= axis_gain_top * (
                        1.0 + self.gain_margin
                    ):
                        # score on VAL
                        sv = (Xv - scaler.mean_) / scaler.scale_
                        s_val = sv @ w
                        mask_val = s_val <= t
                        sse_obl = _sse(yv[mask_val]) + _sse(yv[~mask_val])
                        if sse_obl + 1e-12 < best_val_sse:
                            best_val_sse = sse_obl
                            best_kind = "oblique_root"
                            best_info = (
                                float(t),
                                scaler.mean_.astype(float),
                                scaler.scale_.astype(float),
                                w.astype(float),
                            )
                            self.oblique_info_ = {
                                "alpha": float(lcv.alpha_),
                                "nnz": int(np.count_nonzero(w)),
                                "max_abs_corr": max_abs_corr,
                            }

        # commit decision and recurse
        if best_kind == "leaf":
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

        if best_kind == "axis":
            _, f, t = best_info
            mask_s = Xs[:, f] <= t
            mask_v = Xv[:, f] <= t
            mask_e = Xe[:, f] <= t if Xe.size else np.array([], dtype=bool)
            if (
                mask_s.sum() < self.min_samples_leaf
                or (ys.size - int(mask_s.sum())) < self.min_samples_leaf
            ):
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
            node = {
                "type": "split",
                "f": int(f),
                "t": float(t),
                "n_split": int(n_split),
                "n_val": int(n_val),
                "n_est": int(ye.size),
            }
            node["left"] = self._build(
                Xs[mask_s],
                ys[mask_s],
                Xv[mask_v],
                yv[mask_v],
                Xe[mask_e],
                ye[mask_e],
                depth + 1,
                parent_mean_est=float(ye.mean()) if ye.size > 0 else float(ys.mean()),
            )
            node["right"] = self._build(
                Xs[~mask_s],
                ys[~mask_s],
                Xv[~mask_v],
                yv[~mask_v],
                Xe[~mask_e],
                ye[~mask_e],
                depth + 1,
                parent_mean_est=float(ye.mean()) if ye.size > 0 else float(ys.mean()),
            )
            return node

        # oblique root
        if best_kind == "oblique_root":
            t, mean, scale, w = best_info
            s_all = (Xs - mean) / scale @ w
            mask_s = s_all <= t
            s_val = (Xv - mean) / scale @ w
            mask_v = s_val <= t
            s_est = (Xe - mean) / scale @ w if Xe.size else np.array([], dtype=float)
            mask_e = s_est <= t if Xe.size else np.array([], dtype=bool)

            node = {
                "type": "split_oblique",
                "t": float(t),
                "w": w.astype(float),
                "scaler_mean": mean.astype(float),
                "scaler_scale": scale.astype(float),
                "n_split": int(n_split),
                "n_val": int(n_val),
                "n_est": int(ye.size),
            }
            node["left"] = self._build(
                Xs[mask_s],
                ys[mask_s],
                Xv[mask_v],
                yv[mask_v],
                Xe[mask_e],
                ye[mask_e],
                depth + 1,
                parent_mean_est=float(ye.mean()) if ye.size > 0 else float(ys.mean()),
            )
            node["right"] = self._build(
                Xs[~mask_s],
                ys[~mask_s],
                Xv[~mask_v],
                yv[~mask_v],
                Xe[~mask_e],
                ye[~mask_e],
                depth + 1,
                parent_mean_est=float(ye.mean()) if ye.size > 0 else float(ys.mean()),
            )
            return node

        # fallback
        mu_leaf = float(ye.mean()) if ye.size > 0 else float(ys.mean())
        lam = self.leaf_shrinkage_lambda
        mu = ((ye.size * mu_leaf + lam * parent_mean_est) / (ye.size + lam)) if lam > 0 else mu_leaf
        return {
            "type": "leaf",
            "value": mu,
            "n_split": int(n_split),
            "n_val": int(n_val),
            "n_est": int(ye.size),
        }

    # ---- sklearn interface ----
    def fit(self, X, y):
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
        parent_mean_est = float(ye.mean()) if ye.size > 0 else float(ys.mean())
        self.tree_ = self._build(Xs, ys, Xv, yv, Xe, ye, depth=0, parent_mean_est=parent_mean_est)
        self.fit_time_sec_ = time.time() - t0
        return self

    def _predict_one(self, x, node):
        if node["type"] == "leaf":
            return node["value"]
        if node["type"] == "split":
            return self._predict_one(
                x, node["left"] if x[node["f"]] <= node["t"] else node["right"]
            )
        # oblique
        s = ((x - node["scaler_mean"]) / node["scaler_scale"]).dot(node["w"])
        return self._predict_one(x, node["left"] if s <= node["t"] else node["right"])

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        return _ComparableFloat(r2_score(y, y_pred))

    # convenience
    def count_leaves(self) -> int:
        def _c(nd):
            if nd["type"] == "leaf":
                return 1
            return _c(nd["left"]) + _c(nd["right"])

        return _c(self.tree_)
