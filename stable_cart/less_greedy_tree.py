# stable_cart/less_greedy_tree.py
"""
Unified honest tree implementation for regression and classification with advanced stability
features.

This tree trades some accuracy for substantially improved prediction stability via:
- Honest data partitioning: SPLIT (structure), VAL (validation), EST (leaf estimation)
- Optional oblique root: Linear projections using Lasso (regression) or LogisticRegression
  (classification)
- Honest k-step lookahead: Beam search with multi-step validation when top splits are
  ambiguous
- Leaf smoothing: Shrinkage to parent (regression) or m-estimate (classification)
"""

import operator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Data Classes for Split Results
# ============================================================================


@dataclass(slots=True)
class AxisSplit:
    """Result of axis-aligned split selection."""

    feature: int
    threshold: float


@dataclass(slots=True)
class ObliqueSplit:
    """Result of oblique split selection."""

    threshold: float
    mean_values: np.ndarray  # scaling mean
    scale_values: np.ndarray  # scaling scale
    weights: np.ndarray  # oblique weights


# ============================================================================
# Scoring Functions (Task-Specific)
# ============================================================================


def _sse(y: np.ndarray) -> float:
    """
    Sum of squared errors around mean (regression).

    Parameters
    ----------
    y
        Target values array.

    Returns
    -------
    float
        Sum of squared errors.
    """
    if y.size <= 1:
        return 0.0
    return float(np.var(y) * y.size)


def _gini_impurity(y: np.ndarray) -> float:
    """
    Gini impurity (classification).

    Parameters
    ----------
    y
        Class labels array.

    Returns
    -------
    float
        Gini impurity value.
    """
    if y.size <= 1:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / y.size
    return float(y.size * (1.0 - np.sum(probs**2)))


def _entropy(y: np.ndarray) -> float:
    """
    Entropy (alternative classification criterion).

    Parameters
    ----------
    y
        Class labels array.

    Returns
    -------
    float
        Entropy value.
    """
    if y.size <= 1:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / y.size
    probs = probs[probs > 0]  # avoid log(0)
    return float(-y.size * np.sum(probs * np.log2(probs)))


# ============================================================================
# Utilities
# ============================================================================


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


# ============================================================================
# Unified Tree Implementation
# ============================================================================


class LessGreedyHybridTree(BaseEstimator):
    """
    Unified honest tree for regression and classification with advanced stability features.

    This tree trades some accuracy for substantially improved prediction stability via:

    1. **Honest data partitioning**: Separates data for structure (SPLIT), validation (VAL),
       and leaf estimation (EST) to prevent overfitting
    2. **Optional oblique root**: Linear projections at root using Lasso (regression) or
       LogisticRegression (classification) with correlation gating
    3. **Honest k-step lookahead**: Beam search with multi-step validation when top splits
       are ambiguous (reduces variance from greedy myopia)
    4. **Leaf smoothing**: Shrinkage to parent (regression) or m-estimate (classification)

    Parameters
    ----------
    task
        Type of prediction task.
    max_depth
        Maximum tree depth.
    min_samples_split
        Minimum samples to split a node.
    min_samples_leaf
        Minimum samples per leaf.
    split_frac
        Fraction of data for structure learning (SPLIT).
    val_frac
        Fraction of data for validation (VAL).
    est_frac
        Fraction of data for leaf estimation (EST).
    enable_oblique_root
        Enable linear projections at root node.
    gain_margin
        Minimum gain advantage for oblique vs axis-aligned splits.
    min_abs_corr
        Minimum feature correlation to enable oblique splits.
    oblique_cv
        CV folds for regularization parameter selection.
    beam_topk
        Beam width for candidate tracking.
    ambiguity_eps
        Gain margin for triggering lookahead.
    min_n_for_lookahead
        Minimum samples to enable lookahead.
    root_k
        Lookahead depth at root.
    inner_k
        Lookahead depth at inner nodes.
    leaf_smoothing
        Smoothing parameter (lambda for regression, m for classification).
    classification_criterion
        Split criterion for classification.
    random_state
        Random seed.

    Raises
    ------
    ValueError
        If task is not 'regression' or 'classification'.

    Examples
    --------
    Regression:
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    >>> tree = LessGreedyHybridTree(task='regression', max_depth=4)
    >>> tree.fit(X, y)
    >>> predictions = tree.predict(X)

    Classification:
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> tree = LessGreedyHybridTree(task='classification', max_depth=4)
    >>> tree.fit(X, y)
    >>> probas = tree.predict_proba(X)

    Notes
    -----
    - Honest partitioning reduces effective sample size but prevents overfitting
    - Oblique splits can capture linear relationships but add computational cost
    - Lookahead is expensive but valuable when top splits are ambiguous
    - This algorithm prioritizes stability over maximum accuracy
    """

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        max_depth: int = 5,
        min_samples_split: int = 40,
        min_samples_leaf: int = 20,
        split_frac: float = 0.6,
        val_frac: float = 0.2,
        est_frac: float = 0.2,
        # oblique root
        enable_oblique_root: bool = True,
        gain_margin: float = 0.03,
        min_abs_corr: float = 0.3,
        oblique_cv: int = 5,
        # lookahead (axis-only)
        beam_topk: int = 12,
        ambiguity_eps: float = 0.05,
        min_n_for_lookahead: int = 600,
        root_k: int = 2,
        inner_k: int = 1,
        # leaf smoothing
        leaf_smoothing: float = 0.0,
        # classification-specific
        classification_criterion: Literal["gini", "entropy"] = "gini",
        random_state: int = 0,
    ):
        self.task = task
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
        self.leaf_smoothing = leaf_smoothing
        self.classification_criterion = classification_criterion
        self.random_state = random_state

        # Learned attributes
        self.tree_: dict[str, Any] = {}
        self.classes_: np.ndarray | None = None
        self._global_prior_: float = 0.0

        # Set task-specific functions
        if task == "regression":
            self._loss_fn = _sse
        elif task == "classification":
            if classification_criterion == "gini":
                self._loss_fn = _gini_impurity
            else:
                self._loss_fn = _entropy
        else:
            raise ValueError("task must be 'regression' or 'classification'")

    def _children_sse_vec(
        self, xs: np.ndarray, ys: np.ndarray, min_leaf: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized computation of children loss along sorted feature.

        Parameters
        ----------
        xs
            Sorted feature values.
        ys
            Corresponding target values.
        min_leaf
            Minimum samples per leaf.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Combined loss array and validity mask.
        """
        n = ys.size
        if n < 2 * min_leaf:
            return np.array([]), np.array([], dtype=bool)

        if self.task == "regression":
            # SSE-based (original code)
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

            sseL = np.where(nL > 0, sumL2 - (sumL * sumL) / nL, np.inf)
            sseR = np.where(nR > 0, sumR2 - (sumR * sumR) / nR, np.inf)

            return sseL + sseR, valid

        else:
            # Impurity-based for classification
            # Compute weighted impurity for each split
            idx = np.arange(n - 1)
            valid = xs[:-1] != xs[1:]
            nL = idx + 1
            nR = n - nL
            valid &= (nL >= min_leaf) & (nR >= min_leaf)

            impurities = np.full(n - 1, np.inf, dtype=float)

            for i in np.where(valid)[0]:
                left_impurity = self._loss_fn(ys[: i + 1])
                right_impurity = self._loss_fn(ys[i + 1 :])
                # Weighted by size
                impurities[i] = (nL[i] / n) * left_impurity + (
                    nR[i] / n
                ) * right_impurity

            return impurities, valid

    def _topk_axis_candidates(
        self, Xs: np.ndarray, ys: np.ndarray, topk: int
    ) -> list[tuple[float, int, float]]:
        """
        Find top-k axis-aligned split candidates by gain.

        Parameters
        ----------
        Xs
            Feature array.
        ys
            Target array.
        topk
            Number of top candidates to return.

        Returns
        -------
        list[tuple[float, int, float]]
            List of (gain, feature, threshold) tuples.
        """
        parent_loss = self._loss_fn(ys)
        gains = []
        p = Xs.shape[1]

        for j in range(p):
            order = np.argsort(Xs[:, j], kind="mergesort")
            xs = Xs[order, j]
            ys_ord = ys[order]
            children_loss, valid = self._children_sse_vec(
                xs, ys_ord, self.min_samples_leaf
            )

            if not valid.any():
                continue

            thr = 0.5 * (xs[:-1] + xs[1:])
            idx = np.where(valid)[0]
            g = parent_loss - children_loss[idx]

            for i, gi in zip(idx, g, strict=True):
                gains.append((float(gi), int(j), float(thr[i])))

        if not gains:
            return []

        gains.sort(key=lambda t: t[0], reverse=True)
        return gains[:topk]

    def _val_loss_leaf(self, yv: np.ndarray) -> float:
        """
        Validation loss if we make this a leaf.

        Parameters
        ----------
        yv
            Validation targets.

        Returns
        -------
        float
            Validation loss value.
        """
        return self._loss_fn(yv)

    def _val_loss_after_split(
        self, Xv: np.ndarray, yv: np.ndarray, feat: int, thr: float
    ) -> float:
        """
        Validation loss after applying split.

        Parameters
        ----------
        Xv
            Validation features.
        yv
            Validation targets.
        feat
            Feature index for split.
        thr
            Threshold value for split.

        Returns
        -------
        float
            Validation loss after split.
        """
        mask = Xv[:, feat] <= thr
        nL, nR = mask.sum(), (~mask).sum()
        n = len(yv)

        if nL == 0 or nR == 0:
            return self._val_loss_leaf(yv)

        loss_L = self._loss_fn(yv[mask])
        loss_R = self._loss_fn(yv[~mask])

        # Weighted by size
        return (nL / n) * loss_L + (nR / n) * loss_R

    def _best_kstep_val_loss(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        Xv: np.ndarray,
        yv: np.ndarray,
        depth_remaining: int,
        topk: int,
    ) -> float:
        """
        Best validation loss achievable with k-step lookahead.

        Parameters
        ----------
        Xs
            Split features.
        ys
            Split targets.
        Xv
            Validation features.
        yv
            Validation targets.
        depth_remaining
            Remaining lookahead depth.
        topk
            Number of top candidates to consider.

        Returns
        -------
        float
            Best achievable validation loss.
        """
        if depth_remaining <= 0 or ys.size < self.min_samples_split:
            return self._val_loss_leaf(yv)

        cand = self._topk_axis_candidates(Xs, ys, topk)
        if not cand:
            return self._val_loss_leaf(yv)

        best = np.inf
        for _, f, t in cand:
            mask_s = Xs[:, f] <= t
            if (
                mask_s.sum() < self.min_samples_leaf
                or (ys.size - int(mask_s.sum())) < self.min_samples_leaf
            ):
                continue

            mask_v = Xv[:, f] <= t
            lossL = self._best_kstep_val_loss(
                Xs[mask_s],
                ys[mask_s],
                Xv[mask_v],
                yv[mask_v],
                depth_remaining - 1,
                topk,
            )
            lossR = self._best_kstep_val_loss(
                Xs[~mask_s],
                ys[~mask_s],
                Xv[~mask_v],
                yv[~mask_v],
                depth_remaining - 1,
                topk,
            )

            # Weighted
            nL, nR = mask_v.sum(), (~mask_v).sum()
            n = len(yv)
            tot = (nL / n) * lossL + (nR / n) * lossR if n > 0 else lossL + lossR

            if tot < best:
                best = tot

        if not np.isfinite(best):
            return self._val_loss_leaf(yv)

        return float(best)

    def _fit_oblique_projection(
        self, Xs: np.ndarray, ys: np.ndarray, cv_folds: int
    ) -> tuple[np.ndarray, StandardScaler, float]:
        """
        Fit linear projection for oblique split.

        Parameters
        ----------
        Xs
            Split subset features.
        ys
            Split subset targets.
        cv_folds
            Number of cross-validation folds.

        Returns
        -------
        tuple[np.ndarray, StandardScaler, float]
            Coefficients, scaler, and alpha value.
        """
        scaler = StandardScaler()
        Xs_std = scaler.fit_transform(Xs)

        # Adjust CV for small datasets
        cv_folds = min(cv_folds, Xs.shape[0])
        if cv_folds < 2:
            cv_folds = 2

        if self.task == "regression":
            lcv = LassoCV(cv=cv_folds, random_state=self.random_state)
            lcv.fit(Xs_std, ys)
            w = lcv.coef_.astype(float)
            alpha = float(lcv.alpha_)
        else:
            # Classification: use L1-regularized logistic regression
            lcv = LogisticRegressionCV(
                cv=cv_folds,
                penalty="l1",
                solver="saga",
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=1,
            )
            lcv.fit(Xs_std, ys)

            # Handle coefficient shape (binary vs multi-class)
            if len(lcv.coef_.shape) == 1:
                w = lcv.coef_.astype(float)
            else:
                w = lcv.coef_[0].astype(float)

            # Note: LogisticRegressionCV uses C (inverse reg), not alpha
            alpha = float(1.0 / lcv.C_[0]) if hasattr(lcv, "C_") else 1.0

        return w, scaler, alpha

    def _build(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        Xv: np.ndarray,
        yv: np.ndarray,
        Xe: np.ndarray,
        ye: np.ndarray,
        depth: int,
        parent_mean_est: float,
    ) -> dict[str, Any]:
        """
        Recursively build tree.

        Parameters
        ----------
        Xs
            Split subset features.
        ys
            Split subset targets.
        Xv
            Validation subset features.
        yv
            Validation subset targets.
        Xe
            Estimation subset features.
        ye
            Estimation subset targets.
        depth
            Current tree depth.
        parent_mean_est
            Parent mean estimate.

        Returns
        -------
        dict[str, Any]
            Tree node dictionary.
        """
        n_split = ys.size
        n_val = yv.size

        # Stopping conditions
        if depth >= self.max_depth or n_split < self.min_samples_split:
            return self._make_leaf(ye, ys, parent_mean_est, n_split, n_val)

        # For classification, check purity
        if self.task == "classification" and len(np.unique(ys)) == 1:
            return self._make_leaf(ye, ys, parent_mean_est, n_split, n_val)

        # Default: no split
        best_val_loss = self._val_loss_leaf(yv)
        best_kind = "leaf"
        best_info = None

        # Get axis-aligned candidates
        cand_axis = self._topk_axis_candidates(Xs, ys, self.beam_topk)

        # Immediate axis scoring (k=0)
        if cand_axis:
            loss_list = []
            for _, f, t in cand_axis:
                loss = self._val_loss_after_split(Xv, yv, f, t)
                loss_list.append((loss, f, t))

            loss_list.sort(key=lambda t: t[0])
            best_axis_immediate = loss_list[0]

            if best_axis_immediate[0] + 1e-12 < best_val_loss:
                best_val_loss = best_axis_immediate[0]
                best_kind = "axis"
                best_info = AxisSplit(best_axis_immediate[1], best_axis_immediate[2])

        # Ambiguity gate for lookahead
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

        # Honest k-step lookahead
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
                lossL = self._best_kstep_val_loss(
                    Xs[mask_s],
                    ys[mask_s],
                    Xv[mask_v],
                    yv[mask_v],
                    k_here - 1,
                    self.beam_topk,
                )
                lossR = self._best_kstep_val_loss(
                    Xs[~mask_s],
                    ys[~mask_s],
                    Xv[~mask_v],
                    yv[~mask_v],
                    k_here - 1,
                    self.beam_topk,
                )

                nL, nR = mask_v.sum(), (~mask_v).sum()
                n = len(yv)
                tot = (nL / n) * lossL + (nR / n) * lossR if n > 0 else lossL + lossR

                if tot < best_la[0]:
                    best_la = (tot, f, t)

            if (
                np.isfinite(best_la[0])
                and best_la[0] + 1e-12 < best_val_loss
                and best_la[1] is not None
                and best_la[2] is not None
            ):
                best_val_loss = best_la[0]
                best_kind = "axis"
                best_info = AxisSplit(best_la[1], best_la[2])

        # Oblique root with gating
        if self.enable_oblique_root and depth == 0 and Xs.shape[1] >= 2:
            # Check feature correlations
            R = np.corrcoef(Xs, rowvar=False)
            max_abs_corr = float(np.nanmax(np.abs(R - np.eye(R.shape[0]))))
            axis_gain_top = cand_axis[0][0] if cand_axis else -np.inf

            # Only try oblique if features are correlated
            if max_abs_corr >= self.min_abs_corr:
                try:
                    w, scaler, alpha = self._fit_oblique_projection(
                        Xs, ys, self.oblique_cv
                    )

                    if np.count_nonzero(w) > 0:
                        # Project data
                        mean_ = np.asarray(scaler.mean_)
                        scale_ = np.asarray(scaler.scale_)
                        s = (Xs - mean_) / scale_ @ w
                        order = np.argsort(s, kind="mergesort")
                        ss = s[order]
                        ys_ord = ys[order]

                        # Find best split on projection
                        children_loss, valid = self._children_sse_vec(
                            ss, ys_ord, self.min_samples_leaf
                        )

                        if valid.any():
                            idx = np.where(valid)[0]
                            i = idx[np.argmin(children_loss[idx])]
                            t = 0.5 * (ss[i] + ss[i + 1])

                            parent_loss_split = self._loss_fn(ys)
                            oblique_gain_split = parent_loss_split - children_loss[i]

                            # Check if oblique beats axis by margin
                            if not np.isfinite(
                                axis_gain_top
                            ) or oblique_gain_split >= axis_gain_top * (
                                1.0 + self.gain_margin
                            ):
                                # Score on VAL
                                sv = (Xv - mean_) / scale_
                                s_val = sv @ w
                                mask_val = s_val <= t

                                loss_obl = self._val_loss_after_split_mask(yv, mask_val)

                                if loss_obl + 1e-12 < best_val_loss:
                                    best_val_loss = loss_obl
                                    best_kind = "oblique_root"
                                    best_info = ObliqueSplit(
                                        float(t),
                                        mean_.astype(float),
                                        scale_.astype(float),
                                        w.astype(float),
                                    )
                except Exception:
                    # Oblique split failed, continue with axis
                    pass

        # Commit decision and recurse
        if best_kind == "leaf":
            return self._make_leaf(ye, ys, parent_mean_est, n_split, n_val)

        if best_kind == "axis":
            assert isinstance(best_info, AxisSplit)
            return self._make_axis_split(
                best_info,
                Xs,
                ys,
                Xv,
                yv,
                Xe,
                ye,
                depth,
                parent_mean_est,
                n_split,
                n_val,
            )

        if best_kind == "oblique_root":
            assert isinstance(best_info, ObliqueSplit)
            return self._make_oblique_split(
                best_info,
                Xs,
                ys,
                Xv,
                yv,
                Xe,
                ye,
                depth,
                parent_mean_est,
                n_split,
                n_val,
            )

        # Fallback
        return self._make_leaf(ye, ys, parent_mean_est, n_split, n_val)

    def _val_loss_after_split_mask(self, yv: np.ndarray, mask: np.ndarray) -> float:
        """
        Helper for computing val loss given a mask.

        Parameters
        ----------
        yv
            Validation targets.
        mask
            Boolean mask for split.

        Returns
        -------
        float
            Validation loss after applying mask.
        """
        nL, nR = mask.sum(), (~mask).sum()
        n = len(yv)
        if nL == 0 or nR == 0:
            return self._val_loss_leaf(yv)
        return (nL / n) * self._loss_fn(yv[mask]) + (nR / n) * self._loss_fn(yv[~mask])

    def _make_leaf(
        self,
        ye: np.ndarray,
        ys: np.ndarray,
        parent_mean_est: float,
        n_split: int,
        n_val: int,
    ) -> dict[str, Any]:
        """
        Create a leaf node with task-appropriate value.

        Parameters
        ----------
        ye
            Estimation targets.
        ys
            Split targets.
        parent_mean_est
            Parent mean estimate.
        n_split
            Number of split samples.
        n_val
            Number of validation samples.

        Returns
        -------
        dict[str, Any]
            Leaf node dictionary.
        """
        if self.task == "regression":
            mu_leaf = float(ye.mean()) if ye.size > 0 else float(ys.mean())
            lam = self.leaf_smoothing
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
        else:
            # Classification: compute class probabilities with m-estimate
            if ye.size > 0:
                k = int(ye.sum())
                n = int(ye.size)
            else:
                k = int(ys.sum())
                n = int(ys.size)

            m = self.leaf_smoothing
            p0 = self._global_prior_

            p1 = (k + m * p0) / (n + m) if m > 0 else (k / n if n > 0 else 0.5)

            return {
                "type": "leaf",
                "proba": float(p1),  # P(class=1)
                "n_split": int(n_split),
                "n_val": int(n_val),
                "n_est": int(ye.size),
            }

    def _make_axis_split(
        self,
        best_info: AxisSplit,
        Xs: np.ndarray,
        ys: np.ndarray,
        Xv: np.ndarray,
        yv: np.ndarray,
        Xe: np.ndarray,
        ye: np.ndarray,
        depth: int,
        parent_mean_est: float,
        n_split: int,
        n_val: int,
    ) -> dict[str, Any]:
        """
        Create axis-aligned split node.

        Parameters
        ----------
        best_info
            Best split information tuple.
        Xs
            Split features.
        ys
            Split targets.
        Xv
            Validation features.
        yv
            Validation targets.
        Xe
            Estimation features.
        ye
            Estimation targets.
        depth
            Current depth.
        parent_mean_est
            Parent mean estimate.
        n_split
            Number of split samples.
        n_val
            Number of validation samples.

        Returns
        -------
        dict[str, Any]
            Split node dictionary.
        """
        f, t = best_info.feature, best_info.threshold
        mask_s = Xs[:, f] <= t
        mask_v = Xv[:, f] <= t
        mask_e = Xe[:, f] <= t if Xe.size else np.array([], dtype=bool)

        # Safety check
        if (
            mask_s.sum() < self.min_samples_leaf
            or (ys.size - int(mask_s.sum())) < self.min_samples_leaf
        ):
            return self._make_leaf(ye, ys, parent_mean_est, n_split, n_val)

        node = {
            "type": "split",
            "f": int(f),
            "t": float(t),
            "n_split": int(n_split),
            "n_val": int(n_val),
            "n_est": int(ye.size),
        }

        new_parent = self._compute_parent_for_child(ye, ys)

        node["left"] = self._build(
            Xs[mask_s],
            ys[mask_s],
            Xv[mask_v],
            yv[mask_v],
            Xe[mask_e] if Xe.size else Xe,
            ye[mask_e] if Xe.size else ye,
            depth + 1,
            new_parent,
        )

        node["right"] = self._build(
            Xs[~mask_s],
            ys[~mask_s],
            Xv[~mask_v],
            yv[~mask_v],
            Xe[~mask_e] if Xe.size else Xe,
            ye[~mask_e] if Xe.size else ye,
            depth + 1,
            new_parent,
        )

        return node

    def _make_oblique_split(
        self,
        best_info: ObliqueSplit,
        Xs: np.ndarray,
        ys: np.ndarray,
        Xv: np.ndarray,
        yv: np.ndarray,
        Xe: np.ndarray,
        ye: np.ndarray,
        depth: int,
        parent_mean_est: float,
        n_split: int,
        n_val: int,
    ) -> dict[str, Any]:
        """
        Create oblique split node.

        Parameters
        ----------
        best_info
            Oblique split information tuple.
        Xs
            Split features.
        ys
            Split targets.
        Xv
            Validation features.
        yv
            Validation targets.
        Xe
            Estimation features.
        ye
            Estimation targets.
        depth
            Current depth.
        parent_mean_est
            Parent mean estimate.
        n_split
            Number of split samples.
        n_val
            Number of validation samples.

        Returns
        -------
        dict[str, Any]
            Oblique split node dictionary.
        """
        t, mean, scale, w = (
            best_info.threshold,
            best_info.mean_values,
            best_info.scale_values,
            best_info.weights,
        )

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

        new_parent = self._compute_parent_for_child(ye, ys)

        node["left"] = self._build(
            Xs[mask_s],
            ys[mask_s],
            Xv[mask_v],
            yv[mask_v],
            Xe[mask_e] if Xe.size else Xe,
            ye[mask_e] if Xe.size else ye,
            depth + 1,
            new_parent,
        )

        node["right"] = self._build(
            Xs[~mask_s],
            ys[~mask_s],
            Xv[~mask_v],
            yv[~mask_v],
            Xe[~mask_e] if Xe.size else Xe,
            ye[~mask_e] if Xe.size else ye,
            depth + 1,
            new_parent,
        )

        return node

    def _compute_parent_for_child(self, ye: np.ndarray, ys: np.ndarray) -> float:
        """
        Compute parent value for shrinkage.

        Parameters
        ----------
        ye
            Estimation targets.
        ys
            Split targets.

        Returns
        -------
        float
            Parent value for shrinkage.
        """
        if self.task == "regression":
            return float(ye.mean()) if ye.size > 0 else float(ys.mean())
        else:
            # For classification, return global prior
            return self._global_prior_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LessGreedyHybridTree":
        """
        Fit the tree.

        Parameters
        ----------
        X
            Training features.
        y
            Target values.

        Returns
        -------
        LessGreedyHybridTree
            Fitted estimator.

        Raises
        ------
        ValueError
            If multi-class classification is attempted or data fractions are invalid.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must contain at least one sample.")

        assert (
            0 < self.split_frac < 1 and 0 < self.val_frac < 1 and 0 < self.est_frac < 1
        )
        assert abs((self.split_frac + self.val_frac + self.est_frac) - 1.0) < 1e-8, (
            "split_frac + val_frac + est_frac must sum to 1"
        )

        # Classification-specific setup
        if self.task == "classification":
            y = y.astype(int)
            self.classes_ = np.unique(y)
            if len(self.classes_) > 2:
                raise ValueError(
                    "Multi-class not yet supported. Use binary classification."
                )
            self._global_prior_ = float(y.mean())
        else:
            self._global_prior_ = float(y.mean())

        rng = np.random.default_rng(self.random_state)

        # Honest partition
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

        parent_mean_est = self._compute_parent_for_child(ye, ys)

        self.tree_ = self._build(
            Xs, ys, Xv, yv, Xe, ye, depth=0, parent_mean_est=parent_mean_est
        )

        return self

    def _predict_one(self, x: np.ndarray, node: dict[str, Any]) -> float:
        """
        Predict for single sample.

        Parameters
        ----------
        x
            Single sample features.
        node
            Current tree node.

        Returns
        -------
        float
            Predicted value or probability.
        """
        if node["type"] == "leaf":
            if self.task == "regression":
                return node["value"]
            else:
                return node["proba"]  # Return P(class=1)

        if node["type"] == "split":
            if x[node["f"]] <= node["t"]:
                return self._predict_one(x, node["left"])
            else:
                return self._predict_one(x, node["right"])

        # Oblique
        s = ((x - node["scaler_mean"]) / node["scaler_scale"]).dot(node["w"])
        if s <= node["t"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or values.

        Parameters
        ----------
        X
            Features.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        X = np.asarray(X)

        if self.task == "classification":
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return np.array([self._predict_one(x, self.tree_) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Parameters
        ----------
        X
            Features.

        Returns
        -------
        np.ndarray
            Class probabilities.

        Raises
        ------
        AttributeError
            If called on regression task.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba only available for classification")

        X = np.asarray(X)
        p1 = np.array([self._predict_one(x, self.tree_) for x in X])
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)

        proba = np.column_stack([1 - p1, p1])
        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return R² (regression) or accuracy (classification).

        Parameters
        ----------
        X
            Test features.
        y
            True labels or values.

        Returns
        -------
        float
            Performance metric.
        """
        y = np.asarray(y)

        if self.task == "regression":
            y_pred = self.predict(X)
            return _ComparableFloat(r2_score(y, y_pred))
        else:
            y_pred = self.predict(X)
            return float(accuracy_score(y, y_pred))


# ============================================================================
# Convenience Wrappers
# ============================================================================
