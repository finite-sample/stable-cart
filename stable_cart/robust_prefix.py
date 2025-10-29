# stable_cart/robust_prefix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def _winsorize_fit(X: np.ndarray, q: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-feature (low, high) quantiles for winsorization."""
    lo = np.quantile(X, q[0], axis=0)
    hi = np.quantile(X, q[1], axis=0)
    return lo, hi


def _winsorize_apply(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(X, lo), hi)


def _stratified_bootstrap(
    X: np.ndarray, y: np.ndarray, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Class-stratified bootstrap indices."""
    idxs = []
    for c in np.unique(y):
        cidx = np.where(y == c)[0]
        b = rng.choice(cidx, size=len(cidx), replace=True)
        idxs.append(b)
    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _robust_stump_on_node(
    X_split: np.ndarray,
    y_split: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    B: int,
    subsample_frac: float,
    max_bins: int,
    rng: np.random.RandomState,
) -> Optional[Tuple[int, float]]:
    """
    Choose a stump by 'majority validation-loss winner':
    bag depth-1 trees on bootstraps of the node's SPLIT data, evaluate val log-loss,
    bin thresholds per feature to remove micro-jitter, pick (feature, bin) with lowest median
    val-loss;
    set threshold to the median of thresholds in the winning bin.
    """
    n = len(X_split)
    if n < 10 or len(np.unique(y_split)) < 2 or len(X_val) < 8:
        return None

    # Buckets: (feat, bin_idx, lo, hi) -> list[vloss], list[thr]
    from collections import defaultdict

    bucket_losses: Dict[Tuple[int, int, float, float], List[float]] = defaultdict(list)
    bucket_thresholds: Dict[Tuple[int, int, float, float], List[float]] = defaultdict(list)

    m = max(10, int(subsample_frac * n))
    for _ in range(B):
        Xb, yb = _stratified_bootstrap(X_split, y_split, rng)
        Xb, yb = Xb[:m], yb[:m]
        stump = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(0, 10**9))
        stump.fit(Xb, yb)
        proba = stump.predict_proba(X_val)
        # guard 1-class stumps
        if proba.shape[1] == 1:
            cls = stump.classes_[0]
            p1 = proba[:, 0] if cls == 1 else 1.0 - proba[:, 0]
        else:
            p1 = proba[:, 1]
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        vloss = log_loss(y_val, p1, labels=[0, 1])

        feat = int(stump.tree_.feature[0])
        thr = float(stump.tree_.threshold[0])
        col = X_split[:, feat]
        lo, hi = float(col.min()), float(col.max())
        if hi <= lo:
            bin_idx = 0
        else:
            pos = (thr - lo) / (hi - lo + 1e-12)
            bin_idx = int(np.floor(pos * max_bins))
            bin_idx = max(0, min(max_bins - 1, bin_idx))
        key = (feat, bin_idx, lo, hi)
        bucket_losses[key].append(vloss)
        bucket_thresholds[key].append(thr)

    if not bucket_losses:
        return None
    # Min median val-loss across buckets
    best_key = min(bucket_losses.keys(), key=lambda k: np.median(bucket_losses[k]))
    best_feat = int(best_key[0])
    best_thr = float(np.median(bucket_thresholds[best_key]))
    # clamp into feature range
    lo, hi = best_key[2], best_key[3]
    best_thr = min(max(best_thr, lo), hi)
    return best_feat, best_thr


@dataclass
class RobustPrefixHonestClassifier(BaseEstimator, ClassifierMixin):
    """
    A single-tree classifier that reduces OOS prediction variance by:

    1) **Robust prefix**: lock the top `top_levels` splits via multi-bootstrap
       validation-loss consensus at each node;
    2) **Honest leaves**: fit the leaf structure on SPLIT, estimate leaf probabilities on EST;
    3) **m-estimate smoothing**: stabilise leaf probabilities via `(k + m p0)/(n + m)`;
    4) **Winsorization** and **stratified bootstraps** inside the node-level consensus.

    Parameters
    ----------
    top_levels : int, default=2
        Number of prefix levels to lock using robust split selection.
    max_depth : int, default=6
        Overall tree depth budget (locked prefix + subtree depth).
    min_samples_leaf : int, default=2
        Minimum samples per leaf for the subtrees grown inside regions.
    val_frac : float, default=0.2
        Fraction of training data used as validation for node-level consensus.
    est_frac : float, default=0.4
        Fraction of training data (of each region) used to estimate leaf probabilities (honesty).
    m_smooth : float, default=1.0
        m in the m-estimate smoothing; prior set to global prevalence `p0`.
    winsor_quantiles : tuple(float, float), default=(0.01, 0.99)
        Per-feature quantiles used to winsorize features.
    consensus_B : int, default=12
        Number of bootstrap stumps per node for consensus selection.
    consensus_subsample_frac : float, default=0.8
        Subsample fraction for each stump's training set at the node.
    consensus_max_bins : int, default=24
        Discretisation bins for threshold bucketing.
    random_state : Optional[int], default=None
        RNG seed.

    Notes
    -----
    * Binary classification only (multi-class is a straightforward Dirichlet-smoothed extension).
    * Mirrors the SPLIT / VAL / EST "honest" pattern described in the project docs.
    """

    top_levels: int = 2
    max_depth: int = 6
    min_samples_leaf: int = 2
    val_frac: float = 0.2
    est_frac: float = 0.4
    m_smooth: float = 1.0
    winsor_quantiles: Tuple[float, float] = (0.01, 0.99)
    consensus_B: int = 12
    consensus_subsample_frac: float = 0.8
    consensus_max_bins: int = 24
    random_state: Optional[int] = None

    # learned state
    _lo_: Optional[np.ndarray] = None
    _hi_: Optional[np.ndarray] = None
    _prefix_nodes_: Optional[List[Tuple[int, int, float, int, int]]] = (
        None  # (nid, feat, thr, left_id, right_id)
    )
    _region_models_: Optional[Dict[int, DecisionTreeClassifier]] = None
    _region_leaf_probs_: Optional[Dict[int, Dict[int, float]]] = None  # region -> {leaf_id: p1}
    classes_: Optional[np.ndarray] = None

    def _route_mask(self, X: np.ndarray, path: List[Tuple[int, float, str]]) -> np.ndarray:
        m = np.ones(len(X), dtype=bool)
        for f, t, side in path:
            m &= (X[:, f] <= t) if side == "L" else (X[:, f] > t)
        return m

    def _route_node_ids(self, X: np.ndarray) -> np.ndarray:
        ids = np.zeros(len(X), dtype=int)
        for nid, f, thr, L, R in self._prefix_nodes_ or []:
            if f is None:
                continue
            left = (X[:, f] <= thr) & (ids == nid)
            right = (X[:, f] > thr) & (ids == nid)
            ids[left] = L
            ids[right] = R
        return ids

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) > 2:
            raise ValueError(
                "RobustPrefixHonestClassifier currently supports binary classification only."
            )

        rng = np.random.RandomState(self.random_state)
        # winsorize (store quantiles for inference)
        self._lo_, self._hi_ = _winsorize_fit(X, self.winsor_quantiles)
        Xw = _winsorize_apply(X, self._lo_, self._hi_)

        # honest partition: SPLIT / VAL / EST (global, then routed)
        X_split, X_tmp, y_split, y_tmp = train_test_split(
            Xw,
            y,
            test_size=self.val_frac + self.est_frac,
            random_state=rng.randint(0, 10**9),
            stratify=y,
        )
        rel = (
            (self.est_frac / (self.val_frac + self.est_frac))
            if (self.val_frac + self.est_frac) > 0
            else 0.5
        )
        X_val, X_est, y_val, y_est = train_test_split(
            X_tmp, y_tmp, test_size=rel, random_state=rng.randint(0, 10**9), stratify=y_tmp
        )

        # build locked prefix (level-order)
        self._prefix_nodes_ = []
        node_queue: List[Tuple[int, List[Tuple[int, float, str]]]] = [(0, [])]
        level = 0
        while node_queue and level < self.top_levels:
            next_q = []
            for nid, path in node_queue:
                m_split = self._route_mask(X_split, path)
                m_val = self._route_mask(X_val, path)
                Xs, ys = X_split[m_split], y_split[m_split]
                Xv, yv = X_val[m_val], y_val[m_val]
                if len(Xs) < 30 or len(np.unique(ys)) < 2 or len(Xv) < 15:
                    # terminal at this level
                    self._prefix_nodes_.append((nid, None, None, None, None))
                    continue
                cs = _robust_stump_on_node(
                    Xs,
                    ys,
                    Xv,
                    yv,
                    B=self.consensus_B,
                    subsample_frac=self.consensus_subsample_frac,
                    max_bins=self.consensus_max_bins,
                    rng=np.random.RandomState(rng.randint(0, 10**9)),
                )
                if cs is None:
                    self._prefix_nodes_.append((nid, None, None, None, None))
                    continue
                f, t = cs
                L, R = 2 * nid + 1, 2 * nid + 2
                self._prefix_nodes_.append((nid, f, t, L, R))
                next_q.append((L, path + [(f, t, "L")]))
                next_q.append((R, path + [(f, t, "R")]))
            node_queue = next_q
            level += 1

        # collect terminal paths
        terminal_paths: List[Tuple[int, List[Tuple[int, float, str]]]] = []
        locked = {nid: (f, t, L, R) for (nid, f, t, L, R) in self._prefix_nodes_}

        def gather(nid: int, path: List[Tuple[int, float, str]], lvl: int):
            if nid not in locked or locked[nid][0] is None or lvl == self.top_levels:
                terminal_paths.append((nid, path))
                return
            f, t, L, R = locked[nid]
            gather(L, path + [(f, t, "L")], lvl + 1)
            gather(R, path + [(f, t, "R")], lvl + 1)

        if not self._prefix_nodes_:
            terminal_paths.append((0, []))
        else:
            gather(0, [], 0)

        # train per-region subtrees (structure on SPLIT; leaf probs estimated on EST with
        # m-smoothing)
        remain = max(self.max_depth - self.top_levels, 0)
        self._region_models_ = {}
        self._region_leaf_probs_ = {}
        p0 = float(y.mean())

        for nid, path in terminal_paths:
            m_split = self._route_mask(X_split, path)
            m_est = self._route_mask(X_est, path)
            Xs, ys = X_split[m_split], y_split[m_split]
            Xe, ye = X_est[m_est], y_est[m_est]

            # Structure: subtree on SPLIT
            if remain == 0 or len(ys) < 12 or len(np.unique(ys)) < 2:
                subtree = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
            else:
                subtree = DecisionTreeClassifier(
                    max_depth=remain,
                    min_samples_leaf=max(self.min_samples_leaf, 2),
                    random_state=self.random_state,
                )
            subtree.fit(Xs, ys)
            self._region_models_[nid] = subtree

            # EST leaf probabilities with m-smoothing
            leaf_probs: Dict[int, float] = {}
            # prefer EST; fall back to SPLIT if EST is empty
            Xleaf, yleaf = (Xe, ye) if len(ye) > 0 else (Xs, ys)
            leaves = subtree.apply(Xleaf)
            unique = np.unique(leaves)
            for lid in unique:
                m = leaves == lid
                n_leaf = int(m.sum())
                k_leaf = int(yleaf[m].sum())
                phat = (k_leaf + self.m_smooth * p0) / (n_leaf + self.m_smooth)
                leaf_probs[int(lid)] = float(phat)
            # (optional) ensure every leaf has an entry
            all_leaves = np.unique(subtree.apply(Xs))
            for lid in all_leaves:
                if int(lid) not in leaf_probs:
                    # backfill from SPLIT counts
                    m = subtree.apply(Xs) == lid
                    n_leaf = int(m.sum())
                    k_leaf = int(ys[m].sum())
                    phat = (k_leaf + self.m_smooth * p0) / (n_leaf + self.m_smooth)
                    leaf_probs[int(lid)] = float(phat)

            self._region_leaf_probs_[nid] = leaf_probs

        self.classes_ = np.array([0, 1], dtype=int)
        return self

    # --- sklearn API ---
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            [
                "_lo_",
                "_hi_",
                "_prefix_nodes_",
                "_region_models_",
                "_region_leaf_probs_",
                "classes_",
            ],
        )
        X = np.asarray(X)
        Xw = _winsorize_apply(X, self._lo_, self._hi_)
        ids = self._route_node_ids(Xw)

        proba = np.zeros((len(Xw), 2), dtype=float)
        for nid, subtree in self._region_models_.items():
            mask = ids == nid
            if not mask.any():
                continue
            leaves = subtree.apply(Xw[mask])
            # look up smoothed probs
            p1 = np.array(
                [self._region_leaf_probs_[nid].get(int(lid), 0.5) for lid in leaves], dtype=float
            )
            p1 = np.clip(p1, 1e-7, 1 - 1e-7)
            proba[mask, 1] = p1
            proba[mask, 0] = 1.0 - p1
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)
