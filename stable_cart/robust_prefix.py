# stable_cart/robust_prefix.py

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

# ============================================================================
# Preprocessing utilities
# ============================================================================


def _winsorize_fit(
    X: np.ndarray, q: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return per-feature (low, high) quantiles for winsorization.

    Parameters
    ----------
    X
        Feature array.
    q
        Tuple of (low, high) quantiles for clipping.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (low_quantiles, high_quantiles) arrays.

    Raises
    ------
    ValueError
        If X is empty.
    """
    if X.shape[0] == 0:
        raise ValueError("Cannot winsorize empty array")
    lo = np.quantile(X, q[0], axis=0)
    hi = np.quantile(X, q[1], axis=0)
    return lo, hi


def _winsorize_apply(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Apply winsorization using pre-computed quantiles.

    Parameters
    ----------
    X
        Feature array to winsorize.
    lo
        Low quantile bounds per feature.
    hi
        High quantile bounds per feature.

    Returns
    -------
    np.ndarray
        Winsorized feature array.
    """
    return np.minimum(np.maximum(X, lo), hi)


def _stratified_bootstrap(
    X: np.ndarray, y: np.ndarray, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """
    Class-stratified bootstrap (for classification).

    Parameters
    ----------
    X
        Feature array.
    y
        Class labels array.
    rng
        Random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Bootstrapped (X, y) arrays.
    """
    idxs = []
    for c in np.unique(y):
        cidx = np.where(y == c)[0]
        b = rng.choice(cidx, size=len(cidx), replace=True)
        idxs.append(b)
    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _regular_bootstrap(
    X: np.ndarray, y: np.ndarray, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """
    Regular bootstrap (for regression).

    Parameters
    ----------
    X
        Feature array.
    y
        Target values array.
    rng
        Random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Bootstrapped (X, y) arrays.
    """
    n = len(X)
    idx = rng.choice(n, size=n, replace=True)
    return X[idx], y[idx]


# ============================================================================
# Robust consensus split selection
# ============================================================================


def _robust_stump_regression(
    X_split: np.ndarray,
    y_split: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    B: int,
    subsample_frac: float,
    max_bins: int,
    rng: np.random.RandomState,
) -> tuple[int, float] | None:
    """
    Consensus split selection for regression via bootstrap validation.

    For each bootstrap:
    1. Fit shallow tree on bootstrap sample
    2. Evaluate MSE on validation set
    3. Bin thresholds to remove jitter
    4. Select (feature, threshold) with lowest median validation loss

    Parameters
    ----------
    X_split
        Split subset features.
    y_split
        Split subset targets.
    X_val
        Validation subset features.
    y_val
        Validation subset targets.
    B
        Number of bootstrap samples.
    subsample_frac
        Fraction of data to use per bootstrap.
    max_bins
        Maximum bins for threshold discretization.
    rng
        Random number generator.

    Returns
    -------
    tuple[int, float] | None
        Best (feature, threshold) or None if no valid split.
    """
    n = len(X_split)
    if n < 10 or len(X_val) < 8:
        return None

    from collections import defaultdict

    bucket_losses: dict[tuple[int, int, float, float], list[float]] = defaultdict(list)
    bucket_thresholds: dict[tuple[int, int, float, float], list[float]] = defaultdict(
        list
    )

    m = max(10, int(subsample_frac * n))

    for _ in range(B):
        # Bootstrap sample
        Xb, yb = _regular_bootstrap(X_split, y_split, rng)
        Xb, yb = Xb[:m], yb[:m]

        # Fit shallow tree
        stump = DecisionTreeRegressor(max_depth=1, random_state=rng.randint(0, 10**9))
        stump.fit(Xb, yb)

        # Validation loss
        pred = stump.predict(X_val)
        vloss = mean_squared_error(y_val, pred)

        # Extract split info
        feat = int(stump.tree_.feature[0])
        if feat < 0:  # No split found
            continue
        thr = float(stump.tree_.threshold[0])

        # Bin threshold
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

    # Select bucket with minimum median validation loss
    best_key = min(bucket_losses.keys(), key=lambda k: np.median(bucket_losses[k]))
    best_feat = int(best_key[0])
    best_thr = float(np.median(bucket_thresholds[best_key]))

    # Clamp to feature range
    lo, hi = best_key[2], best_key[3]
    best_thr = min(max(best_thr, lo), hi)

    return best_feat, best_thr


def _robust_stump_classification(
    X_split: np.ndarray,
    y_split: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    B: int,
    subsample_frac: float,
    max_bins: int,
    rng: np.random.RandomState,
) -> tuple[int, float] | None:
    """
    Consensus split selection for classification via bootstrap validation.

    Similar to regression but uses log-loss and stratified bootstrapping.

    Parameters
    ----------
    X_split
        Split subset features.
    y_split
        Split subset class labels.
    X_val
        Validation subset features.
    y_val
        Validation subset class labels.
    B
        Number of bootstrap samples.
    subsample_frac
        Fraction of data to use per bootstrap.
    max_bins
        Maximum bins for threshold discretization.
    rng
        Random number generator.

    Returns
    -------
    tuple[int, float] | None
        Best (feature, threshold) or None if no valid split.
    """
    n = len(X_split)
    if n < 10 or len(np.unique(y_split)) < 2 or len(X_val) < 8:
        return None

    from collections import defaultdict

    bucket_losses: dict[tuple[int, int, float, float], list[float]] = defaultdict(list)
    bucket_thresholds: dict[tuple[int, int, float, float], list[float]] = defaultdict(
        list
    )

    m = max(10, int(subsample_frac * n))

    for _ in range(B):
        # Stratified bootstrap
        Xb, yb = _stratified_bootstrap(X_split, y_split, rng)
        Xb, yb = Xb[:m], yb[:m]

        # Fit shallow tree
        stump = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(0, 10**9))
        stump.fit(Xb, yb)

        # Validation loss
        proba = stump.predict_proba(X_val)
        # Handle single-class stumps
        if proba.shape[1] == 1:  # type: ignore[attr-defined]
            cls = stump.classes_[0]
            p1 = proba[:, 0] if cls == 1 else 1.0 - proba[:, 0]  # type: ignore[call-overload]
        else:
            p1 = proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]  # type: ignore[attr-defined,call-overload]
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)

        # For binary: use log_loss directly
        if len(np.unique(y_split)) == 2:
            vloss = log_loss(y_val, p1, labels=[0, 1])
        else:
            vloss = log_loss(y_val, proba)

        # Extract split info
        feat = int(stump.tree_.feature[0])
        if feat < 0:
            continue
        thr = float(stump.tree_.threshold[0])

        # Bin threshold
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

    best_key = min(bucket_losses.keys(), key=lambda k: np.median(bucket_losses[k]))
    best_feat = int(best_key[0])
    best_thr = float(np.median(bucket_thresholds[best_key]))

    lo, hi = best_key[2], best_key[3]
    best_thr = min(max(best_thr, lo), hi)

    return best_feat, best_thr


# ============================================================================
# Unified Tree Implementation
# ============================================================================


@dataclass
class RobustPrefixHonestTree(BaseEstimator):
    """
    Unified honest tree for regression and classification with robust prefix locking.

    This tree reduces OOS prediction variance through:
    1. **Robust prefix**: Lock top `top_levels` splits via bootstrap consensus
    2. **Honest partitioning**: Structure on SPLIT, leaf estimates on EST
    3. **Smoothing**: Shrinkage (regression) or m-estimate (classification)
    4. **Winsorization**: Reduce impact of extreme feature values

    The key innovation is robust consensus-based split selection at the top of the tree,
    where instability has the largest downstream impact. Lower splits use standard greedy
    methods for efficiency.

    Parameters
    ----------
    task : {'regression', 'classification'}, default='regression'
        Type of prediction task.
    top_levels : int, default=2
        Number of prefix levels to lock using robust consensus.
        Higher values = more stability but slower training.
    max_depth : int, default=6
        Overall tree depth (prefix + subtree).
    min_samples_leaf : int, default=2
        Minimum samples per leaf in subtrees.
    val_frac : float, default=0.2
        Fraction of data for validation in consensus selection.
    est_frac : float, default=0.4
        Fraction of data for honest leaf estimation.
    smoothing : float, default=1.0
        Smoothing parameter:
        - Regression: leaf_shrinkage_lambda (shrink to parent mean)
        - Classification: m_smooth (m-estimate smoothing)
    winsor_quantiles : tuple, default=(0.01, 0.99)
        Quantiles for winsorization clipping.
    consensus_B : int, default=12
        Number of bootstrap samples for consensus.
    consensus_subsample_frac : float, default=0.8
        Subsample fraction per bootstrap.
    consensus_max_bins : int, default=24
        Bins for threshold discretization.
    random_state : int, optional
        Random seed.

    Attributes
    ----------
    task : Literal['regression', 'classification']
        Type of prediction task.
    top_levels : int
        Number of prefix levels to lock using robust consensus.
    max_depth : int
        Overall tree depth (prefix + subtree).
    min_samples_leaf : int
        Minimum samples per leaf in subtrees.
    val_frac : float
        Fraction of data for validation in consensus selection.
    est_frac : float
        Fraction of data for honest leaf estimation.
    smoothing : float
        Smoothing parameter for leaf estimates.
    winsor_quantiles : tuple[float, float]
        Quantiles for winsorization clipping.
    consensus_B : int
        Number of bootstrap samples for consensus.
    consensus_subsample_frac : float
        Subsample fraction per bootstrap.
    consensus_max_bins : int
        Bins for threshold discretization.
    random_state : int | None
        Random seed.
    classes_ : ndarray (classification only)
        Unique class labels.

    Examples
    --------
    Regression:
    >>> tree = RobustPrefixHonestTree(task='regression', top_levels=2, max_depth=5)
    >>> tree.fit(X_train, y_train)
    >>> predictions = tree.predict(X_test)

    Classification:
    >>> tree = RobustPrefixHonestTree(task='classification', top_levels=2, max_depth=6)
    >>> tree.fit(X_train, y_train)
    >>> probas = tree.predict_proba(X_test)

    Notes
    -----
    - Substantially slower than standard trees due to bootstrap consensus
    - Best used when prediction stability is critical (e.g., policy decisions)
    - The honest partitioning prevents overfitting but reduces effective sample size
    - Winsorization helps with outliers but may hide legitimate extreme values
    """

    task: Literal["regression", "classification"] = "classification"
    top_levels: int = 2
    max_depth: int = 6
    min_samples_leaf: int = 2
    val_frac: float = 0.2
    est_frac: float = 0.4
    smoothing: float = 1.0
    winsor_quantiles: tuple[float, float] = (0.01, 0.99)
    consensus_B: int = 12
    consensus_subsample_frac: float = 0.8
    consensus_max_bins: int = 24
    random_state: int | None = None

    # Learned state
    _lo_: np.ndarray | None = None
    _hi_: np.ndarray | None = None
    _prefix_nodes_: (
        list[tuple[int, int | None, float | None, int | None, int | None]] | None
    ) = None
    _region_models_: dict[int, object] | None = None
    _region_leaf_values_: dict[int, dict[int, float]] | None = (
        None  # regression: leaf means
    )
    _region_leaf_probs_: dict[int, dict[int, float]] | None = (
        None  # classification: P(y=1)
    )
    _global_prior_: float | None = (
        None  # p0 for classification, global mean for regression
    )
    classes_: np.ndarray | None = None

    def __post_init__(self):
        """
        Validate parameters.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        if self.task not in ["regression", "classification"]:
            raise ValueError("task must be 'regression' or 'classification'")
        if not 0 < self.val_frac < 1 or not 0 < self.est_frac < 1:
            raise ValueError("val_frac and est_frac must be in (0, 1)")
        if self.val_frac + self.est_frac >= 1:
            raise ValueError("val_frac + est_frac must be < 1")

    def _route_mask(
        self, X: np.ndarray, path: list[tuple[int, float, str]]
    ) -> np.ndarray:
        """
        Apply routing path to get mask for samples reaching a node.

        Parameters
        ----------
        X
            Feature array.
        path
            List of (feature, threshold, side) tuples.

        Returns
        -------
        np.ndarray
            Boolean mask for samples reaching the node.
        """
        m = np.ones(len(X), dtype=bool)
        for f, t, side in path:
            m &= (X[:, f] <= t) if side == "L" else (X[:, f] > t)
        return m

    def _route_node_ids(self, X: np.ndarray) -> np.ndarray:
        """
        Route samples through prefix to get terminal region IDs.

        Parameters
        ----------
        X
            Feature array.

        Returns
        -------
        np.ndarray
            Array of terminal region IDs for each sample.
        """
        ids = np.zeros(len(X), dtype=int)
        for nid, f, thr, L, R in self._prefix_nodes_ or []:
            if f is None:
                continue
            left = (X[:, f] <= thr) & (ids == nid)
            right = (X[:, f] > thr) & (ids == nid)
            ids[left] = L
            ids[right] = R
        return ids

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustPrefixHonestTree":
        """
        Fit the robust prefix honest tree.

        Parameters
        ----------
        X
            Training features.
        y
            Target values.

        Returns
        -------
        RobustPrefixHonestTree
            Fitted estimator.

        Raises
        ------
        ValueError
            If multi-class classification is attempted.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Classification-specific setup
        if self.task == "classification":
            y = y.astype(int)
            self.classes_ = np.unique(y)
            if len(self.classes_) > 2:
                raise ValueError(
                    "Multi-class classification not yet supported. "
                    "Extend with Dirichlet smoothing per class."
                )
            self._global_prior_ = float(y.mean())  # p0
        else:
            self._global_prior_ = float(y.mean())  # global mean

        rng = np.random.RandomState(self.random_state)

        # Winsorize features
        self._lo_, self._hi_ = _winsorize_fit(X, self.winsor_quantiles)
        Xw = _winsorize_apply(X, self._lo_, self._hi_)

        # Honest partition: SPLIT / VAL / EST

        if self.task == "classification":
            X_split, X_tmp, y_split, y_tmp = train_test_split(
                Xw,
                y,
                test_size=self.val_frac + self.est_frac,
                random_state=rng.randint(0, 10**9),
                stratify=y,
            )
            rel = self.est_frac / (self.val_frac + self.est_frac)
            X_val, X_est, y_val, y_est = train_test_split(
                X_tmp,
                y_tmp,
                test_size=rel,
                random_state=rng.randint(0, 10**9),
                stratify=y_tmp,
            )
        else:
            X_split, X_tmp, y_split, y_tmp = train_test_split(
                Xw,
                y,
                test_size=self.val_frac + self.est_frac,
                random_state=rng.randint(0, 10**9),
            )
            rel = self.est_frac / (self.val_frac + self.est_frac)
            X_val, X_est, y_val, y_est = train_test_split(
                X_tmp, y_tmp, test_size=rel, random_state=rng.randint(0, 10**9)
            )

        # Build locked prefix (level-order BFS)
        self._prefix_nodes_ = []
        node_queue: list[tuple[int, list[tuple[int, float, str]]]] = [(0, [])]
        level = 0

        # Select consensus function
        consensus_fn = (
            _robust_stump_classification
            if self.task == "classification"
            else _robust_stump_regression
        )

        while node_queue and level < self.top_levels:
            next_q = []
            for nid, path in node_queue:
                # Route data to this node
                m_split = self._route_mask(np.asarray(X_split), path)
                m_val = self._route_mask(np.asarray(X_val), path)
                Xs, ys = X_split[m_split], y_split[m_split]
                Xv, yv = X_val[m_val], y_val[m_val]

                # Check if node should be terminal
                min_split_size = 30 if self.task == "classification" else 20
                min_val_size = 15 if self.task == "classification" else 8

                if len(Xs) < min_split_size or len(Xv) < min_val_size:
                    self._prefix_nodes_.append((nid, None, None, None, None))
                    continue

                if self.task == "classification" and len(np.unique(ys)) < 2:
                    self._prefix_nodes_.append((nid, None, None, None, None))
                    continue

                # Consensus split selection
                cs = consensus_fn(
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

        # Collect terminal prefix regions
        terminal_paths: list[tuple[int, list[tuple[int, float, str]]]] = []
        locked = {nid: (f, t, L, R) for (nid, f, t, L, R) in self._prefix_nodes_}

        def gather(nid: int, path: list[tuple[int, float, str]], lvl: int):
            if nid not in locked or locked[nid][0] is None or lvl == self.top_levels:
                terminal_paths.append((nid, path))
                return
            f, t, L, R = locked[nid]
            if L is not None and R is not None and f is not None and t is not None:
                gather(L, path + [(f, t, "L")], lvl + 1)
                gather(R, path + [(f, t, "R")], lvl + 1)
            else:
                terminal_paths.append((nid, path))

        if not self._prefix_nodes_:
            terminal_paths.append((0, []))
        else:
            gather(0, [], 0)

        # Fit subtrees in each region
        remain = max(self.max_depth - self.top_levels, 0)
        self._region_models_ = {}

        if self.task == "classification":
            self._region_leaf_probs_ = {}
            self._fit_classification_regions(
                terminal_paths,
                np.asarray(X_split),
                np.asarray(y_split),
                np.asarray(X_est),
                np.asarray(y_est),
                remain,
            )
        else:
            self._region_leaf_values_ = {}
            self._fit_regression_regions(
                terminal_paths,
                np.asarray(X_split),
                np.asarray(y_split),
                np.asarray(X_est),
                np.asarray(y_est),
                remain,
            )

        return self

    def _fit_classification_regions(
        self,
        terminal_paths: list[tuple[int, list]],
        X_split: np.ndarray,
        y_split: np.ndarray,
        X_est: np.ndarray,
        y_est: np.ndarray,
        remain: int,
    ):
        """
        Fit classification subtrees with m-estimate smoothing.

        Parameters
        ----------
        terminal_paths
            List of (node_id, path) for terminal prefix regions.
        X_split
            Features for structure fitting.
        y_split
            Labels for structure fitting.
        X_est
            Features for honest estimation.
        y_est
            Labels for honest estimation.
        remain
            Remaining depth for subtrees.

        Raises
        ------
        ValueError
            If models or leaf probabilities not initialized.
        """
        if self._region_models_ is None or self._region_leaf_probs_ is None:
            raise ValueError("Models or leaf probabilities not initialized")

        p0 = self._global_prior_
        if p0 is None:
            raise ValueError("Global prior not initialized")

        for nid, path in terminal_paths:
            m_split = self._route_mask(np.asarray(X_split), path)
            m_est = self._route_mask(np.asarray(X_est), path)
            Xs, ys = X_split[m_split], y_split[m_split]
            Xe, ye = X_est[m_est], y_est[m_est]

            # Fit structure on SPLIT
            if remain == 0 or len(ys) < 12 or len(np.unique(ys)) < 2:
                subtree = DecisionTreeClassifier(
                    max_depth=1, random_state=self.random_state
                )
            else:
                subtree = DecisionTreeClassifier(
                    max_depth=remain,
                    min_samples_leaf=max(self.min_samples_leaf, 2),
                    random_state=self.random_state,
                )
            subtree.fit(Xs, ys)
            self._region_models_[nid] = subtree

            # Estimate leaf probabilities on EST with m-smoothing
            leaf_probs: dict[int, float] = {}
            Xleaf, yleaf = (Xe, ye) if len(ye) > 0 else (Xs, ys)
            leaves = subtree.apply(Xleaf)

            for lid in np.unique(leaves):
                m = leaves == lid
                n_leaf = int(m.sum())
                k_leaf = int(yleaf[m].sum())
                # m-estimate: (k + m*p0) / (n + m)
                phat = (k_leaf + self.smoothing * p0) / (n_leaf + self.smoothing)
                leaf_probs[int(lid)] = float(phat)

            # Backfill any missing leaves from SPLIT
            all_leaves = np.unique(subtree.apply(Xs))
            for lid in all_leaves:
                if int(lid) not in leaf_probs:
                    m = subtree.apply(Xs) == lid
                    n_leaf = int(m.sum())
                    k_leaf = int(ys[m].sum())
                    phat = (k_leaf + self.smoothing * p0) / (n_leaf + self.smoothing)
                    leaf_probs[int(lid)] = float(phat)

            self._region_leaf_probs_[nid] = leaf_probs

    def _fit_regression_regions(
        self,
        terminal_paths: list[tuple[int, list]],
        X_split: np.ndarray,
        y_split: np.ndarray,
        X_est: np.ndarray,
        y_est: np.ndarray,
        remain: int,
    ):
        """
        Fit regression subtrees with shrinkage.

        Parameters
        ----------
        terminal_paths
            List of (node_id, path) for terminal prefix regions.
        X_split
            Features for structure fitting.
        y_split
            Values for structure fitting.
        X_est
            Features for honest estimation.
        y_est
            Values for honest estimation.
        remain
            Remaining depth for subtrees.

        Raises
        ------
        ValueError
            If models or leaf values not initialized.
        """
        if self._region_models_ is None or self._region_leaf_values_ is None:
            raise ValueError("Models or leaf values not initialized")

        global_mean = self._global_prior_
        if global_mean is None:
            raise ValueError("Global prior not initialized")

        for nid, path in terminal_paths:
            m_split = self._route_mask(np.asarray(X_split), path)
            m_est = self._route_mask(np.asarray(X_est), path)
            Xs, ys = X_split[m_split], y_split[m_split]
            Xe, ye = X_est[m_est], y_est[m_est]

            # Fit structure on SPLIT
            if remain == 0 or len(ys) < 12:
                subtree = DecisionTreeRegressor(
                    max_depth=1, random_state=self.random_state
                )
            else:
                subtree = DecisionTreeRegressor(
                    max_depth=remain,
                    min_samples_leaf=max(self.min_samples_leaf, 2),
                    random_state=self.random_state,
                )
            subtree.fit(Xs, ys)
            self._region_models_[nid] = subtree

            # Estimate leaf values on EST with shrinkage to parent
            leaf_values: dict[int, float] = {}
            Xleaf, yleaf = (Xe, ye) if len(ye) > 0 else (Xs, ys)
            leaves = subtree.apply(Xleaf)

            # Parent mean for this region
            parent_mean = float(yleaf.mean()) if len(yleaf) > 0 else global_mean

            for lid in np.unique(leaves):
                m = leaves == lid
                n_leaf = int(m.sum())
                mu_leaf = float(yleaf[m].mean())

                # Shrinkage: (n*mu + lambda*parent) / (n + lambda)
                if self.smoothing > 0:
                    mu_shrunk = (n_leaf * mu_leaf + self.smoothing * parent_mean) / (
                        n_leaf + self.smoothing
                    )
                else:
                    mu_shrunk = mu_leaf

                leaf_values[int(lid)] = float(mu_shrunk)

            # Backfill from SPLIT if needed
            all_leaves = np.unique(subtree.apply(Xs))
            for lid in all_leaves:
                if int(lid) not in leaf_values:
                    m = subtree.apply(Xs) == lid
                    n_leaf = int(m.sum())
                    mu_leaf = float(ys[m].mean())
                    if self.smoothing > 0:
                        mu_shrunk = (
                            n_leaf * mu_leaf + self.smoothing * parent_mean
                        ) / (n_leaf + self.smoothing)
                    else:
                        mu_shrunk = mu_leaf
                    leaf_values[int(lid)] = float(mu_shrunk)

            self._region_leaf_values_[nid] = leaf_values

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (classification) or values (regression).

        Parameters
        ----------
        X
            Features.

        Returns
        -------
        np.ndarray
            Predicted labels or values.
        """
        if self.task == "classification":
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return self._predict_regression(X)

    def _predict_regression(self, X: np.ndarray) -> np.ndarray:
        """
        Regression prediction implementation.

        Parameters
        ----------
        X
            Features for prediction.

        Returns
        -------
        np.ndarray
            Predicted values.

        Raises
        ------
        ValueError
            If model has not been fitted properly.
        """
        check_is_fitted(
            self,
            [
                "_lo_",
                "_hi_",
                "_prefix_nodes_",
                "_region_models_",
                "_region_leaf_values_",
            ],
        )

        X = np.asarray(X)
        if self._lo_ is None or self._hi_ is None:
            raise ValueError(
                "Model has not been fitted properly - winsorization bounds missing"
            )
        Xw = _winsorize_apply(X, self._lo_, self._hi_)
        ids = self._route_node_ids(Xw)

        predictions = np.zeros(len(Xw), dtype=float)

        if self._region_models_ is None or self._region_leaf_values_ is None:
            raise ValueError("Model has not been fitted properly")

        for nid, subtree in self._region_models_.items():
            mask = ids == nid
            if not mask.any():
                continue

            leaves = subtree.apply(Xw[mask])  # type: ignore[attr-defined]
            vals = np.array(
                [
                    self._region_leaf_values_[nid].get(int(lid), self._global_prior_)
                    for lid in leaves
                ],
                dtype=float,
            )
            predictions[mask] = vals

        return predictions

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
        ValueError
            If model has not been fitted properly.
        """
        if self.task != "classification":
            raise AttributeError(
                "predict_proba only available for classification tasks"
            )

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
        if self._lo_ is None or self._hi_ is None:
            raise ValueError(
                "Model has not been fitted properly - winsorization bounds missing"
            )
        Xw = _winsorize_apply(X, self._lo_, self._hi_)
        ids = self._route_node_ids(Xw)

        proba = np.zeros((len(Xw), 2), dtype=float)

        if self._region_models_ is None or self._region_leaf_probs_ is None:
            raise ValueError("Model has not been fitted properly")

        for nid, subtree in self._region_models_.items():
            mask = ids == nid
            if not mask.any():
                continue

            leaves = subtree.apply(Xw[mask])  # type: ignore[attr-defined]
            p1 = np.array(
                [self._region_leaf_probs_[nid].get(int(lid), 0.5) for lid in leaves],
                dtype=float,
            )
            p1 = np.clip(p1, 1e-7, 1 - 1e-7)

            proba[mask, 1] = p1
            proba[mask, 0] = 1.0 - p1

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
            R² score (regression) or accuracy (classification).
        """
        from sklearn.metrics import accuracy_score, r2_score

        y_pred = self.predict(X)

        if self.task == "regression":
            return r2_score(y, y_pred)
        else:
            return accuracy_score(y, y_pred)


# ============================================================================
# Sklearn-compatible wrappers for explicit task specification
# ============================================================================
