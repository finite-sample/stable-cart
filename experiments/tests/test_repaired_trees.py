import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from experiments.repaired_trees import (
    BootstrapVariancePenalizedTree,
    RobustPrefixHonestTree,
)


def relative_gain(y, left, task):
    if not np.any(left) or np.all(left):
        return 0.0
    if task == "regression":
        parent = np.var(y)
        if parent <= np.finfo(float).eps:
            return 0.0
        child = (
            np.count_nonzero(left) * np.var(y[left])
            + np.count_nonzero(~left) * np.var(y[~left])
        ) / len(y)
    else:
        n_classes = int(np.max(y)) + 1

        def gini(values):
            probabilities = np.bincount(values.astype(int), minlength=n_classes) / len(
                values
            )
            return 1.0 - probabilities @ probabilities

        parent = gini(y)
        if parent <= np.finfo(float).eps:
            return 0.0
        child = (
            np.count_nonzero(left) * gini(y[left])
            + np.count_nonzero(~left) * gini(y[~left])
        ) / len(y)
    return max(0.0, (parent - child) / parent)


def fixed_grid_candidates(X, y, *, task, min_leaf, per_feature):
    candidates = []
    for feature in range(X.shape[1]):
        unique = np.unique(X[:, feature])
        thresholds = unique[:-1] + np.diff(unique) / 2.0
        if per_feature is not None and len(thresholds) > per_feature:
            positions = np.linspace(0, len(thresholds) - 1, per_feature)
            thresholds = thresholds[np.unique(np.rint(positions).astype(int))]
        for threshold in thresholds:
            left = X[:, feature] <= threshold
            if min(np.count_nonzero(left), np.count_nonzero(~left)) < min_leaf:
                continue
            gain = relative_gain(y, left, task)
            if gain > 0:
                candidates.append((feature, float(threshold), gain))
    return candidates


def canonical(X, y):
    keys = tuple(X[:, feature] for feature in range(X.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    return X[order], y[order], order


def tree_structure(node):
    if node.is_leaf:
        return ("leaf", node.depth)
    return (
        node.feature,
        node.threshold,
        tree_structure(node.left),
        tree_structure(node.right),
    )


def test_variance_penalized_root_matches_independent_reference():
    X, y = make_regression(
        n_samples=70,
        n_features=4,
        n_informative=3,
        noise=30,
        random_state=4,
    )
    penalty = 6.0
    n_bootstrap = 24
    per_feature = 10
    seed = 17
    fitted = BootstrapVariancePenalizedTree(
        max_depth=1,
        min_samples_split=10,
        min_samples_leaf=5,
        variance_penalty=penalty,
        n_bootstrap=n_bootstrap,
        max_candidates_per_feature=per_feature,
        random_state=seed,
    ).fit(X, y)

    X_ordered, y_ordered, _ = canonical(X, y)
    candidates = fixed_grid_candidates(
        X_ordered,
        y_ordered,
        task="regression",
        min_leaf=5,
        per_feature=per_feature,
    )
    bootstrap_indices = np.random.default_rng(seed).integers(
        0, len(X), size=(n_bootstrap, len(X))
    )
    scored = []
    for feature, threshold, gain in candidates:
        bootstrap_gains = [
            relative_gain(
                y_ordered[indices],
                X_ordered[indices, feature] <= threshold,
                "regression",
            )
            for indices in bootstrap_indices
        ]
        variance = np.var(bootstrap_gains, ddof=1)
        scored.append((gain - penalty * variance, feature, threshold, gain, variance))
    expected = min(scored, key=lambda row: (-row[0], row[1], row[2]))

    assert fitted.tree_.feature == expected[1]
    assert fitted.tree_.threshold == pytest.approx(expected[2])
    assert fitted.tree_.gain == pytest.approx(expected[3])
    assert fitted.tree_.variance == pytest.approx(expected[4])
    assert fitted.tree_.score == pytest.approx(expected[0])


def test_zero_penalty_is_the_greedy_limiting_case():
    X, y = make_regression(n_samples=100, n_features=5, noise=15, random_state=2)
    model = BootstrapVariancePenalizedTree(
        max_depth=1,
        min_samples_leaf=6,
        min_samples_split=12,
        variance_penalty=0.0,
        max_candidates_per_feature=16,
        random_state=9,
    ).fit(X, y)

    X_ordered, y_ordered, _ = canonical(X, y)
    candidates = fixed_grid_candidates(
        X_ordered,
        y_ordered,
        task="regression",
        min_leaf=6,
        per_feature=16,
    )
    expected = min(candidates, key=lambda row: (-row[2], row[0], row[1]))

    assert (model.tree_.feature, model.tree_.threshold) == expected[:2]
    assert model.tree_.score == pytest.approx(expected[2])
    assert model.tree_.variance == 0.0


def test_variance_penalty_is_operational_not_just_a_veto():
    X, y = make_regression(
        n_samples=80,
        n_features=4,
        n_informative=3,
        noise=40,
        random_state=0,
    )
    common = {
        "max_depth": 1,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "n_bootstrap": 40,
        "max_candidates_per_feature": 12,
        "random_state": 7,
    }
    greedy = BootstrapVariancePenalizedTree(variance_penalty=0.0, **common).fit(X, y)
    penalized = BootstrapVariancePenalizedTree(variance_penalty=20.0, **common).fit(
        X, y
    )

    assert penalized.tree_.feature is not None
    assert (greedy.tree_.feature, greedy.tree_.threshold) != (
        penalized.tree_.feature,
        penalized.tree_.threshold,
    )
    assert penalized.tree_.variance > 0.0
    assert penalized.tree_.score < penalized.tree_.gain


def test_variance_tree_is_target_scale_and_row_order_invariant():
    X, y = make_regression(n_samples=90, n_features=4, noise=20, random_state=8)
    params = {
        "max_depth": 3,
        "min_samples_split": 12,
        "min_samples_leaf": 6,
        "variance_penalty": 4.0,
        "n_bootstrap": 20,
        "max_candidates_per_feature": 12,
        "random_state": 3,
    }
    baseline = BootstrapVariancePenalizedTree(**params).fit(X, y)
    scaled = BootstrapVariancePenalizedTree(**params).fit(X, -7.0 * y)
    permutation = np.random.default_rng(19).permutation(len(X))
    reordered = BootstrapVariancePenalizedTree(**params).fit(
        X[permutation], y[permutation]
    )

    assert tree_structure(baseline.tree_) == tree_structure(scaled.tree_)
    assert tree_structure(baseline.tree_) == tree_structure(reordered.tree_)
    assert np.allclose(baseline.predict(X), reordered.predict(X))
    assert np.allclose(scaled.predict(X), -7.0 * baseline.predict(X))


def test_prefix_root_votes_match_independent_reference():
    X, y = make_regression(
        n_samples=160,
        n_features=5,
        n_informative=4,
        noise=10,
        random_state=0,
    )
    seed = 7
    n_bootstrap = 40
    per_feature = 12
    fitted = RobustPrefixHonestTree(
        max_depth=1,
        min_samples_split=10,
        min_samples_leaf=5,
        prefix_levels=1,
        consensus_threshold=0.2,
        n_bootstrap=n_bootstrap,
        estimation_fraction=0.5,
        max_candidates_per_feature=per_feature,
        random_state=seed,
    ).fit(X, y)

    X_ordered, y_ordered, _ = canonical(X, y)
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(len(X))
    estimation = np.zeros(len(X), dtype=bool)
    estimation[assignment[: round(0.5 * len(X))]] = True
    X_structure = X_ordered[~estimation]
    y_structure = y_ordered[~estimation]
    candidates = fixed_grid_candidates(
        X_structure,
        y_structure,
        task="regression",
        min_leaf=5,
        per_feature=per_feature,
    )
    indices = rng.integers(0, len(X_structure), size=(n_bootstrap, len(X_structure)))
    votes = np.zeros(len(candidates), dtype=int)
    for bootstrap_indices in indices:
        gains = [
            relative_gain(
                y_structure[bootstrap_indices],
                X_structure[bootstrap_indices, feature] <= threshold,
                "regression",
            )
            for feature, threshold, _ in candidates
        ]
        winner = min(
            range(len(candidates)),
            key=lambda index: (
                -gains[index],
                candidates[index][0],
                candidates[index][1],
            ),
        )
        votes[winner] += 1
    supports = votes / n_bootstrap
    eligible = [index for index, support in enumerate(supports) if support >= 0.2]
    expected_index = min(
        eligible,
        key=lambda index: (
            -supports[index],
            -candidates[index][2],
            candidates[index][0],
            candidates[index][1],
        ),
    )
    expected = candidates[expected_index]

    assert (fitted.tree_.feature, fitted.tree_.threshold) == expected[:2]
    assert fitted.tree_.support == pytest.approx(supports[expected_index])
    assert sum(fitted.tree_.vote_supports) == pytest.approx(1.0)
    assert all(0.0 <= support <= 1.0 for support in fitted.tree_.vote_supports)


def test_honest_outcomes_cannot_change_tree_structure():
    X, y = make_regression(n_samples=180, n_features=5, noise=10, random_state=6)
    params = {
        "max_depth": 3,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "prefix_levels": 1,
        "consensus_threshold": 0.1,
        "n_bootstrap": 24,
        "max_candidates_per_feature": 12,
        "random_state": 11,
    }
    baseline = RobustPrefixHonestTree(**params).fit(X, y)
    altered_y = y.copy()
    altered_y[baseline.estimation_indices_] = 1000.0 + np.arange(
        len(baseline.estimation_indices_)
    )
    altered = RobustPrefixHonestTree(**params).fit(X, altered_y)

    assert np.array_equal(baseline.structure_indices_, altered.structure_indices_)
    assert tree_structure(baseline.tree_) == tree_structure(altered.tree_)
    assert not np.allclose(baseline.predict(X), altered.predict(X))


def test_prefix_tree_handles_multiclass_and_aligns_probabilities():
    X, y_codes = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=3,
    )
    labels = np.array(["alpha", "middle", "zeta"])[y_codes]
    model = RobustPrefixHonestTree(
        task="classification",
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        prefix_levels=1,
        consensus_threshold=0.0,
        n_bootstrap=20,
        max_candidates_per_feature=10,
        random_state=4,
    ).fit(X, labels)
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    assert np.array_equal(model.classes_, np.array(["alpha", "middle", "zeta"]))
    assert probabilities.shape == (len(X), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(predictions, model.classes_[probabilities.argmax(axis=1)])


def test_prefix_tree_is_deterministic_and_row_order_invariant():
    X, y = make_regression(n_samples=160, n_features=5, noise=15, random_state=9)
    params = {
        "max_depth": 3,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "prefix_levels": 1,
        "consensus_threshold": 0.1,
        "n_bootstrap": 20,
        "max_candidates_per_feature": 10,
        "random_state": 2,
    }
    first = RobustPrefixHonestTree(**params).fit(X, y)
    second = RobustPrefixHonestTree(**params).fit(X, y)
    permutation = np.random.default_rng(12).permutation(len(X))
    reordered = RobustPrefixHonestTree(**params).fit(X[permutation], y[permutation])

    assert tree_structure(first.tree_) == tree_structure(second.tree_)
    assert tree_structure(first.tree_) == tree_structure(reordered.tree_)
    assert np.array_equal(first.predict(X), second.predict(X))
    assert np.allclose(first.predict(X), reordered.predict(X))


@pytest.mark.parametrize(
    ("estimator", "match"),
    [
        (BootstrapVariancePenalizedTree(variance_penalty=-1), "nonnegative"),
        (RobustPrefixHonestTree(consensus_threshold=1.1), r"in \[0, 1\]"),
        (RobustPrefixHonestTree(estimation_fraction=1.0), "strictly between"),
    ],
)
def test_invalid_repair_parameters_fail_explicitly(estimator, match):
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    with pytest.raises(ValueError, match=match):
        estimator.fit(X, y)


def test_bootstrap_gains_are_invariant_to_large_target_translation():
    from experiments.repaired_trees import _bootstrap_gains, _Candidate

    X = np.arange(12, dtype=float)[:, None]
    y = np.repeat([0.0, 1.0], 6)
    candidates = [_Candidate(feature=0, threshold=5.5, gain=1.0)]
    indices = np.arange(12)[None, :]
    original = _bootstrap_gains(
        X, y, candidates, task="regression", bootstrap_indices=indices
    )
    shifted = _bootstrap_gains(
        X, y + 1e8, candidates, task="regression", bootstrap_indices=indices
    )
    assert np.allclose(original, shifted)
    assert original[0, 0] == pytest.approx(1)


def test_relative_gains_preserve_small_target_scale():
    from experiments.repaired_trees import _bootstrap_gains, _Candidate, _relative_gain

    X = np.arange(12, dtype=float)[:, None]
    y = np.repeat([0.0, 1.0], 6)
    scaled = y * 1e-10
    left = X[:, 0] <= 5.5
    assert _relative_gain(scaled, left, "regression") == pytest.approx(1.0)
    candidates = [_Candidate(feature=0, threshold=5.5, gain=1.0)]
    observed = _bootstrap_gains(
        X,
        scaled,
        candidates,
        task="regression",
        bootstrap_indices=np.arange(12)[None, :],
    )
    assert observed[0, 0] == pytest.approx(1.0)
