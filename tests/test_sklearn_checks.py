"""Run scikit-learn's maintained estimator contract against every task mode."""

from sklearn.svm import LinearSVC
from sklearn.utils.estimator_checks import parametrize_with_checks

from stable_cart import RepresentativeEstimator

ESTIMATORS = [
    RepresentativeEstimator(task="regression", n_candidates=2),
    RepresentativeEstimator(task="classification", n_candidates=2),
    RepresentativeEstimator(
        estimator=LinearSVC(), task="classification", n_candidates=2
    ),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_estimator_contract(estimator, check):
    """Require every applicable upstream estimator check to pass."""
    check(estimator)
