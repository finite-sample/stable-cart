r"""The closed forms in :mod:`stable_cart.linear`, checked against simulation.

A closed form is worth having only if it is right, and the way to find out is to
simulate the thing it claims to describe. Each test below draws many independent
datasets, computes the quantity the hard way, and compares.

Tolerances are derived from each simulation's own standard error rather than
chosen to make the test pass. Where a result is exact rather than asymptotic —
``T1`` needs no distributional assumption at all — the test asserts equality to
floating-point tolerance instead.
"""

from itertools import pairwise

import numpy as np
import pytest

from stable_cart import (
    linear_frontier,
    linear_instability,
    shrinkage_coefficients,
)

N, P, SIGMA = 150, 4, 2.0
REPLICATES = 40_000


@pytest.fixture(scope="module")
def problem():
    """A fixed design, true coefficients, and evaluation points."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, P))
    beta = rng.normal(size=P)
    X_eval = rng.normal(size=(60, P))
    return X, beta, X_eval


@pytest.fixture(scope="module")
def refits(problem):
    """Predictions of least squares refitted on many independent datasets."""
    X, beta, X_eval = problem
    rng = np.random.default_rng(1)
    gram_inv = np.linalg.inv(X.T @ X)
    noise = rng.normal(scale=SIGMA, size=(REPLICATES, N))
    coefs = (noise @ X) @ gram_inv + beta
    return coefs @ X_eval.T


class TestInstabilityIdentities:
    """What the instability measures are, exactly."""

    def test_squared_pairwise_instability_is_twice_the_variance(self, problem):
        """T1, which holds for any procedure and needs no assumptions.

        Two predictions from independent datasets are iid, so their means cancel
        and the expected squared difference is the sum of two variances. Asserted
        to floating-point tolerance because it is an identity, not a limit.
        """
        X, _beta, X_eval = problem

        out = linear_instability(X, X_eval, sigma=SIGMA)

        assert np.allclose(out["s2"], 2.0 * out["variance"], rtol=1e-12)

    def test_variance_matches_simulation(self, problem, refits):
        """T3: the variance of a least-squares prediction is sigma^2 x'(X'X)^-1 x."""
        X, _beta, X_eval = problem

        predicted = linear_instability(X, X_eval, sigma=SIGMA)["variance"]
        observed = refits.var(axis=0)

        # A variance from R draws has relative standard error sqrt(2/(R-1)).
        tolerance = 5 * np.sqrt(2 / (REPLICATES - 1))
        assert np.allclose(observed, predicted, rtol=tolerance)

    def test_pairwise_absolute_measure_matches_simulation(self, problem, refits):
        """T2: E|f_D - f_D'| = (2/sqrt(pi)) * SD under Gaussian sampling."""
        X, _beta, X_eval = problem

        predicted = linear_instability(X, X_eval, sigma=SIGMA)["s1"]
        half = REPLICATES // 2
        observed = np.abs(refits[:half] - refits[half:]).mean(axis=0)

        assert np.allclose(observed, predicted, rtol=0.05)

    def test_centered_mad_matches_simulation(self, problem, refits):
        """Gaussian absolute deviation from the sampling mean has a closed form."""
        X, beta, X_eval = problem

        predicted = linear_instability(X, X_eval, sigma=SIGMA)["centered_mad"]
        observed = np.abs(refits - X_eval @ beta).mean(axis=0)

        assert np.allclose(observed, predicted, rtol=0.05)

    def test_pairwise_and_centered_mad_differ_by_root_two(self, problem):
        """The two Gaussian centered quantities differ by exactly sqrt(2).

        Neither identity manufactures MAPE against an observed fit, which may be
        offset from the sampling center.
        """
        X, _beta, X_eval = problem

        out = linear_instability(X, X_eval, sigma=SIGMA)

        assert np.allclose(out["s1"], np.sqrt(2) * out["centered_mad"], rtol=1e-12)

    def test_centered_mad_is_not_mape_against_an_offset_original(self, problem, refits):
        """The original fitted prediction is random, not the sampling mean."""
        X, _beta, X_eval = problem
        centered = linear_instability(X, X_eval, sigma=SIGMA)["centered_mad"]
        reference = refits.mean(axis=0) + 2.0 * refits.std(axis=0)
        against_reference = np.abs(refits - reference).mean(axis=0)

        assert np.mean(against_reference) > 2.0 * np.mean(centered)

    def test_integrated_s1_is_a_mean_standard_deviation(self, problem):
        """Not the square root of the mean variance — Jensen separates them.

        This is the trap in aggregating a reported MAPE: it is linear in the
        standard deviation, so it cannot be averaged as though it were a variance.
        """
        X, _beta, X_eval = problem
        out = linear_instability(X, X_eval, sigma=SIGMA)

        from stable_cart.linear import PAIRWISE

        mean_of_roots = out["s1_mean"]
        root_of_mean = PAIRWISE * np.sqrt(out["variance_mean"])

        assert mean_of_roots < root_of_mean
        assert not np.isclose(mean_of_roots, root_of_mean, rtol=1e-3)

    def test_rejects_bad_arguments(self, problem):
        X, _beta, X_eval = problem

        with pytest.raises(ValueError, match="non-negative"):
            linear_instability(X, X_eval, sigma=-1.0)
        with pytest.raises(ValueError, match="columns"):
            linear_instability(X, X_eval[:, :-1], sigma=1.0)
        with pytest.raises(ValueError, match="finite"):
            linear_instability(X, X_eval, sigma=np.nan)
        with pytest.raises(ValueError, match="finite"):
            linear_instability(X, np.full_like(X_eval, np.nan), sigma=1.0)

    def test_robust_form_rejects_a_saturated_design(self):
        X = np.eye(4)
        y = np.arange(4.0)

        with pytest.raises(ValueError, match="more observations than columns"):
            linear_instability(X, X, y=y, robust=True)

    def test_robust_form_rejects_a_non_vector_target(self, problem):
        X, _beta, X_eval = problem

        with pytest.raises(ValueError, match="one-dimensional"):
            linear_instability(X, X_eval, y=np.ones((len(X), 2)), robust=True)


class TestFrontier:
    """The Pareto frontier, its slope, and the estimator that achieves it."""

    @pytest.fixture(scope="class")
    def frontier(self, problem):
        """The exact frontier, with beta and sigma supplied rather than estimated."""
        X, beta, _X_eval = problem
        rng = np.random.default_rng(2)
        y = X @ beta + rng.normal(scale=SIGMA, size=N)
        return linear_frontier(X, y, n_points=60, beta=beta, sigma=SIGMA)

    def test_frontier_is_monotone_in_both_coordinates(self, frontier):
        """Buying stability always costs bias — otherwise it is not a frontier."""
        bias2 = [p["bias2"] for p in frontier["points"]]
        variance = [p["variance"] for p in frontier["points"]]

        assert all(a <= b + 1e-12 for a, b in pairwise(bias2))
        assert all(a >= b - 1e-12 for a, b in pairwise(variance))

    def test_s1_averages_pointwise_standard_deviations(self, frontier, problem):
        """Integrated absolute instability is mean(sqrt(v)), not sqrt(mean(v))."""
        X, _beta, _X_eval = problem
        u, _d, _vt = np.linalg.svd(X, full_matrices=False)
        point = frontier["points"][17]
        point_variance = SIGMA**2 * (u**2 @ point["shrinkage"] ** 2)
        expected = 2 / np.sqrt(np.pi) * np.mean(np.sqrt(point_variance))

        assert point["s1"] == pytest.approx(expected)

    def test_no_shrinkage_vector_beats_the_frontier(self, frontier, problem):
        """T4: the closed form really is the constrained optimum.

        At each point, the Lagrangian must be no worse than for any nearby
        shrinkage vector. This is the check that would fail if the derivation
        dropped a factor.
        """
        X, beta, _X_eval = problem
        _u, d, vt = np.linalg.svd(X, full_matrices=False)
        theta_sq = (vt @ beta) ** 2
        rng = np.random.default_rng(3)

        for point in frontier["points"][::12]:
            mu, s = point["mu"], point["shrinkage"]
            best = point["bias2"] + mu * point["variance"]
            for scale in (0.02, 0.08):
                for _ in range(300):
                    other = np.clip(s + scale * rng.normal(size=len(s)), 0.0, 1.0)
                    bias2 = np.sum((1 - other) ** 2 * theta_sq * d**2) / N
                    variance = SIGMA**2 * np.sum(other**2) / N
                    assert best <= bias2 + mu * variance + 1e-12

    def test_the_slope_of_the_frontier_is_minus_mu(self, frontier):
        """T5: mu is the exchange rate, not just a knob."""
        points = frontier["points"]
        for i in range(5, len(points) - 5, 10):
            before, after = points[i - 1], points[i + 1]
            slope = (after["bias2"] - before["bias2"]) / (
                after["variance"] - before["variance"]
            )
            assert slope == pytest.approx(-points[i]["mu"], rel=0.15)

    def test_risk_is_minimized_at_mu_one(self, frontier):
        """T5: risk is bias^2 + variance, so its optimum is where the slope is -1."""
        points = frontier["points"]
        best = min(points, key=lambda p: p["risk"])

        assert best["mu"] == pytest.approx(1.0, rel=0.25)
        assert (
            frontier["risk_optimal"]["risk"] <= min(p["risk"] for p in points) + 1e-12
        )

    def test_ridge_is_optimal_exactly_when_the_signal_is_isotropic(self, problem):
        """T6: plain ridge traces the frontier only under an assumption."""
        X, _beta, _X_eval = problem
        _u, d, _vt = np.linalg.svd(X, full_matrices=False)

        # Isotropic signal: the optimal shrinkage collapses to ridge's.
        theta_sq = np.full(P, 0.7)
        mu, sigma = 1.0, SIGMA
        optimal = d**2 * theta_sq / (d**2 * theta_sq + mu * sigma**2)
        ridge = d**2 / (d**2 + sigma**2 / theta_sq[0])
        assert np.allclose(optimal, ridge)

        # Anisotropic signal: they part company.
        theta_sq = np.linspace(0.05, 3.0, P)
        optimal = d**2 * theta_sq / (d**2 * theta_sq + mu * sigma**2)
        ridge = d**2 / (d**2 + sigma**2 / theta_sq.mean())
        assert not np.allclose(optimal, ridge, rtol=0.05)

    def test_extremes_are_least_squares_and_the_constant(self, problem):
        X, beta, _X_eval = problem
        rng = np.random.default_rng(4)
        y = X @ beta + rng.normal(scale=SIGMA, size=N)

        result = linear_frontier(X, y, n_points=30, mu_max=1e9, beta=beta, sigma=SIGMA)

        assert result["points"][0]["bias2"] == pytest.approx(0.0, abs=1e-12)
        assert result["points"][-1]["variance"] == pytest.approx(0.0, abs=1e-9)

    def test_zero_price_frontier_drops_exactly_null_directions(self):
        X = np.diag([4.0, 3.0, 2.0, 1.0])
        beta = np.array([1.0, 0.0, 0.0, 0.0])

        result = linear_frontier(X, X @ beta, n_points=3, beta=beta, sigma=1.0)
        endpoint = result["points"][0]

        assert endpoint["bias2"] == 0.0
        assert np.count_nonzero(endpoint["shrinkage"]) == 1
        assert endpoint["variance"] == pytest.approx(1 / len(X))

    def test_rejects_bad_arguments(self, problem):
        X, beta, _X_eval = problem
        y = X @ beta

        with pytest.raises(ValueError, match="at least 2"):
            linear_frontier(X, y, n_points=1)
        with pytest.raises(ValueError, match="positive"):
            linear_frontier(X, y, mu_max=0.0)

    @pytest.mark.parametrize("n_points", [2, 9])
    @pytest.mark.parametrize("mu_max", [1e-6, 1e3])
    def test_grid_stays_within_and_ends_at_mu_max(self, problem, n_points, mu_max):
        X, beta, _X_eval = problem
        result = linear_frontier(
            X,
            X @ beta,
            n_points=n_points,
            mu_max=mu_max,
            beta=beta,
            sigma=SIGMA,
        )
        grid = np.array([point["mu"] for point in result["points"]])

        assert len(grid) == n_points
        assert grid[0] == 0.0
        assert np.all(np.diff(grid) > 0)
        assert grid[-1] == pytest.approx(mu_max)
        assert np.all(grid <= mu_max)


class TestShrinkageCoefficients:
    """The estimator behind each frontier point."""

    def test_mu_zero_is_least_squares(self, problem):
        X, beta, _X_eval = problem
        rng = np.random.default_rng(5)
        y = X @ beta + rng.normal(scale=SIGMA, size=N)

        ols, *_ = np.linalg.lstsq(X, y, rcond=None)

        assert np.allclose(shrinkage_coefficients(X, y, mu=0.0), ols)

    def test_mu_zero_needs_no_noise_estimate_for_a_saturated_design(self):
        X = np.array([[1.0, 0.0], [1.0, 1.0]])
        y = np.array([2.0, 5.0])
        expected, *_ = np.linalg.lstsq(X, y, rcond=None)

        assert np.allclose(shrinkage_coefficients(X, y, mu=0.0), expected)

    def test_shrinkage_is_monotone_in_mu(self, problem):
        X, beta, _X_eval = problem
        rng = np.random.default_rng(6)
        y = X @ beta + rng.normal(scale=SIGMA, size=N)

        norms = [
            np.linalg.norm(shrinkage_coefficients(X, y, mu=mu))
            for mu in (0.0, 1.0, 10.0, 1000.0)
        ]

        assert all(a >= b for a, b in pairwise(norms))

    def _held_out_error(self, problem, replicates=400, **kwargs):
        """Mean squared error at held-out points, averaged over datasets."""
        X, beta, X_eval = problem
        truth = X_eval @ beta
        rng = np.random.default_rng(7)
        errors = []
        for _ in range(replicates):
            y = X @ beta + rng.normal(scale=SIGMA, size=N)
            coef = shrinkage_coefficients(X, y, **kwargs)
            errors.append(np.mean((X_eval @ coef - truth) ** 2))
        return float(np.mean(errors))

    def test_the_oracle_optimum_beats_least_squares(self, problem):
        """With the true signal known, mu=1 is the risk optimum and it wins.

        This is the theoretical bound: what shrinkage is worth if you did not
        have to pay to learn where to shrink.
        """
        _X, beta, _X_eval = problem
        ols = self._held_out_error(problem, mu=0.0)
        oracle = self._held_out_error(problem, mu=1.0, beta=beta, sigma=SIGMA)

        assert oracle < ols

    def test_estimating_the_signal_per_direction_loses_to_least_squares(self, problem):
        """The bound is not reachable by plugging in p separate estimates.

        Measured here at roughly +14%, and it stays positive as p grows: the
        formula is optimal given the signal strengths, and estimating p of them
        costs more than the shrinkage saves. Pinned so that nobody restores it
        as the default on the strength of the algebra alone.
        """
        ols = self._held_out_error(problem, mu=0.0)
        per_direction = self._held_out_error(problem, mu=1.0, signal="per_direction")

        assert per_direction > ols

    def test_pooling_the_signal_does_not_lose(self, problem):
        """Pooling estimates one number instead of p, which is ridge, and works.

        At this p the gain is around zero; the held-out advantage grows with p
        (measured -2.1% at p=10, -5.1% at p=50). The requirement here is only
        that the default never does worse than the estimator it replaces.
        """
        ols = self._held_out_error(problem, mu=0.0)
        pooled = self._held_out_error(problem, mu=1.0)

        assert pooled <= ols * 1.01

    def test_rejects_an_unknown_signal_mode(self, problem):
        X, beta, _X_eval = problem

        with pytest.raises(ValueError, match="pooled"):
            shrinkage_coefficients(X, X @ beta, mu=1.0, signal="nonsense")

    def test_rejects_a_negative_price(self, problem):
        X, beta, _X_eval = problem

        with pytest.raises(ValueError, match="non-negative"):
            shrinkage_coefficients(X, X @ beta, mu=-1.0)

    def test_rejects_a_nonfinite_price_or_noise(self, problem):
        X, beta, _X_eval = problem
        y = X @ beta

        with pytest.raises(ValueError, match="finite"):
            shrinkage_coefficients(X, y, mu=np.nan, sigma=1.0)
        with pytest.raises(ValueError, match="finite"):
            shrinkage_coefficients(X, y, mu=1.0, sigma=np.nan)
