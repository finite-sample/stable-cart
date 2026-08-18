r"""Counterexamples that fix the scope of the formal results.

``tests/test_linear.py`` checks that each closed form is right where it was
derived. That is the wrong test for a formal claim: confirmation in the regime a
result was derived in tells you nothing about the regime it will be used in, and
one counterexample disproves it outright.

These tests go looking for the counterexamples. Each asserts a **boundary** — the
place where a claim stops holding — so that the claim cannot quietly re-broaden
later. Three of them encode results that were briefly shipped as unqualified and
are wrong: the per-direction comparison, the :math:`\sqrt2` constant, and the
:math:`\mu=1` optimum.

Every docstring states the number that was measured when the test was written,
so a future failure can be told apart from Monte Carlo drift.
"""

import warnings

import numpy as np
import pytest

from stable_cart import linear_frontier, linear_instability, shrinkage_coefficients
from stable_cart.linear import CENTERED_MAD, PAIRWISE, _shrinkage

SIGMA = 2.0


def held_out_error(p, theta_kind, n=150, reps=400, seed=0):
    """Held-out error of OLS, pooled and per-direction shrinkage.

    ``theta_kind`` shapes the signal **in the design's singular basis**, which is
    where shrinkage acts. Shaping ``beta`` instead proves nothing: a Gaussian
    design rotates a sparse ``beta`` into a dense ``theta``.
    """
    rng = np.random.default_rng(seed + p)
    X = rng.normal(size=(n, p))
    _u, d, vt = np.linalg.svd(X, full_matrices=False)

    theta = np.zeros(p)
    if theta_kind == "dense":
        theta = rng.normal(size=p)
    elif theta_kind == "sparse":
        theta[rng.choice(p, 3, replace=False)] = rng.normal(size=3) * 6
    else:
        theta[0] = 12.0

    beta = vt.T @ theta
    X_eval = rng.normal(size=(300, p))
    truth = X_eval @ beta

    errors = {"ols": [], "pooled": [], "per_direction": []}
    for _ in range(reps):
        y = X @ beta + rng.normal(scale=SIGMA, size=n)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        theta_hat = vt @ coef
        s2 = float(np.sum((y - X @ coef) ** 2) / (n - p))
        unbiased = theta_hat**2 - s2 / d**2
        errors["ols"].append(np.mean((X_eval @ coef - truth) ** 2))
        for mode, theta_sq in (
            ("per_direction", np.maximum(unbiased, 0.0)),
            ("pooled", np.full(p, max(float(unbiased.mean()), 1e-12))),
        ):
            s = _shrinkage(d**2 * theta_sq, 1.0, np.sqrt(s2))
            fitted = vt.T @ (s * theta_hat)
            errors[mode].append(np.mean((X_eval @ fitted - truth) ** 2))
    return {k: float(np.mean(v)) for k, v in errors.items()}


class TestPerDirectionShrinkage:
    """Which shrinkage wins depends on where the signal sits, not on ``p``."""

    @pytest.mark.parametrize("seed", [0, 100, 200])
    def test_per_direction_loses_when_the_signal_is_spread(self, seed):
        """The case that was measured first, and generalized from too freely.

        Per-direction is worse than OLS in all ten draws tried, by +0.3% to
        +22.7%. Only this half is draw-robust: pooled's advantage at p=10 ranges
        from -1.7% to **+1.1%** and is not asserted here, because a single
        favorable draw is how the original over-claim happened.
        """
        out = held_out_error(10, "dense", seed=seed)

        assert out["per_direction"] > out["ols"]

    @pytest.mark.parametrize("seed", [0, 100, 200])
    def test_pooling_pays_reliably_only_at_larger_p(self, seed):
        """At p=50 pooled beat OLS in every draw, by 4.0% to 7.0%."""
        out = held_out_error(50, "dense", seed=seed)

        assert out["pooled"] < out["ols"]

    @pytest.mark.parametrize("theta_kind", ["sparse", "one spike"])
    def test_per_direction_wins_decisively_when_the_signal_is_concentrated(
        self, theta_kind
    ):
        """The counterexample. Shipped docs claimed per-direction loses always.

        Measured at p=10: -46.9% (sparse) and -61.8% (one spike) against OLS,
        both far beyond pooling. The claim is about signal geometry, not ``p``.
        """
        out = held_out_error(10, theta_kind)

        assert out["per_direction"] < out["ols"] * 0.75
        assert out["per_direction"] < out["pooled"]

    def test_sparsity_in_beta_alone_buys_nothing(self):
        """Why the first attempt at this test found no counterexample.

        A Gaussian design's singular basis is a random rotation, so zeros in
        ``beta`` do not survive into ``theta`` and per-direction still loses.
        """
        p, n = 10, 150
        rng = np.random.default_rng(4)
        X = rng.normal(size=(n, p))
        beta = np.zeros(p)
        beta[rng.choice(p, 3, replace=False)] = rng.normal(size=3) * 6
        _u, _d, vt = np.linalg.svd(X, full_matrices=False)

        theta = vt @ beta

        assert np.count_nonzero(np.abs(theta) > 1e-8) == p


class TestShrinkageBoundary:
    """The ``mu = 0`` endpoint, where the formula is 0/0."""

    def test_mu_zero_is_the_operational_least_squares_endpoint(self):
        """The public coefficient path promises ordinary least squares at zero."""
        rng = np.random.default_rng(0)
        p, n = 6, 200
        X = rng.normal(size=(n, p))
        _u, _d, vt = np.linalg.svd(X, full_matrices=False)
        beta = vt.T @ np.array([4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
        y = X @ beta + rng.normal(scale=1.0, size=n)

        result = linear_frontier(
            X, y, n_points=6, beta=beta, sigma=1.0, signal="per_direction"
        )
        start = result["points"][0]

        assert start["bias2"] == pytest.approx(0.0, abs=1e-12)
        assert start["variance"] == pytest.approx(p * 1.0**2 / n, rel=1e-9)
        assert np.all(start["shrinkage"] == 1.0)

    def test_a_small_real_signal_is_not_silently_truncated(self):
        """A hidden relative tolerance cannot distinguish signal from residue."""
        strength = np.array([1.0, 1e-14, 0.0])

        at_zero = _shrinkage(strength, 0.0, 1.0)
        regularized = _shrinkage(strength, 1.0, 1.0)

        assert np.all(at_zero == 1.0)
        assert regularized[1] > 0.0
        assert regularized[2] == 0.0


class TestTheRootTwoConstant:
    """``pairwise = sqrt(2) x compare-to-original`` needs Gaussian sampling."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("two_point", 1.00), ("uniform", 1.33), ("laplace", 1.50)],
    )
    def test_the_constant_is_not_robust_to_the_sampling_distribution(
        self, name, expected
    ):
        """Measured 0.996, 1.332 and 1.497 against the Gaussian 1.4142.

        A symmetric two-point distribution gives 1.00 — the ratio is not even
        bounded near sqrt(2), so the constant is a Gaussian result and not a
        general one.
        """
        rng = np.random.default_rng(0)
        size = 200_000
        draw = {
            "two_point": lambda m: rng.choice([-1.0, 1.0], m),
            "uniform": lambda m: rng.uniform(-1, 1, m),
            "laplace": lambda m: rng.laplace(size=m),
        }[name]
        a, b = draw(size), draw(size)

        ratio = np.mean(np.abs(a - b)) / np.mean(np.abs(a - a.mean()))

        assert ratio == pytest.approx(expected, abs=0.05)
        assert not np.isclose(ratio, np.sqrt(2), atol=0.05) or name == "laplace"

    def test_it_holds_for_a_well_behaved_tree(self):
        """The reassuring half: on independent datasets a tree is close enough.

        Measured 1.4118 against 1.4142 for a depth-4 tree at n=300, so the
        Gaussian approximation is not generally hopeless for the models this
        package ships for.
        """
        pytest.importorskip("sklearn")
        from sklearn.datasets import make_friedman1
        from sklearn.tree import DecisionTreeRegressor

        X_eval = make_friedman1(n_samples=80, noise=1.0, random_state=0)[0]
        preds = []
        for seed in range(400):
            X, y = make_friedman1(n_samples=300, noise=1.0, random_state=seed)
            preds.append(
                DecisionTreeRegressor(max_depth=4, random_state=0)
                .fit(X, y)
                .predict(X_eval)
            )
        drawn = np.array(preds)
        half = len(drawn) // 2

        pairwise = np.abs(drawn[:half] - drawn[half:]).mean()
        centered = np.abs(drawn - drawn.mean(axis=0)).mean()

        assert pairwise / centered == pytest.approx(np.sqrt(2), rel=0.05)


class TestIdentityPreconditions:
    """What T1 needs, and what it does not describe."""

    def test_the_identity_fails_without_independence(self):
        """T1 assumes D and D' are independent, and breaks badly otherwise.

        With draws sharing 90% of their noise the ratio measured 0.010 rather
        than 1.0. Common random numbers across replicates would do this.
        """
        rng = np.random.default_rng(1)
        n, p = 200, 5
        X = rng.normal(size=(n, p))
        gram_inv = np.linalg.inv(X.T @ X)
        X_eval = rng.normal(size=(50, p))

        noise = rng.normal(scale=SIGMA, size=(4000, n))
        shared = 0.9 * noise + 0.1 * rng.normal(scale=SIGMA, size=(4000, n))
        first = (noise @ X) @ gram_inv @ X_eval.T
        second = (shared @ X) @ gram_inv @ X_eval.T

        variance = first.var(axis=0).mean()
        independent = np.mean((first - first[::-1]) ** 2) / (2 * variance)
        dependent = np.mean((first - second) ** 2) / (2 * variance)

        assert independent == pytest.approx(1.0, rel=0.1)
        assert dependent < 0.1

    def test_the_identity_holds_for_pairwise_binary_disagreement(self):
        """T1 is exact for labels, provided the comparison is pairwise.

        E[1(A != B)] = 2 p(1-p) = 2 Var. Measured 0.0947 against 0.0950 at
        p = 0.05, and 0.2402 against 0.2408 at p = 0.14.
        """
        rng = np.random.default_rng(0)
        for probability in (0.05, 0.14, 0.32):
            a = rng.random(200_000) < probability
            b = rng.random(200_000) < probability

            pairwise = float(np.mean(a != b))

            assert pairwise == pytest.approx(
                2 * probability * (1 - probability), abs=0.005
            )

    def test_the_modal_statistic_is_a_different_quantity(self):
        """``per_point`` for classification is not what T1 describes.

        Disagreement with the modal prediction measured 0.0498 against a
        pairwise 0.0947 — roughly half. ``bootstrap_predictions`` now reports
        both, and only ``pairwise`` satisfies the identity.
        """
        pytest.importorskip("sklearn")
        from sklearn.datasets import make_classification
        from sklearn.tree import DecisionTreeClassifier

        from stable_cart import bootstrap_predictions

        X, y = make_classification(
            n_samples=300, n_features=6, n_informative=4, random_state=0
        )
        out = bootstrap_predictions(
            lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
            X[:200],
            y[:200],
            X[200:],
            task="categorical",
            n_bootstrap=400,
            random_state=0,
        )

        share = out["bootstrap"].mean(axis=0)
        assert out["pairwise"].mean() == pytest.approx(
            (2 * share * (1 - share)).mean(), rel=0.1
        )
        assert out["per_point"].mean() < out["pairwise"].mean() * 0.85


class TestVarianceFormulaPreconditions:
    """What ``linear_instability`` assumes, and what happens when it is false."""

    @staticmethod
    def _truth(X, X_eval, beta, scale, rng, reps=4000):
        gram_inv = np.linalg.inv(X.T @ X)
        preds = [
            X_eval @ (gram_inv @ (X.T @ (X @ beta + rng.normal(scale=scale))))
            for _ in range(reps)
        ]
        return float(np.mean(np.var(np.array(preds), axis=0)))

    def test_the_constant_variance_form_is_low_under_heteroskedasticity(self):
        """Measured 0.72x the truth with noise scaling in |x1|.

        Not a defect — the formula assumes E[ee'] = sigma^2 I — but it is a
        precondition, not a technicality.
        """
        rng = np.random.default_rng(0)
        n, p = 300, 5
        X = rng.normal(size=(n, p))
        beta = rng.normal(size=p)
        X_eval = rng.normal(size=(200, p))
        scale = 2.0 * (0.3 + np.abs(X[:, 0]))

        truth = self._truth(X, X_eval, beta, scale, rng)
        plain = linear_instability(X, X_eval, sigma=float(np.sqrt(np.mean(scale**2))))[
            "variance_mean"
        ]

        assert plain < truth * 0.85

    def test_the_robust_form_recovers_it(self):
        """The sandwich estimator measured 0.96x the truth in the same setting."""
        rng = np.random.default_rng(0)
        n, p = 300, 5
        X = rng.normal(size=(n, p))
        beta = rng.normal(size=p)
        X_eval = rng.normal(size=(200, p))
        scale = 2.0 * (0.3 + np.abs(X[:, 0]))

        truth = self._truth(X, X_eval, beta, scale, rng)
        robust = float(
            np.mean(
                [
                    linear_instability(
                        X,
                        X_eval,
                        y=X @ beta + rng.normal(scale=scale),
                        robust=True,
                    )["variance_mean"]
                    for _ in range(40)
                ]
            )
        )

        assert robust == pytest.approx(truth, rel=0.15)

    def test_the_two_forms_agree_under_homoskedasticity(self):
        """A correction that changed the answer when it should not would be worse."""
        rng = np.random.default_rng(0)
        n, p = 300, 5
        X = rng.normal(size=(n, p))
        beta = rng.normal(size=p)
        X_eval = rng.normal(size=(200, p))

        plain = linear_instability(X, X_eval, sigma=2.0)["variance_mean"]
        robust = float(
            np.mean(
                [
                    linear_instability(
                        X,
                        X_eval,
                        y=X @ beta + rng.normal(scale=2.0, size=n),
                        robust=True,
                    )["variance_mean"]
                    for _ in range(40)
                ]
            )
        )

        assert robust == pytest.approx(plain, rel=0.15)

    def test_an_ill_conditioned_design_warns(self):
        """Silently returning a finite number from a singular design is the bad case.

        At condition number 1.7e16 the result was finite, meaningless and
        unannounced.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))
        X[:, -1] = X[:, 0] + 1e-9 * rng.normal(size=300)

        with pytest.warns(RuntimeWarning, match="ill-conditioned"):
            linear_instability(X, X[:20], sigma=1.0)

    def test_a_well_conditioned_design_does_not_warn(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            linear_instability(X, X[:20], sigma=1.0)


class TestFrontierScope:
    """Where the optimality result stops applying."""

    @pytest.mark.parametrize("wide", [False, True])
    def test_rank_deficient_designs_are_rejected(self, wide):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 10)) if wide else rng.normal(size=(80, 5))
        if not wide:
            X[:, -1] = X[:, 0]
        y = rng.normal(size=len(X))

        with pytest.raises(ValueError, match="full column rank"):
            shrinkage_coefficients(X, y, mu=1.0, sigma=1.0)
        with pytest.raises(ValueError, match="full column rank"):
            linear_frontier(X, y, n_points=5, sigma=1.0)

    def test_the_optimum_is_only_over_diagonal_linear_shrinkage(self):
        """Over all procedures the constrained problem is trivial.

        "Return the true beta" has zero bias and zero variance and dominates
        every point of the frontier, so the restriction to the shrinkage family
        is load-bearing and cannot be dropped from the statement.
        """
        rng = np.random.default_rng(0)
        p, n = 5, 200
        X = rng.normal(size=(n, p))
        beta = rng.normal(size=p)
        y = X @ beta + rng.normal(scale=SIGMA, size=n)

        result = linear_frontier(X, y, n_points=20, beta=beta, sigma=SIGMA)
        best = min(point["risk"] for point in result["points"])

        # The oracle procedure that ignores the data entirely.
        assert best > 0.0

    def test_mu_one_is_invariant_to_diagonal_reweighting(self):
        """A strengthening, not a break: the weights cancel in the first-order condition.

        Measured argmin mu = 0.999 under four different diagonal metrics.
        """
        rng = np.random.default_rng(0)
        p, n = 6, 200
        X = rng.normal(size=(n, p))
        beta = rng.normal(size=p)
        _u, d, vt = np.linalg.svd(X, full_matrices=False)
        theta = vt @ beta

        grid = np.geomspace(0.05, 40, 400)
        for weights in (d**2 / n, np.ones(p), 1.0 / d**2, d**4):
            risks = []
            for mu in grid:
                s = _shrinkage(d**2 * theta**2, mu, SIGMA)
                risks.append(
                    np.sum(weights * (1 - s) ** 2 * theta**2)
                    + SIGMA**2 * np.sum(weights * s**2 / d**2)
                )
            assert grid[int(np.argmin(risks))] == pytest.approx(1.0, rel=0.1)

    def test_mu_one_moves_under_a_non_diagonal_metric(self):
        """The break. Measured argmin mu = 1.111 when W is not diagonal.

        Evaluating on a distribution whose second-moment matrix is not diagonal
        in the design's singular basis destroys separability, and both the
        optimum and the optimality of the whole family go with it.
        """
        rng = np.random.default_rng(0)
        p, n = 6, 200
        X = rng.normal(size=(n, p))
        beta = rng.normal(size=p)
        _u, d, vt = np.linalg.svd(X, full_matrices=False)
        theta = vt @ beta

        mixing = rng.normal(size=(p, p))
        weights = vt @ (mixing @ mixing.T / p) @ vt.T
        diagonal = np.diag(weights)

        grid = np.geomspace(0.05, 40, 600)
        risks = []
        for mu in grid:
            s = _shrinkage(d**2 * theta**2, mu, SIGMA)
            bias = (s - 1) * theta
            risks.append(
                float(bias @ weights @ bias)
                + SIGMA**2 * float(np.sum(diagonal * s**2 / d**2))
            )
        best = grid[int(np.argmin(risks))]

        assert not np.isclose(best, 1.0, rtol=0.05)


class TestConversionInPractice:
    """The sqrt(2) conversion in the setting the package actually ships for."""

    def test_the_bootstrap_does_not_center_on_the_original_fit(self):
        """So "vs original" and "vs center" are different measurements.

        For a depth-6 tree the bootstrap mean sat 0.43 SD from the original fit,
        and pairwise/vs-original measured 1.279 rather than 1.414 — about a 10%
        error in the conversion, in exactly the setting the package ships for.
        """
        pytest.importorskip("sklearn")
        from sklearn.tree import DecisionTreeRegressor

        from stable_cart import bootstrap_predictions

        rng = np.random.default_rng(2)
        X = rng.normal(size=(120, 3))
        y = X[:, 0] ** 2 + rng.normal(scale=0.3, size=120)
        X_eval = rng.normal(size=(60, 3))

        out = bootstrap_predictions(
            lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
            X,
            y,
            X_eval,
            n_bootstrap=600,
            random_state=0,
        )
        drawn = out["bootstrap"]
        offset = np.mean(np.abs(drawn.mean(axis=0) - out["original"]))
        spread = np.mean(drawn.std(axis=0))

        assert offset > 0.2 * spread

        pairwise = np.sqrt(out["pairwise"].mean())
        vs_original = out["mape_per_point"].mean()
        assert not np.isclose(
            pairwise / vs_original * np.sqrt(np.pi) / 2, np.sqrt(2), rtol=0.05
        )


def test_the_conversion_constant_is_documented_as_gaussian_only():
    """The two constants differ by sqrt(2) and are labeled as Gaussian results.

    The module docstring has to carry the qualifier, because the constants
    themselves look unconditional and were briefly presented that way.
    """
    import stable_cart.linear as module

    assert pytest.approx(np.sqrt(2)) == PAIRWISE / CENTERED_MAD
    assert "Gaussian" in module.__doc__
