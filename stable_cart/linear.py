r"""The accuracy-stability frontier for linear prediction, in closed form.

Everything else in this package measures stability by resampling, because for a
decision tree there is nothing else to do. For a linear model there is: the
sampling distribution of the coefficients is available in closed form under a
fixed-design Gaussian model. That makes this module a calibration case for the
resampling implementation, not a generally applicable stabilization method.

That is useful twice. It says precisely what the resampling estimate is
estimating, and it gives the only setting in which that estimate can be checked
against a known answer rather than against another estimate.

**The identity underneath all of it.** For independent datasets ``D``, ``D'``
drawn from the same distribution, and *any* procedure at all,

.. math::

    \mathbb{E}\big[(f_D(x) - f_{D'}(x))^2\big] = 2\,\operatorname{Var}_D(f_D(x)),

because the two predictions are iid and their means cancel. No assumption about
the model, the loss, or the noise is used. Squared prediction instability is
twice the variance term of the bias-variance decomposition — so the
accuracy-stability frontier is the bias-variance frontier, drawn on axes a user
can act on. Riley and Collins say as much in words ("minimize the variance
(instability) of predictions"); this module is the arithmetic.

**The constant that is easy to get wrong.** The measure usually reported is the
*absolute* difference, not the squared one. Under Gaussian sampling

.. math::

    \mathbb{E}|f_D(x) - f_{D'}(x)| = \tfrac{2}{\sqrt\pi}\,\sigma_x,
    \qquad
    \mathbb{E}|f_D(x) - \mathbb{E}f_D(x)|
      = \sqrt{\tfrac{2}{\pi}}\,\sigma_x,

where :math:`\sigma_x` is the standard deviation of the prediction at ``x``. The
first compares two independently refitted models; the second compares a refit
with the center of its sampling distribution. They differ by exactly
:math:`\sqrt 2`. The second is **not** Riley and Collins's MAPE against the
observed original model unless that model happens to equal the sampling center.
The nonparametric bootstrap need not be centered there, so the package computes
MAPE directly from resamples instead of manufacturing it from this constant.

References
----------
Riley and Collins, *Stability of clinical prediction models developed using
statistical or machine learning methods*, Biometrical Journal 65(8), 2023.
"""

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "linear_instability",
    "linear_frontier",
    "shrinkage_coefficients",
]

# E|N(0, 2v)| = (2/sqrt(pi)) sqrt(v): two independently refitted models.
PAIRWISE = 2.0 / np.sqrt(np.pi)
# E|N(0, v)| = sqrt(2/pi) sqrt(v): deviation from the sampling mean.
CENTERED_MAD = np.sqrt(2.0 / np.pi)


def _require_full_column_rank(X: NDArray[np.floating]) -> None:
    """Reject designs for which the documented coefficient frontier is undefined."""
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    rank = int(np.linalg.matrix_rank(X))
    n_features = X.shape[1]
    if rank < n_features:
        raise ValueError(
            "X must have full column rank for fixed-design linear calibration; "
            f"got rank {rank} with {n_features} columns"
        )


def linear_instability(
    X: NDArray[np.floating],
    X_eval: NDArray[np.floating],
    sigma: float | None = None,
    y: NDArray[np.floating] | None = None,
    robust: bool = False,
) -> dict[str, Any]:
    r"""
    Analytic prediction instability of least squares, conditional on the design.

    This is the closed form of what
    :func:`~stable_cart.bootstrap_instability` estimates by resampling. With
    :math:`\hat\beta \sim N(\beta, \sigma^2 (X'X)^{-1})`, the prediction at a
    point ``x`` has variance :math:`\sigma^2 x'(X'X)^{-1}x`, and every instability
    measure follows from it.

    Parameters
    ----------
    X
        Training design matrix of shape (n_samples, n_features). Include a column
        of ones if the model has an intercept; this function takes the design as
        given.
    X_eval
        Points at which to evaluate, shape (n_eval, n_features).
    sigma
        Standard deviation of the noise, assumed constant across observations.
        This is the true value, not an estimate — pass
        :math:`\hat\sigma = \sqrt{\mathrm{RSS}/(n-p)}` for the plug-in version.
        Required unless ``robust=True``.
    y
        Training targets. Required when ``robust=True``, which needs residuals.
    robust
        Drop the constant-variance assumption and use the HC0 plug-in estimate
        heteroskedasticity-consistent form
        :math:`(X'X)^{-1}\big(\sum_i x_i x_i' \hat e_i^2\big)(X'X)^{-1}`.
        This branch is an estimated asymptotic covariance, not an exact
        finite-sample result. The assumption is not a technicality: with noise scaling in
        :math:`|x_1|`, the constant-variance formula understates the truth by
        48%, while this one recovers it. Under genuine homoskedasticity the two
        agree, so the cost of using it is only the loss of a known ``sigma``.

    Returns
    -------
    dict[str, Any]
        ``variance`` — per-point prediction variance;
        ``s1`` — per-point :math:`E|f_D(x)-f_{D'}(x)|`, the pairwise measure;
        ``centered_mad`` — per-point
        :math:`E|f_D(x)-E f_D(x)|` under Gaussian sampling;
        ``s2`` — per-point squared pairwise instability, exactly twice
        ``variance``;
        ``variance_mean``, ``s1_mean``, ``centered_mad_mean`` — integrated versions.

    Raises
    ------
    ValueError
        If inputs are nonfinite, if ``sigma`` is negative, if ``X`` and
        ``X_eval`` disagree on width, or if the arguments needed for the requested
        variance form are missing. The robust form also requires more observations
        than columns so residual variation can be estimated.

    Notes
    -----
    A ``RuntimeWarning`` is issued when ``X'X`` is so ill-conditioned that the
    result is not meaningful. A silently finite answer from a near-singular
    design is the dangerous case; an exactly singular one already raises.


    ``s1_mean`` is the mean of the per-point ``s1``, which is proportional to
    :math:`E\sqrt{v(x)}` and **not** to :math:`\sqrt{E v(x)}`. The two differ by
    Jensen's inequality whenever the variance is not constant across evaluation
    points, and the first is what a resampling protocol reports.

    Examples
    --------
    >>> import numpy as np
    >>> from stable_cart import linear_instability
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 4))
    >>> out = linear_instability(X, rng.normal(size=(50, 4)), sigma=2.0)
    >>> bool(np.allclose(out["s2"], 2 * out["variance"]))
    True
    """
    if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
        raise ValueError("sigma must be finite and non-negative")
    X = np.asarray(X, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    if X.ndim != 2 or X_eval.ndim != 2:
        raise ValueError("X and X_eval must be two-dimensional")
    if not np.all(np.isfinite(X_eval)):
        raise ValueError("X_eval must contain only finite values")
    if X.shape[1] != X_eval.shape[1]:
        raise ValueError(f"X has {X.shape[1]} columns but X_eval has {X_eval.shape[1]}")
    _require_full_column_rank(X)
    if robust and y is None:
        raise ValueError("robust=True needs y, to form residuals")
    if robust and X.shape[0] == X.shape[1]:
        raise ValueError(
            "robust=True needs more observations than columns to estimate "
            "residual variation"
        )
    if not robust and sigma is None:
        raise ValueError("pass sigma, or robust=True with y")

    gram = X.T @ X
    condition = float(np.linalg.cond(gram))
    if condition > 1e12:
        warnings.warn(
            f"X'X is ill-conditioned (condition number {condition:.3g}); the "
            "prediction variance below is not meaningful",
            RuntimeWarning,
            stacklevel=2,
        )
    gram_inv = np.linalg.inv(gram)

    if robust:
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim != 1 or len(y_array) != len(X):
            raise ValueError("y must be one-dimensional with one value per row of X")
        if not np.all(np.isfinite(y_array)):
            raise ValueError("y must contain only finite values")
        coef, *_ = np.linalg.lstsq(X, y_array, rcond=None)
        residual = y_array - X @ coef
        meat = (X * residual[:, None] ** 2).T @ X
        covariance = gram_inv @ meat @ gram_inv
        variance = np.einsum("ij,jk,ik->i", X_eval, covariance, X_eval)
    else:
        assert sigma is not None  # guarded above; narrows the type for checkers
        leverage = np.einsum("ij,jk,ik->i", X_eval, gram_inv, X_eval)
        variance = sigma**2 * leverage
    spread = np.sqrt(variance)

    return {
        "variance": variance,
        "s2": 2.0 * variance,
        "s1": PAIRWISE * spread,
        "centered_mad": CENTERED_MAD * spread,
        "variance_mean": float(np.mean(variance)),
        "s1_mean": float(np.mean(PAIRWISE * spread)),
        "centered_mad_mean": float(np.mean(CENTERED_MAD * spread)),
    }


def _shrinkage(strength, mu, sigma):
    """Per-direction shrinkage factors at price ``mu``.

    ``mu=0`` is defined as the ordinary least-squares endpoint. For positive
    ``mu``, an exactly null direction is shrunk to zero. No hidden numerical
    tolerance is used: a package cannot distinguish a tiny real signal from
    floating-point residue without making that threshold part of the model.
    """
    strength = np.asarray(strength, dtype=float)
    if mu == 0:
        return np.ones_like(strength)
    denominator = strength + mu * sigma**2
    factors = np.zeros_like(strength)
    np.divide(strength, denominator, out=factors, where=denominator > 0)
    return factors


def _spectrum(X, y, beta, sigma, signal="pooled"):
    """SVD of the design plus the signal and noise level in that basis.

    ``signal`` decides how the unknown per-direction signal strength is supplied,
    and which choice wins depends on **where the signal sits**, not on ``p``.
    Held-out error against least squares, ``n=150``, ``sigma=2``, with ``theta``
    constructed in the design's singular basis:

    ======  ==========  ================  ================
    p       theta       pooled            per-direction
    ======  ==========  ================  ================
    10      dense       -1.7% to +1.1%    +0.3% to +13.7%
    50      dense       -7.0% to -4.0%    +0.3% to +22.7%
    10      sparse      -0.2%             -46.9%
    50      sparse      -5.4%             -63.8%
    10      one spike   -0.4%             -61.8%
    50      one spike   -4.4%             -67.1%
    ======  ==========  ================  ================

    The dense rows give the range over five draws of the design and signal,
    because a single draw is misleading: pooling's advantage at ``p=10``
    straddles zero and only becomes reliable by ``p=50``.

    Pooling wins when the signal is spread across singular directions, because it
    estimates one number instead of ``p`` and that is exactly ridge regression.
    Per-direction wins — by two thirds — when the signal is concentrated in a few
    directions, because then the truncation to zero is doing real work.

    ``"pooled"`` is the default because a signal spread across directions is the
    common case and it is the safer failure. It is not uniformly better, and an
    earlier version of this docstring claimed it was; the sparse rows above are
    the counterexample, pinned by ``tests/test_linear_adversarial.py``.

    Note that sparsity has to be in the *singular* basis to matter. A Gaussian
    design rotates a sparse ``beta`` into a dense ``theta``, so a sparse
    coefficient vector alone buys nothing here.
    """
    if signal not in ("pooled", "per_direction"):
        raise ValueError("signal must be 'pooled' or 'per_direction'")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be one-dimensional with one value per row of X")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values")
    n, p = X.shape
    _require_full_column_rank(X)
    u, d, vt = np.linalg.svd(X, full_matrices=False)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if sigma is None:
        if n <= p:
            raise ValueError(
                f"cannot estimate sigma with n={n} <= p={p}; pass sigma explicitly"
            )
        residual = y - X @ coef
        sigma = float(np.sqrt(residual @ residual / (n - p)))

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")

    if beta is not None:
        beta = np.asarray(beta, dtype=float)
        if beta.shape != (p,):
            raise ValueError(f"beta must have shape ({p},)")
        if not np.all(np.isfinite(beta)):
            raise ValueError("beta must contain only finite values")
        theta_sq = (vt @ beta) ** 2
    else:
        # theta_hat_j^2 overstates theta_j^2 by sigma^2/d_j^2 in expectation, and
        # the shrinkage is a ratio of the two, so the raw estimate shrinks too
        # little. Subtract the known inflation and truncate at zero.
        unbiased = (vt @ coef) ** 2 - sigma**2 / d**2
        if signal == "pooled":
            theta_sq = np.full(p, max(float(np.mean(unbiased)), 0.0))
        else:
            theta_sq = np.maximum(unbiased, 0.0)

    return u, d, vt, coef, float(sigma), theta_sq


def _bias_variance(s, d, theta_sq, sigma, n):
    """In-sample squared bias and prediction variance of a shrinkage estimator."""
    bias2 = float(np.sum((1.0 - s) ** 2 * theta_sq * d**2) / n)
    variance = float(sigma**2 * np.sum(s**2) / n)
    return bias2, variance


def shrinkage_coefficients(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    mu: float,
    beta: NDArray[np.floating] | None = None,
    sigma: float | None = None,
    signal: str = "pooled",
) -> NDArray[np.floating]:
    r"""
    Coefficients of the estimator that achieves the frontier point at ``mu``.

    Shrinks the least-squares solution along each singular direction by
    :math:`s_j = d_j^2\theta_j^2/(d_j^2\theta_j^2 + \mu\sigma^2)`, which is the
    exact solution of "minimize squared bias subject to a variance budget" —
    see :func:`linear_frontier`.

    Parameters
    ----------
    X
        Full-column-rank design matrix of shape (n_samples, n_features).
    y
        Targets of shape (n_samples,).
    mu
        Price of variance. ``mu=0`` returns least squares; ``mu=1`` minimizes
        risk; larger values buy stability at more than it is worth in accuracy.
    beta
        True coefficients, if known. Estimated from the data when omitted.
    sigma
        True noise level, if known. Estimated as
        :math:`\sqrt{\mathrm{RSS}/(n-p)}` when omitted.
    signal
        How to supply the unknown signal strength when ``beta`` is not given.
        ``'pooled'`` estimates one value for all directions, which makes this
        ridge regression and is safer when signal is spread across directions.
        ``'per_direction'`` estimates each direction separately; its estimation
        cost loses for diffuse signal but it can win decisively when signal is
        concentrated in a few singular directions. See :func:`linear_frontier`
        for the measured boundaries.

    Returns
    -------
    NDArray[np.floating]
        Coefficients of shape (n_features,).

    Raises
    ------
    ValueError
        If ``mu`` or ``sigma`` is negative or nonfinite, any array input is
        nonfinite, or the design is not full column rank.

    Examples
    --------
    >>> import numpy as np
    >>> from stable_cart import shrinkage_coefficients
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 4)); y = X @ np.arange(4.0) + rng.normal(size=200)
    >>> ols = shrinkage_coefficients(X, y, mu=0.0)
    >>> shrunk = shrinkage_coefficients(X, y, mu=5.0)
    >>> bool(np.linalg.norm(shrunk) < np.linalg.norm(ols))
    True
    """
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and non-negative")
    # The OLS endpoint does not depend on a noise estimate. Passing zero here
    # keeps all design, target, signal, beta, and explicit-sigma validation in
    # one place without attempting RSS / (n - p) for a saturated design.
    spectrum_sigma = 0.0 if mu == 0 and sigma is None else sigma
    _u, d, vt, coef, sigma_hat, theta_sq = _spectrum(X, y, beta, spectrum_sigma, signal)
    s = _shrinkage(d**2 * theta_sq, mu, sigma_hat)
    return vt.T @ (s * (vt @ coef))


def linear_frontier(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    n_points: int = 50,
    mu_max: float = 1000.0,
    beta: NDArray[np.floating] | None = None,
    sigma: float | None = None,
    signal: str = "pooled",
) -> dict[str, Any]:
    r"""
    Trace an oracle or plug-in frontier for fixed-design linear prediction.

    Among estimators that shrink each singular direction of the design, the one
    with the least squared bias at a given prediction variance is

    .. math::

        s_j(\mu) = \frac{d_j^2\theta_j^2}{d_j^2\theta_j^2 + \mu\sigma^2},

    with :math:`\mu` the Lagrange multiplier on the variance budget. Sweeping
    :math:`\mu` from 0 to :math:`\infty` traces the whole frontier, from the
    minimum-variance zero-bias endpoint to the zero-variance constant. The first
    point equals least squares when every singular direction carries signal; it
    drops exactly null directions otherwise.

    Two consequences are worth having in front of you. The slope of the frontier
    is :math:`dB/dV = -\mu` exactly, so :math:`\mu` *is* the exchange rate — the
    units of squared bias you pay per unit of variance you buy. And since risk is
    :math:`\sigma^2 + B + V`, risk is minimized exactly at :math:`\mu = 1`.
    Everything to the stable side of that point costs strictly more accuracy than
    it saves; that is not a matter of taste, it is where the slope crosses one.

    Parameters
    ----------
    X
        Full-column-rank design matrix of shape (n_samples, n_features). Include
        a column of ones for an intercept.
    y
        Targets of shape (n_samples,).
    n_points
        Number of points along the frontier, spaced geometrically in ``mu``.
    mu_max
        Largest ``mu`` to trace. The frontier approaches the constant predictor
        as ``mu`` grows.
    beta
        True coefficients, if known — used by tests and by simulation studies to
        get the oracle fixed-design frontier rather than the plug-in curve.
    sigma
        True noise level, if known.
    signal
        How the unknown signal strength is supplied when ``beta`` is omitted;
        see :func:`shrinkage_coefficients`. ``'pooled'`` (the default) makes the
        traced path the ridge path.

    Returns
    -------
    dict[str, Any]
        ``points`` — a list of dicts, each with ``mu``, ``bias2``, ``variance``,
        ``risk`` (excess over the noise floor, i.e. ``bias2 + variance``),
        ``s1`` and ``s2`` instability, ``exchange_rate`` (equal to ``mu``), and
        ``shrinkage`` (the per-direction factors);
        ``risk_optimal`` — the point at ``mu = 1``;
        ``sigma`` — the noise level used, estimated if it was not supplied.

    Raises
    ------
    ValueError
        If ``n_points`` is below 2, ``mu_max`` is not finite and positive,
        ``sigma`` is negative or nonfinite, an array input is nonfinite, or the
        design is not full column rank.

    Notes
    -----
    The separability that makes the closed form exact needs the evaluation metric
    to be diagonal in the design's singular basis. Risk here is therefore
    *in-sample* prediction risk, weighted by :math:`X'X/n` — the standard choice,
    and the one under which :math:`s_j(\mu)` above is exactly optimal.

    When ``beta`` and ``sigma`` are not supplied they are estimated from the same
    data, so the returned frontier is optimistic near :math:`\mu = 0` in the same
    way any in-sample curve is.

    Examples
    --------
    >>> import numpy as np
    >>> from stable_cart import linear_frontier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 5)); y = X @ np.arange(5.0) + rng.normal(size=200)
    >>> out = linear_frontier(X, y, n_points=20)
    >>> out["risk_optimal"]["mu"]
    1.0
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if not np.isfinite(mu_max) or mu_max <= 0:
        raise ValueError("mu_max must be finite and positive")

    u, d, _vt, _coef, sigma_hat, theta_sq = _spectrum(X, y, beta, sigma, signal)
    n = np.asarray(X).shape[0]
    strength = d**2 * theta_sq

    def at(mu):
        s = _shrinkage(strength, mu, sigma_hat)
        if mu == 0:
            # The Pareto endpoint is the mu -> 0+ limit: directions known to
            # carry exactly no signal can be removed without bias. The public
            # coefficient function separately defines mu=0 as operational OLS.
            s = np.where(strength > 0, 1.0, 0.0)
        bias2, variance = _bias_variance(s, d, theta_sq, sigma_hat, n)
        point_variance = sigma_hat**2 * (u**2 @ s**2)
        return {
            "mu": float(mu),
            "bias2": bias2,
            "variance": variance,
            "risk": bias2 + variance,
            "s2": 2.0 * variance,
            "s1": PAIRWISE * float(np.mean(np.sqrt(point_variance))),
            "exchange_rate": float(mu),
            "shrinkage": s,
        }

    positive_count = n_points - 1
    positive_grid = (
        np.array([mu_max])
        if positive_count == 1
        else np.geomspace(mu_max * 1e-6, mu_max, positive_count)
    )
    grid = np.concatenate([[0.0], positive_grid])
    return {
        "points": [at(mu) for mu in grid],
        "risk_optimal": at(1.0),
        "sigma": sigma_hat,
    }
