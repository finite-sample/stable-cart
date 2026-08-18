.. _theory:

What the instability quantities mean
====================================

The package estimates the distribution of predictions produced by a specified
model-building procedure when its training sample changes. The procedure, the
resampling scheme, the prediction representation, and the evaluation cases are
all part of the estimand.

MAPE against the original fit
-----------------------------

Let :math:`D_0` be the observed training data, :math:`D_b` a bootstrap sample,
and :math:`f_D(x)` the prediction produced after fitting the complete procedure
to :math:`D`. The per-case mean absolute prediction error is

.. math::

   \frac{1}{B}\sum_{b=1}^B |f_{D_b}(x) - f_{D_0}(x)|.

The package averages this quantity over evaluation cases to report ``mape``.
For class labels, the absolute difference is replaced by an indicator that the
two labels disagree. This makes the result invariant to class names.

Pairwise instability
--------------------

Pairwise instability compares two independent refits. For numeric predictions,

.. math::

   E[(f_D(x) - f_{D'}(x))^2] = 2\operatorname{Var}(f_D(x)).

The identity needs only independent and identically distributed predictions.
The implementation estimates it with twice the sample variance, which uses all
bootstrap refits. For categorical labels, it computes the probability that two
distinct bootstrap refits disagree. For probability vectors, it computes the
expected squared Euclidean distance between the vectors.

MAPE and pairwise instability answer different questions. MAPE is conditional
on the fitted model that the user actually has. Pairwise instability describes
the spread between two new fitted models and does not privilege the observed
fit.

Classification representations
------------------------------

Class labels have no numerical spacing. The variance of codes such as 0, 1, and
100 changes when the classes are renamed and is not a classification stability
measure.

Label disagreement measures whether the selected class changes. Probability
instability measures how the full risk distribution changes even when the
selected class stays fixed. The implementation aligns probability columns with
the estimator's ``classes_`` attribute before comparing them.

Resampling defines the question
-------------------------------

Regression uses the ordinary pairs bootstrap. Classification conditions only
on a draw containing at least two observed classes, because many standard
classifiers are undefined on a one-class fit. Class prevalence otherwise varies
across refits; freezing it through stratification can materially understate
probability instability. The returned rejected-draw count makes this
conditioning visible.
If a supplied fitting procedure cannot fit a multiclass resample that retains
at least two observed classes but omits another rare class, the audit fails
rather than silently substituting the original data or changing the resampling
distribution.

Groups, time dependence, clusters, and survey designs need resampling schemes
that respect their sampling process. The current API does not yet implement
those schemes. Treat ordinary row resampling as wrong for those data rather than
as a harmless default.

Monte Carlo uncertainty
-----------------------

The returned standard errors quantify error from approximating the bootstrap
distribution with finitely many refits. MAPE uses variation across refits.
Pairwise instability uses a delete-one jackknife standard error for the same
all-pairs U-statistic used by the point estimate.

These standard errors do not quantify how much the instability estimate would
change across new original datasets. More bootstrap refits reduce Monte Carlo
error but cannot remove that dataset-level uncertainty.

Fixed-design linear calibration
-------------------------------

``stable_cart.linear_instability`` supplies an analytic calibration
case. Under a fixed design, homoskedastic Gaussian errors, and known noise scale,
least-squares prediction variance at :math:`x` is

.. math::

   \sigma^2 x'(X'X)^{-1}x.

The expected absolute difference between two independent Gaussian refits is
:math:`2\sqrt{v(x)/\pi}`. The expected absolute deviation from the sampling mean
is :math:`\sqrt{2v(x)/\pi}`.

The second expression is not MAPE against the observed fitted model unless that
model equals the sampling mean. The analytic function therefore calls it
``centered_mad`` and leaves MAPE to the bootstrap calculation. Its ``robust``
branch is an HC0 plug-in covariance estimate, not an exact finite-sample result.

The companion linear frontier is oracle when true coefficients and noise scale
are supplied and a plug-in curve otherwise. Its optimality applies only to
diagonal linear shrinkage under the stated fixed-design metric. It is a check on
the measurement code, not the package's proposed stabilization method.
