.. _api_reference:

API Reference
=============

Two halves. The **measurement** functions work on any scikit-learn estimator and
are the ones to reach for first — you cannot choose an operating point on the
accuracy-stability tradeoff without seeing it. The **estimators** are trees that
try to sit further up that curve.

.. currentmodule:: stable_cart

.. _measuring_stability:

Measuring prediction stability
------------------------------

The protocol is Riley and Collins, *Stability of clinical prediction models
developed using statistical or machine learning methods*, Biometrical Journal
65(8), 2023: refit the whole model-building step on bootstrap resamples of the
training data and compare each refitted model's predictions with the original
model's, **for the same individuals**.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ~bootstrap_instability
   ~bootstrap_predictions
   ~stability_frontier
   ~pareto_front

.. note::

   Instability on its own is meaningless: a model that ignores its training
   data scores a perfect zero. Read every number here next to an accuracy
   measure, which is why :func:`stability_frontier` reports both and returns the
   Pareto set rather than a winner.

.. _plots:

Plots
-----

Install with ``pip install "stable-cart[plots]"``. Each takes an ``ax`` and
returns it, so they compose into a figure you control.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ~plot_prediction_instability
   ~plot_mape_by_prediction
   ~plot_stability_frontier

.. _tree_estimators:

Tree estimators
---------------

:class:`StableTree` is the one to start with: every one of its parameters is
measured to change a prediction, and it handles multi-class classification. The
other four combine several stability primitives at once and three of them are
binary-classification only.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ~StableTree
   ~LessGreedyHybridTree
   ~BootstrapVariancePenalizedTree
   ~RobustPrefixHonestTree
   ~CentroidTree

.. _evaluation_functions:

Comparing already-fitted models
-------------------------------

A different question from resampling instability: whether a *set* of fitted
models agree with each other. A model can agree with its peers and still move
wildly under resampling.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ~prediction_stability
   ~evaluate_models

Base classes and research APIs
------------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ~BaseStableTree
   ~SplitCandidate
   ~StabilityMetrics
   ~SplitStrategy
   ~create_split_strategy
