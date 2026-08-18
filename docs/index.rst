Prediction instability audits
=============================

This package measures how much a fitted model's predictions change when its
training data change. It repeats the complete model-building procedure on
bootstrap samples and preserves the resulting prediction distribution for every
evaluation case.

The package works with arbitrary estimators and pipelines. It does not claim to
stabilize them. Its job is to measure instability correctly, expose where it is
concentrated, and compare candidate procedures without mistaking a constant
predictor for a success.

Quick start
-----------

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.linear_model import RidgeCV
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler

   from stable_cart import bootstrap_instability

   X, y = make_regression(n_samples=400, n_features=10, random_state=0)

   result = bootstrap_instability(
       lambda: make_pipeline(StandardScaler(), RidgeCV()),
       X[:300],
       y[:300],
       X[300:],
       n_bootstrap=200,
       random_state=0,
   )

   print(result["mape"], result["mape_standard_error"])

The factory returns a fresh, unfitted procedure. Put preprocessing, feature
selection, and tuning inside that procedure if they should be repeated in the
audit.

Start with :doc:`operating_point` for the evaluation workflow and
:doc:`theory` for the quantities being estimated.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   operating_point
   theory
   scope

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Links

   GitHub repository <https://github.com/finite-sample/stable-cart>
   PyPI package <https://pypi.org/project/stable-cart/>
