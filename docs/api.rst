.. _api_reference:

API reference
=============

The top-level API measures instability for arbitrary fitted procedures and
includes one representative-model selector plus fixed-design linear
calibration tools. Inclusion means the implementation and workflow are
supported; it does not imply a universal stabilization guarantee.

.. currentmodule:: stable_cart

Bootstrap audits
----------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   bootstrap_predictions
   bootstrap_instability

``bootstrap_predictions`` returns every refitted prediction and the per-case
statistics. ``bootstrap_instability`` returns aggregate summaries and Monte
Carlo standard errors.

Model-selection frontier
------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   stability_frontier
   pareto_front

The score used to construct a frontier is a validation score. It is not a final
test-set performance estimate.

Tree-structure audits
---------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   split_features
   split_feature_paths
   explanation_instability
   root_agreement
   path_agreement

Read structural instability beside prediction instability. A consistently
shallow or inaccurate tree can have perfectly stable structure.

Plots
-----

Install plotting dependencies with ``pip install "stable-cart[plots]"``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   plot_prediction_instability
   plot_mape_by_prediction
   plot_stability_frontier

Supported estimator and analytic tools
---------------------------------------

``RepresentativeEstimator`` supports multiclass classification and selects a
single fitted candidate by validation-set prediction centrality. Its observed
stability benefit is task dependent. The linear functions are exact or
calibrated calculations under their documented fixed-design assumptions; they
are not general estimators.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RepresentativeEstimator
   linear_instability
   linear_frontier
   shrinkage_coefficients
