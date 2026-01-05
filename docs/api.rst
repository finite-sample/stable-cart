.. _api_reference:

API Reference
=============

This page documents the complete API for stable-cart. All tree estimators follow the scikit-learn API
and support both regression and classification through a unified ``task`` parameter.

.. currentmodule:: stable_cart

Tree Estimators
---------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ~LessGreedyHybridTree
   ~BootstrapVariancePenalizedTree  
   ~RobustPrefixHonestTree

Evaluation Functions
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ~prediction_stability
   ~evaluate_models

Base Classes and Advanced APIs
------------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ~BaseStableTree
   ~SplitCandidate
   ~StabilityMetrics
   ~SplitStrategy
   ~create_split_strategy
