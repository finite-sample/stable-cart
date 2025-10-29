API reference
=============

This project provides unified tree estimators that follow the familiar scikit-learn API. 
All classes support both regression and classification through a unified ``task`` parameter.
The implementation lives in the ``stable-cart`` directory, and all classes and 
functions are automatically documented from their docstrings.

Unified Tree Estimators
------------------------

These are the main estimators that work for both regression and classification:

.. autoclass:: stable_cart.LessGreedyHybridTree
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: stable_cart.BootstrapVariancePenalizedTree
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: stable_cart.RobustPrefixHonestTree
   :members:
   :inherited-members:
   :show-inheritance:

Base Classes
------------

For advanced users and researchers:

.. autoclass:: stable_cart.BaseStableTree
   :members:
   :inherited-members:
   :show-inheritance:

Utility Classes
---------------

.. autoclass:: stable_cart.SplitCandidate
   :members:
   :show-inheritance:

.. autoclass:: stable_cart.StabilityMetrics
   :members:
   :show-inheritance:

.. autoclass:: stable_cart.SplitStrategy
   :members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: stable_cart.prediction_stability

.. autofunction:: stable_cart.evaluate_models

.. autofunction:: stable_cart.create_split_strategy
