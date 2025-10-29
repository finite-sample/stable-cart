API reference
=============

This project packages estimators that follow the familiar scikit-learn API. 
The implementation lives in the ``stable-cart`` directory. All classes and 
functions are automatically documented from their docstrings.

Estimators
----------

.. autoclass:: stable_cart.LessGreedyHybridRegressor
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: stable_cart.BootstrapVariancePenalizedRegressor
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: stable_cart.RobustPrefixHonestClassifier
   :members:
   :inherited-members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: stable_cart.prediction_stability

.. autofunction:: stable_cart.evaluate_models
