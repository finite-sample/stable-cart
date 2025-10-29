API reference
=============

This project provides unified tree estimators that follow the familiar scikit-learn API. 
All classes support both regression and classification through a unified ``task`` parameter.

Unified Tree Estimators
------------------------

These are the main estimators that work for both regression and classification:

.. currentmodule:: stable_cart

.. autoclass:: LessGreedyHybridTree
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: BootstrapVariancePenalizedTree
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: RobustPrefixHonestTree
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Base Classes
------------

For advanced users and researchers:

.. autoclass:: BaseStableTree
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Evaluation Functions
--------------------

These functions help assess model performance and prediction stability:

.. autofunction:: prediction_stability

.. autofunction:: evaluate_models

Advanced Classes for Researchers
---------------------------------

Internal classes for advanced customization and research:

.. autoclass:: SplitCandidate
   :members:
   :show-inheritance:

.. autoclass:: StabilityMetrics
   :members:
   :show-inheritance:

.. autoclass:: SplitStrategy
   :members:
   :show-inheritance:

.. autofunction:: create_split_strategy
