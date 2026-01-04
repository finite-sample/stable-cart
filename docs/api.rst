.. _api_reference:

API Reference
=============

This project provides unified tree estimators that follow the familiar scikit-learn API. 
All classes support both regression and classification through a unified ``task`` parameter.

.. _tree_estimators:

Unified Tree Estimators
------------------------

These are the main estimators that work for both regression and classification.
All classes inherit from :class:`BaseStableTree` and support both ``task='regression'`` and ``task='classification'``:

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

.. _base_classes:

Base Classes
------------

For advanced users and researchers who want to extend the functionality or understand the underlying architecture:

.. autoclass:: BaseStableTree
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

.. _evaluation_functions:

Evaluation Functions
--------------------

These functions help assess model performance and prediction stability.
Use these to compare different tree algorithms or measure the effectiveness of stability features:

.. autofunction:: prediction_stability

.. autofunction:: evaluate_models

.. _advanced_classes:

Advanced Classes for Researchers
---------------------------------

Internal classes for advanced customization and research.
These provide the building blocks for creating custom stability algorithms:

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
