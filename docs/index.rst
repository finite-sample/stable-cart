Welcome to stable-cart's documentation!
=======================================

**stable-cart** provides unified tree estimators with enhanced prediction stability for both regression and classification tasks. All trees follow the familiar scikit-learn API while incorporating advanced stability features.

Key Features
------------

🌳 **Unified Architecture**: Single classes handle both regression and classification via ``task`` parameter

🎯 **Enhanced Stability**: Multiple stability primitives reduce prediction variance across training runs

📊 **sklearn Compatible**: Works seamlessly with pipelines, cross-validation, and grid search

Quick Start
-----------

.. code-block:: python

   from stable_cart import LessGreedyHybridTree
   from sklearn.datasets import make_classification
   
   # Works for both regression and classification
   X, y = make_classification(n_samples=1000, n_features=10)
   
   tree = LessGreedyHybridTree(task='classification', max_depth=5)
   tree.fit(X, y)
   predictions = tree.predict(X)

See the :ref:`tree_estimators` section for a complete list of available estimators and the :ref:`evaluation_functions` for assessing model stability.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api

.. toctree::
   :maxdepth: 1
   :caption: Links:
   
   GitHub Repository <https://github.com/finite-sample/stable-cart>
   PyPI Package <https://pypi.org/project/stable-cart/>
