Welcome to stable-cart's documentation!
=======================================

**stable-cart** provides individual decision trees with stability-focused modifications to the tree-building process. All trees follow the familiar scikit-learn API while incorporating advanced stability features.

Key Features
------------

🌳 **Individual Decision Trees**: Single trees (not ensembles) with stability-focused modifications

🎯 **Stability Mechanisms**: Multiple approaches including data partitioning, consensus, and bootstrap methods

📊 **sklearn Compatible**: Works seamlessly with pipelines, cross-validation, and grid search

🔬 **Analysis Tools**: Bootstrap variance measurement for evaluating prediction stability

Quick Start
-----------

.. code-block:: python

   from stable_cart import LessGreedyHybridTree
   from sklearn.datasets import make_classification
   
   # Generate sample data
   X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
   
   # Create and train a stable tree
   tree = LessGreedyHybridTree(
       task='classification',
       max_depth=6,
       min_samples_leaf=2,
       split_frac=0.9,
       val_frac=0.05,
       est_frac=0.05,
       random_state=42
   )
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
   :maxdepth: 2
   :caption: Examples:

   notebooks/index

.. toctree::
   :maxdepth: 1
   :caption: Links:
   
   GitHub Repository <https://github.com/finite-sample/stable-cart>
   PyPI Package <https://pypi.org/project/stable-cart/>
