Basic Usage of Stable-CART
==========================

This section demonstrates the basic usage of stable-cart decision trees for both classification and regression tasks.

Setup
-----

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_classification, make_regression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, mean_squared_error

   from stable_cart import LessGreedyHybridTree, BootstrapVariancePenalizedTree, RobustPrefixHonestTree

Classification Example
----------------------

Let's start with a simple classification task using the LessGreedyHybridTree.

.. code-block:: python

   # Generate synthetic classification data
   X_clf, y_clf = make_classification(
       n_samples=1000,
       n_features=20,
       n_informative=10,
       n_redundant=10,
       n_clusters_per_class=1,
       random_state=42
   )

   X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
       X_clf, y_clf, test_size=0.3, random_state=42
   )

   print(f"Classification dataset shape: {X_clf.shape}")
   print(f"Classes: {np.unique(y_clf)}")

Train and Predict
-----------------

.. code-block:: python

   # Train LessGreedyHybridTree for classification
   clf_tree = LessGreedyHybridTree(
       task='classification',
       max_depth=6,
       min_samples_split=10,
       random_state=42
   )

   clf_tree.fit(X_train_clf, y_train_clf)

   # Make predictions
   y_pred_clf = clf_tree.predict(X_test_clf)
   clf_accuracy = accuracy_score(y_test_clf, y_pred_clf)

   print(f"Classification Accuracy: {clf_accuracy:.3f}")

Regression Example
------------------

Now let's try a regression task with the same tree, just changing the task parameter.

.. code-block:: python

   # Generate synthetic regression data
   X_reg, y_reg = make_regression(
       n_samples=1000,
       n_features=20,
       n_informative=10,
       noise=10,
       random_state=42
   )

   X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
       X_reg, y_reg, test_size=0.3, random_state=42
   )

   # Train LessGreedyHybridTree for regression
   reg_tree = LessGreedyHybridTree(
       task='regression',
       max_depth=6,
       min_samples_split=10,
       random_state=42
   )

   reg_tree.fit(X_train_reg, y_train_reg)

   # Make predictions
   y_pred_reg = reg_tree.predict(X_test_reg)
   reg_mse = mean_squared_error(y_test_reg, y_pred_reg)

   print(f"Regression MSE: {reg_mse:.2f}")

Comparing Different Tree Types
------------------------------

Let's compare the performance of different stable tree implementations on the same classification task.

.. code-block:: python

   # Define different tree models
   trees = {
       'LessGreedyHybridTree': LessGreedyHybridTree(
           task='classification',
           max_depth=6,
           random_state=42
       ),
       'BootstrapVariancePenalizedTree': BootstrapVariancePenalizedTree(
           task='classification',
           max_depth=6,
           random_state=42
       ),
       'RobustPrefixHonestTree': RobustPrefixHonestTree(
           task='classification',
           max_depth=6,
           random_state=42
       )
   }

   results = {}

   for name, tree in trees.items():
       # Train the tree
       tree.fit(X_train_clf, y_train_clf)
       
       # Make predictions
       y_pred = tree.predict(X_test_clf)
       accuracy = accuracy_score(y_test_clf, y_pred)
       
       results[name] = accuracy
       print(f"{name}: {accuracy:.3f}")

Key Takeaways
-------------

1. **Unified API**: All stable-cart trees use the same interface - just change the ``task`` parameter for classification vs regression.

2. **Multiple Algorithms**: Different tree types implement various stability primitives for enhanced prediction reliability.

3. **sklearn Compatible**: All trees follow scikit-learn conventions and can be used in pipelines, grid search, etc.

4. **Stability Focus**: These trees are designed to reduce prediction variance across different training runs.