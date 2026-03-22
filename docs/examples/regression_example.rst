Regression with Stable-CART Trees
==================================

This example demonstrates how to use stable-cart trees for regression tasks, including comparison with standard scikit-learn decision trees.

Setup and Data Generation
--------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.datasets import make_regression, load_diabetes
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.metrics import mean_squared_error, r2_score

   from stable_cart import LessGreedyHybridTree, BootstrapVariancePenalizedTree

   # Generate synthetic regression data
   X, y = make_regression(
       n_samples=500,
       n_features=10,
       n_informative=5,
       noise=15,
       random_state=42
   )

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
   )

   print(f"Dataset shape: {X.shape}")
   print(f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")

Model Training and Comparison
-----------------------------

.. code-block:: python

   # Define models
   models = {
       'Standard DecisionTree': DecisionTreeRegressor(
           max_depth=8,
           min_samples_split=10,
           random_state=42
       ),
       'LessGreedyHybridTree': LessGreedyHybridTree(
           task='regression',
           max_depth=8,
           min_samples_split=10,
           random_state=42
       ),
       'BootstrapVariancePenalizedTree': BootstrapVariancePenalizedTree(
           task='regression',
           max_depth=8,
           min_samples_split=10,
           random_state=42
       )
   }

   results = {}

   for name, model in models.items():
       # Train the model
       model.fit(X_train, y_train)
       
       # Make predictions
       y_pred = model.predict(X_test)
       
       # Calculate metrics
       mse = mean_squared_error(y_test, y_pred)
       r2 = r2_score(y_test, y_pred)
       
       results[name] = {'mse': mse, 'r2': r2, 'predictions': y_pred}
       
       print(f"{name}:")
       print(f"  MSE: {mse:.2f}")
       print(f"  R²:  {r2:.3f}")
       print()

Real-World Example: Diabetes Dataset
------------------------------------

.. code-block:: python

   # Load diabetes dataset
   diabetes = load_diabetes()
   X_diabetes, y_diabetes = diabetes.data, diabetes.target

   print(f"Diabetes dataset shape: {X_diabetes.shape}")
   print(f"Feature names: {diabetes.feature_names}")
   print(f"Target range: [{y_diabetes.min():.1f}, {y_diabetes.max():.1f}]")

   X_train_db, X_test_db, y_train_db, y_test_db = train_test_split(
       X_diabetes, y_diabetes, test_size=0.3, random_state=42
   )

   # Train stable tree on diabetes data
   stable_tree = LessGreedyHybridTree(
       task='regression',
       max_depth=6,
       min_samples_split=5,
       random_state=42
   )

   stable_tree.fit(X_train_db, y_train_db)
   y_pred_db = stable_tree.predict(X_test_db)

   # Calculate performance
   mse_db = mean_squared_error(y_test_db, y_pred_db)
   r2_db = r2_score(y_test_db, y_pred_db)

   print(f"Diabetes Dataset Results:")
   print(f"MSE: {mse_db:.2f}")
   print(f"R²: {r2_db:.3f}")

Summary
-------

This example demonstrated:

1. **Easy Task Switching**: The same tree class can handle regression by setting ``task='regression'``

2. **Competitive Performance**: Stable-cart trees often match or exceed standard decision tree performance

3. **Real-World Application**: The trees work well on real datasets like the diabetes progression dataset

4. **Stability Focus**: These trees are designed to provide more consistent predictions across different training runs

The key advantage of stable-cart trees is their enhanced prediction stability, which becomes especially important in production environments where consistent model behavior is crucial.