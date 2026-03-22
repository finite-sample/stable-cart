Classification with Stable-CART Trees
=====================================

This example demonstrates how to use stable-cart trees for classification tasks, including multi-class problems and performance analysis.

Setup
-----

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.datasets import make_classification, load_wine, load_digits
   from sklearn.model_selection import train_test_split, cross_val_score
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

   from stable_cart import LessGreedyHybridTree, BootstrapVariancePenalizedTree, RobustPrefixHonestTree

Binary Classification Example
-----------------------------

.. code-block:: python

   # Generate binary classification data
   X_binary, y_binary = make_classification(
       n_samples=1000,
       n_features=10,
       n_informative=5,
       n_redundant=3,
       n_classes=2,
       n_clusters_per_class=1,
       flip_y=0.1,  # Add some noise
       random_state=42
   )

   X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
       X_binary, y_binary, test_size=0.3, random_state=42
   )

   print(f"Binary dataset shape: {X_binary.shape}")
   print(f"Class distribution: {np.bincount(y_binary)}")

   # Train different models on binary classification
   binary_models = {
       'Standard DecisionTree': DecisionTreeClassifier(
           max_depth=8,
           min_samples_split=10,
           random_state=42
       ),
       'LessGreedyHybridTree': LessGreedyHybridTree(
           task='classification',
           max_depth=8,
           min_samples_split=10,
           random_state=42
       ),
       'BootstrapVariancePenalizedTree': BootstrapVariancePenalizedTree(
           task='classification',
           max_depth=8,
           min_samples_split=10,
           random_state=42
       )
   }

   binary_results = {}

   for name, model in binary_models.items():
       # Train the model
       model.fit(X_train_bin, y_train_bin)
       
       # Make predictions
       y_pred = model.predict(X_test_bin)
       
       # Calculate accuracy
       accuracy = accuracy_score(y_test_bin, y_pred)
       
       binary_results[name] = {'accuracy': accuracy, 'predictions': y_pred}
       
       print(f"{name}: {accuracy:.3f}")

Multi-class Classification: Wine Dataset
-----------------------------------------

.. code-block:: python

   # Load wine dataset
   wine = load_wine()
   X_wine, y_wine = wine.data, wine.target

   print(f"Wine dataset shape: {X_wine.shape}")
   print(f"Number of classes: {len(wine.target_names)}")
   print(f"Class names: {wine.target_names}")
   print(f"Class distribution: {np.bincount(y_wine)}")

   # Split the data
   X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
       X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
   )

   # Train stable trees on wine dataset
   wine_models = {
       'LessGreedyHybridTree': LessGreedyHybridTree(
           task='classification',
           max_depth=10,
           min_samples_split=5,
           random_state=42
       ),
       'BootstrapVariancePenalizedTree': BootstrapVariancePenalizedTree(
           task='classification',
           max_depth=10,
           min_samples_split=5,
           random_state=42
       ),
       'RobustPrefixHonestTree': RobustPrefixHonestTree(
           task='classification',
           max_depth=10,
           min_samples_split=5,
           random_state=42
       )
   }

   wine_results = {}

   for name, model in wine_models.items():
       # Train the model
       model.fit(X_train_wine, y_train_wine)
       
       # Make predictions
       y_pred_wine = model.predict(X_test_wine)
       
       # Calculate accuracy
       accuracy = accuracy_score(y_test_wine, y_pred_wine)
       
       wine_results[name] = {'accuracy': accuracy, 'predictions': y_pred_wine}
       
       print(f"{name}:")
       print(f"  Accuracy: {accuracy:.3f}")
       print()

   # Show detailed results for the best model
   best_model_name = max(wine_results.keys(), key=lambda k: wine_results[k]['accuracy'])
   best_predictions = wine_results[best_model_name]['predictions']

   print(f"\\nDetailed results for {best_model_name}:")
   print(classification_report(y_test_wine, best_predictions, target_names=wine.target_names))

Cross-Validation Analysis
-------------------------

.. code-block:: python

   # Perform cross-validation
   cv_results = {}

   for name, model in wine_models.items():
       # 5-fold cross-validation
       cv_scores = cross_val_score(model, X_wine, y_wine, cv=5, scoring='accuracy')
       
       cv_results[name] = {
           'mean': cv_scores.mean(),
           'std': cv_scores.std(),
           'scores': cv_scores
       }
       
       print(f"{name}:")
       print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
       print(f"  Individual scores: {cv_scores}")
       print()

High-Dimensional Classification: Digits Dataset
-----------------------------------------------

.. code-block:: python

   # Load digits dataset (subset for faster execution)
   digits = load_digits()
   X_digits, y_digits = digits.data, digits.target

   # Use a subset for faster computation
   subset_size = 500
   indices = np.random.choice(len(X_digits), subset_size, replace=False)
   X_digits_subset = X_digits[indices]
   y_digits_subset = y_digits[indices]

   print(f"Digits dataset shape: {X_digits_subset.shape}")
   print(f"Number of classes: {len(np.unique(y_digits_subset))}")
   print(f"Classes: {np.unique(y_digits_subset)}")

   X_train_dig, X_test_dig, y_train_dig, y_test_dig = train_test_split(
       X_digits_subset, y_digits_subset, test_size=0.3, random_state=42
   )

   # Train stable tree on digits
   digits_tree = LessGreedyHybridTree(
       task='classification',
       max_depth=15,
       min_samples_split=5,
       random_state=42
   )

   digits_tree.fit(X_train_dig, y_train_dig)
   y_pred_dig = digits_tree.predict(X_test_dig)

   digits_accuracy = accuracy_score(y_test_dig, y_pred_dig)
   print(f"Digits Classification Accuracy: {digits_accuracy:.3f}")

   # Show classification report
   print("\\nClassification Report:")
   print(classification_report(y_test_dig, y_pred_dig))

Summary
-------

This example demonstrated stable-cart trees for classification:

1. **Binary Classification**: Simple two-class problems with good performance

2. **Multi-class Classification**: Real-world wine dataset with three classes

3. **Cross-validation**: Robust performance evaluation across multiple folds

4. **High-dimensional Data**: Digits dataset with 64 features and 10 classes

**Key Benefits:**

- **Unified API**: Same interface for binary and multi-class classification
- **Competitive Performance**: Often matches or exceeds standard decision trees
- **Enhanced Stability**: More consistent predictions across training runs
- **sklearn Integration**: Works seamlessly with scikit-learn workflows

The stable-cart trees provide a reliable alternative to standard decision trees with enhanced prediction stability, making them ideal for production environments where consistent model behavior is important.