Stability Analysis with Stable-CART
===================================

This example demonstrates how to analyze prediction stability using the evaluation functions provided in stable-cart.

Understanding Prediction Stability
----------------------------------

Prediction stability measures how consistent a model's predictions are across different training runs on the same out-of-sample data. This is crucial for production environments where consistent model behavior is important.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_classification, make_regression
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
   from sklearn.ensemble import RandomForestClassifier

   from stable_cart import (
       LessGreedyHybridTree, 
       BootstrapVariancePenalizedTree,
       prediction_stability,
       evaluate_models
   )

   # Generate test data
   X_test, y_test = make_classification(
       n_samples=200,
       n_features=10,
       n_informative=5,
       n_classes=2,
       random_state=42
   )

   print(f"Test data shape: {X_test.shape}")
   print(f"Class distribution: {np.bincount(y_test)}")

Helper Functions
----------------

.. code-block:: python

   # Function to train multiple models with different seeds
   def train_multiple_models(model_class, X_train_data, y_train_data, n_models=10, **model_kwargs):
       """Train multiple models with different random seeds."""
       models = []
       
       for seed in range(n_models):
           if 'random_state' in model_kwargs:
               model_kwargs['random_state'] = seed
           model = model_class(**model_kwargs)
           model.fit(X_train_data, y_train_data)
           models.append(model)
       
       return models

   def get_predictions_from_models(models, X_test):
       """Get predictions from a list of models."""
       predictions = []
       for model in models:
           pred = model.predict(X_test)
           predictions.append(pred)
       return predictions

Stability Comparison: Standard vs Stable Trees
----------------------------------------------

.. code-block:: python

   # Generate training data with some noise
   X_train, y_train = make_classification(
       n_samples=500,
       n_features=10,
       n_informative=5,
       n_classes=2,
       flip_y=0.1,  # Add noise
       random_state=123  # Different from test data
   )

   # Define models to compare
   model_configs = {
       'Standard DecisionTree': {
           'class': DecisionTreeClassifier,
           'kwargs': {'max_depth': 8, 'min_samples_split': 10, 'random_state': 42}
       },
       'LessGreedyHybridTree': {
           'class': LessGreedyHybridTree,
           'kwargs': {'task': 'classification', 'max_depth': 8, 'min_samples_split': 10, 'random_state': 42}
       },
       'BootstrapVariancePenalizedTree': {
           'class': BootstrapVariancePenalizedTree,
           'kwargs': {'task': 'classification', 'max_depth': 8, 'min_samples_split': 10, 'random_state': 42}
       }
   }

   n_models = 20
   stability_results = {}

   for name, config in model_configs.items():
       print(f"Training {n_models} {name} models...")
       
       # Train multiple models
       models = train_multiple_models(
           config['class'], X_train, y_train, n_models, **config['kwargs']
       )
       
       # Get predictions
       predictions = get_predictions_from_models(models, X_test)
       
       # Calculate stability
       stability_score = prediction_stability(predictions)
       stability_results[name] = {
           'stability': stability_score,
           'predictions': predictions
       }
       
       print(f"  Stability score: {stability_score:.4f}")
       print()

   print("\\nStability Comparison (Higher is Better):")
   for name, result in stability_results.items():
       print(f"{name}: {result['stability']:.4f}")

Model Evaluation with evaluate_models Function
----------------------------------------------

The ``evaluate_models`` function provides comprehensive performance metrics for multiple models.

.. code-block:: python

   # Train single best models for evaluation
   evaluation_models = {
       'Standard_DT': DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42),
       'Stable_LessGreedy': LessGreedyHybridTree(task='classification', max_depth=8, min_samples_split=10, random_state=42),
       'Stable_Bootstrap': BootstrapVariancePenalizedTree(task='classification', max_depth=8, min_samples_split=10, random_state=42),
       'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
   }

   # Train models
   trained_models = {}
   for name, model in evaluation_models.items():
       model.fit(X_train, y_train)
       trained_models[name] = model

   # Evaluate models
   model_predictions = {}
   for name, model in trained_models.items():
       pred = model.predict(X_test)
       model_predictions[name] = pred

   # Use evaluate_models function
   evaluation_results = evaluate_models(
       y_true=y_test,
       predictions_dict=model_predictions,
       task='classification'
   )

   print("Model Evaluation Results:")
   print("=" * 50)
   for model_name, metrics in evaluation_results.items():
       print(f"\\n{model_name}:")
       for metric_name, value in metrics.items():
           print(f"  {metric_name}: {value:.4f}")

Stability Analysis for Regression
---------------------------------

.. code-block:: python

   # Generate regression data
   X_reg_train, y_reg_train = make_regression(
       n_samples=400,
       n_features=8,
       noise=15,
       random_state=456
   )

   X_reg_test, y_reg_test = make_regression(
       n_samples=100,
       n_features=8,
       noise=15,
       random_state=789
   )

   # Regression models
   reg_model_configs = {
       'Standard DecisionTree': {
           'class': DecisionTreeRegressor,
           'kwargs': {'max_depth': 8, 'min_samples_split': 10, 'random_state': 42}
       },
       'LessGreedyHybridTree': {
           'class': LessGreedyHybridTree,
           'kwargs': {'task': 'regression', 'max_depth': 8, 'min_samples_split': 10, 'random_state': 42}
       }
   }

   reg_stability_results = {}
   n_reg_models = 15

   for name, config in reg_model_configs.items():
       print(f"Training {n_reg_models} {name} models for regression...")
       
       # Train multiple models
       models = train_multiple_models(
           config['class'], X_reg_train, y_reg_train, n_reg_models, **config['kwargs']
       )
       
       # Get predictions
       predictions = get_predictions_from_models(models, X_reg_test)
       
       # Calculate stability using prediction_stability
       stability_score = prediction_stability(predictions)
       reg_stability_results[name] = {
           'stability': stability_score,
           'predictions': predictions
       }
       
       # Calculate prediction variance
       pred_array = np.array(predictions)
       pred_variance = np.var(pred_array, axis=0)
       mean_variance = np.mean(pred_variance)
       
       print(f"  Stability score: {stability_score:.4f}")
       print(f"  Mean prediction variance: {mean_variance:.2f}")
       print()

   print("\\nRegression Stability Comparison:")
   for name, result in reg_stability_results.items():
       print(f"{name}: {result['stability']:.4f}")

Summary
-------

This example demonstrated comprehensive stability analysis using stable-cart:

**Key Functions:**

1. **``prediction_stability(predictions)``**: Measures how consistent predictions are across multiple models
   
   - Higher scores indicate more stable predictions
   - Works for both classification and regression

2. **``evaluate_models(y_true, predictions_dict, task)``**: Comprehensive model evaluation
   
   - Provides accuracy, precision, recall, F1 for classification
   - Provides MSE, MAE, R² for regression

**Key Insights:**

- **Stable-cart trees typically show higher prediction stability** than standard decision trees
- **Lower prediction variance** leads to more reliable model behavior in production
- **Stability vs Performance trade-off**: Stable models may sacrifice slight performance for consistency
- **Real-world relevance**: Stability is crucial for applications where consistent predictions matter

**When to Use Stable Trees:**

- Production environments requiring consistent model behavior
- Applications where prediction variance matters more than marginal accuracy gains
- Situations with noisy or limited training data
- When model interpretability and reliability are priorities