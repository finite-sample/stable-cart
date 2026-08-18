.. _operating_point:

Audit and compare fitted procedures
===================================

An instability audit has one unit of analysis: the complete procedure that turns
training data into predictions. A useful audit repeats every data-dependent step
that would change if the sample changed.

Define the procedure
--------------------

Pass a zero-argument factory that returns a fresh estimator or pipeline.

.. code-block:: python

   from sklearn.linear_model import RidgeCV
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler

   procedure = lambda: make_pipeline(StandardScaler(), RidgeCV())

If feature selection or hyperparameter tuning is part of model development, put
it inside the returned procedure. Keeping a tuned parameter fixed asks the
narrower question of how estimation varies after tuning.

The audit's ``random_state`` controls resampling, not randomness inside the
estimator. Set estimator seeds in the factory when the target is sampling
variation alone or when exact reproducibility matters. Leave them random only
when algorithmic randomness is intentionally part of the procedure being
audited.

Measure the fitted procedure
----------------------------

.. code-block:: python

   from stable_cart import bootstrap_instability

   result = bootstrap_instability(
       procedure,
       X_train,
       y_train,
       X_evaluation,
       task="continuous",
       n_bootstrap=500,
       random_state=0,
   )

   print(result["mape"], result["mape_standard_error"])
   print(result["pairwise_mean"], result["pairwise_standard_error"])

MAPE compares bootstrap refits with the original fitted model. Pairwise
instability compares two independently refitted models. The Monte Carlo standard
errors measure uncertainty from using a finite number of bootstrap refits. They
do not measure uncertainty across possible original datasets.

For categorical outcomes, ``prediction_method="predict"`` measures class-label
disagreement. ``prediction_method="predict_proba"`` measures movement in the
full aligned probability vector. Class labels are names, so their numerical
spacing never enters either calculation.

Inspect individual predictions
------------------------------

.. code-block:: python

   from stable_cart import bootstrap_predictions, plot_mape_by_prediction

   raw = bootstrap_predictions(
       procedure,
       X_train,
       y_train,
       X_evaluation,
       n_bootstrap=500,
       random_state=0,
   )

   plot_mape_by_prediction(raw)

The raw result keeps one prediction per bootstrap refit and evaluation case.
For multiclass probabilities it keeps one additional class dimension. This is
the evidence behind every summary and lets users compute a domain-specific
quantity without refitting the models.

Construct a validation frontier
-------------------------------

.. code-block:: python

   from sklearn.tree import DecisionTreeRegressor
   from stable_cart import stability_frontier

   result = stability_frontier(
       lambda **params: DecisionTreeRegressor(random_state=0, **params),
       {"max_depth": [2, 4, 8], "ccp_alpha": [0.0, 0.001, 0.01]},
       X_train,
       y_train,
       X_eval=X_validation,
       y_eval=y_validation,
       n_bootstrap=200,
       random_state=0,
   )

   for point in result["frontier"]:
       print(point["score"], point["instability"], point["params"])

The function returns all configurations and the nondominated subset. The same
bootstrap seed is used for each configuration, which makes the comparison
paired. The returned score remains a validation score because the frontier was
selected with it. Report final performance on an untouched test set or through
an outer resampling loop.

Avoid false wins
----------------

A procedure that ignores its training data has zero instability. A useful audit
therefore reports predictive performance beside instability and applies any
minimum acceptable performance before interpreting the frontier.

Twenty bootstrap refits can expose a large problem, but they cannot support a
precise estimate. Increase the number of refits until the returned Monte Carlo
standard error is small relative to the differences that would change a
decision.
