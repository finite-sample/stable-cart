.. _operating_point:

Choosing your operating point
=============================

There is no most stable model worth having. A predictor that ignores its
training data has zero instability and no use, so stability is only ever a
question asked jointly with accuracy — and the answer is a *curve*, not a
number. This page is about reading that curve on your own data and picking a
point on it.

Step 1: find out where you are
------------------------------

Before changing anything, measure the model you already have.

.. code-block:: python

   from stable_cart import bootstrap_instability

   bootstrap_instability(
       lambda: DecisionTreeRegressor(max_depth=8, min_samples_leaf=10),
       X_train, y_train, X_test,
       n_bootstrap=50,
   )
   # {'instability_mean': 0.103, 'instability_p90': 0.238,
   #  'instability_max': 1.255, 'mape': 0.275}

Read ``mape`` in the units of your target. On California housing the target is
in hundreds of thousands of dollars, so 0.275 means a household's predicted
value moves by about $27,500 depending on which sample the model happened to be
fitted on. Read ``instability_p90`` next: it is the tenth of cases the mean is
hiding.

The factory takes no arguments and returns an *unfitted* estimator. That is the
whole model-building step being repeated, which is the point of the protocol —
if you tune hyperparameters inside it, tuning is part of what gets resampled,
which is more honest and more expensive.

Step 2: find out where you *could* be
-------------------------------------

.. code-block:: python

   from stable_cart import stability_frontier

   result = stability_frontier(
       lambda **kw: DecisionTreeRegressor(min_samples_leaf=10, random_state=0, **kw),
       {"max_depth": [3, 5, 8], "ccp_alpha": [0.0, 0.005, 0.05]},
       X_train, y_train, n_bootstrap=20, random_state=0,
   )

   for point in result["frontier"]:
       print(point["accuracy"], point["instability"], point["params"])

::

   R2=0.651  instability=0.107  {'ccp_alpha': 0.0,   'max_depth': 8}
   R2=0.594  instability=0.079  {'ccp_alpha': 0.0,   'max_depth': 5}
   R2=0.592  instability=0.078  {'ccp_alpha': 0.005, 'max_depth': 8}
   R2=0.583  instability=0.074  {'ccp_alpha': 0.005, 'max_depth': 5}
   R2=0.526  instability=0.057  {'ccp_alpha': 0.005, 'max_depth': 3}
   R2=0.483  instability=0.053  {'ccp_alpha': 0.05,  'max_depth': 8}

That is the exchange rate: halving the instability costs 12.5 points of R².
Everything not in ``frontier`` is in ``points`` and is strictly worse on one axis
or both — there is never a reason to choose it.

The comparison is *paired*: one set of bootstrap index sets is drawn and reused
for every configuration, so two configurations are never separated by which
resamples they happened to get.

Step 3: compare families, not just settings
-------------------------------------------

``stability_frontier`` is model-agnostic on purpose. Put pruning and
:class:`~stable_cart.StableTree` on the same axes and let the data decide.

.. code-block:: python

   from stable_cart import StableTree, plot_stability_frontier

   frontiers = {
       "CART (pruned)": stability_frontier(
           lambda **kw: DecisionTreeRegressor(min_samples_leaf=10, random_state=0, **kw),
           {"max_depth": [3, 4, 5, 6, 8], "ccp_alpha": [0.0, 0.001, 0.005, 0.02, 0.05]},
           X_train, y_train, n_bootstrap=20, random_state=0,
       ),
       "StableTree": stability_frontier(
           lambda **kw: StableTree(task="regression", min_samples_leaf=10,
                                   n_consensus=12, random_state=0, **kw),
           {"max_depth": [3, 4, 5, 6, 8],
            "consensus_threshold": [0.0, 0.3, 0.6],
            "leaf_shrinkage": [0.0, 2.0, 10.0]},
           X_train, y_train, n_bootstrap=20, random_state=0,
       ),
   }
   plot_stability_frontier(frontiers)

Give both families the same complexity knob. Fixing one family's depth while
sweeping the other's makes the fixed one look worse for a reason nobody can act
on.

Which knob to turn, and why it depends on your data
---------------------------------------------------

A tree's prediction variance has two sources: **which splits it chose**, and
**what value it put in each leaf**. Measured on this package's synthetic
regimes, the leaf component is 2–11% of the total when the signal is nearly
noiseless and 40–90% when noise is high.

That decides the knob:

- **Noisy data** → the leaves are the problem. Reach for ``leaf_shrinkage``
  (:class:`~stable_cart.StableTree`) or plain pruning, which puts more rows in
  each leaf. Averaging the split decision cannot help much, because the
  structure was not what was moving.
- **Clean, low-noise data** → the structure is the problem. Reach for
  ``consensus_threshold`` and ``n_consensus``, which average the split decision
  across resamples.

``experiments/knob_study.py`` measures this directly on a data-generating
process you can vary.

What the numbers cannot tell you
--------------------------------

Whether the trade is worth taking. A clinical model that has to be defended to
a regulator and a churn model retrained nightly sit at very different points on
the same curve, and no statistic in this package distinguishes them. The tool's
job is to make the curve visible and refuse to pick for you.

Two things to watch:

- **The stable-and-useless corner.** A Pareto frontier always contains it,
  because nothing is more stable than a constant. Check that every point you are
  considering clears a real accuracy floor first.
- **Instability is not error.** A model can be perfectly stable and perfectly
  wrong. These measurements say nothing about bias, and are not a substitute for
  a held-out score.
