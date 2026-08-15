Welcome to stable-cart's documentation!
=======================================

**stable-cart** answers two questions about a single decision tree: how much do
its predictions move when the training data is resampled, and what would it cost
in accuracy to make them move less.

Why it matters
--------------

Fit a tree, resample the training data, fit it again. The second tree predicts
something different for the same individual — often very different. On
California housing a depth-8 tree moves a household's predicted value by
$27,500 on average, and by $125,000 at the extreme. Averaging predictions fixes
this and leaves you with a forest; if you have to ship one readable model, you
need something else.

Key features
------------

**Measurement first**: the resampling protocol of Riley and Collins (2023),
implemented for any scikit-learn estimator. The R package ``pminternal`` does
this; scikit-learn had no equivalent.

**The frontier, not a winner**: :func:`~stable_cart.stability_frontier` sweeps a
parameter grid and returns the configurations no other configuration beats on
both accuracy and stability — including plain ``DecisionTreeRegressor``, because
the honest answer is often that pruning wins.

**A tree whose splits are averaged**: :class:`~stable_cart.StableTree` averages
the split *decision* over bootstrap replicates rather than averaging
predictions, so the output is still one tree you can read.

Quick start
-----------

.. code-block:: python

   from sklearn.datasets import fetch_california_housing
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeRegressor

   from stable_cart import bootstrap_instability

   X, y = fetch_california_housing(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

   bootstrap_instability(
       lambda: DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=0),
       X_train, y_train, X_test,
       n_bootstrap=50, random_state=0,
   )
   # {'instability_mean': 0.102, 'instability_p90': 0.236,
   #  'instability_max': 1.234, 'mape': 0.275}

``mape`` is the headline: a model fitted on a resample predicts $27,500 away
from what the model fitted on all the data predicts for the same household.

Start at :ref:`measuring_stability`; :ref:`tree_estimators` lists the trees, and
:ref:`plots` the three figures.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Guides:

   operating_point

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Links:
   
   GitHub Repository <https://github.com/finite-sample/stable-cart>
   PyPI Package <https://pypi.org/project/stable-cart/>
