API reference
=============

This project packages a couple of estimators that follow the familiar
scikit-learn API. The implementation lives in the ``stable-cart``
directory. The sections below summarise the main entry points and their
responsibilities.

LessGreedyHybridRegressor
-------------------------

:Location: ``stable-cart/less_greedy_tree.py``

A drop-in replacement for tree-based regressors that trades a little raw
accuracy for greatly improved stability. Key features include:

* Honest data partitioning into ``SPLIT``, ``VAL`` and ``EST`` subsets.
* An optional oblique (linear) root split backed by Lasso-based feature
  projections.
* Honest lookahead with beam search when multiple candidate splits look
  equally attractive.
* Leaf-value shrinkage to limit overfitting and reduce variance.

BootstrapVariancePenalizedRegressor
-----------------------------------

:Location: ``stable-cart/bootstrap_variance_tree.py``

A tree regressor that explicitly penalizes bootstrap prediction variance
during split selection. This approach encourages the tree to make splits
that lead to more stable predictions across different bootstrap samples
of the training data. Key features include:

* Bootstrap variance estimation on validation set during split evaluation.
* Configurable variance penalty weight to balance accuracy vs. stability.
* Fast bootstrap trees for efficient variance computation.
* Full compatibility with sklearn pipelines and cross-validation.


Utility helpers
---------------

:Location:
   ``stable-cart/less_greedy_tree.py``
   ``stable-cart/evalutation.py``

The implementation also ships with a collection of helper routines that
handle variance calculations, evaluation utilities and benchmarking
support used in the accompanying notebooks.
