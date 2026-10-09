# Repair plan for the removed tree candidates

Plan frozen before comparative outcomes were inspected: 2026-08-17.

## Why the earlier verdict changes

The deleted implementations did not perform their advertised split selection.
That invalidates the code and every result produced by it. It does not, by
itself, invalidate the underlying algorithms. The two ideas below therefore
return to the research queue. They remain outside the shipped package until
their implementation and comparative evidence clear the gates in
`VALIDATION_STANDARD.md`.

## Bootstrap variance-penalized split selection

At a node, enumerate the same admissible axis-aligned candidates used by the
unpenalized research tree. Let `g(s; D)` be the impurity reduction of split `s`
on node data `D`, divided by the parent impurity. For every candidate, draw `B`
row-bootstrap samples of the node and calculate

`score(s) = g(s; D) - lambda * Var_b[g(s; D_b)]`.

Choose the candidate with the largest positive score. Ties are resolved by
feature index and threshold. The bootstrap samples are shared across candidates
at a node. This is uncertainty in split quality, not prediction variance.

Correctness identities:

1. `lambda=0` returns the unpenalized greedy split on the identical candidate
   set.
2. Rescaling a regression target by a nonzero constant does not change the
   structure.
3. A slow independent root calculation reproduces the selected split and its
   score.
4. The same seed is bit-identical; row order does not change the fitted tree
   when covariate rows are distinct.
5. Injecting a high-variance rival can make a positive penalty choose a
   different candidate; otherwise the mechanism is not operational.

## Robust-prefix honest split selection

Partition rows independently of their outcomes into a structure sample and an
estimation sample. At nodes above `prefix_levels`, enumerate candidates once on
the structure rows. For each of `B` bootstrap resamples of those rows, give one
vote to the candidate with the greatest relative impurity gain. Select the
candidate with greatest support, breaking ties by full-sample gain, only if its
support is at least `consensus_threshold`. If no candidate clears the threshold,
stop. At and below `prefix_levels`, use ordinary greedy selection on the
structure sample. Estimate terminal predictions only from estimation-sample
outcomes; an empty honest leaf uses the global estimation-sample prior.

Correctness identities:

1. Every bootstrap replicate contributes exactly one vote, so support is in
   `[0, 1]` and supports sum to one.
2. A slow independent root vote calculation reproduces the selected split.
3. `prefix_levels=0` removes consensus selection while retaining honesty.
4. Changing estimation-sample outcomes cannot change the learned structure.
5. Multiclass probabilities align with `classes_` and sum to one.
6. The same seed is bit-identical; row order does not change the fitted tree
   when covariate rows are distinct. Exact duplicate covariate rows have no
   outcome-independent identity with which to assign them individually to an
   honest subsample.

## Comparative validation

Only after the identities pass, compare each repaired method with tuned,
cost-complexity-pruned CART using identical train, validation, test, and outer
bootstrap draws. Also include the following diagnostic ablations:

- unpenalized research tree for the variance method;
- honest greedy tree (`prefix_levels=0`) for the prefix method.

The primary outcome is all-pairs test prediction instability, with labels for
classification and predictions normalized by the test-target standard deviation
for regression. Report predictive score beside instability. Hyperparameters are
chosen using training and validation data only. The unit of uncertainty is an
independently generated dataset.

An estimator clears the comparative package gate only if its familywise 98.33%
interval shows lower instability and its mean score loss is no more than 0.5
percentage points of accuracy or 1% of the CART test-target variance in MSE.
Classification and regression are separate claims. A general claim requires
both. Runtime is reported and never omitted from the verdict.

## Prior-art boundary

Bootstrap tree-stability assessment and honest sample splitting are established
ideas. Athey and Imbens (2016) supply the direct precedent for honest leaf
estimation, although for causal trees. Bootstrap frequencies of split variables
and cut points are also established diagnostics. Until a closer search and
independent review say otherwise, neither repaired combination is claimed as
new.
