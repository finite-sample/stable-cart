# Frozen screen for the repaired tree candidates

Frozen before comparative outcomes were inspected: 2026-08-17.

This screen decides whether either repaired idea warrants a full confirmatory
study. It cannot put an estimator in the package and cannot support a paper
claim.

## Dataset collection

Use six independently generated datasets from each of six processes, for 36
independent units:

- binary logistic classification with one dominant feature;
- noisy XOR classification;
- three-class softmax classification;
- one-step regression;
- redundant-feature regression;
- Friedman-1 regression.

Each unit has 400 development rows and 600 untouched test rows. Split the
development data 2:1 into training and validation data. Seeds are written to the
result artifact. A failed fit aborts the screen; it is never silently dropped.

## Arms and tuning budgets

Primary comparator: cost-complexity-pruned CART. No forest arm.

Each primary arm receives eight configurations and is evaluated on the same six
inner bootstrap samples:

- CART: depth in `{3, 5}`, minimum leaf size in `{5, 15}`, and pruning alpha at
  zero or the median positive alpha on the training pruning path;
- bootstrap variance penalty: depth in `{3, 5}`, minimum leaf size in `{5, 15}`,
  and penalty in `{1, 8}`;
- robust honest prefix: depth in `{3, 5}`, minimum leaf size in `{5, 15}`, and
  consensus threshold in `{0, 0.2}`, with one consensus level.

The repaired arms use 16 bootstrap samples per node and at most 16 candidate
thresholds per feature. All trees use `min_samples_split=30`. Within each arm,
choose the lowest-instability configuration whose validation score is no worse
than 0.01 accuracy or 0.02 R-squared below the best CART configuration. If no
configuration qualifies, choose the arm's highest-score configuration and mark
it ineligible.

Refit each selected configuration on eight paired outer row-bootstrap samples of
the full development set and evaluate once on the untouched test set.

## Outcomes

Primary instability is mean all-pairs label disagreement for classification and
mean all-pairs squared prediction difference divided by test-target variance for
regression. Report mean accuracy or R-squared beside it. Dataset effects are the
repaired-arm value minus the CART value, plus the symmetric percent difference.
The independent dataset is the unit of uncertainty.

For mechanism diagnosis, evaluate two untuned test-stage ablations using the
selected repaired configuration:

- set the variance penalty to zero;
- set prefix levels to zero while retaining honest leaf estimation.

These isolate the penalty and consensus prefix respectively. They are not
additional comparators and do not receive their own tuning searches.

## Screen decision

Advance a task-method pair to a fresh-seed confirmatory study only if all four
conditions hold in this fixed screen:

1. mean normalized instability difference from CART is negative;
2. at least two of the three process means are negative;
3. mean score loss is within the declared validation tolerance;
4. the named mechanism is no worse than its ablation on mean instability.

The screen is deliberately permissive: intervals may be wide with six datasets
per process. Failure kills the present specification. Passing only licenses a
larger frozen study; it is not evidence for package inclusion.
