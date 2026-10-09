# Persistent-trunk falsification plan

Status: revised and frozen before the first run on 2026-08-17. Version 1 had an
independent pre-outcome audit before any generator or outcome code existed. The
audit correctly required exact data-generating equations, a gate sensitive to
the frozen threshold as well as its feature, a zero-instability rule, and a
precision justification. This version incorporates those changes. It remains a
repository research screen, not a package inclusion test or a paper result.

## Question

When the upper decision rule is expected to persist across scheduled retrains,
does freezing that trunk while relearning lower subtrees provide a useful
score-instability operating point? The method is allowed to be conditional on
an observable trunk-stability diagnostic. It is not allowed to claim a general
benefit under structural drift.

The independent unit is one generated reference/update problem. Results are
aggregated with equal weight across units and processes, never across test rows
or model-refit pairs as if those were independent datasets.

## Workflows

All trees use CART with maximum total depth 4, `min_samples_leaf=10`, Gini or
squared-error impurity, and `ccp_alpha=0`. This is a bounded, unpruned CART
comparison rather than a claim against every tuned CART workflow.

1. `refit_cart`: relearn the complete tree on every update sample.
2. `frozen_tree`: fit one depth-4 tree on the reference sample and never update
   it.
3. `refreshed_leaves`: preserve every split in the reference depth-4 tree and
   update only the prediction in each terminal leaf.
4. `persistent_trunk`: fit a depth-1 trunk on the reference sample, preserve
   its feature and threshold exactly, and relearn a depth-3 CART subtree within
   each trunk leaf on every update sample.

The fourth workflow is the candidate. The first is its unconstrained baseline.
The middle two determine whether the candidate adds anything beyond refusing
to change or refreshing constants inside a completely fixed partition.

## Data-generating regimes

The screen crosses three outcomes—binary classification, three-class
classification, and regression—with two regimes and 24 independently seeded
units per cell (144 units total).

- `stable`: the reference, validation-update, final-update, and test populations
  share the same dominant upper split on feature 0.
- `moving`: the reference population uses feature 0, while all later update and
  test populations use feature 3 as the dominant upper split. Descendant signal
  remains available in both regimes. This is an abrupt structural break, not
  ordinary sampling noise.

Every feature vector has eight independent standard-normal coordinates. Define
`s(j) = 2 * 1[x_j > 0] - 1` and
`d(j) = 1[x_j > 0] * s(1) + 1[x_j <= 0] * s(2)`. The reference population uses
`j=0`; the stable update population uses `j=0`; and the moving update population
uses `j=3`.

- Binary classification draws `y ~ Bernoulli(logit^-1(2.5*s(j) + 1.25*d(j)))`.
- Three-class classification draws from softmax logits
  `(-2.4*s(j) + 1.1*s(1), 2.4*s(j) + 1.1*s(2), 1.3*s(4))`.
- Regression uses `y = 4*s(j) + 2*d(j) + epsilon`, where
  `epsilon ~ Normal(0, 1.5^2)`.

Each unit contains 600 reference observations, eight validation-update samples
of 400 observations, one unlabeled gate sample of 1,000 update-population
feature vectors, 16 final-update samples of 400 observations, and an untouched
test set of 1,000 observations from the update population. The same update
samples and test cases are used by every workflow. Seeds are derived by taking
the first four bytes of SHA-256 over the JSON encoding of
`(20260817, process, regime, unit, stage, replicate)`; units are numbered 0
through 23. No generated outcome was viewed when these values were chosen.

## Outcomes

The primary instability outcome is all-pairs prediction distance across the 16
final updates: label disagreement for classification and mean squared
prediction difference divided by test-outcome variance for regression. Score is
test accuracy or R-squared, averaged across the same updates. Classification
also reports aligned probability-vector squared distance and log loss as
secondary diagnostics, so stable hard labels cannot hide large probability
movement.

For each unit, candidate-versus-CART effects are the symmetric instability
difference `2 * (candidate - CART) / (candidate + CART)` and the raw score
difference. Score tolerances are 0.01 accuracy and 0.02 R-squared. A unit is
`useful` when candidate instability does not exceed CART and its score is no
more than the tolerance below CART.

If both instability values are at most `1e-12`, the symmetric effect is zero.
If only their sum is at most `1e-12`, execution fails rather than returning an
unstable ratio.

Intervals resample independent units within each outcome-by-regime cell and
then average cells with equal weight. There are 20,000 deterministic stratified
resamples. The report also gives process-level results and every unit-level row.

## Observable applicability rule

Before final updates are evaluated, fit depth-4 CART on each validation-update
sample. For each fitted root, compare its binary routing of the unlabeled gate
sample with the frozen reference root, taking the larger agreement after
allowing left/right label reversal. The gate accepts persistence when at least
six of eight roots have routing agreement of at least 0.90. This tests the
partition induced by both feature and threshold and uses no final-update
predictions or test outcomes.

The gate is considered informative only if it accepts at least 20 of 24 stable
units and rejects at least 20 of 24 moving units, separately in every outcome
cell, and each corresponding two-sided 95% Wilson lower bound exceeds 0.65.
Conditional performance is reported for accepted units, but the known regime
label is never an input to the gate.

## Advance rule

The candidate advances to a new, independently seeded confirmation only if all
of the following hold in this screen:

1. In the stable regime, the upper 95% interval bound for the equally weighted
   candidate-versus-CART instability effect is below zero.
2. In the stable regime, the lower 95% interval bound for score difference is
   above the negative task-specific tolerance in every outcome cell.
3. At least 54 of 72 stable units are useful.
4. In the stable regime, candidate mean score exceeds both `frozen_tree` and
   `refreshed_leaves` in every outcome cell. This guards against a fake stability
   win obtained merely by refusing useful updates.
5. The validation routing-agreement gate clears its cell-specific count and
   Wilson-bound requirements.

Moving-regime score harm, instability, and gate errors are reported regardless
of the verdict. Passing this screen would support only a conditional candidate,
not shipment. Failing any rule stops work on this specification unless a new
mechanism—not a tuned threshold on these outcomes—is proposed.

With 24 independent units, a paired cell comparison has about 80% power at a
two-sided 5% level for a standardized mean effect of roughly 0.57 under a
Gaussian approximation; the equal-weight stable aggregate over 72 units can
resolve roughly 0.33 standard deviations. Smaller effects are treated as
unresolved, not absent. Interval coverage is checked in tests against simulated
Gaussian paired effects before the results are interpreted.

## Correctness checks required before interpretation

- The learned trunk feature and threshold are bit-for-bit unchanged by every
  update.
- Routing and child-subtree predictions match an independent manual
  reconstruction.
- The refreshed-leaf baseline preserves all reference splits and independently
  reconstructed leaf means or class modes.
- Binary, multiclass, and regression updates work; unseen update leaves have a
  documented global fallback.
- Classification probability columns are aligned to the reference class order
  even when an update-side subtree does not observe every class.
- Row order and label renaming do not change the induced partitions.
- Identical update data make persistent-trunk predictions deterministic.
- All committed summaries reconstruct exactly from unit-level rows and record
  this plan's SHA-256 digest and the complete settings.

## Interpretation boundary

This synthetic screen can show that the mechanism behaves as intended under a
known stable trunk and that an observable diagnostic distinguishes one stark
break. It cannot establish prevalence in real deployments, robustness to
gradual drift, or a general guarantee. No estimator is added to `stable_cart`
from this run.
