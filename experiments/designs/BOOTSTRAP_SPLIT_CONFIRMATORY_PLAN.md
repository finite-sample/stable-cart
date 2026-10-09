# Confirmatory validation plan: bootstrap-aggregated tree splits

Frozen on 2026-08-17 before running the experiments specified below.

This is a self-attested prospective plan, not a preregistration. The earlier
seven-dataset screen has been read. It selected configurations and evaluated
them on the same test split, so it is excluded from the confirmatory evidence.
Its mixed estimates informed the magnitude ranges and the decision to use
pruned CART as the only primary comparator.

## Question and estimand

Among single trees trained on independent row samples, does selecting split
features and thresholds through bootstrap votes reduce prediction instability
relative to validation-tuned, pruned CART without a material loss in untouched
test performance?

The uncertainty unit is an independently generated dataset. Each dataset has a
development sample and an untouched test sample. Configurations are chosen with
an internal development split. Repeated paired ordinary row-bootstrap fits on
the full development sample estimate conditional instability after
configuration selection. The same resampled row indices are used in both arms;
classification bootstraps do not fix class counts.

The primary dataset-level effect is

```text
100 * 2 * (instability_bootstrap_split - instability_cart)
        / (instability_bootstrap_split + instability_cart).
```

Negative values favor `BootstrapSplitTree`. If both instabilities are zero, the
effect is zero. The study reports raw instability and test-score differences as
well.

## Comparator and equal tuning budget

The comparator is scikit-learn CART. A random forest is not included because it
does not solve the single-tree deployment problem.

Both arms receive 12 distinct configurations and the same 16 paired validation
bootstrap samples.

CART crosses:

- `max_depth` in `{3, 5}`;
- `min_samples_leaf` in `{5, 10}`;
- `ccp_alpha` in `{0, median positive path alpha, 80th-percentile positive path
  alpha}`.

It uses `min_samples_split=20`, matching the bootstrap-split arm, so stricter
small-node stopping cannot masquerade as an effect of bootstrap voting.

The alpha values are derived separately from CART's cost-complexity path for
each `(max_depth, min_samples_leaf)` pair on the inner training sample. The two
positive values are distinct observed path alphas nearest the 50th and 80th
percentiles of the distinct positive path. A dataset fails closed if a pair
does not supply two distinct positive alphas; duplicate configurations do not
count toward the tuning budget.

`BootstrapSplitTree` crosses:

- `max_depth` in `{3, 5}`;
- `min_samples_leaf` in `{5, 10}`;
- `consensus_threshold` in `{0, 0.3, 0.5}`;

It uses `leaf_shrinkage=0`, `min_samples_split=20`, `n_consensus=16`, and
`max_candidates=40` throughout. Fixing shrinkage at zero makes the comparison
about bootstrap split aggregation and consensus stopping, rather than mixing
those mechanisms with terminal-prediction shrinkage.

## Configuration selection

The development sample contains 600 cases. A stratified 400/200 split is used
for classification and an ordinary 400/200 split for regression. Each
configuration is fitted on the same 16 bootstrap samples of the 400 cases
and evaluated on the same 200 validation cases.

The score floor is CART's best mean validation score minus 0.01 accuracy for
classification or 0.02 R2 for regression. Within each arm, the selected
configuration has the lowest validation instability among configurations that
clear that common floor. Ties favor higher score, then the lexical configuration
label. If no bootstrap-split configuration clears the floor, its highest-score
configuration is selected and the dataset is recorded as ineligible. It is not
dropped. CART is eligible by construction.

Validation instability is all-pairs class-label disagreement for classification
and all-pairs mean squared prediction distance for regression. Validation score
is mean accuracy or R2 across the 16 refits.

After selection, both chosen configurations are fitted on the same 20 bootstrap
samples of all 600 development cases. Their predictions and scores are measured
on 1,000 untouched test cases.

## Data-generating processes

There are 20 independently generated datasets in each of six synthetic cells.
All features not otherwise specified are independent standard normal variables.

1. `classification_unique`: 10 features and
   `Pr(y=1)=logit_inverse(3*x0)`;
2. `classification_redundant`: latent `z`, two observed proxies
   `x0=z+0.25*e0` and `x1=z+0.25*e1`, eight noise features, and
   `Pr(y=1)=logit_inverse(2*z)`;
3. `classification_interaction`: 10 features,
   `y=1(x0*x1>0)`, with 8% independently flipped labels;
4. `regression_unique_step`: 10 features and
   `y=4*1(x0>0)+Normal(0,1)`;
5. `regression_redundant`: the same two-proxy construction and
   `y=3*z+Normal(0,1)`;
6. `regression_friedman1`: the standard 10-feature Friedman-1 process with
   Gaussian noise 1.

The unique-signal cells are mechanism controls. A method intended to resolve
unstable split choice should not claim a large benefit where one split is
already clear. The redundant and interaction cells inject competing split
choices where bootstrap aggregation could help.

## Outcomes and uncertainty

Primary test instability is all-pairs label disagreement for classification and
all-pairs squared prediction distance for regression. Secondary outcomes are
binary probability-vector squared distance, root-feature disagreement, mean
leaf count, mean fit time, and the paired test-score difference.

Conditional Monte Carlo error uses ten disjoint pairs of final bootstrap fits.
It is diagnostic only. Within each cell, 95% percentile bootstrap intervals use
20,000 deterministic resamples of the 20 independent datasets. Grand and task
summaries give each declared process equal weight and resample datasets within
process. The general, classification, and regression claims use 98.33%
intervals (Bonferroni correction across all three claims). Raw differences are
retained alongside the bounded symmetric percentage effect. A practical-effect
guard uses the raw label-disagreement difference for classification and the raw
prediction-MSE difference divided by test-outcome variance for regression.

## Hypotheses and decision rules

The expected broad effect is a 5% to 15% reduction in instability. The expected
effect is 0% to 5% in the unique-signal controls and 10% to 25% in the redundant
and interaction cells. These ranges are deliberately falsifiable; the earlier
screen was mixed and does not justify a stronger prior.

Broad support requires all of the following:

1. the grand mean effect is at most -5% and its familywise 98.33% interval is
   below zero;
2. at least four of six cell means favor `BootstrapSplitTree`;
3. no cell mean exceeds a 10% instability increase;
4. at least 90% of datasets in every cell clear the CART-derived validation
   score floor;
5. the lower interval endpoint for the mean test-score difference is at least
   -0.01 in classification and -0.02 in regression;
6. within every cell, the lower 95% interval endpoint for the mean test-score
   difference is at least -0.01 for classification or -0.02 for regression;
7. the equally weighted normalized raw effect is at most -0.002 and its 98.33%
   interval is below zero; this requires at least a 0.2 percentage-point drop in
   classification disagreement or a regression drop equal to 0.2% of outcome
   variance on the common dimensionless scale;
8. no configuration fit fails.

Classification-specific and regression-specific support use the same effect,
eligibility, score, and failure rules within their three cells, and require at
least two of three cell means to favor `BootstrapSplitTree`. General and task
effect and task-average score checks use the familywise 98.33% intervals; cell
score checks remain at 95%.

This is a comparison of the configured procedures, not causal identification
of voting as the sole mechanism. The unique-signal cells test scope and
heterogeneity; they are not exclusion restrictions on the package decision.

If broad support passes, the estimator earns a general experimental-package
recommendation within the tested scope. If only one task passes, it earns a
task-specific recommendation. If neither task passes, the package recommendation
is to remove `BootstrapSplitTree` rather than retain an indefinitely exploratory
method. A score-instability tradeoff that misses the guardrail is reported as a
failure of the claimed free improvement.

## External descriptive replication

The frozen procedure is repeated on 16 prespecified 70/30 splits of breast
cancer, binary digits, and diabetes. These splits overlap, so their intervals
are descriptive and cannot replace the independent synthetic datasets. They do
not change the synthetic decision.

## Falsification and audit checks

- An A/A comparison of a prediction array with itself must return zero effect
  and zero paired delta.
- The fast all-pairs calculation must match a slow unordered-pair loop.
- Renaming binary classes must not change label disagreement.
- Selected configurations must be recoverable from the stored validation rows
  without test outcomes.
- An ineligible bootstrap-split arm must remain in the final rows.
- JSON and CSV summaries must agree, and summaries must regenerate from raw
  dataset rows.
- Fit failures are recorded and fail the confirmatory decision.
- The A/A, slow-pair, class-renaming, equal-effective-grid, and ineligible-arm
  checks execute inside the experiment before any study unit and fail closed.

The experiment will be run once with master seed `20260818` and
`--confirmatory`. That mode enforces every frozen count and setting, rejects an
existing output directory, and records plan, runner, estimator, lockfile, Python,
NumPy, and scikit-learn identities. Code defects may be fixed, but the defect,
fix, and invalidated run will be recorded. A smoke test must use a different
master seed from the confirmatory run.

## Pre-outcome amendment

The original plan had SHA-256
`a26fc4e5f266ae4692717ddacb96a547394bde8342e93e70f5b5b49240c7a90e`.
Before any confirmatory outcome was run, an independent code-and-design review
identified four defects: terminal shrinkage confounded the proposed mechanism;
CART pruning alphas came from only one depth/leaf pair and duplicates counted
as budget; eight refits weakly ranked 12 configurations; and the fallback and
score-harm rules were too permissive. The changes above fix shrinkage at zero,
derive distinct CART alphas within each pair, increase validation refits to 16,
add cell-level score guards and multiplicity control, and execute the frozen
falsification checks inside the runner. A second pre-outcome review aligned the
classification resampling with the ordinary row-bootstrap estimand and made the
observed-alpha selection rule literal. Its follow-up review led to matched
small-node stopping, three-claim multiplicity control, a normalized absolute
effect guard, strict run-once execution and provenance, task-scoped failure
handling, and post-write artifact regeneration checks. The distinct-seed smoke
run preceded these amendments and is excluded from all evidence and design
choices.
