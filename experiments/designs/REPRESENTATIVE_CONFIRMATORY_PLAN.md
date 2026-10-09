# Confirmatory validation plan: representative model selection

Frozen on 2026-08-17 before running the experiments specified below.

This is a self-attested prospective plan, not a preregistration. Earlier pilot
results informed the design: six fixed examples suggested that choosing the
validation-set prediction medoid can reduce the retraining instability of one
deployed model relative to choosing the candidate with the best validation
score. Those examples are excluded from the confirmatory evidence below.

## Question and estimand

Given an identical pool of models trained from bootstrap resamples, does
selecting the candidate closest to the validation-set mean prediction yield a
less unstable deployed model than selecting the candidate with the best
validation score, without a material loss in test performance?

The unit of uncertainty is an independently generated synthetic dataset. For
each dataset, repeated outer bootstrap fits estimate the conditional
instability of each selection rule. The primary dataset-level effect is

```text
100 * 2 * (instability_representative - instability_best_validation)
        / (instability_representative + instability_best_validation).
```

Negative values favor representative selection. If both instabilities are
zero, the effect is defined as zero. Results will also report raw instability
and performance differences.

## Candidate pool and selection rules

Every comparison uses the exact same candidate models.

- `representative`: candidate closest to the mean validation prediction. For
  classification, distance is mean squared probability distance. For
  regression, it is root mean squared prediction distance; this has the same
  ranking as squared prediction distance.
- `best_validation`: candidate with the highest validation accuracy or R2.
- `random`: one candidate selected by a seed fixed independently of outcomes.
- `ensemble`: mean candidate prediction, reported as context rather than as a
  deployable-single-model comparator.

Each outer fit uses 12 candidates, an 80/20 internal train/validation split,
and stratified resampling for classification. Ties are resolved by the first
candidate index. Candidate failures stop the cell and are reported; no failed
fit is silently excluded.

## Data-generating processes

There are 24 independently generated datasets in each synthetic cell. Each has
500 training and 1,000 untouched test cases. The six processes are:

1. easy binary classification: 12 features, 6 informative, 2 redundant,
   class separation 1.5, label noise 0.02;
2. hard binary classification: the same dimensions, class separation 0.5,
   label noise 0.10;
3. four-class classification: 12 features, 8 informative, 2 redundant, one
   cluster per class, class separation 0.8, label noise 0.05;
4. linear regression: 12 features, 8 informative, Gaussian outcome noise 20;
5. Friedman-1 regression: 10 features and Gaussian outcome noise 1;
6. heteroscedastic regression: 10 independent standard-normal features,
   mean `3*x0 - 2*x1 + x2*x3`, and noise standard deviation
   `0.5 + 1.5*abs(x0)`.

Generation seeds are derived only from the cell and dataset indices. The
earlier pilot seeds and fixed pilot datasets are not reused.

## Estimator families

Each process is evaluated with one tree and one smoother base estimator.

- Classification: a decision tree (`max_depth=5`, `min_samples_leaf=10`) and a
  standardized logistic regression (`C=1`, `max_iter=2000`).
- Regression: a decision tree (`max_depth=5`, `min_samples_leaf=10`) and a
  standardized ridge regression (`alpha=1`).

This deliberately tests whether any benefit is confined to unstable trees.
There are 12 primary synthetic cells: six processes by two estimator families.

## Repetition, outcomes, and uncertainty

For every independent dataset, 20 paired outer bootstrap fits are made. All
rules see the same outer resample, internal split, and candidate pool.

Primary instability is the all-pairs mean test-set prediction distance across
the 20 deployed models:

- classification: mean squared Euclidean distance between class-probability
  vectors;
- regression: mean squared difference between predictions.

Secondary outcomes are classification label disagreement, test accuracy, test
R2, instability versus random selection, ensemble instability, and the
association between validation centrality and test centrality within each
candidate pool. Conditional Monte Carlo error is estimated from 10 disjoint
pairs of outer fits. It is diagnostic only and is not substituted for
dataset-level uncertainty.

Within each cell, 95% percentile bootstrap intervals for the mean dataset-level
effect and mean performance difference use 20,000 deterministic resamples of
the 24 independent datasets. The synthetic grand summary gives each of the 12
declared cells equal weight and resamples datasets within cell. Tree and linear
family summaries give each applicable data-generating process equal weight.

## Decision rules

Broad support requires all of the following:

1. the 95% interval for the equally weighted synthetic grand mean instability
   effect is below zero;
2. at least 8 of 12 cell means favor representative selection;
3. no cell has a mean instability increase greater than 10%;
4. the lower 95% interval endpoint for mean representative-minus-best test
   accuracy is at least -0.01 in classification, and the corresponding R2
   endpoint is at least -0.02 in regression.

Tree-specific or linear-specific support uses the same interval and performance
guardrails within that prespecified family and requires at least 4 of its 6
cell means to favor representative selection, with no cell mean above 10%.
Effects smaller than 5% in magnitude will be described as small even if their
interval excludes zero. Results that miss these rules will be reported as
mixed or unsupported, not repaired by changing the grid or dropping cells.

## External descriptive replication

After the synthetic analysis, the same frozen procedure is run on 20
prespecified 70/30 splits of scikit-learn breast-cancer, wine, and diabetes
datasets, for both estimator families. Splits overlap, so their intervals are
descriptive and cannot replace the independent-dataset synthetic analysis.
They test transfer to real covariate and outcome distributions.

## Falsification and audit checks

- With one candidate, representative and best-validation selection must be
  identical.
- Reordering class labels must not change probability-distance instability.
- Recomputing all-pairs instability with a slow pair loop must reproduce the
  stored summaries.
- JSON and flat CSV evidence must agree exactly up to serialized precision.
- The implementation records the complete configuration and raw
  dataset-level rows so conclusions can be regenerated without consulting
  prose.

The experiment will be run once at the frozen settings. Code defects may be
fixed, but the defect, fix, and any invalidated run will be recorded.
