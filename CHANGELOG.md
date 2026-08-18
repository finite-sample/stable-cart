# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 3.0.0 - 2026-08-17

### Added

- A complete public-API workflow on built-in regression and multiclass datasets,
  exercised by the test suite and runnable from an installed wheel.
- `RepresentativeEstimator` for selecting one fitted candidate by validation-set
  prediction centrality.
- Fixed-design linear calibration through `linear_instability`,
  `linear_frontier`, and `shrinkage_coefficients`.
- Multiclass probability instability, tree-structure diagnostics, Monte Carlo
  standard errors, and score-instability frontier plots.

### Changed

- `RepresentativeEstimator` clones any supplied scikit-learn-compatible
  estimator and passes scikit-learn's maintained estimator checks in
  classification and regression modes.
- `stability_frontier` accepts explicit validation data, reports generic
  validation scores, and uses paired bootstrap samples across configurations.
- Classification probability columns are aligned through each estimator's
  `classes_` attribute, including when a bootstrap resample omits a class.

### Fixed

- Classification metrics no longer depend on numeric class encoding.
- Pairwise instability uses every bootstrap refit; numeric variance uses
  `ddof=1`; its Monte Carlo error now uses the delete-one jackknife for the same
  all-pairs U-statistic reported as the point estimate.
- Classification audits use the ordinary pairs bootstrap so class-prevalence
  variation is measured instead of conditioned away.
- Pairwise Monte Carlo error remains numerically stable when predictions share a
  large common offset.
- `RepresentativeEstimator` preserves DataFrame columns for wrapped pipelines
  and deterministically seeds random-state parameters in nested estimators.
- Fixed-design linear calibration rejects wide and collinear designs instead of
  returning undefined shrinkage coefficients or misleading frontiers.
- Linear frontier grids now stay within the requested range and include
  `mu_max`, including for two-point and sub-`1e-3` frontiers.
- Linear frontiers use the Pareto-optimal zero-price limit, dropping exactly null
  signal directions without bias, while `shrinkage_coefficients(mu=0)` remains
  ordinary least squares.
- The `mu=0` coefficient endpoint returns ordinary least squares without
  requiring an irrelevant residual noise estimate.
- Robust fixed-design calibration rejects saturated designs that cannot estimate
  residual variation.
- Validation frontiers normalize single-column targets and reject multioutput
  targets instead of allowing broadcasting to corrupt scores.
- Probability-vector MAPE plots report probability-vector units rather than
  class-label disagreement.
- Classification-label plots encode and display arbitrary string class names
  instead of assuming labels are numeric.
- Fixed-design linear tools reject nonfinite prices, noise scales, coefficients,
  targets, and evaluation points instead of returning plausible NaN results.
- Tree feature paths follow scikit-learn's learned routing direction for missing
  values.
- `RepresentativeEstimator` exposes `predict_proba` only when its configured
  classifier supports probability predictions.
- Validation frontiers use scikit-learn's finite R-squared convention for
  constant validation targets.
- One-class classification bootstrap samples are redrawn under one explicit,
  estimator-independent policy, with rejected draws and fit counts reported.
- Frontier plots recompute the Pareto set for the instability quantity shown on
  the axis.
- Fixed-design linear formulas now distinguish sampling-mean absolute deviation
  from bootstrap MAPE and compute the stability frontier on the documented
  pointwise scale.

### Removed

- Experimental custom tree estimators and their private split-strategy
  machinery. They are not part of the supported package.
- Thin metric wrappers, deprecated aliases, and compatibility wrappers for the
  removed estimators.
- Generated documentation artifacts and invalid historical benchmark claims.

## [2.0.0] - 2026-08-15

- Reoriented the project around model-agnostic prediction-instability audits.
- Added bootstrap audits, score-instability frontiers, and plotting functions.
- Removed implementations and benchmark results that did not support their
  advertised behavior.

[2.0.0]: https://github.com/finite-sample/stable-cart/releases/tag/v2.0.0
