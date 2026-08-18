# Supported package scope

`stable-cart` audits the prediction instability of scikit-learn-compatible
estimators and model-building pipelines. The supported top-level API is:

| Workflow | Public objects |
|---|---|
| Bootstrap prediction audit | `bootstrap_predictions`, `bootstrap_instability` |
| Validation score-instability frontier | `stability_frontier`, `pareto_front` |
| Tree-structure audit | `split_features`, `split_feature_paths`, `explanation_instability`, `root_agreement`, `path_agreement` |
| Plots | `plot_prediction_instability`, `plot_mape_by_prediction`, `plot_stability_frontier` |
| Representative selection | `RepresentativeEstimator` |
| Fixed-design linear calibration | `linear_instability`, `linear_frontier`, `shrinkage_coefficients` |

The bootstrap functions refit the complete procedure supplied by the user.
Classification supports class-label disagreement and aligned multiclass
probability vectors. Aggregate estimates include Monte Carlo standard errors,
and the raw per-refit predictions remain available for case-level inspection.

`RepresentativeEstimator` returns one member of a bootstrap-fitted candidate
pool, selected by validation-set prediction centrality. Evidence for lower
instability is task dependent, so the class does not claim to stabilize every
estimator or dataset. The linear functions are calibration tools under their
documented fixed-design assumptions, not general guarantees.

Current resampling assumes independent rows. Clustered, grouped,
time-dependent, and survey data require a design-appropriate resampling scheme
that this release does not implement. A stability frontier is a validation
object; final predictive performance must be measured on untouched data or in
an outer resampling loop.

The package intentionally has no experimental estimator namespace, deprecated
aliases, or compatibility wrappers for removed tree implementations.
