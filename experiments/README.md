# Experiments with single decision trees

A CART tree can change its first split after a small change in the training data.
Every descendant then sees different observations. These experiments ask whether
we can choose, grow, or update one tree with better accuracy or more stable
predictions. Accuracy and complexity accompany every stability comparison:
a constant prediction is stable, but often useless.

The studies include improvements, tradeoffs, and failed ideas. An ensemble can
serve as a reference; the proposed fitted predictor remains one tree. Neural
network selection and standalone linear-model studies are outside this collection.

## Selecting a representative tree

Fit several candidate trees, average their validation predictions, and keep the
candidate closest to that average. This changes selection, not CART's splitting
rule. Choosing a model close to an ensemble predates this repository
([Ferri et al., 2002](https://link.springer.com/chapter/10.1007/3-540-36182-0_16)).
The question here is whether the choice survives independent evaluation and what
it costs in accuracy.

For prediction vectors p₁,…,pₖ and their mean p̄,

    Σⱼ ‖pᵢ − pⱼ‖² = K ‖pᵢ − p̄‖² + Σⱼ ‖pⱼ − p̄‖².

The last term does not depend on i. The closest candidate to the mean therefore
also minimizes total squared distance to the candidate pool. That identity
explains what selection optimizes. It does not guarantee greater accuracy or
lower variance across newly generated pools.

The rerun uses 144 independently generated datasets, six processes, and 24
datasets per process. Each dataset supplies 20 paired outer bootstrap fits and
12 candidates per fit. Classification resamples preserve observed class counts. Representative selection, best validation score, random selection, and
the ensemble use the same pools and an untouched final test set.

Representative selection lowers the equally weighted symmetric instability
contrast by **8.27%** (95% interval: **7.32% to 9.19%** lower). The classification
accuracy difference is **−0.99 percentage points** (−1.16 to −0.82); the regression
R² difference is **+0.0066** (+0.0042 to +0.0090). The classification interval does
not satisfy the original one-percentage-point accuracy margin.

“Symmetric contrast” means 200(A−B)/(A+B), with zero assigned when both measures
are zero. It is not ordinary percentage reduction. Intervals resample independent
datasets within each fixed process. Real-data repeated splits are descriptive
because they overlap. The outer bootstrap can duplicate observations across the
selector's internal training/validation split; the independent final test set
evaluates that implemented workflow, not selection with a fresh validation sample.

Run `uv run python -m experiments.representative_confirmatory`.
The earlier `selection_rules_experiment` is a fixed-dataset pilot, with its own
results under `results/selection_rules/`.

## Growing and updating trees

| Experiment | Intervention and comparator | Finding |
|---|---|---|
| `bootstrap_split_confirmatory` | Bootstrap feature votes, median thresholds, and leaf shrinkage versus validation-tuned, pruned CART | Classification instability rises. The regression symmetric contrast improves, but the absolute-change interval includes zero. |
| `repaired_trees_screen` | Penalize variable split gains, or use an honest robust prefix | Neither method passed the screening rule. Zero-penalty and fixed-prefix checks isolate what each implementation changes. |
| `persistent_trunk_screen` | Freeze the root and refit descendants; compare full refitting, a frozen tree, and refreshed leaves | A stable root sometimes helps relative to full CART refitting, but refreshing leaves in the whole fixed partition is the stronger comparator in the stable regimes. |
| `knob_study` | Raise the required support for a split, or shrink leaf values toward their parent | The measured leaf-variance share did not reliably choose the better intervention: 2/4 matches for separated trees, 1/4 for aliased signals, and 0/4 for smooth signals. |

On the 60 independent regression datasets, bootstrap split voting improves R²
by **0.0061** (95% interval: **0.0027 to 0.0093**) and improves the symmetric
instability contrast by **16.35%** (**11.83% to 20.56%**). On the 60 classification
datasets, the symmetric instability contrast worsens by **30.55%**. The original
regression inclusion rule still fails because the absolute instability-change
interval includes zero. That decision does not negate the measured accuracy gain.

The bootstrap-voting baseline in `knob_study` already uses twelve votes. Its
support intervention changes when growth stops; it does not switch averaging on.
Its twelve regimes are exploratory, with one training dataset per regime.
The repaired-tree screen has only six independent datasets per process. Task
intervals resample datasets within each fixed process; six units per process
still make these coarse screening estimates.

The persistent-root simulation deliberately makes stable and moving roots easy
to distinguish. Successful gating there does not establish successful detection
of gradual or subtle drift.

Run the named module with `uv run python -m experiments.MODULE`.
For the knob study, pass `--dgp tree_separated`, `--dgp aliased`, or `--dgp smooth`
and `--output results/knob_study/NAME`. `bootstrap_split_eval` and `frontier_eval`
provide exploratory score/stability comparisons. These sweeps use evaluation
scores to choose configurations, so their apparent gains need independent
confirmation. Both arms remain visible when only one meets the score target.

## Separating structure and leaf variance

`variance_budget` fits a partition on one sample and estimates its leaf values
on fresh samples. Empty leaves use the fresh sample's overall mean. This is an
honest-leaf estimator; ordinary CART uses the same observations for both tasks.
Its variance decomposition cannot be read as a decomposition of ordinary CART.

Let mₛ be the mean prediction across L fresh leaf samples for partition s, and
wₛ their sample variance, using L−1 in the denominator. At each evaluation point,

    Ŵ = meanₛ(wₛ)
    B̂ = sample_varianceₛ(mₛ) − Ŵ/L
    T̂ = Ŵ + B̂.

The sample mean mₛ contains inner-simulation noise with expected variance W/L.
Subtracting it estimates variation in the conditional mean across partitions.
Negative estimates of B are possible and remain visible. Consequently an
estimated component share can exceed one; truncation would conceal simulation
error. Uncertainty resamples entire outer structures, conditional on the common
evaluation sample. Replicating evaluation rows does not increase the effective
number of simulated structures.

The 216 regimes vary signal structure, noise, sample size, and minimum leaf size.
Each uses 25 partitions and ten fresh leaf samples per partition. Results report
honest-leaf accuracy, ordinary CART accuracy and variance, and the component
estimates. Compressed sufficient statistics reconstruct the components and their
standard errors without fitting another tree.

Run `uv run python -m experiments.variance_budget`.

## Comparing greedy and optimized trees

`margin_study` changes the coefficient of a second signal. It is a controlled
simulation, but that coefficient also changes prediction difficulty and useful
tree complexity. It is not an intervention on a literal impurity margin. A
clearly best root split does not guarantee a globally optimal full tree.

Both solver studies learn quantile cutpoints on fitting observations, apply
those cutpoints to independent evaluation observations, and give greedy trees
the optimized tree's leaf budget. They retain raw-feature CART as a comparator.
The binary representation itself can reduce accuracy substantially. Actual leaf
counts remain visible because a common budget does not force a common size.

[GOSDT](https://github.com/ubc-systopia/gosdt-guesses) minimizes penalized training
misclassification; CART greedily reduces Gini impurity. The comparison therefore
changes the objective as well as search. Each fit records termination status,
objective bounds, and whether optimality was certified. Timed-out fits are
incumbent solutions, not demonstrated optima. If the majority-class error is at
most the leaf penalty, a constant tree is already optimal: any nonconstant tree
pays at least two leaf penalties. This case is certified analytically.

The signal sweep uses twelve independent training samples per setting. The
five-dataset study uses fifteen paired bootstrap fits per fixed development/test
split. Whole-fit bootstrap intervals are conditional on the evaluation sample;
they do not estimate uncertainty over new datasets. With only twelve or fifteen
fits, disagreement intervals are approximate; comparing marginal intervals is
not a paired test of an algorithm difference. Compressed predictions permit
summary reconstruction. These small studies illustrate algorithm differences;
they do not establish a generally superior search rule.

The five-dataset rerun gives the following GOSDT-minus-binarized-CART accuracy
differences and paired 95% intervals. Intervals are conditional and unadjusted
across these exploratory comparisons.

| Dataset | Accuracy difference (percentage points) | Certified fits |
|---|---:|---:|
| synth_easy | +0.71 (-0.62, +2.22) | 11/15 |
| synth_hard | +1.24 (-0.13, +2.76) | 0/15 |
| breast_cancer | +0.74 (+0.27, +1.29) | 8/15 |
| wine_binary | -2.22 (-5.13, +0.68) | 15/15 |
| digits_binary | +0.06 (+0.00, +0.19) | 15/15 |

GOSDT has lower prediction disagreement than binarized CART on two of the five
datasets. In the signal sweep, the representation matters: at delta=1 the Bayes
boundary is x0=0, but the four population quantile cutpoints omit zero. Even an
optimized tree on that binary representation cannot recover the Bayes rule.

GOSDT needs an older scikit-learn API. Its environment is locked separately:

```sh
uv run --project experiments/solver python -m experiments.margin_study
uv run --project experiments/solver python -m experiments.optimal_tree_premise
make test-solver
```

## Reproduction

Run commands from the repository root. `uv sync --all-groups` installs the main
environment. `make lint`, `make test`, and `make ci-docker` cover package and
experiment code. Tests include scalar split-gain references, target translation,
class relabeling, one-candidate equality, nested-variance reference models, and
reconstruction of saved summaries. The solver tests compare a tiny search with
an exhaustive objective calculation and verify constant-tree handling.

`results/` holds the current evidence for each study. `designs/` retains the
original study specifications and inclusion rules; their hashes identify the
protocol used by a run. The representative specification also records the
original, wider model scope.
The CART subset and repaired reruns are not new blinded confirmations. Existing
outcomes were known before consolidation. The variance correction, fitting-only
cutpoints, solver status reporting, and numerical split-scoring repair were
reviewed before interpreting the reruns. Experimental estimators stay in this
directory and do not extend the package API.
