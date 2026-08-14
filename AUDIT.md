# stable-cart correctness audit

**Date:** 2026-08-14 · **Commit audited:** `1729d22` · **Method:** three independent
passes — Claude (this session), `codex` (GPT-5.2-codex), `agy` (Gemini 3.1 Pro) — each
given an identical brief, then every external finding re-run against the live repo
before it entered this report.

**Headline:** the documented stability primitives do not reach the code that builds the
tree. Twenty-one to twenty-three of each estimator's constructor parameters change
nothing at all, two of the three "different" algorithms are bit-identical, and on
standard regression benchmarks the trees returned a single leaf (R² ≈ 0 against 0.82 for
`DecisionTreeRegressor`). The published variance-reduction numbers were an artifact of
that: a model that predicts a constant has zero prediction variance, which the benchmark
scored as a large "variance reduction". Fixing the split search (C3) restores near-CART
accuracy and flips the headline metric negative — but only mildly, and not uniformly: on
the regenerated benchmark the three methods each win on 4–5 of 11 datasets and lose badly
on the rest (C4). `CentroidTree` is a separate story: its candidate pool was twenty copies
of one tree, and fixing that turns the selection rule into a real +18% effect — though
still not enough to beat a plain CART (C5, C6).

Everything below was reproduced by running it. Repros assume `uv run python` at the
repo root.

**Status:** C3 is fixed in this working tree (test-first, `tests/test_split_candidates.py`);
C2 largely resolves as a consequence — see the before/after table. C1 and everything else
is reported only, awaiting a decision. Re-measured after the C3 fix: C1 is unchanged —
still 20–24 inert parameters per estimator, and BootstrapVariancePenalizedTree and
RobustPrefixHonestTree still produce bit-identical predictions on matched settings.

---

## Critical

### C1. Documented feature switches never reach the split strategy — **FIXED**

**File:** `stable_cart/base_stable_tree.py:581` (`_create_split_strategy`)

Every exported estimator leaves `split_strategy=None`, so `_create_split_strategy` takes
the `else` branch and returns `HybridStrategy(focus=self.algorithm_focus, task=...,
random_state=...)`. That branch forwards **no feature switches and no numeric settings**.
The `create_split_strategy(...)` branch that does forward them is unreachable unless the
user passes `split_strategy=` explicitly — which is a `BaseStableTree` parameter absent
from all three unified subclasses' signatures.

Consequence: oblique splits, lookahead/beam search, consensus, threshold binning,
ambiguity gating, variance penalties and gain margins are all fixed at whatever
`HybridStrategy` hardcodes for the chosen `algorithm_focus`.

**Repro** — flip each parameter and count how many move a single prediction:

```python
# scratchpad/flag_effect.py in full; the result per class:
LessGreedyHybridTree[classification]        2 live, 21 inert
LessGreedyHybridTree[regression]            2 live, 21 inert
BootstrapVariancePenalizedTree[both]        2 live, 23 inert
RobustPrefixHonestTree[classification]      3 live, 21 inert
```

The only parameters that change anything are `enable_stratified_sampling`,
`enable_winsorization`, `leaf_smoothing`/`smoothing`. Inert ones include
`variance_penalty`, `n_bootstrap`, `top_levels`, `consensus_samples`,
`enable_oblique_splits`, `enable_lookahead`, `beam_width`, `enable_threshold_binning`.

**Sharpest form** — two classes hardcode `algorithm_focus="stability"`, so they are the
same estimator:

```python
SHARED = dict(
    random_state=42,
    max_depth=5,
    min_samples_leaf=20,
    val_frac=0.2,
    est_frac=0.2,
    enable_stratified_sampling=True,
    enable_winsorization=False,
)
b = BootstrapVariancePenalizedTree(task="regression", leaf_smoothing=1.0, **SHARED).fit(
    X, y
)
r = RobustPrefixHonestTree(task="regression", smoothing=1.0, **SHARED).fit(X, y)
np.array_equal(b.predict(X), r.predict(X))  # -> True, 0/500 rows differ
```

And the namesake parameters do nothing:

```
variance_penalty=0.0 / 1.0 / 100.0 / 1e6   -> identical predictions
n_bootstrap=2 / 10 / 50                    -> identical predictions
top_levels=1 / 2 / 5                       -> identical predictions
```

**Fix applied.** `_create_split_strategy` now assembles the strategy graph from the
documented switches — `enable_prefix_consensus` → `ConsensusStrategy`,
`enable_oblique_splits` → `ObliqueStrategy`, `enable_lookahead` → `LookaheadStrategy`,
`enable_explicit_variance_penalty` → `VariancePenalizedStrategy` — each receiving its
numeric settings, composed over an `AxisAlignedStrategy` base that carries the
tie-breaking, margin-veto and threshold-binning parameters.

Covered by `tests/test_strategy_wiring.py`: 8 tests, all failing before the change. They
assert both that a flag puts its strategy in the graph *and* that numeric settings arrive
(`consensus_samples=7` reaches the strategy as 7), plus one behavioural test that a
documented flag actually changes predictions. This is the check whose absence let 20+
inert parameters ship.

**Measured effect.** Live parameters per estimator, before → after:

| estimator | before | after |
|---|---|---|
| LessGreedyHybridTree | 2 live / 21 inert | 6–10 live / 13–17 inert |
| BootstrapVariancePenalizedTree | 2 live / 23 inert | 8–15 live / 10–17 inert |
| RobustPrefixHonestTree | 3 live / 21 inert | 10–14 live / 10–14 inert |

**This also dissolves the bit-identical finding above.** `BootstrapVariancePenalizedTree`
and `RobustPrefixHonestTree` now differ (116/500 rows on classification, 500/500 on
regression). They were identical only because `algorithm_focus` was the sole input to the
strategy; they are genuinely different estimators once their parameters are honoured, so
the "one of them is redundant" removal no longer follows.

**Still inert after the fix** — 10–17 parameters per estimator, including `top_levels`,
`consensus_subsample_frac`, `enable_beam_search_for_consensus`,
`enable_bootstrap_variance_tracking`, `variance_tracking_samples`. These are the kill-list
candidates: a parameter that cannot change a prediction should be deleted, not documented.

---

### C2. Regression fits collapse to a single leaf — **largely resolved by the C3 fix**

**File:** consequence of C1 + C3.

```python
X, y = make_regression(
    n_samples=500, n_features=10, n_informative=6, noise=2.0, random_state=0
)
LessGreedyHybridTree(task="regression", random_state=42).fit(X, y)
# unique predictions = 1,  R² = -0.002        (root node type == 'leaf')
DecisionTreeRegressor(max_depth=5, random_state=42)
# unique predictions = 30, R² = +0.815
```

Same on `make_friedman1`. At n=500 and n=2000 the root is a leaf; only at n=10000 does it
split at all, reaching R² = 0.073. Classification is unaffected (0.84–0.92 accuracy).

This is visible in the repo's own committed benchmark: `r2_mean` is 0.038 for
BootstrapVariancePenalized and 0.128 for LessGreedyHybrid against 0.570 for CART.

**After fixing C3** (same seeds, same defaults, `DecisionTreeRegressor(max_depth=5)` for
reference):

| dataset | model | R² before | R² after | sklearn |
|---|---|---:|---:|---:|
| make_regression | LessGreedyHybrid | −0.002 | **+0.764** | +0.815 |
| make_regression | BootstrapVariancePenalized | −0.002 | **+0.509** | +0.815 |
| make_regression | RobustPrefixHonest | −0.002 | **+0.666** | +0.815 |
| friedman1 | LessGreedyHybrid | −0.000 | **+0.607** | +0.829 |
| friedman1 | BootstrapVariancePenalized | −0.000 | **+0.431** | +0.829 |
| friedman1 | RobustPrefixHonest | −0.000 | **+0.658** | +0.829 |

The trees now split (4–40 distinct predictions instead of 1). Classification is
essentially unchanged (0.84 → 0.84 for Bootstrap, 0.84 → 0.82 for LessGreedy), which is
consistent with the bug biting hardest where a good threshold sits away from a feature's
low tail. **The committed benchmark artifacts and every README number derived from them
are now stale** and must be regenerated.

---

### C3. Axis-aligned split search only examines the lowest feature values — **FIXED**

**File:** `stable_cart/stability_utils.py:1084` — *found by codex*

The loop evaluates only the first `splits_per_feature` sorted thresholds and then sorts
that truncated low-tail set, so the search never sees the middle or upper range of a
feature.

```python
X = np.arange(100, dtype=float).reshape(-1, 1)
y = (X[:, 0] >= 50).astype(int)  # the only correct threshold is 49.5
BaseStableTree(
    task="classification",
    max_depth=1,
    min_samples_split=2,
    min_samples_leaf=1,
    enable_honest_estimation=False,
    enable_validation_checking=False,
    algorithm_focus="speed",
    random_state=0,
).fit(X, y)
# stable-cart threshold =  9.5, accuracy = 0.6
# sklearn     threshold = 49.5, accuracy = 1.0
```

**Fix applied.** `_find_candidate_splits` now scores *every* admissible midpoint of each
feature through a new vectorized helper `_all_split_gains` (one sort plus cumulative
sums, O(n log n) per feature instead of O(n) per threshold), then keeps that feature's
**best** `max_candidates // n_features` thresholds. The per-feature budget — which keeps
one feature from crowding out the others — is unchanged; only the selection rule changed,
from "the numerically lowest" to "the highest-gain".

Covered by `tests/test_split_candidates.py` (5 tests, all failing before the change).
One of them is an equivalence check: the vectorized gain must equal the original
per-mask `_evaluate_split_gain` for every returned candidate, on both regression and
classification data — so the speedup cannot silently change the scoring. The
regression/classification predicate is now shared (`_is_regression_target`) so the two
paths cannot drift apart.

---

### C4. With C3 fixed, the stability claim inverts

The benchmark was regenerated after the C3 fix on the same 14 datasets, same seed 42,
same 20 bootstrap samples. CART, CART_Pruned and RandomForest come back **bit-identical**
(`pred_variance_mean` 311.5981 → 311.5983), which is the control: only stable-cart's own
code changed.

Accuracy recovers to roughly CART parity:

| model | R² before | R² after | CART |
|---|---:|---:|---:|
| LessGreedyHybrid | 0.128 | **0.564** | 0.570 |
| BootstrapVariancePenalized | 0.038 | **0.494** | 0.570 |
| RobustPrefixHonest | 0.034 | **0.487** | 0.570 |

And the headline metric changes sign. `variance_reduction_pct` versus CART, where
positive means *more stable than CART*:

| model | before | after |
|---|---:|---:|
| LessGreedyHybrid | **+55.4%** | **−16.4%** |
| BootstrapVariancePenalized | **+35.9%** | **−108.1%** |
| RobustPrefixHonest | −11.7% | **−98.7%** |
| CART_Pruned (plain sklearn) | +12.1% | +12.1% |
| RandomForest | −2.8% | −2.8% |

**Correction to an earlier draft of this section.** Those pooled figures were *means* of an
unbounded ratio and overstated the case; `variance_reduction_pct` has mean −2.7% but median
**+81.7%** for RandomForest (std 306%, min −1065%), so the mean inverts the sign of an
obvious effect. On medians the reduction is −3.0% / −11.1% / −15.2%, not −16% / −108%.

More importantly the pooled number hides the real pattern — these methods win on some
datasets and fail badly on others:

| dataset | LessGreedy | Bootstrap | RobustPrefix | CART_Pruned |
|---|---:|---:|---:|---:|
| friedman3 | +45.8 | +68.8 | **+84.5** | +58.5 |
| xor_nonlinear | +13.7 | +12.1 | +34.5 | +0.5 |
| high_dim_sparse | −0.1 | +0.4 | +13.8 | −0.0 |
| california_housing | +58.5 | −45.1 | −37.7 | +20.9 |
| friedman2 | −90.2 | **−544.9** | −500.5 | 0.0 |
| quadrant_interaction | −3.0 | −497.3 | −502.5 | +24.1 |
| breast_cancer | −128.5 | −95.4 | −127.1 | −10.6 |

Wins: 4/11, 4/11, 5/11 — a coin flip with a fat left tail, not a uniform failure. The
sharper problem is that `CART_Pruned` wins 9/14 and `RandomForest` 13/14: sklearn's own
cost-complexity pruning is a more reliable stability tool than any of the three, at one
argument. **No method should be removed on this evidence alone** — that needs the frontier
comparison at matched accuracy, which has not been run.

Partial root cause for the catastrophic rows: **these trees are not scale-invariant,
although a CART is.** On `friedman2` (feature sd spanning 0.29 to 469.6) CART's instability
is byte-identical raw vs standardized, while standardizing the features moves
BootstrapVariancePenalized from −551% to −115% and RobustPrefixHonest from −638% to −195%.
That is C-M4 (`enable_feature_standardization` is a no-op) biting in production: the
methods have never been evaluated as designed.

Scale does **not** generalize as an early-warning rule, and the hypothesis was tested and
refuted: `friedman2` and `friedman3` have identical scale ratios (log10 = 3.21) and land at
−545% and **+69%**; Spearman rho is +0.15 to +0.37 with p > 0.26 at n=11, the opposite sign
to the hypothesis; and `quadrant_interaction` is −497% at a scale ratio of ~1. With 11
datasets no reliable predictor can be fitted. The dependable form of "early detection" is
to measure rather than predict — run `bootstrap_instability` for the candidate model and a
pruned CART on the user's own data and compare.

**This does not prove the underlying ideas fail.** C1 is still open: the honest
partitioning, consensus and variance-penalty machinery is still not reachable from the
constructor, so what was measured is "honest-partitioned CART with a hardcoded strategy
preset", not the documented method. It does mean no number currently published about
this package survives.

The classification side still has the M6 problem: the stable methods produce results on
**2** of 5 datasets (they raise on multiclass), so their accuracy column is not
comparable to CART's.

---

### C4b. How much of the instability is *greedy* search? Only a fifth of it.

The root-cause story — structure churn dominates, and greedy myopic split selection causes
it — was tested against a globally-optimal solver (GOSDT, `experiments/optimal_tree_premise.py`).
An exact solver does not tie-break greedily, so if greedy search were the culprit its trees
should be dramatically more stable. Three arms, since GOSDT needs binarized features and
binarization is itself a threshold-snapping step: raw greedy CART, greedy CART on the same
binarized features (the control), and GOSDT.

| dataset | binarized greedy | GOSDT | GOSDT vs greedy | accuracy greedy → GOSDT |
|---|---:|---:|---:|---|
| synth_easy | 0.1240 | 0.1120 | +9.7% | 0.880 → 0.820 |
| synth_hard | 0.1484 | 0.1320 | +11.1% | 0.707 → 0.740 |
| breast_cancer | 0.0452 | 0.0519 | −14.7% | 0.924 → 0.906 |
| wine_binary | 0.0564 | 0.0427 | **+24.2%** | 0.949 → 0.949 |
| digits_binary | 0.00856 | 0.00734 | +14.3% | 0.991 → 0.982 |

**GOSDT is more stable on 4 of 5** — so the gate passes directionally — but the effect is
**10–24%**, and three of the four wins come with an accuracy drop, so part of it is just
the usual accuracy-for-stability trade. Set that against bagging, which buys **62–92%** on
the same metric. (`breast_cancer` hit the 30s solver limit on 4 of 15 fits, so that row is
not a truly optimal tree.)

That magnitude is the finding. The strong form of the root cause — "structure churn is
caused by greedy myopia" — does **not** survive: perfect search recovers only about a fifth
of what averaging does. Most of the churn is intrinsic to fitting **one discrete structure to
a finite sample**, not an artifact of the greedy heuristic.

**Correction to the 89–98% figure quoted above.** Two problems, both since fixed in
`experiments/variance_budget.py`:

1. *The decomposition was invalid.* It held the structure fixed at a reference tree fitted on
   a **different** sample, so the two quantities were not nested and the "share" could exceed
   100% — observed at 122.7%. The corrected version conditions on the structure and applies
   the law of total variance, so the parts sum to the total exactly.
2. *The number was a regime, not a fact.* Across a 216-cell grid of DGP × σ × n × leaf size,
   the leaf share spans **1.1% to 90.8%, median 66.4%**, and exceeds 40% in 148 of 216 cells.
   Noise is what governs it: at σ=0.1 the leaf share is 6–7%, at σ=10 it is 76–83%.

Re-measured with the corrected method, the three real regression datasets give leaf shares of
3–24% — so structure does dominate **there**, and the original qualitative reading of those
datasets survives. What does not survive is the generalisation. The practical consequence
inverts with noise: in low-noise data, split-selection cleverness is the right lever; in noisy
data, leaf shrinkage is, because leaf estimation is then 70–90% of the variance. That makes
`leaf_smoothing` the right tool for noisy data rather than the near-useless knob called for in
an earlier draft of this report.

The practical consequence is a ceiling: no single-tree method should be expected to
approach ensemble stability, because only averaging over the data perturbation removes the
intrinsic part. A single tree can be made *modestly* more stable; it cannot be made as
stable as a forest while remaining one tree. Any README claim should be scaled to that.

It also sharpens what Phase 1c should be: stability selection helps not by searching better
but by *averaging the split decision over resamples* — moving the averaging from prediction
space into split-choice space. That is a different mechanism from optimal search, and it is
the one still worth testing.

---

### C5. `CentroidTree`'s candidate pool was N copies of one tree — **FIXED**

**File:** `stable_cart/centroid_tree.py:167` (`fit`)

Candidates were built by varying `random_state` alone on identical training data:

```python
seed = int(rng.integers(0, 2**31 - 1))
tree = tree_class(**{**base_params, "random_state": seed})
tree.fit(X_train, y_train)  # same X_train for every candidate
```

A decision tree is deterministic given (data, parameters); `random_state` only breaks
ties between *exactly equal* splits, which with continuous features essentially never
occur. Measured spread across a 20-candidate pool (`max_depth=6, min_samples_leaf=20`):
**1.2e-14** — floating-point noise. Selecting the member closest to the mean of twenty
identical trees is arithmetic, not a method. Every selection rule scored within 0.3% of
picking at random, on every dataset.

**Fix applied.** New `bootstrap_candidates=True` parameter (default): each candidate is
fitted on its own resample of the training split, class-stratified for classification so
no candidate loses a class. Covered by `tests/test_candidate_pool_diversity` — five tests,
four failing before the change, including one that pins the old behaviour under
`bootstrap_candidates=False` so the failure mode cannot return unnoticed.

**Effect, measured with a paired design** (one pool per replicate, every rule applied to
that same pool; 7 datasets × 8 independent outer seeds × 50 replicates):

| selection rule | mean vs random pick | 95% CI | wins |
|---|---:|---|---:|
| **centroid** | **+18.13%** | [15.92, 20.31] | 55/56 |
| medoid (Banerjee) | +17.27% | [15.13, 19.41] | 55/56 |
| best_val (control) | +12.11% | [9.55, 14.90] | 54/56 |

Accuracy *improves* alongside: +0.0187 [+0.0151, +0.0228]. The centroid and `best_val`
intervals do not overlap, so proximity-to-the-ensemble beats "select something" — the
mechanism is real, not an artifact of choosing from a pool.

### C6. …but `CentroidTree` still does not beat a plain CART

The gain in C5 is against a *random pick from its own pool*. Against the baseline a user
actually has — one `DecisionTreeRegressor`/`Classifier` fitted on all the data — it is a
wash or worse, because manufacturing the pool costs what the selection recovers:

| diversity mechanism | regression vs CART | classification vs CART |
|---|---:|---:|
| bootstrap rows | −16.2% | +0.6% |
| random subspace (all rows) | −51.6% | +4.0% |
| both (bagging) | −43.1% | −4.6% |

The reason is structural, and it is the caveat flagged from the representative-tree
literature: **a plain CART fitted on fixed data is already deterministic**, so it has no
selection variance to remove. The instability that matters is variance under *training-data*
resampling, and choosing a representative member of an ensemble built from one training
set cannot reduce it — averaging can (the `ensemble` rule is 51–92% more stable), but an
average is not a single tree.

Banerjee et al. motivate representative trees by **interpretability** — recovering one
readable tree that summarises a forest — not by prediction stability. That framing is
supported here; the stability framing is not. Recommend repositioning `CentroidTree`
accordingly rather than deleting it.

### C7. The CentroidTree experiment measured nothing

**File:** `experiments/centroid_experiment.py:270`

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=..., random_state=seed
)
...
pred_variance = np.mean(np.var(seed_predictions_array, axis=0))
```

`X_test` changes with `seed`, so row `j` is a different data point in every row of that
array; the variance is taken across unrelated points. The classification branch computes a
modal prediction the same way. Shapes match, so nothing errors. This is the metric behind
the README's CentroidTree table and the CHANGELOG's "~3% variance reduction". **Not yet
fixed** — superseded in practice by `experiments/selection_rules_experiment.py`, which
measures instability correctly.

Related gap, now closed: the package shipped `prediction_stability` (agreement among a set
of fitted models) but no measure of instability under resampling — the quantity every
claim in the README is about. `stable_cart.bootstrap_instability` now provides it, with
tests including the identity that a model ignoring its training data scores exactly zero.

---

## High

### H1. Consensus threshold is overwritten by the split threshold

**File:** `stable_cart/stability_utils.py:159` — *found by codex*

`threshold` (documented as "minimum consensus threshold for accepting a split") is
rebound while unpacking `(feature_idx, split_threshold)`, so the requested threshold is
lost.

```python
X = np.arange(40, dtype=float).reshape(-1, 1)
y = (X[:, 0] > 19).astype(int)
bootstrap_consensus_split(X, y, n_samples=30, threshold=0.5, random_state=0)
# -> accepted a candidate with consensus_support = 0.267 against a requested 0.5
bootstrap_consensus_split(X + 100, y, n_samples=30, threshold=0.5, random_state=0)
# -> None.  Translating the feature by a constant flips acceptance.
```

**Fix:** keep the argument under a distinct name (`consensus_threshold`).

### H2. `consensus_support` can exceed 1.0

**File:** `stable_cart/stability_utils.py:133` — *found by codex*

Multiple thresholds from one bootstrap replicate collapse to the same binned key and are
counted as separate votes, so a "fraction of bootstraps agreeing" comes out at 9.17.

```python
best, _ = bootstrap_consensus_split(
    np.arange(100.0).reshape(-1, 1),
    (np.arange(100) >= 50).astype(int),
    n_samples=12,
    max_candidates=20,
    threshold=0.5,
    enable_quantile_binning=True,
    max_bins=2,
    random_state=0,
)
best.consensus_support  # -> 9.166666666666666
```

**Fix:** deduplicate `(feature, binned_threshold)` within a replicate before tallying.

### H3. `RobustPrefixHonestTree` breaks `clone`, `get_params`, `set_params`

**File:** `stable_cart/unified_robust_prefix_tree.py:216` — *found by agy and independently here*

`smoothing` is forwarded to the base as `leaf_smoothing=smoothing` but never stored as
`self.smoothing`, and sklearn's `get_params()` reads the attribute named in the signature.

```python
clone(RobustPrefixHonestTree(smoothing=0.5))
# AttributeError: 'RobustPrefixHonestTree' object has no attribute 'smoothing'
cross_val_score(RobustPrefixHonestTree(task="regression"), X, y, cv=3)  # same error
```

Every sklearn meta-estimator — `GridSearchCV`, `Pipeline`, `cross_val_score` — is
therefore unusable with this class. *(Correction to agy's write-up: it reported that
`smoothing` is "never passed to the parent". It is passed; it is only not stored.)*

**Fix:** `self.smoothing = smoothing` (keep the existing forward).

### H4. `CentroidTree` does not forward `task` to its candidate trees

**File:** `stable_cart/centroid_tree.py:159` — *found by agy*

Candidates are built as `tree_class(**{**base_params, "random_state": seed})`, with no
`task=`. Since stable-cart trees default to `task="regression"`, the combination the
README advertises ("Works with any sklearn-compatible tree class (CART,
LessGreedyHybridTree, etc.)") fails:

```python
CentroidTree(
    base_tree_class=LessGreedyHybridTree,
    task="classification",
    n_candidates=2,
    random_state=0,
).fit(X, y)
# ValueError: predict_proba is only available for classification tasks   (raised inside fit)
```

Workaround today is `base_params={"task": "classification"}`. *(agy reported this as
raising at `predict_proba`; it actually raises during `fit`.)*

**Fix:** forward `task=self.task` when the base class accepts it.

---

## Medium

### M1. `enable_gain_margin_logic` is reported as `False` whatever you pass

**Files:** `unified_less_greedy_tree.py:196`, `unified_bootstrap_variance_tree.py:216`

Both subclasses declare the parameter with default `True` and forward it as
`enable_margin_vetoes=`, leaving the base's own `enable_gain_margin_logic` at `False`.
`get_params()` reads the latter, so `clone()` silently builds a *different* estimator
than the one you configured — which is what `cross_val_score` and `GridSearchCV` fit.

```python
m = LessGreedyHybridTree(enable_gain_margin_logic=True)
m.enable_gain_margin_logic, m.get_params()["enable_gain_margin_logic"]  # (False, False)
```

(Both flags are also never read anywhere — see C1.)

### M2. No sklearn estimator tags: nothing is a classifier or a regressor

**Files:** `base_stable_tree.py:28`, `centroid_tree.py:26`

`BaseStableTree` inherits `BaseEstimator` only; `CentroidTree` inherits **both**
`ClassifierMixin` and `RegressorMixin`, which cancel out. Under sklearn 1.7.2 every
class/task combination reports `is_classifier=False, is_regressor=False`, so
`check_cv` picks `KFold` instead of `StratifiedKFold` for classification.

```python
for cls in (
    BaseStableTree,
    LessGreedyHybridTree,
    BootstrapVariancePenalizedTree,
    RobustPrefixHonestTree,
    CentroidTree,
):
    is_classifier(cls(task="classification")), is_regressor(cls(task="regression"))
    # (False, False) for all five
```

**Fix:** task-aware `__sklearn_tags__` rather than static mixins.

### M3. `classification_criterion="entropy"` is ignored

**File:** `stability_utils.py:1127` — *found by codex*

```python
g = BaseStableTree(classification_criterion="gini", **kw).fit(X, y)
e = BaseStableTree(classification_criterion="entropy", **kw).fit(X, y)
g.tree_ == e.tree_  # True — identical trees
# sklearn gini vs entropy differ on 3 predictions for the same data
```

### M4. `enable_feature_standardization` performs no transformation

**File:** `base_stable_tree.py:537` — *found by codex*

```python
BaseStableTree(enable_feature_standardization=True)._preprocess_features(X)
# returns X unchanged: means [2., 20.], stds [0.82, 8.16]
```

### M5. Predictions depend on feature-column order

**File:** `stability_utils.py:131` — *found by agy and codex*

Consensus votes are tallied in a dict whose insertion order follows column order, and
ties are then broken by first-seen. Permuting columns (and predicting on the permuted
matrix) changes predictions in **6 of 8** random permutations tested:

```python
p1 = RobustPrefixHonestTree(task="classification", random_state=42).fit(X, y).predict(X)
p3 = (
    RobustPrefixHonestTree(task="classification", random_state=42)
    .fit(X[:, cp], y)
    .predict(X[:, cp])
)
np.array_equal(p1, p3)  # False, 5/100 differ
```

Note this contradicts the package's own `enable_deterministic_tiebreaks` default.

### M6. README classification accuracies compare different dataset sets

**File:** `README.md` "Performance Comparison" table

The stable trees raise `"Multi-class classification not yet supported"` on all three
multiclass datasets (`iris`, `wine`, `digits_multiclass`), so their accuracy column is a
mean over **2** datasets while CART's and RandomForest's is a mean over **5**:

```
non-null accuracy count per model (classification)
  BootstrapVariancePenalized   2   0.8073
  LessGreedyHybrid             2   0.9886     <- README: "0.99 ± 0.00"
  CART                         5   0.8870     <- README: "0.89 ± 0.09"
  RandomForest                 5   0.9489
```

`benchmark_results/summary_statistics.csv` nevertheless records `n_datasets = 5` for
every model. The table's most flattering cell — LessGreedyHybrid beating CART by 10
points — is a selection artifact.

Related: the "Avg Variance Reduction" column rewards degenerate models. A tree that
returns one constant has zero prediction variance, which is scored as a ~94% reduction
for BootstrapVariancePenalized (R² = 0.038).

*The rest of that table does reconcile to the CSVs* — R², accuracy and their standard
deviations all match `summary_statistics.csv` to the displayed rounding.

### M7. CentroidTree README table is one dataset presented as an average

**File:** `README.md` "CentroidTree Results" — *found by agy and codex*

The table says "Results from experiments with 30 random seeds across synthetic and real
datasets", but every cell matches `synth_classification_easy` alone:

| Method | README | synth_classification_easy | mean over all 4 clf datasets |
|---|---|---|---|
| CART | 0.85 / 43.4% | 0.8533 / 43.38% | 0.8630 / 39.17% |
| CentroidTree-CART-20 | 0.85 / 43.3% | 0.8467 / 43.29% | 0.8592 / 39.44% |
| CentroidTree-LessGreedy-20 | 0.82 / 42.4% | 0.8218 / 42.36% | 0.7824 / 29.60% |
| LessGreedyHybridTree | 0.82 / 42.2% | 0.8151 / 42.18% | 0.7833 / 30.37% |
| RandomForest-20 | 0.89 / 43.0% | 0.8944 / 42.96% | 0.9066 / 38.58% |

### M8. CHANGELOG's "~3% variance reduction" is 0.2% in the committed data

**File:** `CHANGELOG.md:14` — *found by codex*

`results/centroid_experiment/improvements.csv`, regression, `CentroidTree-CART-20`:
mean `variance_reduction_pct` = **0.2027%**.

---

## Low

- **L1. `CentroidTree.score(..., sample_weight=)` raises** (`centroid_tree.py:278`,
  *found by agy*) — `BaseStableTree.score()` takes no `sample_weight`, so the forward is a
  `TypeError` whenever the base tree is a stable-cart tree. Works with sklearn base trees.
- **L2. `proximity_metric` is silently substituted** (`centroid_tree.py:303`) — asking for
  `"rmse"` on a classification task gives disagreement scores with no warning;
  `rmse` is the constructor default while `task="classification"` is also the default.
- **L3. Fitted attributes created in `__init__`** — `tree_`, `classes_`, `n_classes_`
  exist before `fit`, and no estimator sets `n_features_in_` (*found by agy*).
- **L4. User-input validation uses bare `assert`** (`less_greedy_tree.py:1161-1170`,
  `bootstrap_variance_tree.py:702-711`) — the `split_frac + val_frac + est_frac == 1`
  check disappears under `python -O`. Should be `raise ValueError`.

---

## Raised but not filed

- **Row-order dependence.** All five estimators change predictions when training rows are
  permuted with `random_state` held fixed (25–100% of rows). agy and codex both filed it.
  It is real and reproducible, and `sklearn.tree.DecisionTreeClassifier` *is* row-invariant
  on the same data — but it follows directly from partitioning by array position
  (`train_test_split` on indices), which is the documented honest-partitioning design
  rather than a coding slip. Worth a documented caveat; listing it as a defect alongside
  C1–C3 would dilute them.
- **`check_estimator` failure counts.** codex measured 16 failures for LessGreedy and
  Bootstrap, 38 for RobustPrefix, 3 for Centroid, 15 for Base. Almost all are loud
  API/validation failures already covered by M2/L3, so they are not separate findings.
- **agy's mechanism for H3 and H4** was wrong in both cases (see the corrections inline).
  The defects are real; the explanations were not, which is why each was re-run here.

## What could not be verified

- The benchmark was **not** re-run from scratch; all claim-reconciliation is against the
  committed CSVs in `benchmark_results/` and `results/`.
- No "all regularization off ⇒ ordinary CART" reference exists for LessGreedy or
  RobustPrefix — there is no single master strength parameter — so that limiting-case
  identity could not be tested for them.
- Whether the legacy modules (`less_greedy_tree.py`, `bootstrap_variance_tree.py`,
  `robust_prefix.py`) implement the primitives correctly was not audited in depth. They
  are not what the exported classes use.
- **Independence caveat:** both external auditors ran in the same disposable snapshot.
  agy finished first and wrote its report there, so codex could in principle have seen it.
  codex flagged the stray files itself and said it did not rely on them; its five novel
  findings (C3, H1, H2, M3, M4) appear in no other report.

## Checks that passed

Same-seed determinism (bit-identical, all five estimators, both tasks); `score()` equals a
hand-computed R²/accuracy; `classes_` / `predict_proba` column order / `predict` labels
agree, including string labels; `CentroidTree(n_candidates=1)` exactly matches the seeded
plain base tree; every proximity metric matches a hand recomputation and `selected_index_`
is always `argmin(candidate_scores_)`; all five documented `CentroidTree` attributes exist
across task × metric × base-class grids; the README "Performance Comparison" numbers
reconcile to the CSVs (their *interpretation* is the problem, not their arithmetic).
