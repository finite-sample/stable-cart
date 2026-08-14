# Prediction stability and explanation stability are not the same thing

A pre-analysis plan. Nothing in the comparison runs until this is frozen and reviewed.

## Context

The tree-stability literature — and every claim in this repo — measures **prediction**
stability: how much do predictions move when the training data is perturbed. But a single
decision tree is chosen *because* it is readable. Its deliverable is the explanation. Nobody
here has measured whether the explanation holds still.

The measurements from this session point that way:

- 89–98% of prediction instability was structure choice, 2–11% leaf noise
- yet the **root feature was chosen identically 100% of the time**, while levels 2–3 agreed
  only 5–25% — the tree churns in a way that barely moves predictions
- exact optimization (GOSDT) removed 10–24% of instability where bagging removed 62–92%

**These are not results, and the plan must not treat them as ones.** Each is a measurement at
particular settings — four regression datasets at `min_samples_leaf=20` for the decomposition,
five binary-classification datasets for GOSDT. The leaf share already moves across those four
(diabetes 10–11%, friedman1 4–6%), which is a 2× spread and a warning that the quantity is a
**function of the DGP and the fit settings, not a constant**. Quoting "95% structural" as a
finding would be exactly the error this repo already made once.

So the study's job is to predict the function rather than report an average. There is a clean
handle for the leaf component: a leaf holding `m` samples estimates its mean with variance
σ²/m, so with `L` leaves and `n` training points the leaf-only instability should go as

    leaf-only instability  ≈  σ² · L / n

which is directly testable by sweeping σ and n, and implies the leaf share is *large* in
noisy, small-n, many-leaf regimes and *small* in the opposite one. The existing numbers are
consistent with it — friedman1 at n=800 → n=3000 halved the leaf component (0.258 → 0.122)
while the structure component fell much less — but that is one comparison, not a test.

The greedy-versus-optimal gap has an analogous conditioning variable: the **split margin**. If
the best and second-best splits are well separated, greedy and optimal agree by construction
and the gap must vanish; the gap can only open where splits are close. That variable is a knob
in simulation, and predicting the gap from it is the interesting claim.

**Claim:** prediction stability and explanation stability dissociate. A method can stabilize
one while leaving the other untouched, the two metrics can rank methods differently, and the
field has been optimizing the one that is not the product.

**Why simulation is primary:** only there do we know the true tree, can draw genuinely
independent training samples instead of bootstrap approximations, and can construct data that
is *ambiguous by construction* — which is what turns the dissociation from an observation into
a proof. Real datasets appear as a confirmation section.

## Estimand

For estimator `m`, data-generating process `d`, sample size `n`:

> The difference in **explanation instability** between estimator `m` and a plain CART,
> over trees fitted to independent samples of size `n` drawn from `d`, evaluated at **matched
> out-of-sample accuracy**, aggregated as the median over independent DGP draws and reported
> as a paired interval.

with the same quantity defined for **prediction instability**, and the contrast of the two
being the object of the paper.

Matched accuracy is not a detail. Every comparison in this repo so far has been confounded by
it — a method that is simply more regularised looks more stable. Each method is swept over its
regularisation path and instabilities are compared where accuracy is equal.

## Metrics

**Prediction instability** — `stable_cart.bootstrap_instability` (already shipped and tested),
plus a true-resampling variant that draws fresh samples from the DGP rather than bootstrapping.

**Explanation instability** — primary: mean pairwise Jaccard distance between the multisets of
split features at depth ≤ 3, over pairs of independently fitted trees. Secondary: root-feature
agreement rate; path agreement (do two fits route the same test point through the same feature
sequence); threshold-aware Jaccard. Prior art for tree distances is Banerjee et al. (2012);
these are chosen for interpretability rather than novelty.

**Accuracy** — R² or accuracy on a fixed held-out sample from the same DGP.

**Structure recovery** (simulation only) — did the fit find the true splits.

## Pre-specified hypotheses

Signs and magnitude ranges committed in advance. An effect we cannot bound in advance is one
we cannot be surprised by.

| # | hypothesis | predicted | falsified if |
|---|---|---|---|
| H1 | **Leaf-only instability follows σ²·L/n.** Sweep σ ∈ {0.1,1,3,10}, n ∈ {250,1000,4000,16000}, `min_samples_leaf` ∈ {5,20,100} | log-log slope −1 in n and +1 in σ², within [−1.2,−0.8] and [0.8,1.2] | slopes outside those bands — the leaf component is not simple sampling noise and the decomposition needs rethinking |
| H1b | **The leaf share is not a constant.** There exist settings in the sweep where it exceeds 40% and settings where it is under 5% | both regimes observed | leaf share stays in a narrow band — then it *is* roughly a constant, and the "structure dominates" framing generalises more than expected |
| H2 | **Dissociation.** On DGP-B, prediction instability is low while explanation instability is high | pred < 25% of DGP-C level; expl > 75% of DGP-C level | both metrics move together — the central claim fails |
| H3 | **Rank reversal.** The ordering of methods differs between the two metrics on at least one DGP | ≥1 reversal | orderings identical everywhere — dissociation is real but useless |
| H4 | **The greedy-vs-optimal gap is governed by the split margin δ**, not a constant. Sweep δ from well-separated to near-tied | gap → 0 as δ grows (greedy = optimal by construction); gap largest at small δ but still below bagging's reduction at every δ | the gap is flat in δ — then something other than search ambiguity drives it and the mechanism story is wrong. If the gap ever exceeds bagging's, the ceiling argument fails outright |
| H5 | Medoid selection from a bagged pool improves **explanation** stability more than prediction stability | expl gain > pred gain | no differential — Banerjee's interpretability motivation is unsupported |
| H6 | Prefix stability selection reduces explanation instability | 15–40% at ≤ 2 accuracy points | below 15% or accuracy cost > 2 points — drop the mechanism |
| H7 | Instability decays with n; the leaf component decays faster than the structure component | leaf ~n⁻¹, structure slower | structure decays as fast — it is ordinary sampling noise, not a selection problem |

## Data-generating processes, chosen as controls

| DGP | construction | purpose | pre-specified expectation |
|---|---|---|---|
| **A** `tree_separated` | true depth-3 tree, well-separated splits, low noise | **positive control** | both instabilities → 0 as n grows. If explanation instability does not, our *metric* is broken, not the method |
| **B** `tree_tied` | true tree splits on X1; X2 is an exchangeable near-copy | **the dissociation case** | predictions stable (both features induce the same partition), explanation unstable (the tree flips between them). Only constructible in simulation |
| **C** `smooth` | Friedman1 — no true tree, many near-equivalent structures | the realistic case | both moderately unstable |
| **D** `noise_only` | y ⟂ X | **placebo** | accuracy 0 for every method; anything scoring "stable" here is degenerate. This is precisely the failure that produced the +55% claim in this repo, and it must be a standing control |
| **E** `correlated` | features at ρ = 0.9 | realistic version of B | dissociation, attenuated |
| **F** `margin_sweep` | true tree whose best and second-best root splits differ by a controlled margin δ | isolates the variable behind H4 | greedy-vs-optimal gap shrinks to zero as δ grows |

Every DGP is instantiated across the **σ × n × leaf-size grid** of H1, because the point of the
study is that these quantities are functions of the regime. Any number reported without its
regime is not a result.

DGP-D earns its place: a constant predictor has zero prediction variance, which is how a
broken estimator scored as the most stable method in the shipped benchmark. Every table
reports instability *beside* accuracy so that failure cannot recur silently.

## Methods, and what we are dropping

Carried forward — each has either evidence or a clean mechanism:

- **CART** (baseline) and **cost-complexity pruned CART** (`ccp_alpha` swept) — currently the
  most reliable stabiliser, 9/14 wins
- **Bagging / RF** — the ceiling reference; not a single tree, included to bound what is achievable
- **GOSDT** — exact-search reference, classification subset only (binarized features)
- **Prefix stability selection** — the one untested mechanism aimed at the dominant component,
  and it works by averaging the *split decision* over resamples rather than by searching better
- **Medoid / centroid selection from a bagged pool** — dead as a prediction-stability device
  (C6), but H5 tests it in the frame its own literature actually claims
- **Surviving stable-cart estimators**, post-C1

Dropped, with the measurement that killed them: distillation from a forest (0–9%), threshold
snapping to a quantile grid (−7% to +9%), leaf smoothing as a stability device (addresses
2–11% of the variance by construction).

## Inference

- **Replication unit is an independent DGP draw**, not a bootstrap resample. Paired across
  methods within a draw, since all methods see the same data.
- Report paired intervals via bootstrap over draws. **Never a bare mean of a ratio** — this
  repo has been misled twice by exactly that (the README's +55.4%, and my own −16%/−108%).
- **The words "significant" and "not significant" do not appear.** Every result is an interval
  and what it rules out.
- **Design analysis before running:** pilot with R=20 draws to estimate the variance of the
  paired difference, then choose R so the interval on a 10% instability difference is tight
  enough to decide H2/H3. Report the MDE. If the required R is infeasible, say so and reduce
  the claim rather than run underpowered.
- **Bootstrap-vs-truth check:** with independent draws available, quantify the bias of the
  bootstrap instability estimate. If it is biased, that affects every published number that
  uses it, including this repo's benchmark.

## Freeze and audit — before any comparison runs

1. Commit `pap.md` (this plan, in the repo) and tag it. No method-vs-method number is computed
   before that commit exists.
2. **Self-audit** with `audit-analysis` in own mode over the design.
3. **Two independent readers**, different model families, on the frozen PAP:

   ```bash
   codex exec --sandbox read-only "<prompt>"
   agy --mode plan --dangerously-skip-permissions --print-timeout 45m \
     --model gemini-3.1-pro-high -p "<prompt>"
   ```

   Ask each for exactly three things: **the strongest rival explanation for the dissociation,
   the three assumptions most likely to be wrong, and anything methodologically out of date.**
   Not a code review — that reliably returns style notes instead of the design flaw.
4. **Re-derive every finding they raise before acting on it.** In this session roughly half of
   the delegated findings were right about the defect and wrong about the mechanism.
5. Only then run the pre-specified comparison. Anything found afterwards is labelled
   exploratory, and deviations from the PAP get a table: what changed, why, and what the
   pre-specified version showed.

## Build

- `pap.md` — this plan, committed and tagged first
- `stable_cart/explanation_stability.py` — the new metrics (Jaccard, root agreement, path
  agreement), shipped and tested like `bootstrap_instability` was
- `experiments/dgps.py` — the five DGPs with known ground truth
- `experiments/stability_study.py` — the driver: DGP × method × n × metric, independent draws
- `experiments/frontier.py` — the regularisation sweeps that make matched-accuracy possible
- Reuse: `stable_cart.bootstrap_instability`, the paired-pool harness and `pool_spread` guard in
  `experiments/selection_rules_experiment.py`, `experiments/optimal_tree_premise.py` for the
  GOSDT arm
- Prerequisites already queued: **1b** — fix H1 (consensus threshold overwritten by the split
  threshold) and H2 (`consensus_support` can exceed 1.0), since prefix stability selection is
  built on that code and would inherit both

## Verification

- Positive control A must show both instabilities → 0 with n. If not, stop: the metric is wrong.
- Placebo D must show every method at chance accuracy, and any method with low instability there
  is reported as degenerate rather than stable.
- Negative control: prefix stability selection at π=0 must reproduce plain greedy CART exactly.
- Every explanation metric gets a unit test with a hand-constructed pair of trees whose distance
  is known by inspection.
- Sanity check that survived contact with reality this session: the full ensemble must be the
  most stable row in every prediction-instability table. It is what caught the degenerate pool.
- `uv run pytest`, `uv run ruff check .`, `uv run pyright`, `preen check` stay clean.

## What would change the conclusions

- H2 failing is fatal to the paper's framing. The fallback is **not** "the ceiling result",
  which as noted above is a regime-dependent measurement rather than a finding; the fallback is
  H1/H4 — the laws governing when each component dominates — which survive H2 either way.
- The level-agreement numbers come from 4 regression datasets at one depth setting. If the
  100%-root / 5–25%-deep pattern does not replicate across the DGPs and depths, the motivating
  fact is weaker than stated and the introduction has to change.
- If H1's σ²·L/n law fits badly, the whole decomposition framing is on sand and the honest paper
  is a smaller one about the metrics.
- If the bootstrap estimate turns out unbiased and the true-resampling numbers match it, the
  methodological contribution shrinks to the metrics themselves.
