# A variance budget for decision-tree instability

> **STATUS: ABANDONED as a paper, 2026-08-14.** All three claims failed review. The
> decomposition was published five months before this plan was written — Mustafa Cavus,
> *Decomposing Observational Multiplicity in Decision Trees: Leaf and Structural Regret*,
> [arXiv:2603.11701](https://arxiv.org/abs/2603.11701), 12 March 2026 — which defines leaf
> regret as within-leaf finite-sample variability and structural regret as variability from
> tree-structure instability, with a formal decomposition and statistical guarantees. Geurts
> and Wehenkel (ECML 2000) are a direct antecedent, separating threshold, structure and leaf
> estimation variance and re-estimating leaves against a fixed structure. Verified from
> source.
>
> Independently, the implementation was wrong in ways that matter: the reported Monte Carlo
> standard errors were computed across 4,000 evaluation points sharing the same fitted trees
> and understate the true run-to-run variability by **11.7×** on the leaf component (measured
> over 20 reruns of one cell). Re-running a single cell gives leaf shares from **27.4% to
> 52.4%**, so the per-cell numbers cannot support the distinctions drawn from them. The
> plug-in decomposition is also biased at S=20, L=8 (`ddof=0` within-structure variance;
> between-structure variance containing ~1/L of the leaf variance), and the estimand matches
> an *honest* tree rather than ordinary CART, where the same outcomes choose the splits and
> set the leaf means.
>
> Claim 2 was never tested: matching `max_leaf_nodes` to GOSDT's output collapsed both arms
> to stumps, and the structural metric discarded thresholds, so "identical structure" was
> close to vacuous. Claim 3 was an unbounded universal negative contradicted by verified
> literature (Geurts and Wehenkel on threshold averaging; Bertsimas, Dunn and Paskov on
> robust optimisation).
>
> What survives is the package audit in `AUDIT.md` and the fixes it produced, not a paper.
> The remainder of this document is kept as the record of what was planned and why it failed.

Pre-analysis plan, **v3**. v1 frozen at `pap-v1` (`bc9b35d`); v2 superseded before execution.
The deviations table records every change. No method-versus-method number has been computed.

## What changed, and why the headline moved

v1 and v2 claimed that prediction stability and explanation stability dissociate. **That claim
is nineteen years old.** Verified from the primary source — Dwyer and Holte, *Decision Tree
Instability and Active Learning*, ECML 2007:

> "two types of stability are examined: semantic and structural stability… For a learner to be
> structurally stable, a stronger condition must be satisfied, namely, the hypotheses that it
> creates from closely related data sets must be **syntactically similar**. Thus, structural
> stability is a sufficient condition for semantic stability, **but the converse is not true**."

They also propose a partition-based "region stability" measure, anticipating the ARI metric v2
added, and note that earlier measures (Discrepant, Common) collapse when trees differ at the
root. Wang et al. (2018) restate the same distinction with region compatibility.

So the dissociation is **setup, not contribution**. What our measurements support and the
literature does not already contain:

1. **A variance budget.** Decompose tree prediction instability into a leaf-estimation
   component and a structure-selection component, and characterise which dominates as a
   function of noise, sample size and leaf size. The existing literature establishes that trees
   are unstable; it does not say *which part* of the tree is responsible, or when.
2. **An attribution of structure churn to search.** How much of the structure component is the
   greedy heuristic's fault, measured against an exact solver, versus intrinsic to estimating a
   discrete structure from a finite sample. Pilot: exact search recovered 10–24% where bagging
   recovered 62–92%.
3. **A ceiling, with its mechanism.** The consequence for method design: which stabilisation
   mechanisms can possibly work, and how much they can possibly buy.
4. **Stability metrics are gameable by degenerate models**, with a documented instance: a
   published package scored a constant predictor as its most stable method, because a constant
   has zero prediction variance. This is the tree analogue of Yeh et al. (NeurIPS 2019) showing
   that optimising explanation sensitivity alone favours a vacuous constant explanation.

## Estimand

For estimator `m`, data-generating process `d`, sample size `n`, noise `σ`:

> The share of `m`'s prediction instability attributable to leaf estimation rather than
> structure selection, over trees fitted to independent samples from `d`, with instability
> measured at fixed evaluation points and normalised by `Var(y)`; reported as a median over
> independent draws with a Monte Carlo standard error.

and, for the attribution:

> The reduction in structure-selection instability obtained by replacing greedy search with
> exact optimisation, as a function of the split margin δ, at matched out-of-sample accuracy.

## Metrics — established ones, not invented ones

| quantity | measure | source |
|---|---|---|
| prediction (semantic) instability | pointwise prediction variance across independent fits, **normalised by `Var(y)`** | agreement-style, cf. Dwyer & Holte |
| feature-choice stability | **Nogueira's Φ̂** — the only measure satisfying all five properties including correction for chance | Nogueira, Sechidis & Brown, JMLR 18 (2018) |
| partition (region) stability | ARI between induced leaf partitions on a fixed evaluation set | region stability, Dwyer & Holte 2007; co-clustering distance, Banerjee et al. 2012 |
| structural distance | Banerjee's covariate-use and terminal-node co-clustering distances | Banerjee et al., Stat Med 2012 |
| **fidelity** | recovery of the true splits **modulo equivalent representations** (aliased features count as recovered) | required by Yeh et al. 2019 — stability without fidelity is satisfied by a constant |

Raw multiset-Jaccard is **demoted to a sensitivity analysis**: verified from JMLR 18 that it is
not chance-corrected and varies with the number of features selected.

The fidelity column is not optional. A stump that always tests the same feature is perfectly
stable and useless; the degenerate-predictor bug that motivated this study is the same failure.
**No stability number is reported without fidelity and accuracy beside it.**

## Pre-specified hypotheses

Ordinal where the metric is a ratio. Pilot-derived percentage bands were miscalibrated.

| # | hypothesis | predicted | falsified if |
|---|---|---|---|
| **H1** | The leaf component decays **faster in n** than the structure component | leaf strictly faster on ≥ 4 of 6 DGPs | structure decays as fast — instability is ordinary sampling noise, not a selection problem, and the budget framing collapses |
| **H1b** | The leaf **share** is regime-dependent | ∃ grid settings with share > 40% **and** ∃ with share < 5% | share stays in a narrow band — a *stronger* result, reported as such |
| H1c | descriptive: log-log slopes of leaf instability in n and σ² | near −1 and +1 | reported either way; exact `σ²L/n` is not expected to hold, since leaf membership is itself estimated and the numerator is `Var(Y│leaf)`, not `σ²` |
| **H2** | The greedy-vs-exact gap is **decreasing in the split margin δ**, and below bagging's reduction at every δ | monotone decreasing; gap < bagging at all δ | flat in δ — the ambiguity mechanism is wrong. Gap ≥ bagging anywhere — the ceiling claim fails |
| H3 | Method orderings differ between prediction and structural stability on ≥ 1 DGP, **using chance-corrected measures** | ≥ 1 reversal outside its interval | identical orderings — the 2007 distinction has no consequence for method choice, which is worth reporting |
| H4 | Medoid selection from a bagged pool improves **structural** stability more than prediction stability | structural gain > prediction gain | no differential — Banerjee's interpretability motivation is unsupported on its own metric |
| H5 | Prefix stability selection reduces structural instability versus CART at matched accuracy **and matched fidelity** | strictly lower on ≥ 4 of 6 DGPs | not lower on a majority — drop the mechanism |

## DGPs

| DGP | construction | role |
|---|---|---|
| **A** `tree_separated` | true depth-3 tree, well-separated splits | **positive control**: all instabilities → 0 with n, else the metric is broken |
| **B** `aliased` | true tree on X0, X1 an exact copy | representational non-identifiability; fidelity is scored modulo equivalence here |
| **C** `smooth` | linear / Friedman, no true tree | approximation ambiguity |
| **D** `noise_only` | y ⟂ X | **placebo**: chance accuracy for all; anything "stable" here is degenerate |
| **E** `correlated` | ρ = 0.9 | realistic B |
| **F** `margin_sweep` | controlled margin δ between best and second-best root split | isolates H2's mechanism |

Instantiated over σ ∈ {0.1, 1, 3, 10} × n ∈ {250, 1000, 4000, 16000} × `min_samples_leaf` ∈
{5, 20, 100}. **A number without its regime is not a result.**

## Inference — Monte Carlo practice

Following Morris, White & Crowther (Stat Med 2019):

- Data-generating mechanism and performance estimands defined explicitly (above).
- **Repetitions chosen from required Monte Carlo precision**, not convention: pilot at R=20,
  compute the MC SE of each estimand, then set R so the SE is small relative to the effect the
  hypothesis must resolve.
- **A Monte Carlo SE reported for every performance measure.** No point estimate without one.
- Replication unit is an independent training draw from a fixed DGP; these are Monte Carlo
  repetitions, **not** independent DGPs, so generalisation claims across DGP families are made
  by replicating across the parameter grid, not by pooling draws.
- Exact empirical quantiles of paired differences. No bootstrap where independent draws exist.
- No significance language: every result is an interval and what it rules out.
- **Bootstrap-vs-truth check:** quantify the bias of bootstrap instability against
  independent-draw instability, since every published number in this area uses the former.

## Matched accuracy — and its limits

Both reviewers flagged that equal accuracy does not imply equal regularisation: a shallow and a
deep tree can share held-out accuracy while differing in how many opportunities they have to
churn. Therefore:

- match on accuracy **and** report complexity (leaf count, depth) alongside;
- present the accuracy–complexity–stability surface, not a single matched point;
- require rank conclusions to hold **across the high-accuracy region**, not at one crossing;
- use separate samples for matching and evaluation.

## Build

- `stable_cart/explanation_stability.py` — Jaccard, root agreement, path agreement **(shipped, 9 tests)**; add Nogueira Φ̂, partition ARI, Banerjee co-clustering
- `stable_cart/evaluation.py` — `bootstrap_instability` **(shipped, 6 tests)**; add normalised and independent-draw variants
- `experiments/dgps.py` — six DGPs with ground truth and a recovery oracle
- `experiments/variance_budget.py` — the headline study: decomposition over the σ × n × leaf grid
- `experiments/margin_study.py` — H2, reusing `experiments/optimal_tree_premise.py` for the GOSDT arm
- `experiments/frontier.py` — accuracy–complexity–stability surfaces
- Prerequisite: fix the consensus defects (AUDIT H1, H2) before prefix stability selection

## Verification

- Positive control A: all instabilities → 0 with n, else stop.
- Placebo D: chance accuracy everywhere; low instability there reported as degeneracy.
- Negative control: prefix stability selection at π=0 reproduces greedy CART exactly.
- Every metric unit-tested against a hand-constructed case with a known value.
- The full ensemble must be the most stable row in every prediction-instability table.
- `uv run pytest`, `ruff check`, `pyright`, `preen check` clean.

## Deviations

| # | change | source |
|---|---|---|
| D1 | Prediction instability normalised by `Var(y)` | own design check; neither reviewer raised it |
| D2 | Partition/region stability added as a first-class metric | agy; independently recommended by codex |
| D3 | Ratio hypotheses made ordinal | agy |
| D4 | Exact quantiles replace bootstrap over draws | agy |
| D5 | `σ²L/n` demoted from gate to description | agy and codex; codex supplied the corrected form `Σ p_ℓ Var(Y│ℓ) E(1/N_ℓ)` |
| **D6** | **Headline changed from "dissociation exists" to the variance budget** | codex, verified against Dwyer & Holte 2007 — the dissociation is a known result and cannot be the contribution |
| **D7** | **Jaccard demoted; Nogueira Φ̂ promoted** | codex, verified against JMLR 18 (2018): Jaccard is not chance-corrected |
| **D8** | **Fidelity required beside every stability number** | codex, via Yeh et al. 2019 — stability alone is maximised by a vacuous constant |
| D9 | Matched accuracy supplemented by complexity matching and a surface | both reviewers |
| D10 | Monte Carlo SEs and precision-driven R | codex, via Morris, White & Crowther 2019 |

Provenance: `agy` (gemini-3.1-pro-high) and `codex` reviewed the frozen v1 independently. agy's
central criticism — that the dissociation would be an aliasing artifact with the partition
unchanged — was **not supported** when I checked it (partition ARI 0.354, not ≈1.0; root
agreement 1.00, so the alias flip it requires does not occur). Codex's central criticism was
supported and is fatal to v1's framing; both of its load-bearing citations were verified from
the primary PDFs rather than accepted.
