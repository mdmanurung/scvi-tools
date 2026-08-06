# Review — .scratch/mrtotalvi-z-vs-totalvi/ — 2026-07-27

**Scope**: whole analysis directory (all files treated as added)
**Files reviewed**: 8 scripts/docs + 6 result TSVs + 1 training log
**Sub-agents run**: 6 (stats-causal, data-pipeline-leakage, bioinformatics, llm-failure-modes,
doc-schema-fidelity, code-quality)

The analysis compares MrTotalVI-LN's `u`/`z` embeddings against stock TOTALVI-LN on two datasets
(schisto 10k human CITE-seq; the RDX-03 `canonical_human` fixture) and questions the validity of
RDX-03's effective-rank collapse gate.

**Headline: three of the analysis's own conclusions do not survive review.** The central claim —
that RDX-03's gate "rejects the more state-informative representation" — rested on a metric that
cannot support it. A denominator-free retest (F1) was run during synthesis and gives a materially
different answer. Two further major findings (F2, F5) invalidate specific supporting claims.

## Key decisions in this analysis

- **Effective-rank metric** — `exp(entropy of covariance eigenspectrum)`, replicated from
  `benchmarks/mrtotalvi/diagnostics.py`. Verified byte-equivalent by two agents. Informational.
- **State-alignment metric** — multivariate η² = `trace(between)/trace(total)`, used as the sole
  quantitative discriminator between "healthy compression" and "information loss". See **F1**.
- **Stock TOTALVI baseline n** — n=1 model reused across three seed rows for the scIB comparison.
  See **F3**; a 3-seed replacement was mid-training at review time (seed 0 has since completed at
  effective rank 17.60, early-stopped at epoch 141).
- **Cell-matching strategy** — every embedding restricted to the shared barcode intersection before
  measurement, and `train_totalvi_baseline.py` matches the *training* set too. Verified sound.
- **Label granularity** — `celltype` (schisto) and `cell_label_l2` (20 states, RDX-03) as the sole
  ground truth. Already caveated in FINDINGS.md.
- **Label supervision asymmetry** — the schisto MrTotalVI models registered `labels_key="celltype"`;
  stock TOTALVI did not. Undisclosed. See **F2**.
- **Tail cutoff** — PC 11–20 as "the dimensions MrTotalVI doesn't span", chosen to mirror the gate's
  0.5×dim split rather than derived from the measured 6.4-unit rank gap. See **F6**.
- **MIN_CELLS = 200** for within-sample donor×timepoint groups. Transparently reported.

## Questions for the analyst

- Is this destined for the RDX-03 reviewer as a gate-validity challenge, or is it internal
  orientation? The bar for F1 differs sharply between those.
- Should the schisto comparison be re-run with `labels_key` omitted (F2), or is the
  supervised-vs-unsupervised framing acceptable if disclosed?
- For the DA use case specifically, is within-state structure the thing you actually need
  preserved? If so, no metric here measures it, and that gap matters more than the rank gap.
- Are donor and timepoint separate factors for your purposes, or is donor×timepoint the unit?
  The "donor" column in the RDX-03 table is actually donor×timepoint (F7).
- Is `cell_label_l2` the right resolution to adjudicate collapse, given 20 states bound
  between-state variation to ≤19 dimensions?

## Findings

### Statistics & causal inference

#### Major

##### F1. Multivariate η² cannot distinguish "gained state signal" from "discarded other variance"
`.scratch/mrtotalvi-z-vs-totalvi/rdx03_collapse_character.py:129-132`
```python
# Sum_pc var_share_pc * eta2_pc == multivariate eta^2: the fraction of the
# embedding's total variance explained by group means. Rotation-invariant,
# so it is not inflated by concentrating variance into few directions.
"var_weighted_eta2": float((etas * share).sum()),
```
**Why it matters here**: the rotation-invariance argument is correct but answers the wrong
objection — it establishes basis-independence *within* one embedding, not comparability *across*
two embeddings with different total variance. η² = trace(between)/trace(total) rises whenever
non-state variance shrinks, with no floor: collapse every cell onto its state centroid and η² → 1
at vanishing rank. Since the compression under investigation is exactly a shrinkage of non-state
variance, the metric is entangled with the effect it is being used to adjudicate. FINDINGS.md's own
reasoning about the donor column (that a construction shrinking the denominator makes the number
near-definitional) applies with equal force to the state column, but the document explicitly exempts
it. This was the sole evidence for the escalation-worthy claim that RDX-03's gate rejects better
representations.
**Fix**: replaced with a denominator-free kNN state-recovery test
(`knn_state_recovery.py`, stratified 5-fold, k=15, on the same sealed held-out cells). Result:

| embedding | eff. rank | macro-F1 raw | macro-F1 z-scored |
|---|---|---|---|
| B1 factual_z (stock TOTALVI) | 18.25 | **0.811** ± 0.005 | 0.799 ± 0.015 |
| B2 factual_z (MrTotalVI) | 7.14 | 0.798 ± 0.013 | **0.843** ± 0.011 |
| B2 u (MrTotalVI) | 5.76 | 0.751 ± 0.012 | 0.753 ± 0.014 |

The correct conclusion is narrower than the one drawn: B2's `z` recovers state *comparably* to B1
despite spanning 2.5× fewer effective dimensions (slightly worse raw, better after z-scoring) —
supporting "compression, not damage" — but it is **not** more state-informative, and B2's `u` at
rank 5.76 is meaningfully **worse** (≈7% relative, >4 seed-SDs). So the gate is not simply invalid:
at the lowest ranks there is real information loss.

#### Minor

##### F3. "Indistinguishable" asserted from an eyeball, against an n=1 arm
`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:23`
```
The `z` vs stock gap is +0.0007 against a seed std of 0.0050 — indistinguishable.
```
**Why it matters here**: no test is run, and the comparison implicitly assumes TOTALVI's unmeasured
seed variance resembles MrTotalVI's. The gap is genuinely tiny and the limitation is disclosed in
the Caveats section, so this is minor — but the in-flight 3-seed baseline should convert it into a
proper two-sample comparison.
**Fix**: finish `train_totalvi_baseline.py` + `compare_with_3seed_baseline.py`, then report a CI on
the mean difference.

##### F6. "Not extra cell-type signal" overstates the tail result
`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:80`
```
So TOTALVI's extra ~6 effective dimensions are **not** extra cell-type signal.
```
**Why it matters here**: in absolute terms TOTALVI's tail carries *more* celltype-associated
variance (29.8% × 0.124 ≈ 3.7%) than MrTotalVI's (14.0% × 0.166 ≈ 2.3%). The correct statement is
weaker per-unit-variance association, not absent signal.
**Fix**: reword to per-unit-variance and cite the absolute shares.

### Bioinformatics

#### Major

##### F2. Undisclosed label supervision confounds every schisto cell-type claim
`schisto_citeseq/.../train_multiseed.py:138-150` (the `mrtotalvi_ln` variant analysed here)
```python
MrTotalVI.setup_anndata(
    adata, ..., labels_key="celltype",
)
model = MrTotalVI(..., u_prior="mog")  # pin: this variant is defined as LN+MoG
```
**Why it matters here**: verified in `_components.py:283` — `resolved_k = n_labels if n_labels > 1`
— so registering `labels_key` makes the MoG mixture count equal the label count, which activates the
label-conditioned logit offset (`u_prior_label_weight` default **10.0**) in `build_u_prior`. The
MrTotalVI models were told each cell's cell type during training; stock TOTALVI
(`train_model.py:230-235`, no `labels_key`) was not. Every schisto cell-type-alignment claim — the
scIB bio-conservation edge and the per-PC η²_celltype tail table — compares a label-supervised model
against an unsupervised one on the label it was supervised with. This could account for some or all
of the reported advantage.
**Fix**: disclose it, and either re-run with `labels_key` omitted or reframe as
supervised-vs-unsupervised. **Scope note**: RDX-03 is unaffected —
`convergence_runner.py:535-542` registers no `labels_key`, verified.

##### F5. `factual_z` is described backwards, invalidating the donor-column dismissal
`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:123-126`
```
B2's `factual_z` is MrTotalVI's counterfactual z, evaluated at a fixed reference
sample — a construction whose explicit purpose is to remove donor variance.
Its lower donor eta^2 (0.036 vs 0.067) is close to definitional, not empirical.
```
**Why it matters here**: `_model.py:899` and `:932` state `give_z=True` returns z "using the cell's
**actual** donor index" — factual means at the observed sample, contrasted with the separate opt-in
`get_counterfactual_latent` API (ADR-0007), which this comparison never calls. Nothing in the
construction mechanically suppresses donor variance, so the lower donor η² is a genuine empirical
result that was wrongly talked down. Ironically this *understated* MrTotalVI.
**Fix**: correct the description and either restore the donor result as empirical or drop the
dismissal.

### Documentation & schema fidelity

#### Major

##### F4. New user-guide text states a parameter is a no-op when it is not
`docs/user_guide/models/mr_multimodal.md` (section added this session)
```
`init_prior_from_data` and `freeze_prior_after_init` both default to `False`
and are ignored under `u_prior="mog"`.
```
**Why it matters here**: verified — `_model.py:228` gates only `init_prior_from_data` to
`vamp`; `freeze_prior_after_init` is passed through unconditionally at `:274`, and
`_components.py:243-245` documents that under MoG it freezes `u_prior_means`/`u_prior_scales` at
their **random initialisation**. A user following this doc could freeze an uninitialised MoG prior
believing the flag is inert. This is shipped user-facing documentation, not a scratch note.
**Fix**: state that `init_prior_from_data` is vamp-only, while `freeze_prior_after_init` applies to
both priors and under MoG freezes randomly-initialised means/scales.

#### Minor

##### F8. "Reproduce" section omits 3 of the 4 scripts behind the tables
`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md` (Reproduce section) names only
`effective_rank_matched.py`, though `rank_decomposition.py`, `what_the_extra_dims_encode.py` and
`rdx03_collapse_character.py` produce most of the reported numbers.
**Fix**: list all scripts in run order.

##### F9. Docstring cites a function that does not exist
`.scratch/mrtotalvi-z-vs-totalvi/effective_rank_check.py:36` refers to
`diagnostics._summarize_embedding`; the real function is `latent_diagnostics`. Math verified correct.

##### F10. Residual-variance triple is sorted, not in seed order
`FINDINGS.md:49` gives "0.47% / 2.86% / 6.82%" ascending, while every other list in the document is
s0/s1/s2. Values correct, order misleading.

### Data pipeline & leakage

No major findings. Independently verified: all latent TSVs have unique duplicate-free barcode
indices; MrTotalVI's 97,954 cells are an exact subset of TOTALVI's 125,706; the `REFERENCE_INDEX`
trick in `train_totalvi_baseline.py` reproduces the pre-existing
`modeling_keep_known_timepoint == True` filter exactly (set-equality confirmed); RDX-03 train/
validation indices are asserted disjoint and complete upstream, and both B1 and B2 use identical
`external_indexing`, so the held-out boundary is intact.

#### Minor

##### F7. "donor" column is actually donor × timepoint
`.scratch/mrtotalvi-z-vs-totalvi/rdx03_collapse_character.py:70`
```python
for name, key in (("donor", fixture.sample_key), ("batch", fixture.technical_batch_key)):
```
**Why it matters here**: `fixture.sample_key` is literally `"donor_timepoint"` for this fixture
(`convergence_runner.py:361`), so the reported "donor η²" mixes donor and timepoint variance. The
argument survives, the label overstates precision.
**Fix**: rename the column `donor_timepoint`.

### LLM coding antipatterns

No major findings. The reviewer independently recomputed nearly every number in FINDINGS.md against
the TSVs and source JSONs and found matches to 2–3 significant figures, including the residual
variance shares, within-sample per-seed ranks, the RDX-03 ELBO-descent table, and the
`lr-Adam=0.004` vs configured `lr=1e-3` distinction. The unexecuted
`compare_with_3seed_baseline.py` was traced and no blocking bug found. `train_totalvi_baseline.py`'s
config was verified identical to `train_multiseed.py:54-66` and `train_model.py:225-242`.

### Code quality

#### Minor

##### F11. Volatile run path will silently break
`.scratch/mrtotalvi-z-vs-totalvi/rdx03_collapse_character.py:26-30` hardcodes a run directory whose
name embeds the `.tmp-1oncvmjo` suffix that the run's own publication step strips on sealing. Once
sealed, `path.exists()` returns False and the script dies later with an unrelated `KeyError` on an
empty DataFrame rather than a clear "run has sealed, update the path".
**Fix**: glob for both the `.tmp-*` and sealed forms, and fail loudly if neither resolves.

##### F12. `effective_rank` duplicated 6×, `eta_squared` 3×
No shared home. `benchmarks/mrtotalvi/diagnostics.py` is the authoritative implementation but is
inside the byte-verified frozen tree, so the fix is a small local helper module in the scratch dir —
**not** an edit or restructuring of `diagnostics.py`.

##### F13. Dead code
`compare_with_3seed_baseline.py:116,138` computes a full groupby/agg then discards it via `_ = agg`.

##### F14. Duplicated magic numbers
`MIN_CELLS = 200` defined twice; the PC head/tail split (11 of 20) hardcoded two different ways
(literal slice vs function default) rather than derived from `n_dim`.

## What was checked but is fine

- **Data pipeline & leakage**: every barcode join, the train/validation boundary, and the
  training-set matching verified correct against the data on disk, not just read.
- **LLM antipatterns**: no hallucinated APIs, no silent-failure fallbacks, no untraceable numbers;
  claimed-identical configs verified line by line.
- **Statistics**: rotation-invariance of multivariate η² verified algebraically and numerically
  (the defect in F1 is a different property); no multiple-comparison issue since nothing is a
  hypothesis test; the 37% rank gap survives the n=1 caveat on effect-size grounds and is
  independently reproduced by the within-sample recomputation.
- **Bioinformatics**: TOTALVI training protocol (counts layer, protein obsm, batch key) matches
  standard practice and the config it claims to replicate; label-coarseness and within-state
  caveats already honestly stated.

## Notes

- **F1, F2 and F5 share one root cause**: each is a place where a claim about MrTotalVI's
  representational advantage rests on something other than a direct, denominator-free, like-for-like
  measurement. F1 and F2 inflate the advantage; F5 deflates it. Fixing all three means the
  writeup's conclusions move in both directions.
- **F4 is the only finding touching shipped artifacts** and should be fixed first — it is
  user-facing documentation asserting a parameter is inert when it silently freezes a randomly
  initialised prior. Related to already-tracked P2-006 but distinct.
- **Frozen-tree constraint**: `src/scvi/**/*.py` and `benchmarks/mrtotalvi/` are byte-verified by a
  live 48-fit experiment; every fix above is confined to `.scratch/` and `docs/`.

## Remediation applied 2026-07-27

| Finding | Status |
|---------|--------|
| F1 η² confound | **Fixed** — `knn_state_recovery.py` added and run; FINDINGS.md conclusion rewritten to the narrower, supported claim |
| F2 label supervision | **Disclosed** — prominent warning block added to the schisto section, with the RDX-03 scope exclusion |
| F4 docs no-op claim | **Fixed** — `mr_multimodal.md` now states `freeze_prior_after_init` applies to MoG and freezes a random init |
| F5 factual_z description | **Fixed** — retraction added; donor result restored as empirical |
| F6 tail overstatement | **Fixed** — reworded to per-unit-variance with absolute shares |
| F7 donor mislabel | **Fixed** — column renamed `donor_timepoint` in FINDINGS.md |
| F8 Reproduce section | **Fixed** — all six scripts listed in run order, with the `LD_LIBRARY_PATH` requirement |
| F9 docstring reference | **Fixed** — now cites `latent_diagnostics` |
| F10 seed ordering | **Fixed** — residual triple reordered to s0/s1/s2 |
| F11 volatile run path | **Documented** — noted in Reproduce; not yet code-fixed |
| F3, F12, F13, F14 | **Open** — F3 resolves when the 3-seed baseline finishes (seed 0 done: rank 17.60) |

Logged as **L-094** (variance-fraction metrics cannot compare embeddings with different total
variance) and **L-095** (`labels_key` silently makes the MoG prior label-supervised).
