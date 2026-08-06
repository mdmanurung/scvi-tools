# Review — .scratch/mrtotalvi-z-vs-totalvi/ (second pass) — 2026-07-28

**Scope**: whole analysis directory; emphasis on the 8 files added since the 2026-07-27 pass
**Files reviewed**: 15 scripts + FINDINGS.md + 12 TSV/log/PNG artifacts
**Sub-agents run**: 6
**First pass**: `.living/outputs/reviews/2026-07-27-mrtotalvi-z-vs-totalvi.md`

## First-pass fixes: all held

F1 (η² retraction), F2 (label-supervision disclosure), F4 (`mr_multimodal.md` parameter claim,
re-verified against `_model.py:228,274` and `_components.py:299-303`), F5, F6, F7, F9, F10 all
verified still correct. F11 (`resolve_run_dir()`) works and fails loudly. F13 dead code gone.
F14 partially — `MIN_CELLS` centralised, tail split now derived from `n_dim`.

**F8 regressed** — the Reproduce section was fixed for six scripts and this session added six more
without updating it. See F6 below.

## Key decisions in this analysis

- **Timepoint-signal operationalisation** — global pooled kNN purity at k=15 vs a Simpson
  baseline, with no per-celltype or same-sample decomposition. **See F1** — this is the evidentiary
  basis for a project-wide integration decision.
- **Multigrate likelihood** — `losses=["nb","nb"]` with no `size_factor_key`. **See F2.**
- **Best-F1-across-resolutions** as the fair comparator (the L-097 fix). Applied symmetrically to
  both arms; cluster counts per resolution are near-identical between arms, so the
  multiple-comparison opportunity is symmetric. Sound, with one caveat (F7).
- **Seed discovery** over `range(6)` rather than a fixed (0,1,2), because Multigrate's arms have
  different surviving seed sets. Correct, but the resulting n asymmetry isn't carried into the
  tables (F8).
- **Multigrate hyperparameters left at published defaults** — documented rationale against
  candidate-specific retuning. Sound.

## Questions for the analyst

- Is the "infection response is compositional" claim load-bearing for the project, or incidental?
  F1 changes its evidence but not its direction — that matters only if the claim gets published.
- Should Multigrate remain a headline comparator given F2 makes its protein likelihood
  non-comparable, or be demoted to a sanity check?
- Are the 10 donors biological replicates for your purposes? Every cohort-level claim here is
  cell-weighted, not donor-weighted.
- Is `.scratch/` the final home for this, or does it become a manuscript figure? Several minors
  (F7–F12) only matter under publication.

## Findings

### Bioinformatics

#### Major

##### F1. The "compositional, not state-defining" claim rests on a statistic inflated by same-sample leakage — and blind to rare states
`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md` (Integrating on sample section)
```
Timepoint ... sits at **0.287–0.322 against a 0.258 random baseline in every arm, including `z`**.
Cell-level neighbourhood structure barely tracks infection timepoint at all
```
**Why it matters here**: same-donor×timepoint neighbours are *trivially* same-timepoint, and
`knnP_dtp` (0.077–0.154, measured one row up) is not subtracted. Decomposing
`(knnP_time − knnP_dtp)/(1 − knnP_dtp)` gives cross-sample timepoint purity of **0.197–0.232 in
every arm — below the 0.2575 baseline, not above it** (verified). Separately, a global pooled
statistic at k=15 can only resolve a state above roughly 5% of cells; every rare population this
document treats seriously (Plasma 0.13%, Plasmablasts 0.06%, Late erythroid 0.04%, DC1 0.01%) is
far below that floor, so the metric cannot see the populations most likely to carry a
timepoint-specific state. The corrected numbers make the compositional reading *stronger*, but the
stated evidence was not valid.
**Fix**: report the decomposed cross-sample purity; add the detection-floor caveat; run
`knn_purity(timepoint)` grouped by celltype to test at the resolution where a state would appear.

##### F2. Multigrate's protein likelihood is scaled by an RNA-derived size factor — undisclosed confound
`.scratch/mrtotalvi-z-vs-totalvi/train_multigrate.py:104-114`
```python
MultiVAE.setup_anndata(work, rna_indices_end=rna_end, categorical_covariate_keys=[args.integrate_on])
model = MultiVAE(work, integrate_on=args.integrate_on, z_dim=Z_DIM, losses=["nb", "nb"])
```
**Why it matters here**: verified in the installed package —
`multigrate/utils/_utils.py:35-41` derives `size_factors` from **RNA counts only**, and
`multigrate/module/_multivae_torch.py:672-679` applies that same tensor to *every* modality's NB
mean, protein included. TOTALVI/MrTotalVI never tie protein to an RNA library size — they use
`NegativeBinomialMixture` with learned background/foreground. So a cell's ADT likelihood is
anchored to its transcriptional depth, which correlates with cell type and activation. Multigrate
"wins on every within-run measure" in this document; some of that could be this confound. Direction
unknown without an ablation, but the confound is verified, not speculative.
**Fix**: disclose alongside Multigrate's other caveats (NaN divergence, donor/batch collinearity);
it is a tool limitation, not a fixable bug here.

### Documentation & schema fidelity

#### Major

##### F3. Donor×timepoint random baseline is wrong (0.025 → 0.032)
`FINDINGS.md` (Integrating on sample section): "random-baseline kNN purity is **0.025** for
donor×timepoint". Recomputed on the actual 97,954-cell analysis set (38 groups): **0.0320**. The
0.025 came from the full 125,706-cell object with 48 groups — the wrong cell set. The sibling
0.258 timepoint baseline reproduces exactly (0.2575), so only this one is affected.
**Fix**: 0.025 → 0.032.

##### F4. Per-celltype table miscounts and drops a real result
`FINDINGS.md` (best-achievable recovery table): claims "*(14 types within ±0.01)*". Recomputed from
`resolution_sweep.tsv`: **8 types outside ±0.01, 16 inside**. The table omits **Tem/Effector helper
T cells (+0.029, n=2,962)** — a gain comparable to Tem/Trm which *is* shown — and lists DC2
(−0.009) as an outside-noise loss when it is inside.
**Fix**: add Tem/Effector helper T, move DC2 into the within-noise group, correct the count to 16.

##### F5. Superseded n=1 tail numbers still asserted, contradicting the corrected section
`FINDINGS.md:127,133` — the "Is the higher rank better?" table still carries the n=1 stock-TOTALVI
tail values (0.124 / 0.052) and concludes "3.7% vs 2.3%", while the earlier 3-seed section
(explicitly headed "supersedes the n=1 table below") gives 0.150 / 0.057 and the later text uses
"4.4% vs 2.3%". Two contradictory absolute-share conclusions in one document, and unlike the two
deliberately-retained superseded sections this one carries no provenance marker.
**Fix**: update the table to 3-seed values or mark it superseded like the others.

##### F6. Reproduce + Caveats stale again (F8 regression)
`FINDINGS.md`: "Stock TOTALVI's schisto row is pending seed 2" (it is complete, `n_pairs=3`);
"still in progress" for the 3-seed baseline scripts whose outputs are now the document's primary
evidence; Caveats still describes the n=1 problem as open; the runnable block omits
`cross_seed_stability.py`, `u_space_clustering.py`, `per_celltype_resolution.py`,
`resolution_sweep.py`, `train_multigrate.py`, `plot_umap_mog_vs_vamp.py` — most of the tables and
the only figure. The volatile-path note still tells readers to "update `RUN`", a variable removed
by the F11 fix.

#### Minor

##### F7. `best F1 across resolutions` is a max over a noisier curve for MoG
Per-seed max over 5 correlated noisy grid points inflates the arm with the more volatile curve.
MoG's plasma-cell F1 at res 1.0 is 0.438/0.821/0.482 across seeds; its per-seed-max variance is
consistently larger than VampPrior's. Doesn't flip the ranking; magnitudes are optimistic.

##### F8. Multigrate (batch) stability is a single seed pair, tabled beside 3-pair rows
`cross_seed_stability.tsv`: `n_pairs=1`, sd columns empty. The "worst Procrustes disparity of all
seven arms" superlative rests on that one cell (0.257 vs the reliable donor arm's 0.247).
Conclusion survives via the 3-pair donor arm.
**Fix**: show `n_pairs`; qualify the superlative.

##### F9. "More reproducibly" from three sd ratios at n=3
An 8× sd ratio on 2 df is within chance. The mean deltas already carry the finding.

##### F10. `train_multigrate.py` summary TSV/JSON overwritten on partial reruns
Currently shows 1 of 3 donor seeds and 2 of 3 batch attempts; the full picture survives only in raw
logs, and donor seed 2's crash predates the try/except so it isn't in any summary.

##### F11. NMI(donor,batch)=0.796 has no backing script
Computed inline, never persisted. It underpins a stated limitation.

### Code quality

#### Minor

##### F12. `_common.py` adoption is not durable
`train_multigrate.py` — written *after* `_common.py` — re-declares `PUB`/`PREPARED`/`OUT`/
`LATENT_DIR` and re-implements `effective_rank` byte-for-byte. The k=15 constant exists under three
names (`K`, `KNN_K`, `N_NEIGHBORS`) across five scripts. Multivariate-η² composition is
reimplemented three times; best-cluster-F1 twice (files written 7 minutes apart). The F12 pattern
recurring one abstraction level up.

## What was checked but is fine

- **Data pipeline**: `organize_multimodal_anndatas` row/var order verified *empirically* with a
  synthetic out-of-order case; `rna_indices_end=10000` correct against the 10,130-feature joint
  object; scvi-tools `_make_data_loader` defaults to `shuffle=False` so latent order matches
  `obs_names`. Subsamples are one fixed-seed draw reused across all arms. `knn_state_recovery.tsv`
  shows identical `n_cells=9370`/`n_states=20` on all 9 rows.
- **Stats**: Welch t/df/CI recomputed and internally consistent; unequal-variance choice correct
  (0.53 vs 1.18). The `whiten()` transform is correct. The "knnP_cell flat" null holds (within-arm
  sd 0.0005–0.0074 vs 0.012 between-arm spread).
- **APIs**: every Multigrate and scanpy call verified against installed versions, including
  `get_model_output()` → `obsm["X_multigrate"]` at `_multivae.py:330`.
- **Traceability**: essentially every table independently recomputed from source TSVs and matched,
  except F3/F4/F5.
- **`venv-multigrate/`** is self-`.gitignore`d — confirmed no commit risk.

## Notes

- **F1, F3, F4, F5 share a root cause**: numbers or baselines computed on one cell set / at one
  point in the session, then carried into prose after the underlying set or evidence changed. The
  analysis has strong *internal* recomputation discipline but weak *revision* discipline.
- **F1 is the third instance of the L-094/L-097 class** — a statistic whose denominator or
  operating point encodes the very thing under test. That is now a pattern worth a convention, not
  another learning entry.
