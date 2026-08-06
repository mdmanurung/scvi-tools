# Review — `.scratch/mrtotalvi-z-vs-totalvi` (pass 3, ultra) — 2026-07-28

**Scope**: `.scratch/mrtotalvi-z-vs-totalvi/` — 15 Python scripts, `FINDINGS.md`, and their
`.tsv`/`.log`/`.png` artefacts. Third review pass after two prior passes
([2026-07-27](2026-07-27-mrtotalvi-z-vs-totalvi.md), 14 findings;
[2026-07-28 pass 2](2026-07-28-mrtotalvi-z-vs-totalvi-pass2.md), 12 findings) already fixed the
obvious issues.
**Files reviewed**: 15 scripts + `FINDINGS.md` + all `.tsv` artefacts (byte/number-level
recomputation against `human_prepared_10k.h5ad`, `schisto_human.h5ad`, and every cited `.tsv`).
**Sub-agents run**: 5 (stats-causal, data-pipeline-leakage, bioinformatics, llm-failure-modes,
doc-schema-fidelity, code-quality — six dimensions, code-quality and stats-causal converged on
one shared finding, folded together below) + adversarial verification pass + completeness critic
(re-run this pass after the first run of this synthesis returned degenerate placeholder output —
see "Completeness critic" below for the real result).

Bar for this pass was explicitly high: two prior passes already removed the obvious defects, so
survival required independent recomputation against source data, not re-reading of prose. Six
raw candidate findings were killed outright at the verification stage (see the "Full verdict
detail" the synthesizer was given — one `stats-causal` candidate on `knnP_celltype` flatness was
demoted from "confirmed" to "drop" for resting on an underpowered n=3, naive-pooling paired test
that the reviewing agent itself called fragile).

## Remediation applied

Since this report was first drafted, the analyst has fixed two of the six original findings in
`FINDINGS.md` directly:

- **F1** (cross-sample timepoint statistic compared against the wrong null) — fixed. Line 528 now
  reads `vs conditional null 0.233` in place of the old `(base 0.258)` comparison, and the prose
  explicitly walks through the `(0.2575 − 0.0320)/(1 − 0.0320) = 0.233` derivation and states that
  Vamp-`u` is "at null, not below it." Verified directly in the current file.
- **F3** (u-space scorecard omitted the two MoG-favoring stability metrics) — fixed. The scorecard
  at lines 393–403 now includes `raw CKA` and `Procrustes disparity` rows (both marked MoG wins),
  the effective-rank row is struck through and annotated "not a quality criterion," and the prose
  explicitly states "Cross-seed stability is a 2–2 split, not a VampPrior sweep," including a
  meta-note that "an earlier version of this table... omitted the two MoG wins." Verified directly
  in the current file.

**F2, F4, F5, F6 remain open** — re-verified against the current file (not re-described below;
see each finding's own text, preserved verbatim from the original pass):
- F2 (celltype provenance): still only the generic "coarse label" caveat at line 146; no
  CellTypist/supersession disclosure anywhere in the file (grepped for `celltypist`, `supersed`,
  `coarse label`).
- F4 (stale n=1 stock-TOTALVI values): `15.62 ± 0.66` still unmarked at line 113; `17.28` still
  unmarked at line 258.
- F5 (stale table-header baseline): line 505 still reads `(base 0.025)`, three lines below the
  corrected prose.
- F6 (missing stock-TOTALVI row in schisto cross-seed table): the schisto block at lines 308–312
  still lists only the four MrTotalVI-LN rows.

## Key decisions in this analysis

- **Null/baseline specification for compositional statistics** — the document derives corrected
  numerators for nested categorical comparisons (donor×timepoint nested in timepoint) but has
  twice reused an un-derived baseline for the corrected statistic instead of deriving the matching
  conditional null. See F1 (now fixed — see "Remediation applied").
- **Ground-truth label source for all cell-type recovery metrics** — every celltype-keyed number
  in the document (scIB scores, η²_celltype, knnP_celltype, the per-celltype F1 recovery table)
  is computed against an automated CellTypist ensemble label that the project's own curation
  documentation says was superseded for final results. See F2 (open).
- **Metric selection for the u-space "VampPrior beats MoG default" recommendation** — the
  cross-seed-stability scorecard reports 4 of 6 computed stability metrics and both omitted ones
  favor the losing arm (MoG). See F3 (now fixed).
- **Reuse of a stock-TOTALVI baseline across document revisions** — the document explicitly
  supersedes an n=1 stock-TOTALVI baseline with a 3-seed one in its lead table, but the n=1 value
  resurfaces unmarked in two later tables. See F4 (open).
- **Propagation discipline for numeric corrections** — three independent reviewers (across three
  different review dimensions) each independently found the same already-once-fixed baseline
  value stale in a table header. See F5 (open).
- **Comparator-arm completeness in cross-seed tables** — the schisto half of the cross-seed
  stability table omits the stock-TOTALVI comparator row that the RDX-03 half above it retains,
  and that stock row happens to be the best-stability arm in the schisto data. See F6 (open).
- **Whether "MoG wins on geometry" is independent evidence or the same compression artifact under
  a new name** — the two scorecard rows that give MoG its remaining clean wins (centroid
  separation, within-cluster dispersion) are the same trace-ratio construction as η², and move in
  lockstep with effective rank across every arm. See F7 (new, minor — does not change the
  recommendation, which already discounts this axis for Leiden).
- **Whether the central "cluster on `u`" recommendation has been validated at the point a reader
  will actually act on it** — it rests entirely on pre-clustering kNN-neighbourhood purity;
  neither script that actually runs Leiden at multiple resolutions ever scores the resulting
  clusters for donor purity, despite already having donor identity available and already looping
  over the same resolutions/seeds. See F8 (new, major).

## Questions for the analyst

- Given that `knnP_celltype` is itself ≈0.81 (far from the marginal celltype mix), is the marginal
  Simpson index even the right *null family* for the cross-sample timepoint statistic, or does the
  document need a celltype-stratified null (`knnP_celltype`-conditioned) rather than just a
  corrected donor×timepoint-conditioned one? This bears on F1 (now largely resolved, but the
  question about null *family* rather than just null *value* is still open).
- The lineage-specific `L1`/`L2`/`L3` annotation exists for a related object
  (`schisto_human_merged_annotated.h5ad`) but was never merged into the object this analysis uses.
  Is re-running the key celltype-recovery comparisons against it (even as a spot-check on the
  rare/contested populations singled out in the document — Plasma cells, CD16− NK, DC1) worth
  doing before these numbers are treated as final? See F2.
- For the u-space clustering recommendation specifically, which class of cross-seed-stability
  metric is actually decision-relevant — the variance-weighted/centroid measures (raw CKA,
  Procrustes) or the neighbourhood/direction-set measures (whitened CKA, kNN Jaccard)? The
  document's fixed version now states a rationale (Leiden uses the kNN graph) but this was added
  after the fact — worth an explicit sign-off that this is the intended justification. See F3.
- Is there a plan to sweep the whole document once more for any other place a "superseded, n=1"
  number might have been pasted before the corrected value existed — two such instances were found
  in independent tables this pass (F4), beyond the one pass 2 already fixed?
- Before treating "cluster on `u`" as settled operational guidance: is there appetite to add a
  single post-hoc donor-purity check on the Leiden clusters already produced by
  `resolution_sweep.py`/`per_celltype_resolution.py`, at the resolution used for the F1 tables?
  The raw cluster assignments and donor labels both already exist; only the crosstab is missing.
  See F8.

## Findings

### Statistics & causal inference / code quality

#### Major

##### F1. Cross-sample timepoint statistic is compared against the wrong (unconditioned) null — **FIXED**

`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:504–523`
```markdown
Same-donor×timepoint neighbours are *trivially*
same-timepoint, so `knnP_timepoint` contains `knnP_dtp` inside it. Removing that trivial part,
`(knnP_time − knnP_dtp) / (1 − knnP_dtp)` gives cross-sample timepoint purity:

| representation | raw knnP_time | knnP_dtp | **cross-sample** (base 0.258) |
...
Every arm lands at **0.197–0.232 — *below* the 0.258 baseline, not above it.**
```
**Why it matters here**: The numerator was correctly decomposed (removing the same-`donor×timepoint`
component that trivially inflates `knnP_timepoint`), but the *baseline it's compared to* was not
decomposed the same way — 0.258 (`Σq²_time`) is the null for the raw statistic, not for the
conditional one. Since `donor×timepoint` is nested inside `timepoint` (same `dtp` ⟹ same
`timepoint`), the matching conditional null is `(Σq²_time − Σp²_dtp)/(1 − Σp²_dtp) ≈ 0.233`, using
the document's own two baseline numbers (0.2575 and 0.032) computed independently on the same
97,954-cell / 38-group analysis set (recomputed directly from `human_prepared_10k.h5ad`: 0.031974
and 0.257506, matching the document exactly). Against 0.233 rather than 0.258, MrTotalVI-LN-Vamp
`u` (0.232) is statistically indistinguishable from baseline rather than "below" it, and every
arm's margin below baseline shrinks substantially (from the stated 0.026–0.061 gap to roughly
0.001–0.036). This sits directly under the document's central "cluster on `u`" recommendation,
and is a fresh instance of the exact error class this project already has a name for (C-004: a
statistic whose denominator/operating point encodes the effect under test) — recurring *inside*
the fix pass 2 introduced for the same error class. Pass 2's own review text explicitly re-verified
the wrong comparison ("0.197–0.232 in every arm — below the 0.2575 baseline... verified") without
catching it, so this genuinely survived two passes. It was independently caught this pass by two
separate review dimensions (stats-causal and code-quality) using the same derivation, which is a
reliability boost, not double-counting — both are folded into this single finding.
**Why it matters here (impact)**: does not reverse the qualitative "compositional, not
state-defining" conclusion (no arm sits *above* either null), but it does retract "the corrected
numbers point the same way more strongly" and changes Vamp-`u` — the arm the document's headline
recommendation favors — from "clearly below baseline" to "at baseline."
**Fix**: replace 0.258 with the derived conditional null (~0.233) in the table header and prose at
lines 509–523; reframe Vamp-`u` (0.232) as statistically indistinguishable from the corrected null
rather than "below baseline"; retract "points the same way more strongly."

**Status: fixed.** Verified in the current file (line 528: `vs conditional null 0.233`; prose now
derives 0.233 explicitly and states Vamp-`u` is "at null, not below it").

### Bioinformatics

#### Major

##### F2. Ground-truth `celltype` label is an automated, self-documented-as-superseded ensemble call, undisclosed beyond a generic "coarse label" caveat — **OPEN**

`schisto_citeseq/analysis/final-assembly/outputs/schisto_human.h5ad` (`obs['celltype']`), consumed
throughout `.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md`
```text
# schisto_citeseq/curation/cards/celltypist-annotation.md:31
CellTypist calls used as first-pass labels in assemble_master.R; superseded by MrVI-based
lineage-specific annotations for final results.
```
**Why it matters here**: `obs['celltype']` traces verbatim through `assemble_master.R:38–92` to
the CellTypist `Immune_All_Low` ensemble vote, which the project's own curation card says was
replaced elsewhere ("for final results") by a lineage-specific `L1`/`L2`/`L3` hierarchy that was
never merged back into the object this analysis uses. Directly recomputed on `schisto_human.h5ad`:
full 3-model agreement (`human_n_agree==3`) holds for only 82.7% of cells (16.3% at 2/3, 0.9% at
an effective coin-flip 1/3), `celltype_confidence` 25th percentile is 0.544, and `celltype` differs
from the single-model `celltype_raw` call for 20.7% of cells. Every celltype-keyed number in
`FINDINGS.md` — scIB bio-conservation scores, the η²_celltype tail table, `knnP_celltype`, and the
per-celltype/resolution-sweep F1 recovery table that anchors "seven of eight favour VampPrior" —
measures recovery of this specific, project-acknowledged-superseded label, not the project's
best-available cell identity. For the rare populations the document scrutinises individually
(Plasma cells n=126–320, CD16− NK n=434–573, DC1 n=10), ensemble label noise is a real alternative
explanation for the cross-seed F1 variance currently attributed entirely to embedding/prior
choice, and is nowhere disclosed beyond "celltype is a coarse label."
**Fix**: disclose the CellTypist/superseded provenance and quantify disagreement
(`human_n_agree`, `celltype_confidence`) for at least the rare types singled out; if the
lineage-specific `L1`/`L2`/`L3` hierarchy is available for the same 97,954 cells, prefer it or use
it as a sensitivity check on the key recovery comparisons before treating current numbers as
settled.

**Status: open.** Verified against the current file — line 146 still reads only "`celltype` is a
coarse label," with no CellTypist/supersession disclosure anywhere (grepped for `celltypist`,
`supersed`, `coarse label`).

### LLM coding antipatterns

#### Major

##### F3. u-space "VampPrior beats the MoG default" scorecard omits the two stability metrics that favor MoG — **FIXED**

`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:391–406`
```markdown
| criterion | MoG u | Vamp u | winner |
|-----------|-------|--------|--------|
| donor mixing (knnP_donor) | 0.273 | **0.225** | Vamp |
| cross-seed stability (kNN Jaccard) | 0.145 | **0.189** | Vamp |
| cross-seed stability (whitened CKA) | 0.663 | **0.712** | Vamp |
...
VampPrior is better on both donor-invariance and seed-reproducibility.
```
**Why it matters here**: `cross_seed_stability.tsv` — the same table, same script, same u-vs-u
comparison — also has `cka_raw` (MoG 0.935 vs Vamp 0.808, MoG wins) and `procrustes_disparity`
(MoG 0.171 vs Vamp 0.243, lower-is-better, MoG wins), confirmed against
`FINDINGS.md:298–312` itself two sections earlier ("Raw CKA drops... Procrustes disparity
worsens... 0.171 to 0.243 for u"). The scorecard includes neither row. The document states its own
rule for exactly this pair 50 lines earlier ("Report both or neither" for raw vs. whitened CKA),
then breaks it in the one table that feeds a project-level recommendation aligned with the
document's own preferred conclusion (cluster on `u`, prefer VampPrior). Including effective rank as
a scored row compounds this: the document elsewhere (lines 122, 228, and its own "Is the higher
rank better?" section) repeatedly argues higher effective rank is *not* evidence of a better
representation, yet the scorecard counts it as a point for Vamp anyway. Correcting the tally with
the two omitted rows and setting rank aside as contested gives a genuine 3-3-1 split on cross-seed
stability (MoG wins on variance-weighted/centroid measures, Vamp wins on neighbourhood/
direction-set measures) — not the near-sweep the prose ("VampPrior is better on both
donor-invariance and seed-reproducibility") implies. This does not require reversing the section's
ultimate recommendation (donor mixing and kNN Jaccard are plausibly the most decision-relevant
measures for graph-based clustering), but the current framing overstates how one-sided the
cross-seed-stability evidence is.
**Fix**: add raw-CKA and Procrustes rows to the scorecard (MoG wins both, per the document's own
numbers at lines 298–312); either drop the effective-rank row or relabel it "not a quality
criterion, see 'Is the higher rank better?'"; reword the "VampPrior is better on both..." sentence
to state the split honestly (MoG wins on variance-weighted stability; Vamp wins on donor-invariance
and neighbourhood-based stability).

**Status: fixed.** Verified in the current file — lines 393–403 now include `raw CKA` and
`Procrustes disparity` rows (MoG wins both, bolded), effective rank is struck through and
annotated "not a quality criterion," and lines 405–416 state the honest 2–2 split plus a
meta-note that "an earlier version of this table... omitted the two MoG wins."

### Documentation & schema fidelity

#### Major

##### F4. Stale n=1 stock-TOTALVI values reused unmarked in two later tables, after being explicitly superseded earlier in the same document — **OPEN**

`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:48–52` (correct, marked) vs. `:111–115` and `:254–258`
(stale, unmarked)
```markdown
<!-- line 50, correct 3-seed value, explicitly headed "supersedes the n=1 table below" -->
| totalVI_ln (3 seeds) | **17.08 ± 0.53** | **15.49 ± 0.51** |

<!-- line 113, same quantity, no supersession marker -->
| totalVI_ln (stock) | 15.62 ± 0.66 | — |

<!-- line 258, same quantity, no supersession marker -->
| stock TOTALVI (RDX-03 / here) | n/a | 18.0–18.4 / 17.28 |
```
**Why it matters here**: `within_sample_rank.tsv` gives exactly `15.6245 ± 0.6593` for the n=1-era
stock rank quoted at line 113 — this is the pre-3-seed value, unlike the co-tabulated MrTotalVI
rows (9.55/9.11) which *are* the current 3-seed values (verified against `comparison_3seed.tsv`).
The stated deltas (−38.9%/−41.7%) reproduce only against the stale 15.62 denominator; against the
correct, already-computed-elsewhere 15.49 they are −38.3%/−41.2%. Likewise, 17.28 at line 258 is
the n=1 pooled rank the document itself marks superseded two sections earlier (line 87), while
17.08 is used correctly at line 50 — but the marking wasn't carried to its reuse at line 258. This
is precisely the failure mode pass 2 already flagged and fixed once (its F5, for a different
table) recurring in two other tables pass 2 didn't check. A reader has no textual cue to distrust
these two tables — unlike the ones that do carry "superseded"/"n=1" markers — so the wrong
percentages and reconciliation figure are the ones most likely to get quoted going forward.
**Fix**: replace 15.62±0.66 with 15.49±0.51 (recompute deltas to −38.3%/−41.2%) at line 113, and
17.28 with 17.08 at line 258 — or add the same "superseded, n=1" marker used at line 87 if keeping
the old numbers for provenance is intentional.

**Status: open.** Verified against the current file — `15.62 ± 0.66` still unmarked at line 113;
`17.28` still unmarked at line 258 (line 50 still correctly shows `17.08 ± 0.53`).

#### Minor

##### F5. Random-baseline table header still shows the value pass 2 already retracted, three lines below the corrected prose — independently caught by three review dimensions — **OPEN**

`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:488–490`
```markdown
random-baseline kNN purity on the 97,954-cell analysis
set is **0.032** for donor×timepoint (38 groups) and **0.258** for timepoint:

| representation | knnP_dtp ↓ (base 0.025) | knnP_timepoint (base 0.258) |
```
**Why it matters here**: pass 2's F3 fixed the random-donor×timepoint-baseline number from 0.025
to 0.032 (the correct value, recomputed independently here as 0.031974 on the full set / 0.032014
on the 30k kNN subsample) — but only in the prose sentence two lines above; the table header three
lines below still reads "0.025", the only remaining occurrence in the document (grep-confirmed).
Three independent review dimensions this pass (data-pipeline-leakage, bioinformatics,
doc-schema-fidelity) each caught this same regression separately, which is a strong reliability
signal but doesn't change the stakes: the number is supplementary context for a table whose
reported values (0.077–0.154) are well above either baseline either way, so no downstream
conclusion changes. One reviewer graded this major on the grounds that it's a direct, unambiguous
self-contradiction rather than a matter of interpretation — a fair point about clarity, but by the
Major/Minor bar (does the result become invalid or misleading in a way that matters) this is a
cosmetic propagation gap on a non-load-bearing baseline, not a Major.
**Fix**: change "(base 0.025)" to "(base 0.032)" at line 490 to match the corrected prose at line
488.

**Status: open.** Verified against the current file — line 505 (numbering shifted slightly since
the F1/F3 fixes were applied) still reads `(base 0.025)`.

### Code quality

#### Minor

##### F6. Schisto half of the cross-seed-stability table omits the stock-TOTALVI comparator row that the RDX-03 half retains — **OPEN**

`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:302–312`
```markdown
| **RDX-03 canonical_human** (9,370 cells) | | | | | |
| B1 factual_z (stock TOTALVI) | 18.25 | 0.875 | **0.832** | 0.158 | 0.303 |
...
| **schisto** (97,954 cells) | | | | | |
| MrTotalVI-LN-MoG u | 9.90 | **0.935** | 0.663 | 0.171 | 0.145 |
```
**Why it matters here**: `cross_seed_stability.tsv` has a `TOTALVI-LN (stock)` row for schisto
(rank 17.08, CKA raw 0.888, CKA whitened 0.824, Procrustes 0.175, kNN Jaccard 0.213 — all
`n_pairs=3`), which reappears correctly in a later Multigrate comparison table
(`FINDINGS.md:437`), but is silently dropped from the schisto block of this table even though the
RDX-03 block directly above keeps its own stock row. That dropped row is in fact the best-stability
arm shown anywhere in the schisto data (highest whitened CKA and kNN Jaccard of any row in the
table) — directly relevant to the surrounding prose's "u is consistently the least stable
representation... whatever is unstable in MrTotalVI lives in u" framing, currently checkable only
by cross-referencing a table 130 lines further down.
**Fix**: add the `TOTALVI-LN (stock) | 17.08 | 0.888 | 0.824 | 0.175 | 0.213` row to the schisto
block (values already in `cross_seed_stability.tsv`); the "`u` is least stable" claim still holds
with it restored.

**Status: open.** Verified against the current file — the schisto block at lines 308–312 still
lists only the four MrTotalVI-LN rows.

### Completeness-critic findings (new this pass)

#### Minor

##### F7. "MoG wins on geometry" rests on the same trace-ratio construction as η², and moves in lockstep with effective rank

`.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:391–403` (scorecard rows `centroid separation`,
`within-cluster dispersion`) and `:418–423` ("Beyond stability, MoG wins on **geometry**...")

**Summary**: `separation_ratio` (between-centroid distance / within-cluster distance) and
`within_dispersion` — the two scorecard rows the document credits as clean, uncontested MoG wins —
are structurally the same construction as multivariate η², the metric this same document
demonstrates elsewhere (three times: the `z`-tail rewrite, the RDX-03 kNN retest, and F3's own
rank-descoring above) is mechanically inflated by whichever embedding happens to be more
compressed, with no denominator-free floor. Neither prior pass, nor the document's own three
self-corrections of η²-based claims, ever applied that same scrutiny to these two rows.

**Evidence**: recomputed directly from `u_space_clustering.tsv` (20 arm×seed rows: TOTALVI stock,
MoG u/z, Vamp u/z, Multigrate donor/batch): `effective_rank` correlates at r=−0.87 with
`eta2_celltype`, r=−0.64 with `separation_ratio`, and `eta2_celltype` correlates at r=−0.92 with
`within_dispersion`. The three headline arms move together exactly as compression predicts:
TOTALVI stock (rank 17.08, η²_celltype 0.370, separation 1.878) → Vamp `u` (rank 11.62, η² 0.430,
separation 1.701) → MoG `u` (rank 9.90, η² 0.575, separation 2.186).

**Why it matters**: does not change the section's actual recommendation — the document already
discounts this axis for the decision at hand ("Leiden operates on the kNN graph, not on UMAP
coordinates or centroid distances... at the graph level the two are equivalent"). What it changes
is the implicit corollary a reader may still draw: that MoG has a genuinely better-separated
embedding for *some* other purpose (visualization, non-graph downstream use), with graph-clustering
irrelevance being the only reason to set it aside. The correlation data says the "win" itself is
likely the same compression artifact under a different name, not an independently real property of
MoG's embedding — the same caveat the document already applies to η², just not yet to these two
rows.

**Fix**: add a one-line note next to "MoG wins on geometry" that `separation_ratio`/
`within_dispersion` correlate with effective rank (r=−0.64/−0.92 respectively) and are not
established as independent of the compression artifact already discounted for η²; this affects
interpretation, not the recommendation.

**Confidence**: high on the correlation numbers (directly recomputed); medium on how much
interpretive weight to give it, since the document's actual decision doesn't rest on this axis.

#### Major

##### F8. The central "cluster on `u`" recommendation is validated only pre-clustering — no script ever scores the actual Leiden output for donor purity

`resolution_sweep.py`, `per_celltype_resolution.py` (no `donor` reference anywhere in either
script — grep-confirmed); `FINDINGS.md:365–425` ("Clustering on `u` for sample-robustness" and
"For `u`-space clustering, VampPrior beats the MoG default")

**Summary**: the document's headline decision — cluster on `u` rather than `z`, and prefer
VampPrior over the MoG default, specifically *for graph-based clustering* — is supported entirely
by pre-clustering kNN-neighbourhood purity (`knnP_donor`, computed in `u_space_clustering.py`
directly on a kNN graph over the embedding). The two scripts that actually run Leiden — at 5
resolutions × 3 seeds × 2 priors, to produce the per-celltype F1 recovery tables the document
relies on elsewhere — never compute or report donor purity of the resulting clusters, even though
donor identity is already in `obs` and the same resolution/seed loop already exists for a
different purpose.

**Evidence**: `grep -n donor resolution_sweep.py per_celltype_resolution.py` returns nothing —
confirmed directly. `u_space_clustering.py`'s `knnP_donor` (0.273 MoG-`u` vs 0.225 Vamp-`u`) is the
only donor-fragmentation evidence anywhere in the directory, and it is computed before any
clustering step, never on the Leiden partitions themselves.

**Why it matters**: kNN-neighbourhood purity is a reasonable proxy but is not the outcome a reader
will act on. A modest purity gap (0.273 vs 0.225, both far from the 1.0 "fully donor-pure"
extreme) is compatible with very different downstream cluster structures — from a handful of
borderline-mixed clusters to several outright single-donor clusters — and the two are not
distinguishable from the purity number alone. Since a reader will use this document to choose a
clustering space and prior for a live atlas, whether `u` still throws off donor-specific clusters
*at the resolution actually used for cell typing* is exactly the failure mode the whole `u`-vs-`z`
design intent exists to prevent (cf. the Svensson post the document itself cites), and it was never
measured even though the raw Leiden cluster assignments already exist on disk from the F1-recovery
runs.

**Fix**: re-run `resolution_sweep.py` or `per_celltype_resolution.py` with one added computation
per resolution/seed/arm — e.g. `pd.crosstab(adata.obs['leiden'], adata.obs['donor']).apply(lambda
r: r.max()/r.sum(), axis=1)` — and report the fraction of clusters with donor-max-share above a
threshold (e.g. 0.8), for MoG-`u` vs Vamp-`u`, at the resolution(s) already used for the celltype
F1 comparison. This could confirm the recommendation outright, or reveal it needs qualification —
either way it closes a gap between the evidence collected and the claim made.

**Confidence**: high that the check does not currently exist anywhere in the directory; the
severity reflects that this is the evidentiary basis for the document's single most actionable
recommendation, not a numeric error.

## What was checked but is fine

- **Statistics & causal inference**: the 3-seed effective-rank Welch t-tests, tail-eta² table, and
  the `knnP_celltype` "flat across arms" characterization were all reproduced to 2–3 significant
  figures against `comparison_3seed.tsv` / `totalvi_baseline_3seed.json`; a candidate finding on a
  small paired z/u gap in `knnP_celltype` was tested computationally and killed — it rests on an
  n=3, naive cross-architecture pooling that doesn't reach significance within either prior alone.
- **Data pipeline & leakage**: Multigrate's disclosed seed substitutions, the 30k/40k visualization
  subsamples (never used for per-celltype claims), Leiden `random_state` usage, and RDX-03's
  held-out-only validation split were all re-verified directly against source code and found sound;
  no new leakage path found.
- **Bioinformatics**: all prior-pass fixes (labels_key warning, `factual_z` retraction,
  donor_timepoint baseline correction, the 8-outside/16-inside per-celltype recount, the Multigrate
  size-factor caveat) confirmed still present and correct; the schisto/RDX-03 label-provenance
  concern (F2) is correctly scoped to schisto only — RDX-03's `cell_label_l2` fixture is a separate,
  digest-sealed object unconnected to the CellTypist pipeline.
- **LLM coding antipatterns**: every prior-pass fix (F1–F14, pass-2 F1–F12) re-verified still
  holding; the "best F1 across resolutions" max-inflation issue (already flagged by pass 2) is
  correctly scoped as not flipping any ranking; no new hallucinated-API or silent-fallback issues
  found beyond F3 above.
- **Documentation & schema fidelity**: the decomposed cross-sample timepoint-purity numbers, the
  per-celltype resolution-sweep counts, the RDX-03 rank-gate retest, and the `_common.py`
  effective-rank/eta² implementations were reproduced exactly across every script that reimplements
  them (`rdx03_collapse_character.py`, `u_space_clustering.py`, `train_multigrate.py`); no numerical
  divergence between duplicated implementations.
- **Code quality**: the k=15 kNN constant remains duplicated across 6 scripts (already flagged by
  pass 2, not re-reported); `train_totalvi_baseline.py`'s summary-overwrite pattern mirrors an
  already-flagged sibling issue in `train_multigrate.py` but has never actually triggered (all 3
  seeds present in both files); `resolve_run_dir()` correctly resolves the in-flight run directory
  with no evidence of concurrent rewriting during analysis.
- **Completeness critic (this pass)**: one candidate gap was investigated and *not* escalated to a
  finding — a possible cross-document duplication between `FINDINGS.md`'s "the default appears to
  have been set on a metric that does not govern graph-based clustering" discussion (lines 418–425)
  and `todo/TODO_REGISTRY.md`'s P2-008, an already-open, high-priority TODO item recording almost
  the identical conclusion and numbers (`knnP` 0.225 vs 0.273; Jaccard 0.189 vs 0.145) with the
  additional detail that the original session-67 T/NK comparison artifact is confirmed
  unrecoverable. Verified P2-008 exists with this content. This is a real coordination gap (two
  write-ups of the same open question, uncited to each other) but it is not a defect *in*
  `FINDINGS.md` — the document's own recommendation to "re-run this before trusting the default" is
  still correct advice, it's just that P2-008 already establishes the re-run isn't currently
  possible (no surviving artifact). Below the bar for a numbered finding; worth a one-line
  cross-reference (`FINDINGS.md` → P2-008) so a reader doesn't waste time on a re-run the project
  already knows can't be done as stated.

## Completeness critic

*(Replaces the prior run's degenerate, placeholder-only output — see the "Notes" section below.)*

**Unexamined areas surfaced and their disposition:**

1. **The u-space clustering scorecard's "geometry" metrics are not denominator-free** — escalated
   to **F7** (minor) after independent recomputation confirmed the claimed correlations
   (r=−0.87/−0.64/−0.92) against `u_space_clustering.tsv`.
2. **No script ever computes donor purity on the actual Leiden clusters** — escalated to **F8**
   (major) after confirming via `grep` that neither `resolution_sweep.py` nor
   `per_celltype_resolution.py` references `donor` at all, despite both already running Leiden
   across the same resolution/seed grid used for the celltype F1 tables.
3. **`FINDINGS.md`'s session-67 re-run recommendation duplicates an existing, more-detailed TODO
   item (P2-008) without cross-referencing it** — investigated, confirmed real, but kept out of the
   numbered findings (see "What was checked but is fine" above) since it is a documentation/
   coordination gap rather than a defect in the analysis itself, and doesn't meet the "genuinely
   matters" bar for a third-pass finding.

**End-to-end consistency**: every quantitative table spot-checked against its source TSV in this
pass and the two prior passes reproduced exactly, with the exception of the two still-open stale
numbers (F4, F5). The one place internal consistency breaks in a way arithmetic-checking alone
would not catch is structural: part of the u-space "geometry" evidence (F7) is not, in fact,
independent of the compression artifact the rest of the document is built around avoiding.

**Would a reader be misled?** Mildly, on two specific points, both now captured as F7/F8: (a) a
reader who has internalized the document's repeated η² warning would not expect it to also apply
to "centroid separation"/"within-cluster dispersion," since those are framed as a distinct,
seemingly more intuitive geometric measure; (b) a reader would reasonably assume the pre-clustering
donor-purity gap settles the donor-robustness question for the *actual* Leiden clusters used
elsewhere in the document, when that specific check was never run despite the material to run it
already existing on disk. Everything else checked in this pass and the two prior ones held up
exactly as written.

## Notes

Two things worth naming explicitly since they recur across F1, F4, and F5:

1. **A single root cause spans three of six original findings.** F1 and (independently) a
   code-quality finding both converged on the same cross-sample-null miscalibration and were folded
   into one finding above rather than double-counted (now fixed). F4 and F5 are a matched pair —
   both are the "correction applied in prose, not propagated to an adjacent table" pattern, just at
   different severities (F4's stale values are load-bearing percentages that get quoted; F5's is a
   supplementary baseline that changes no conclusion). Both remain open. If a fourth pass runs, the
   single highest-leverage action is not another line-by-line check but a document-wide grep for
   every number this analysis has ever revised (0.025, 15.62, 17.28, and any others) to confirm none
   survive unmarked elsewhere — this is the third generation of exactly this defect class (C-004)
   and it keeps recurring in new locations even as each specific instance gets fixed.
2. **F2 (celltype provenance) is the one original finding that neither prior pass's checklist was
   built to catch** — it required tracing data provenance through a sibling repository
   (`schisto_citeseq/`) rather than re-deriving a statistic already in `FINDINGS.md`. It remains
   open, and its fix (re-running key comparisons against the lineage-specific annotation, if
   available) is genuinely out of scope for a text-only correction pass.
3. **The completeness critic run for this synthesis pass had previously returned degenerate,
   placeholder ("test") output** in an earlier run of this same step, meaning F1–F6 were derived
   entirely from the six dimension sub-agents with no gap-check at all. This pass re-ran the critic
   properly; it surfaced two findings that genuinely survived scrutiny (F7, F8) and one item
   investigated and correctly *not* escalated (the P2-008 cross-reference). Anyone running a fourth
   pass should confirm the critic step returns substantive content before trusting a "nothing else
   to check" verdict from it — this is now the second time that check has mattered.
4. **On severity discipline for this pass**: per the explicit instruction to be ruthless given this
   is a third pass, F7 was kept at minor (it doesn't change any conclusion, only an interpretive
   corollary) and the P2-008 duplication was kept out of the numbered findings entirely (real, but
   a documentation-coordination issue, not a defect in the analysis). F8 is the one new finding
   escalated to major, because it identifies that the document's single most actionable
   recommendation has an evidentiary gap at the exact point a reader would act on it, using material
   that already exists on disk — this is a "genuinely matters" gap, not a nitpick.

---

## Completeness critic — re-run (the first attempt returned placeholder output)

The synthesis agent could not incorporate these (session limit); appended by the analyst.

### F7 (Major). Two scorecard "geometry" rows are a fourth C-004 instance
`u_space_clustering.py::compactness()` computes between-centroid / within-cluster distance in raw
Euclidean latent space — structurally the same `trace(between)/trace(within)` form L-094 showed is
mechanically inflated by compression. Verified across all 20 arm × seed rows:

| | vs effective_rank | vs η²_celltype | vs knnP_celltype |
|---|---|---|---|
| separation_ratio | −0.64 | +0.86 | +0.27 |
| within_dispersion | +0.85 | −0.92 | −0.29 |

Both track compression and the known-artefact η², not cell-type recovery. **"MoG wins on geometry"
was not established.** Applying the same test to stability metrics (n=10) also removed
**whitened CKA (r=+0.63), which favoured VampPrior** — so the correction cuts both ways. Clean
scorecard: 2–2 with one tie, not 5–2. **Remediated** in FINDINGS.md.

### F8 (Major). The central recommendation was never tested on actual clusters
`resolution_sweep.py` and `per_celltype_resolution.py` run Leiden (5 resolutions × 3 seeds × 2
priors) but `grep -n donor` returns nothing in either. So no artefact answers the literal question a
reader will act on: **among the Leiden clusters this document already computed, what fraction are
donor-pure?** The recommendation rests entirely on *pre-clustering* kNN neighbourhood purity
(0.273 vs 0.225 — both far from 1.0), which cannot distinguish "a few borderline-mixed clusters"
from "several outright donor-specific clusters". That is precisely the failure mode the `u`/`z`
design exists to prevent. **Open** — the raw material already exists.
Check: `pd.crosstab(leiden, donor).apply(lambda r: r.max()/r.sum(), axis=1)`, report the fraction of
clusters with donor-max-share > 0.8, MoG-u vs Vamp-u.

### F9 (Minor). FINDINGS.md recommends a remediation its own TODO says is impossible
FINDINGS.md advises re-running the session-67 T/NK comparison "on the original data"; P2-008 (same
day, high, open) records that the artifact "has no surviving file anywhere". The recommendation is
not executable as stated, and the two documents independently derive the same conclusion without
citing each other. **Open** — reconcile or merge.

### End-to-end consistency

The critic independently re-derived the 3-seed Welch table, both per-PC tail tables, and the
corrected resolution sweep (8 of 24 outside ±0.01, DC2 inside at −0.0092) — all reproduced exactly.
It also resolved the 0.163-vs-0.166 and 0.124-vs-0.119 discrepancies between the two tail tables as
a legitimate average-then-ratio vs ratio-then-average difference, not an error. The only structural
inconsistency found is F7.

## Process note

The first run of this critic returned literal `"test"` in every schema field — type-valid garbage.
The synthesis agent flagged it rather than writing around it. Re-running with a mandatory
tool-use procedure and an explicit anti-placeholder rule produced the three findings above,
including the highest-value finding of the entire three-pass review. **A schema-conformant response
is not evidence of work having been done.**

---

## Closure log — 2026-07-28 (analyst)

| Finding | Status |
|---------|--------|
| F1 conditional null (0.233 not 0.258) | **Fixed** — table + prose corrected; Vamp u is *at* null, not below |
| F2 `celltype` provenance | **Assessed & disclosed.** CellTypist 3-model ensemble, curation card marks it superseded — real and now documented. Two of the finding's figures were computed on the wrong cell set (confidence 25th pct is 0.693 not 0.544; disagreement 19.2% not 20.7% on the 97,954 analysis cells). Its proposed remedy is unavailable — no L1/L2/L3 hierarchy in this object, and `obs['final_label']` is donor_timepoint, not a cell-type call. **Sensitivity checks run**: `knnP_celltype` moves ≤0.002 on 3/3-agreement cells; the per-celltype sweep keeps the same sign for all 6 types present in both sets, but magnitudes move up to 5× and the two most extreme full-cell results (Late erythroid +0.191, Plasmablasts −0.024) have **no cells** at 3/3 agreement. Claim revised to sign-only. |
| F3 scorecard omissions | **Fixed** — raw CKA and Procrustes restored (both favour MoG); effective rank descored |
| F4 stale n=1 tail values | **Fixed** — table row marked superseded; prose now quotes the 3-seed 4.4% vs 2.3% |
| F5 stale 0.025 baseline in table header | **Fixed** — 0.032 |
| F6 stale Reproduce/Caveats | **Fixed** in pass 2 remediation |
| F7 geometry metrics are compression artefacts | **Fixed** — separation/dispersion descored; whitened CKA discounted on VampPrior's side too; clean scorecard is 2–2 with one tie |
| F8 cluster-level donor purity never measured | **Closed by measurement** — `cluster_donor_purity.py`. No arm produces patient-specific clusters (0–1.3% of clusters, TOTALVI included); `u` cuts excess donor structure ~30%; positive control (`z` worse than `u`) passed |
| F9 unexecutable remediation | **Fixed** — reconciled with P2-008, which takes precedence |

All pass-1, pass-2 and pass-3 findings are now closed or explicitly scoped. The one substantive
caveat carried forward: per-celltype recovery **magnitudes** are not stable to label quality, only
their sign.
