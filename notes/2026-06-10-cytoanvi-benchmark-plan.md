---
date: 2026-06-10
topic: cytoanvi-benchmark
branch: feat/cytoanvi
status: plan
summary: Real-data benchmarking plan for CytoANVI using the two CytoVI vignette datasets (Roider BNHL multi-panel; Nuñez PBMC batch). Closes the "no real-data validation" gap from 2026-06-01-cytoanvi.md.
---

# CytoANVI benchmarking plan (real data from the CytoVI vignettes)

**Why:** the implementation passes 20/20 synthetic tests but has **no real-data validation**
(`notes/2026-06-01-cytoanvi.md`). Synthetic accuracy is meaningless (independent ref/query). This
plan benchmarks CytoANVI on the datasets the CytoVI vignettes themselves use, so the comparison is
against CytoVI's own published workflow — not a substituted dataset.

## Datasets (from the CytoVI vignettes)

### D1 — Roider BNHL (CytoVI *advanced* tutorial) — PRIMARY
The advanced vignette is exactly CytoANVI's target scenario: **two antibody panels, a shared
backbone, and annotation transfer from an annotated panel to an unannotated one.**
- Source: Roider et al. 2024, Nat Cell Biol (doi 10.1038/s41556-024-01358-2); B-cell non-Hodgkin
  lymphoma tumor-infiltrating T cells, 33 donors.
- Load (preprocessed `.h5ad`, already arcsinh + scaled + subsampled):
  ```python
  import scanpy as sc
  from scvi.external import cytovi
  p1 = sc.read("roider_p1.h5ad", backup_url="https://figshare.com/ndownloader/files/56891468")
  p2 = sc.read("roider_p2.h5ad", backup_url="https://figshare.com/ndownloader/files/56891471")
  adata = cytovi.merge_batches([p1, p2], batch_key="panel_batch")  # builds _nan_mask
  ```
- Shape: 2 panels × 12 markers (+FSC/SSC), **10 shared backbone markers**, ~9,966 cells, 4 batches,
  33 patients. **Cell-type labels in panel 1 only.**

### D2 — Nuñez PBMC (CytoVI *batch-correction* tutorial) — SECONDARY
Single-panel, fully-labelled, two batches of one donor — clean for label-transfer accuracy and
batch-mixing without panel divergence.
- Source: Nuñez/Schmid/Power et al. 2023, Nat Immunol (SARS-CoV-2 vaccine study).
- Load (FCS):
  ```python
  b1 = cytovi.read_fcs("Nunez_PBMCs_batch1.fcs", remove_markers=["Time", "LD", "-"])  # file 55982654
  b2 = cytovi.read_fcs("Nunez_PBMCs_batch2.fcs", remove_markers=["Time", "LD", "-"])  # file 55982657
  # then cytovi.transform_arcsinh(cofactor≈2000) -> cytovi.scale([0,1]) -> merge_batches
  ```
- Shape: 35 antibody markers, 2 batches, 1 donor, ~20k cells, **11 labelled populations**.

## Benchmark tasks (each maps to a CytoANVI feature + a baseline)

The baseline throughout is **CytoVI + its k-NN label transfer**
(`CYTOVI` → `impute_categories_from_reference`) — the vignette's own method. CytoANVI's claim is
that a *trained* semi-supervised classifier on the M1+M2 latent beats k-NN voting and shapes a
better latent.

| # | Question | Design | Metric | CytoANVI API |
|---|----------|--------|--------|--------------|
| **B1** | Does the semi-supervised classifier transfer labels better than CytoVI k-NN? | D2 (fully labelled). 5-fold: hold out X% of labels as the unlabeled category, train, predict held-out. | accuracy, **macro-F1** (class-imbalanced), per-class recall | `setup_anndata(labels_key, unlabeled_category)` → `train` → `predict` |
| **B2** | Is the latent better integrated (batch) while preserving biology? | D2 latent, both methods. | **iLISI/kBET** (batch mixing) + **cLISI/ARI/NMI** (bio) via scib-metrics; report the trade-off | `get_latent_representation` |
| **B3** | Panel-divergent mapping: panel-1 reference → panel-2 query. | D1. Train CytoANVI on panel 1 (labelled). `prepare_query_anndata(panel2, ref)` → `load_query_data` → `predict` panel-2 labels. Panel 2 is unlabelled, so validate by (a) marker-consistency of predicted types on panel-2 backbone markers, (b) agreement with CytoVI k-NN transfer, (c) held-out **within panel 1** for a hard accuracy number. | accuracy (panel-1 holdout); concordance vs k-NN (panel 2) | `prepare_query_anndata` + `load_query_data` + `predict` |
| **B4** | Continual case-control update preserves reference/control latent while surfacing case signal. | D1 split by donor/condition into reference + healthy-control + case query (needs a case/control axis — see Open decisions). `load_query_data_with_replay`. | **reference-latent drift** (latent of replay cells before vs after), batch mixing of mapped query, recovery of a known case-enriched population; sweep `ewc_importance` | `load_query_data_with_replay` + `train(plan_kwargs={ewc_importance})` |
| **B5** | Does `get_uncertainty` flag novel (out-of-reference) cells? | D1/D2: hold out one cell type from the reference entirely; map a query containing it. | AUROC of BI uncertainty for held-out-type vs seen cells | `get_uncertainty` |
| **B6** | Tune λ (`ewc_importance`) for CytoVI's intensity likelihood. | B4 sweep λ ∈ {0, 1, 10, 100, 1000}; the paper's 100 was RNA/scANVI. | curve of case-signal vs reference-drift; pick the knee | (output: a documented default) |

## Metrics & tooling
- **Label transfer:** `sklearn` accuracy / `f1_score(average="macro")` / `classification_report`.
- **Integration:** `scib-metrics` (iLISI, cLISI, kBET, graph-connectivity, ARI, NMI, silhouette);
  the `scib` / `scib-gpu` conda envs already exist. Report the standard scib aggregate
  (batch-correction vs bio-conservation).
- **Uncertainty:** `roc_auc_score` of `get_uncertainty` vs held-out-type indicator.
- **Continual:** reference-latent drift = mean L2 between replay-cell latents pre/post update;
  case-signal = differential-abundance recovery (`differential_abundance`) of a known population.

## Harness layout
```
benchmarks/cytoanvi/
  data.py        # figshare download + cache (pooch), load D1/D2, build _nan_mask
  tasks.py       # B1–B6 as functions returning a metrics dict
  baselines.py   # CytoVI + impute_categories_from_reference wrapper
  run.py         # CLI: --dataset {roider,nunez} --task {b1..b6} --seed --out results.json
  metrics.py     # scib-metrics + sklearn wrappers
README.md        # exact reproduction commands
```
- Run in the `scvi-test` (or `scib-gpu`) conda env via the existing
  `PYTHONPATH=src LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python` invocation.
- Seeds fixed (`scvi.settings.seed`); each task ≥3 seeds; results to JSON + a summary table.

## Success criteria (pre-registered)
- **B1:** CytoANVI macro-F1 ≥ CytoVI k-NN on D2 holdout (target: +0.03 absolute, the bar for "the
  classifier earns its keep").
- **B2:** CytoANVI within ±0.02 cLISI of CytoVI (no biology lost) at equal-or-better iLISI.
- **B3:** panel-1 holdout accuracy > chance and ≥ k-NN; panel-2 concordance with k-NN ≥ 0.7.
- **B4/B6:** a λ exists where reference-drift stays low **and** case population is recovered; record
  it as the CytoVI-specific default (replaces the paper's 100).
- **B5:** held-out-type AUROC > 0.7.

## Open decisions (need input)
1. **Case/control axis for B4.** Roider BNHL is all-tumor; there's no built-in healthy control.
   Options: (a) treat one donor subset as pseudo-control, (b) use a different dataset with explicit
   case/control, (c) defer B4 to a dataset Mikhael provides. B1–B3, B5 don't need this.
2. **Scope/order:** start with D2 (B1, B2 — fastest, fully labelled), then D1 (B3, B5)?
3. **Compute:** GPU (untested path per lab notes) vs CPU for the first pass.

## Provenance
CytoVI advanced tutorial (Roider data): https://docs.scvi-tools.org/en/1.4.1/tutorials/notebooks/cytometry/CytoVI_advanced_tutorial.html
CytoVI batch-correction tutorial (Nuñez data): https://docs.scvi-tools.org/en/1.4.0/tutorials/notebooks/cytometry/CytoVI_batch_correction_tutorial.html
