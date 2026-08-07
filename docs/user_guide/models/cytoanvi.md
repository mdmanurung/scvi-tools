# CytoANVI

**CytoANVI** [^ref1] (Python class {class}`~cytoanvi.CytoANVI`) is a semi-supervised extension of
{class}`~scvi.external.CYTOVI` for antibody-based single-cell data (flow cytometry, mass cytometry,
CITE-seq protein). It follows the same design pattern as {class}`~scvi.model.SCANVI` [^ref2] extends
{class}`~scvi.model.SCVI`: a shared CytoVI protein encoder/decoder plus a classifier head and a
partially observed label objective (M1+M2 hierarchy), while keeping CytoVI's batch correction,
missing-marker masking, and scArches [^ref3] query mapping.

The {doc}`/usage_readiness` matrix is authoritative. Source/API coverage, installed-artifact
acceptance, scientific evidence, and human promotion are separate states.

The advantages of CytoANVI are:

- Transfers cell-type labels to unlabeled cells with a trained classifier, not only k-NN in latent space.
- Integrates labeled reference and unlabeled query in one model (semi-supervised training).
- Inherits CytoVI panel-aware query prep, imputation, differential abundance/expression, and latent integration.
- Quarantines the former TTA novelty estimator after a below-chance multi-seed result.
- Supports experimental EWC-plus-replay reference updates with required query controls.

The limitations of CytoANVI include:

- Requires at least some labeled cells per type you want to predict (or an unlabeled category for unknowns).
- The classifier reads the **backbone** latent; types separated only by panel-specific markers may be under-resolved.
- Effectively requires a GPU for training on large cytometry panels.
- Does **not** use CytoVI's label-conditioned mixture-of-Gaussians prior (semi-supervised M1+M2 replaces it).

```{topic} Related tutorials:
- `vignettes/cytoanvi_example_reference_query.py` (tracked label-transfer and panel-mapping fixture)
- `vignettes/cytoanvi_treearches_synthetic.py` (tracked treeArches engineering fixture)
- {doc}`/user_guide/models/cytovi` (shared cytometry preprocessing and tasks)
- {doc}`/user_guide/models/scanvi` (semi-supervised VI background)
```

## Preliminaries

CytoANVI expects the same inputs as CytoVI:

- A transformed protein matrix (typically arcsinh + min-max scaled) in a layer (default `"scaled"`).
- A `batch_key` for batch / replicate correction.
- A `labels_key` with a dedicated **unlabeled category** string (e.g. `"Unknown"`) for cells without annotations.

For overlapping antibody panels, set up the reference with a `nan_layer` (see
{func}`scvi.external.cytovi.merge_batches`) so panel-specific markers can be masked during query
mapping.

For real flow or mass cytometry data, handle instrument-level preprocessing before CytoANVI:

- Apply compensation / spillover correction upstream for fluorescence cytometry.
- Use an arcsinh cofactor appropriate for the technology and staining panel.
- Scale each marker consistently across reference and query; do not fit query scaling independently
  if that changes the reference distribution.
- Harmonize marker names before merging panels and inspect missing-marker masks after padding.
- Keep biological labels and the unlabeled category as explicit strings; avoid missing values in
  the label column passed to `setup_anndata`.

Approximate runtime expectations:

| Workflow | Hardware | Expected use |
|----------|----------|--------------|
| Synthetic tutorial | CPU or GPU | API smoke and docs examples |
| Small real dataset | GPU preferred, CPU possible | Method familiarization and parameter checks |
| Full cytometry atlas | CUDA GPU | Publication-scale training and query mapping |
| Full Roider benchmark | A100 40 GB class GPU | Release validation; use batch size around 8192 |

## Relation to CytoVI

| Feature | CytoVI | CytoANVI |
|---------|--------|----------|
| Latent integration | yes | yes |
| Label transfer | k-NN in latent (`impute_categories_from_reference`) | `predict()` classifier |
| Semi-supervised training | optional label-informed **prior** only | classifier + partial labels |
| Query / scArches | `load_query_data` | `prepare_query_anndata` + `load_query_data` (panel-aware nan mask) |
| Novelty/OOD | — | No stable supported API; legacy TTA entry points fail closed |
| Continual update | — | Experimental EWC + replay; replay and `control_adata` are required |

You can warm-start from a trained CytoVI model with {meth}`~cytoanvi.CytoANVI.from_cytovi_model`.

## Descriptive model

CytoANVI adds an M1+M2 two-level latent hierarchy (following scANVI [^ref2]) on top of CytoVI's
protein encoder/decoder:

- **z1** — the shared backbone latent ($z_1 \in \mathbb{R}^d$), inferred by $q(z_1 \mid x, s)$
  (same encoder as CytoVI).
- **z2** — a label-conditioned latent, inferred by $q(z_2 \mid z_1, y)$ with a standard Normal
  prior $p(z_2) = \mathcal{N}(0, I)$; decoded back to z1 space by $p(z_1 \mid z_2, y)$.
- **Classifier** — a shallow network $q(y \mid z_1)$ that maps z1 to per-label cell-type
  probabilities.

For **labeled** cells the training loss augments the variational ELBO with a cross-entropy term
scaled by `classification_ratio` ($\alpha$, default 50):

$$
\mathcal{L}_{\text{labeled}} = \mathcal{L}_{\text{ELBO}}(y) + \alpha \cdot \mathrm{CE}\!\left(y,\; q(y \mid z_1)\right)
$$

For **unlabeled** cells the ELBO is marginalized over all labels using soft classifier weights,
so unlabeled cells still train the classifier through soft-label averaging:

$$
\mathcal{L}_{\text{unlabeled}} = \sum_y q(y \mid z_1)\,\mathcal{L}_{\text{ELBO}}(y) + \mathrm{KL}\!\left[q(y \mid z_1) \;\|\; p_{\text{prior}}(y)\right]
$$

**Per-cell nan-masking.** When a `nan_layer` is present (multi-panel data), the per-marker
reconstruction loss is multiplied element-wise by a binary mask before summation, so absent markers
do not contribute to the likelihood gradient on a per-cell basis. See {doc}`/user_guide/models/cytovi`
for the protein observation model and preprocessing background.

## API readiness and limitations

Use `from cytoanvi import CytoANVI` as the default import path. The previous
`scvi.external.CytoANVI` and `scvi.external.cytoanvi` paths are intentionally not part of the
current API.

| Surface | Status | Notes |
|---------|--------|-------|
| `cytoanvi.CytoANVI` | Conditional core | Setup, training, prediction, latent, save/load, and query mapping require dataset-specific independent validation. |
| `get_uncertainty` / indirect TTA replay selection | No-go | Stable entry points raise. Explicit experimental TTA code is historical method-development surface only. |
| `cytoanvi.CytoANVAE` | Stable advanced | Exported for advanced module-level inspection and extension; most users should instantiate `CytoANVI`. |
| `cytoanvi.hierarchy` | Optional-extra | Importable without scHPL; scHPL workflows require `pip install cytoanvi[cytoanvi-hierarchy]`. |
| `cytoanvi.mapping_qc` | Experimental optional-extra | Requires exactly mapQC 0.1.1; no automatic acceptance threshold is promoted. |
| `load_query_data_with_replay` | Experimental | `replay_adata` and `control_adata` are required; reloaded continual state cannot train until replay is reconstructed. |
| AnnBatch, FlowSOM, RAPIDS, benchmark tasks | Benchmark-only | These backends are CLI/evaluation infrastructure and are not part of the model API. |

Supported stable workflows on synthetic and unit-test coverage are flat label transfer,
CytoVI warm-start, panel-aware reference/query mapping, HCE prediction after an explicit hierarchy,
mapping-QC delegation, paired RNA/CyTOF preprocessing, and core save/load inference. These are
engineering states, not scientific promotion. Continual replay/EWC is experimental/no-go for
consequential updates; stable TTA novelty is no-go.

## Historical exploratory benchmark diagnostics

These unsealed results predate the frozen P2 protocol. They are useful for diagnosing methods, but
they do not supply independent labels, a second cohort, frozen margins, installed-artifact
acceptance, review authority, or promotion. The usage-readiness matrix remains authoritative.

| Task | Dataset | Metric | CytoANVI | Baseline | Verdict |
|------|---------|--------|----------|----------|---------|
| B1 label transfer | Roider BNHL | macro-F1 | **0.9317 ± 0.0022** | CytoVI-kNN 0.8928 ± 0.0034 | Retrospective delta; P2 margin unfrozen |
| B1 label transfer | Nuñez PBMC | macro-F1 | 0.9751 ± 0.0003 | CytoVI-kNN 0.9581 ± 0.0007 | Retrospective near-ceiling diagnostic |
| B2 batch integration | Nuñez PBMC | scib batch delta | no observed regression | CytoVI latent | Retrospective diagnostic; no frozen margin |
| B3 (p1) cross-panel | Roider BNHL | panel-1 holdout macro-F1 | **0.828 ± 0.015** | — | Exploratory only; independent labels absent |
| B3 (p2) cross-panel | Roider BNHL | inter-method concordance | 0.671 ± 0.008 | CytoVI-kNN | Concordance, not target accuracy or a promotion gate |
| B5 novelty detection | Roider BNHL | mean AUROC | historical mean below 0.5 | CytoVI kNN-OOD comparator | Historical negative; stable TTA is no-go |
| B8 HCE vs flat CE | Nuñez PBMC / Hao CITE-seq | lineage-coherence delta | fewer cross-lineage errors (see below) | flat CE | Exploratory coherence diagnostic, not fine accuracy |

### Limitations of the current evidence

- **Label circularity (B1, B3).** The CytoVI-kNN baseline uses the same encoder that produced
  the Leiden reference labels, so both B1 and B3 overstate CytoANVI's advantage relative to a
  truly independent annotator.
- **B3 p2 is concordance, not accuracy.** No independent manually-gated panel-2 ground-truth
  labels exist, so the 0.671 number measures agreement between two methods that share an encoder,
  not correctness against biology.
- **B5 novelty is a genuine negative result.** CytoANVI's test-time-augmentation uncertainty does
  not detect held-out cell types better than a trivial kNN-distance baseline in CytoVI latent, and
  the held-out "novel" types are proxy Leiden clusters rather than manually gated populations.
- **B8/HCE helps lineage coherence, not fine accuracy.** HCE is *mathematically identical to flat
  CE for leaf-only labels* — it only engages when some cells are labelled at coarse (internal)
  nodes (partial/mixed-granularity annotation; Nuñez's +0.086 came from a populated coarse node).
  On a genuinely deep tree (Hao CITE-seq ADT, 8 internal nodes, 3 seeds) the fine leaf-F1 effect
  is within noise, but HCE *consistently reduces cross-lineage errors* (all seeds, ~2.75σ, scaling
  with the coarse-label fraction). The magnitude is small (~9% relative) because rich ADT panels
  already give ~99% lineage accuracy. Report B8 with a lineage-level / hierarchical metric, not
  leaf macro-F1, and frame it around partial annotation.

B2 on the full Roider cohort, B9 mapping-QC (universal Nuñez rejection despite strong transfer),
and continual-update tasks B4/B6 (pseudo-batch plumbing) are not positive publication evidence.

## Quick start (label transfer)

```python
from cytoanvi import CytoANVI

CytoANVI.setup_anndata(
    adata,
    layer="scaled",
    batch_key="batch",
    labels_key="cell_type",
    unlabeled_category="Unknown",
)

model = CytoANVI(adata, y_prior="empirical")  # optional: class-imbalance prior
model.train(max_epochs=1000)

adata.obsm["X_CytoANVI"] = model.get_latent_representation()
adata.obs["pred_cell_type"] = model.predict()
```

Hold out a fraction of labels by setting those cells to `"Unknown"` before `setup_anndata`, then
compare `model.predict()` on held-out cells to ground truth.

## Paired scRNA + CyTOF

For same-study cohorts with **paired donors** (shared `sample_id`, not same-cell barcodes), use
{func}`scvi.external.cytovi.prepare_paired_cytoanvi` to harmonize immune markers, scale RNA,
run **scennep** (SNN-weighted pseudobulk on RNA), and merge with pre-scaled CyTOF. Partial pairing
is supported: some donors may be RNA-only or CyTOF-only, as long as at least one `sample_id` appears
in both modalities.

| Requirement | RNA | CyTOF |
|-------------|-----|-------|
| `obs["sample_id"]` | required | required |
| `obs["celltype"]` | required (eval; masked to Unknown for training) | required (training labels) |
| Expression | built internally (`scaled` → `scennep`) | `layers["scaled"]` required |

CyTOF must be arcsinh-transformed and min-max scaled first (see {doc}`/user_guide/models/cytovi`):

```python
from cytoanvi import CytoANVI
from scvi.external.cytovi import prepare_paired_cytoanvi

adata, markers = prepare_paired_cytoanvi(rna, cytof)

CytoANVI.setup_anndata(
    adata,
    layer="scaled",
    batch_key="modality",
    sample_key="sample_id",
    labels_key="celltype",
    unlabeled_category="Unknown",
)
model = CytoANVI(adata, y_prior="empirical")
model.train(max_epochs=1000)

adata.obsm["X_CytoANVI"] = model.get_latent_representation()
adata.obs["pred"] = model.predict()
```

Smoke test: `python vignettes/rna_cytof_cocluster.py --smoke` (synthetic data, no download).
Benchmark task B7: `python -m benchmarks.cytoanvi.run --dataset paired-rna-cytof --task b7`.

## Training details

- Default training uses {class}`~scvi.train.SemiSupervisedTrainingPlan` (labeled + unlabeled minibatches).
- `classification_ratio` (default `50`, set via `train(plan_kwargs={"classification_ratio": ...})`)
  balances the semi-supervised classification loss against the ELBO. Higher values emphasize label
  transfer accuracy; lower values can improve batch mixing in the latent (benchmark B2 tradeoff).
- `y_prior="empirical"` sets the label prior from observed label frequencies (Laplace-smoothed); use
  it only after evaluation. In 0.2.0 it is resolved from the realized training indices, never held-
  out validation labels. Default is uniform.
- Every non-null `adversarial_classifier` is rejected before trainer construction; no adversarial
  objective exists in this packet.
- Only `latent_distribution="normal"` is supported.
- For overlapping panels, ensure `encode_backbone_only=True` (default when a nan mask is present).

### Integration vs label transfer (batch–bio tradeoff)

CytoANVI shapes the latent with both reconstruction and the classifier on labeled cells. Completed
manifest artifacts should be used to quantify biological conservation and batch mixing for each
dataset; do not generalize from smoke or vignette-only runs. Tune `classification_ratio` and
`y_prior` if batch correction is the primary goal.

## Tasks

### Dimensionality reduction

```python
latent = model.get_latent_representation()
adata.obsm["X_CytoANVI"] = latent
```

Same API as CytoVI; the latent is shaped by both reconstruction and the classifier on labeled cells.

### Cell-type prediction

```python
pred = model.predict()
prob = model.predict(soft=True)  # per-class probabilities
```

Unlabeled cells (category `unlabeled_category`) receive predictions; labeled cells can be evaluated
by masking labels during training.

### Hierarchical cross-entropy (opt-in)

By default CytoANVI uses flat cross-entropy. To train with
[hierarchical cross-entropy](https://github.com/microsoft/hce-classification) (HCE), supply a
parent→children ontology or reachability matrix. HCE is **off** until you set a matrix explicitly.

```python
# Option A: parent → children edges (observed labels only; excludes unlabeled_category)
edges = {
    "T cells": ["CD4 T", "CD8 T"],
    "Myeloid": ["Monocyte", "DC"],
}
model.set_hierarchy(edges)

# Option B: precomputed (n_labels, n_labels) reachability matrix at construction
model = CytoANVI(adata, reachability_matrix=matrix)

model.train(max_epochs=200)

# Hierarchy-consistent predictions (requires set_hierarchy first)
hier_pred = model.predict_hierarchical()
hier_scores = model.predict_hierarchical(soft=True)
# Leaf labels only (skip coarse parents in argmax)
hier_leaf = model.predict_hierarchical(leaf_only=True)
```

Reachability is persisted across `save`/`load`. Soft hierarchical outputs are
hierarchy-adjusted scores, not normalized probabilities: an ancestor score includes descendant
mass. The unlabeled category is never a hierarchy node.

#### When HCE helps

| Source | Non-trivial hierarchy when… |
|--------|----------------------------|
| `set_hierarchy(edges)` | Coarse types (e.g. `"T cells"`) are **observed model labels** in the edge dict |
| `set_hierarchy_from_schpl(tree)` | A coarse type is an scHPL **internal** node whose name matches a model label (or use explicit `label_map`) |

If scHPL maps only **sibling fine types** (no shared coarse label in the model), the reachability
matrix is often the identity — HCE then matches flat CE for those labels. For batch-specific scHPL
leaves (`celltype-batch`), pass an explicit `label_map` rather than relying on prefix matching.

**Fail-fast:** `predict_hierarchical()` raises `ValueError` if no hierarchy is set.
`set_hierarchy` raises on label mismatches, invalid DAGs, or wrong matrix shape — there is no silent
fallback to flat CE.

See ADR `docs/adr/0003-cytoanvi-hce-schpl-hierarchy.md`.

### treeArches workflow (optional scHPL)

Learn, update, and predict cell-type hierarchies on CytoANVI latents with
[scHPL](https://schpl.readthedocs.io/) [^ref4] (treeArches-style). Install the optional extra:

```bash
pip install cytoanvi[cytoanvi-hierarchy]
```

Import helpers from the optional module (not the top-level `cytoanvi` package):

```python
from cytoanvi.hierarchy import (
    latent_to_anndata,
    learn_hierarchy,
    update_hierarchy,
    predict_schpl,
    set_hierarchy_from_schpl,
    run_tree_arches_pipeline,
)
```

Calling these without scHPL installed raises `ImportError` with the pip command above.

#### Recommended workflow

1. **Reference integration** — train CYTOVI (~1000 epochs), then
   `CytoANVI.from_cytovi_model` and fine-tune (~200 epochs). Optionally call `set_hierarchy` first
   if you have a user ontology for HCE.
2. **Learn hierarchy** — `latent_to_anndata(model, adata)` → `learn_hierarchy` on reference latent
   (concatenate `cell_type + "-" + batch` for unique labels across studies).
3. **Optional HCE alignment** — `set_hierarchy_from_schpl(model, tree)` then a short retrain.
4. **Query surgery** — `prepare_query_anndata` (if panel divergent) → `load_query_data` → train.
5. **Post-integration hierarchy** — combined latent → `update_hierarchy` (labeled query) or
   `predict_schpl` (unlabeled query). Compare with `predict_hierarchical()` and flat `predict()`.

Or call `run_tree_arches_pipeline(...)` to chain steps 2–5 in one explicit invocation (each stage
validates inputs; no silent skip on failure).

For `mode="update"`, provide latents via one of: pre-built `combined_latent`, `combined_adata` plus
`query_model`, or `reference_adata` + `query_adata` + `query_model` (concatenates with
`adata.concatenate` — add batch metadata before concat if you need provenance columns).

Tracked executable engineering fixture: `vignettes/cytoanvi_treearches_synthetic.py`. The richer
Markdown tutorial lives in an external uninitialized gitlink and its required repair is recorded in
the usage-readiness packet's `external-submodule-tutorial-handoff.md`; it is not part of the 0.2.0
artifact lineage.

### Warm-start from CytoVI

```python
from cytoanvi import CytoANVI
from scvi.external import CYTOVI
from scvi.external import cytovi as cytovi_pp

LAYER = "scaled"
LABELS_KEY = "cell_type"
UNLABELED = "Unknown"

# Preprocess cytometry intensities and keep unannotated / held-out cells explicitly unlabeled.
cytovi_pp.transform_arcsinh(adata)
cytovi_pp.scale(adata)
adata.obs[LABELS_KEY] = adata.obs[LABELS_KEY].where(adata.obs[LABELS_KEY].notna(), UNLABELED)
adata.obs.loc[held_out_cells, LABELS_KEY] = UNLABELED
adata.obs[LABELS_KEY] = adata.obs[LABELS_KEY].astype(str)

CYTOVI.setup_anndata(
    adata,
    layer=LAYER,
    batch_key="batch",
    labels_key=LABELS_KEY,
)
cytovi_model = CYTOVI(adata)
cytovi_model.train(max_epochs=1000)

anvi = CytoANVI.from_cytovi_model(
    cytovi_model,
    unlabeled_category="Unknown",
    labels_key=LABELS_KEY,
)
anvi.train(max_epochs=200)  # fine-tune classifier + semi-supervised head

# Optional query mapping. If the query has no labels, add/fill the label column as unlabeled.
if LABELS_KEY not in query_adata.obs:
    query_adata.obs[LABELS_KEY] = UNLABELED
else:
    query_adata.obs[LABELS_KEY] = query_adata.obs[LABELS_KEY].fillna(UNLABELED).astype(str)
query_model = CytoANVI.load_query_data(query_adata, anvi)
query_model.train(max_epochs=200)
query_adata.obs["cytoanvi_pred"] = query_model.predict()
```

Direct CytoANVI tutorials that call `CytoANVI.setup_anndata(...)` and then `CytoANVI(...)`
remain valid direct-training examples; they are not the CYTOVI warm-start workflow above.

### Panel-divergent query mapping (Roider-style)

When the query panel differs from the reference (shared backbone + panel-specific markers):

```python
query = CytoANVI.prepare_query_anndata(query_adata, reference_model=model, inplace=False)
query_model = CytoANVI.load_query_data(query, model)
query_model.train(max_epochs=200)
query.obs["pred"] = query_model.predict()
```

`prepare_query_anndata` pads missing markers and writes a **nan mask** so padded zeros are not
treated as real intensities. The reference must have been trained with a genuine backbone/panel split.

### Novelty / OOD status

Stable TTA novelty entry points fail closed. The available three-seed result was below chance
(mean AUROC 0.484), whereas latent kNN was informative. No replacement is promoted. An explicitly
named experimental TTA method retains deterministic per-cell masks for historical reproduction,
but its output must not be used as a supported novelty or replay-selection decision.

### Query mapping QC (optional mapQC)

After query surgery, score mapping quality on CytoANVI latents with
[mapQC](https://github.com/theislab/mapqc) (scores **> 2** = far from reference). Install:

```bash
pip install "cytoanvi[cytoanvi-mapping-qc]"  # resolves exactly mapQC 0.1.1
```

Import helpers from the optional module (not the top-level `cytoanvi` package):

```python
from cytoanvi.mapping_qc import (
    build_mapqc_anndata,
    evaluate_mapqc,
    run_mapqc_on_cytoanvi,
)

joint = query_model.score_query_mapping(
    reference_controls,
    query_adata,
    sample_key="patient",
    n_nhoods=50,
    k_min=50,
    k_max=200,
)
stats = evaluate_mapqc(joint, case_control_key="status", case_cats=["case"], control_cats=["control"])
```

Use the **trained query model** after surgery (not the reference-only model when query batches differ).

**Requirements:** reference = **controls only**; query must include **matched control cells**
(and optionally case cells). At least **3 reference samples** in `sample_key`. mapQC runs on the
joint embedding in ``obsm["X_CytoANVI"]`` — not on raw protein intensities.

**Fail-fast:** missing optional extra → `ImportError`; too few reference samples → `ValueError`.
mapQC is never run automatically during training or `load_query_data`.

mapQC remains advisory: it rejected every Nuñez query in one benchmark despite strong label
transfer, so no automatic gate or threshold is promoted. It is not applicable to unlabeled
panel-only queries without matched controls (e.g. Roider panel-2).

### Continual update (experimental)

For updating a reference with new query cohorts while limiting catastrophic forgetting (cscanvi-style):

```python
from cytoanvi import CytoANVI

# Replay membership must be frozen independently of TTA/outcome scores.
replay = reference_adata[predeclared_replay_indices].copy()
# healthy controls from the query (~5–10%)
controls = query_adata[query_adata.obs["status"] == "healthy"].copy()

updated = CytoANVI.load_query_data_with_replay(
    query_adata,
    reference_model=model,
    replay_adata=replay,
    control_adata=controls,  # required
)
updated.train(max_epochs=200, plan_kwargs={"ewc_importance": 100.0})  # λ — retune for CytoVI
```

`ewc_importance` (= λ) is **not** a constructor argument; pass it at train time. The paper used
`λ=100` for scANVI/RNA; CytoVI's intensity likelihood has different Fisher magnitudes, so λ must be
retuned (see benchmark task B6).

After `save`/`load`, continual models retain the EWC anchor, Fisher importances, and combine rule,
but they do **not** retain replay batches. Calling `train()` in that state raises before trainer
construction. Rebuild query surgery with both
`load_query_data_with_replay(..., replay_adata=..., control_adata=...)` before training.

See ADR `docs/adr/0002-cytoanvi-continual-follows-paper-not-code.md` for design notes.

## Nuñez PBMC labels (benchmark D2)

Vignette Nuñez FCS files do **not** include cell-type annotations. For paper-aligned eleven-type
labels, use the CytoVI tutorial workflow (train CytoVI → Leiden on latent → manual cluster map):

```bash
PYTHONPATH=src:. python -m benchmarks.cytoanvi.annotate_nunez \
  --data-dir data --out data/nunez_annotated.h5ad --max-epochs 100
```

Then point CytoANVI at `obs["cell_type"]` from that file. For publication-grade benchmarks, pass
`--require-annotated-nunez` to the benchmark CLI so Leiden proxy labels are not used.

## Inherited CytoVI methods

CytoANVI subclasses CytoVI and retains (among others):

- {meth}`~scvi.external.CYTOVI.get_normalized_expression` — denoised / batch-corrected protein expression
- {meth}`~scvi.external.CYTOVI.differential_expression`
- {meth}`~scvi.external.CYTOVI.differential_abundance`

Multi-panel references are built with {func}`scvi.external.cytovi.merge_batches` before
`CytoANVI.setup_anndata`.

Refer to {doc}`/user_guide/models/cytovi` for preprocessing cofactors and mathematical background.

## Citation

If you use CytoANVI in your research, please cite [^ref1].

[^ref1]:
    Manurung et al. _CytoANVI: annotation-aware variational inference for antibody-based single-cell
    cytometry._ Manuscript in preparation (2026).

[^ref2]:
    Xu C, Lopez R, Mehlman E, Regier J, Jordan MI, Yosef N. (2021).
    _Probabilistic harmonization and annotation of single-cell transcriptomics data with deep
    generative models._
    [Molecular Systems Biology](https://www.embopress.org/doi/full/10.15252/msb.20209620).

[^ref3]:
    Lotfollahi M, Naghipourfar M, Luecken MD, Khajavi M, Büttner M, Wagenstetter M, Avsec Ž,
    Gayoso A, Yosef N, Interlandi M, Rybakov S, Misharin AV, Theis FJ. (2022).
    _Mapping single-cell data to reference atlases by transfer learning._
    [Nature Biotechnology](https://www.nature.com/articles/s41587-021-01001-7).

[^ref4]:
    Michielsen L, Reinders MJT, Mahfouz A. (2021).
    _Hierarchical progressive learning of cell identities in single-cell data._
    [Nature Communications](https://www.nature.com/articles/s41467-021-23774-w).
