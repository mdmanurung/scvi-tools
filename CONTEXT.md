# scvi-tools — CytoANVI domain language

Glossary for the CytoANVI line of work: a semi-supervised, annotation-aware extension of CytoVI
for antibody-based single-cell cytometry (flow, mass cytometry, CITE-seq protein). Created lazily
as terms are resolved; scoped to this feature, not the whole library.

## Language

**Protein intensity likelihood**:
The decoder emission distribution over antibody-measured marker *intensities* — Normal or Beta in
CytoVI. _Avoid_: count likelihood, NB/ZINB (those are RNA-count models and do not apply here).

**Unlabeled category**:
The single label value (e.g. "Unknown") marking cells with no ground-truth annotation; remapped
to the last integer code by `LabelsWithUnlabeledObsField`. _Avoid_: missing label, NA label.

**Observed labels**:
The cell-type labels excluding the unlabeled category; their count is the model's `n_labels`
(registry `n_labels − 1`). _Avoid_: known classes, ground truth (ambiguous).

**M1+M2 hierarchy (z1 / z2)**:
The two-level latent of scANVI carried into CytoANVI — z1 is the shared representation the
classifier reads; z2 is the per-label latent with prior N(0,I). _Avoid_: just "latent" when the
level matters.

**Label-conditioned mixture prior (GMM prior)**:
CytoVI's alternative label-shaping mechanism (one Gaussian component per label). In CytoANVI it is
**off** (see ADR 0001). _Avoid_: conflating with the M2 prior.

**Panel / backbone markers / missing markers**:
A panel is one antibody marker set; backbone markers are those shared across panels; missing
markers are unmeasured in a given panel, masked via `nan_layer` and encoded only on the backbone.
_Avoid_: features, genes.

**Reference / query / surgery**:
The reference is the trained labeled model; a query is a new dataset mapped onto it via
scArches-style surgery (`load_query_data`); label transfer assigns observed labels to query cells.
_Avoid_: train/test (different meaning).

**Continual update (Phase 2)**:
Incremental case–control atlas building (`cscanvi`-style) that fine-tunes on a query while
anchoring to healthy controls and a replay buffer, avoiding catastrophic forgetting. _Avoid_:
retraining, plain fine-tuning.

**ContinualUpdate (module)**:
The single module that owns a configured continual update: the reference **anchor** (`old_params`),
the reference Fisher and the query-control Fisher, the combine rule (`F_reference ∘ F_query_ctrl`),
and the **replay buffer**. Constructed at surgery (`load_query_data_with_replay`); held by
`CytoANVAE` as one object (present = active, absent = inactive). Exposes the drift penalty (unscaled
— λ stays a train-time `ewc_importance`) and the next replay minibatch. _Avoid_: "EWC state",
"continual state" (loose attributes — the point is they no longer are).

## Relationships

- A **CytoANVI reference** is built from labeled + **unlabeled-category** cells over **observed
  labels**, with annotation-aware shaping from the **M1+M2 hierarchy** (not the **GMM prior**).
- A **query** is integrated onto a **reference** via **surgery**; **label transfer** assigns
  **observed labels** to its cells (classifier `predict()`, or CytoVI's k-NN
  `impute_categories_from_reference`).
- **Missing markers** across **panels** are handled on the **backbone**; the classifier reads
  **z1**, so it stays valid when only backbone markers are encoded.

## Example dialogue

> **Dev:** "For the query cells with no annotation, what label do we give them at setup?"
> **Domain expert:** "The **unlabeled category** — one reserved value. The model only ever
> predicts **observed labels**, never that one."
> **Dev:** "And the latent shaping comes from CytoVI's mixture prior?"
> **Domain expert:** "No — in CytoANVI that's off. Shaping comes from the **M1+M2 hierarchy** plus
> the classifier. See ADR 0001."

## Flagged ambiguities

- "label" meant both the full registry categories (incl. unlabeled) and the predictable classes —
  resolved: the latter are **observed labels** (`n_labels`); the registry total is `n_labels + 1`.
- "prior" was ambiguous between CytoVI's **GMM prior** and the M2 **z2 prior N(0,I)** — resolved:
  CytoANVI uses the latter; the former is disabled (ADR 0001).
