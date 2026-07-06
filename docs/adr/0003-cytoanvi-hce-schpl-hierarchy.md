# CytoANVI hierarchy: opt-in HCE and optional scHPL treeArches

CytoANVI supports two optional hierarchy features:

1. **Hierarchical cross-entropy (HCE)** — hierarchy-aware classifier training via an explicit
   reachability matrix (Microsoft [hce-classification](https://github.com/microsoft/hce-classification)).
2. **treeArches-style workflows** — learn, update, and predict cell-type hierarchies on CytoANVI
   latents with [scHPL](https://schpl.readthedocs.io/), behind
   `pip install cytoanvi[cytoanvi-hierarchy]`.

Both are **opt-in**. Flat cross-entropy remains the default when no reachability matrix is set.

## Decision

### HCE is core, activated only by an explicit matrix

- `CytoANVAE.classification_loss` uses flat `F.cross_entropy` when `reachability_matrix_` is `None`.
- HCE activates only after the user calls `CytoANVI.set_hierarchy(...)` or passes
  `hierarchy_edges` / `reachability_matrix` at construction.
- Reachability is built from a user-supplied parent→children DAG
  (`build_reachability_matrix` in `cytoanvi._hce`). No ontology auto-fetch, no scHPL import in
  the HCE path.

### scHPL is an optional extra

- treeArches helpers live in the top-level `cytoanvi.hierarchy` module.
- Functions lazy-import scHPL and raise `ImportError` with
  `pip install cytoanvi[cytoanvi-hierarchy]` when missing.
- `pyproject.toml` defines `cytoanvi-hierarchy = ["scHPL"]`.

### Fail-fast, no silent fallbacks

| Situation | Behavior |
|-----------|----------|
| `predict_hierarchical()` without hierarchy | `ValueError` with fix instruction |
| `set_hierarchy` label mismatch | `ValueError` listing unknown or missing labels |
| scHPL function without extra installed | `ImportError` with pip command |
| `set_hierarchy_from_schpl` unmapped leaves | silently ignored (absent from reachability mapping); only ambiguous assignment (2+ candidates) raises `ValueError` |
| Both `hierarchy_edges` and `reachability_matrix` at init | `ValueError` |
| Invalid reachability shape or non-DAG edges | `ValueError` with expected vs actual |

**Never:** silently fall back to flat CE when hierarchy was requested but invalid; silently skip scHPL;
auto-install dependencies.

### Hierarchy excludes the unlabeled category

Reachability rows/columns index **observed labeled categories only** (registry categories minus
`unlabeled_category`). The unlabeled string is not a node in the ontology and is not included in
HCE or `predict_hierarchical` propagation.

## Considered options

- **HCE always on when labels exist.** Rejected — forces users to supply or infer a DAG; breaks
  existing flat-CE benchmarks and workflows.
- **Both HCE and scHPL behind one optional extra.** Rejected — HCE has no third-party deps; keeping
  it in core avoids gating a zero-dependency loss.
- **Stateful orchestrator class (`CytoANVIHierarchyAtlas`).** Rejected — plain functions plus one
  optional `run_tree_arches_pipeline(...)` with explicit parameters is simpler and easier to test.

## Consequences

- Default CytoANVI behavior is unchanged for users who never call hierarchy APIs.
- Hierarchy-aware training requires an explicit ontology or a scHPL-learned tree converted via
  `set_hierarchy_from_schpl`.
- Integration (CytoVI → CytoANVI, scArches surgery) stays in existing APIs; hierarchy docs describe
  recommended step order without new training wrappers.
