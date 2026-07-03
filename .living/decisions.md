# Decisions Log

Record non-obvious choices made during development. One entry per decision. Reference IDs in commit messages and learnings where applicable.

## Format

```
### D-NNN — [YYYY-MM-DD] Title
**Context**: What situation prompted this decision.
**Decision**: What was chosen.
**Rationale**: Why, including alternatives considered.
**Consequences**: What this commits us to or rules out.
**Status**: active | superseded | revisited
```

---

### D-001 — [2026-06-01] Use M1+M2 hierarchy (scANVI-style), not GMM prior
**Source**: `docs/adr/0001-cytoanvi-m2-over-gmm-prior.md`
**Context**: CytoVI uses a GMM prior for label shaping. CytoANVI needed a classification objective.
**Decision**: Use scANVI's M1+M2 two-level latent (z1 classifier input, z2 per-label) instead of GMM. `CytoANVAE` mirrors `TOTALANVAE(SupervisedModuleClass, CytoVAE)` and forces `prior_mixture=False` in the semi-supervised path.
**Rationale**: GMM prior is unsupervised shaping; M1+M2 adds a discriminative loss directly. GMM alone gives no label-transfer accuracy guarantee. The M2 loss ignores `generative_outputs["pz"]`, so an active GMM prior is "dead weight at best and double-counts label shaping at worst."
**Rejected alternatives**: (a) GMM + classifier head only, no M2 — diverges from scANVI/totalANVI, won't align with Phase-2 csCANVI port; (b) keep GMM *and* M2 — double-counts label shaping, novel/unvalidated.
**Consequences**: CytoANVI is architecturally closer to scANVI than CytoVI. Label-conditioned GMM off by default. **Gotcha**: `prior_mixture=False` forced in `CytoANVAE.__init__` — the apparently-unused GMM code path is deliberate, not an oversight.
**Status**: active

### D-002 — [2026-06-10] Benchmark at max_epochs=1000, full cohorts only
**Context**: Prior results used vignette subsamples (100 epochs) — not publication-grade.
**Decision**: All Track-B benchmarks run at max_epochs=1000 on full Nuñez and Roider cohorts.
**Rationale**: Convergence at 100 epochs is insufficient; scib metrics need full-cohort cell counts.
**Consequences**: Each benchmark job takes ~hours on GPU. Gate publication on these results.
**Status**: active

### D-003 — [2026-06-17] Continual update via EWC + replay; follow the paper, not the released code
**Source**: `docs/adr/0002-cytoanvi-continual-follows-paper-not-code.md`
**Context**: B4/B6 tasks require integrating case-control query without catastrophic forgetting. The csCANVI released code and the bioRxiv Methods disagree on implementation details.
**Decision**: `ContinualUpdate` module combines Fisher-weighted EWC + experience replay. When paper and code disagree, **follow the paper**. Specifics: (1) experience replay rehearses ~20% reference cells in the ELBO (released code only uses replay for Fisher, never rehearses); (2) EWC weight = Hadamard product of reference-replay Fisher and query-control Fisher, `combine_type="product"` default; (3) query controls are **required** (must exist in both reference and query); (4) TTA masks **50%** of features for uncertainty (not 15%).
**Rationale**: EWC is theoretically grounded; replay alone risks mode collapse on small query sizes. Paper is the primary source; repo/README are not substitutes (Mikhael's instruction).
**Consequences**: λ (`ewc_importance`) is a new hyperparameter; B6 sweeps it. Fisher computation costs one extra forward pass. Default λ≠100 (paper used 100 for RNA/scANVI, but CytoVI intensity-likelihood Fisher scale differs — must retune; see B6).
**Status**: active

### D-004 — [2026-06-18] `encoder_marker_mask_` persisted on save/load
**Context**: `prepare_query_anndata` was failing when called from a saved model path (mask not reconstructable without original training data).
**Decision**: Persist the encoder marker mask as a model attribute alongside other state dicts.
**Rationale**: Without persistence, any surgery from disk was broken. Simple fix with no downsides.
**Consequences**: Old model checkpoints (pre-fix) need re-training if using path-based surgery.
**Status**: active

### D-005 — [2026-06-29] HCE and scHPL hierarchy are opt-in, matrix-gated; fail-fast on misuse
**Source**: `docs/adr/0003-cytoanvi-hce-schpl-hierarchy.md`
**Context**: CytoANVI needed to support cell-type hierarchies for structured label transfer. Two mechanisms were considered: HCE (hierarchical cross-entropy) and scHPL treeArches workflows.
**Decision**: (1) **HCE is core** but matrix-gated — `CytoANVAE.classification_loss` uses flat `F.cross_entropy` when `reachability_matrix_ is None`; activated explicitly via `set_hierarchy()` / `hierarchy_edges` / `reachability_matrix`. Hierarchy excludes the unlabeled category. (2) **scHPL treeArches** is an optional extra: lazy-import, behind `pip install scvi-tools[cytoanvi-hierarchy]`. **Fail-fast policy**: raise `ValueError` on `predict_hierarchical()` without hierarchy, `set_hierarchy` label mismatch, unmapped scHPL leaves, both `hierarchy_edges`+`reachability_matrix` at init, invalid shape/non-DAG; `ImportError` (with pip cmd) when extra missing. **Never** silently fall back to flat CE or skip scHPL.
**Rejected alternatives**: HCE always-on; both behind one extra; stateful `CytoANVIHierarchyAtlas` orchestrator.
**Consequences**: Users must opt in explicitly. Any code path that expects flat CE by default still works; HCE only activates when a reachability matrix is set.
**Status**: active

### D-006 — [2026-06-29] CytoANVI promoted to top-level package; `scvi.external.cytoanvi` removed
**Source**: `docs/adr/0004-cytoanvi-top-level-package.md`
**Context**: CytoANVI started as `scvi.external.cytoanvi` during prototyping. As it matured, the external namespace became misleading.
**Decision**: CytoANVI is a **top-level package** — `cytoanvi.CytoANVI`, `cytoanvi.CytoANVAE`, `cytoanvi.hierarchy`, `cytoanvi.mapping_qc`. The old `scvi.external.cytoanvi` / `scvi.external.CytoANVI` export and package path are **removed**.
**Consequences**: **Breaking change** — unpickling/model-loading of old artifacts that reference the previous class module path will fail. Tests, tutorials, benchmarks, and API docs import from `cytoanvi`. Benchmark harness stays under `benchmarks/cytoanvi` (not part of the public package). Extras keep names `scvi-tools[cytoanvi-hierarchy]`, `scvi-tools[cytoanvi-mapping-qc]`.
**Status**: active

### D-007 — [2026-06-29] Manifest-mode aggregation for publication; recursive --input is exploratory only
**Source**: `.scratch/cytoanvi-benchmark/PRD.md` (2026-06-28 decisions table)
**Context**: `aggregate_results.py` supports both recursive `--input` (gathers all JSONs under a path) and manifest mode (explicit `publication_manifest.json` allowlist). Recursive mode accidentally picked up smoke/synthetic/stale-roider results alongside e1000 full-cohort results.
**Decision**: Publication aggregation uses `publication_manifest.json` exclusively. Recursive `--input` is for exploratory inspection only and must not be used to generate publication claims.
**Consequences**: Any result JSON not listed in `publication_manifest.json` is silently excluded from publication aggregation. This is a feature, not a bug.
**Status**: active

### D-XXX — [2026-06-29] Cancel B8 job 25107490; resubmit as 25108052 after second review fixes
**Context**: Second code review found two B8 correctness issues after job 25107490 had been running for 2+ hours.
**Decision**: Cancel and resubmit rather than let invalid results land as publication artifacts.
- **#3 leaf_held bias**: `delta_hierarchical_vs_flat_macro_f1` was evaluated over all held cells; cells with internal-node true labels always score wrong under `predict_hierarchical(leaf_only=True)`, biasing delta against HCE. Fix: exclude internal-label cells from evaluation (see L-012).
- **#8 HCE routing**: Hand-rolled HCE setup/train can drift from flat arm's config; both arms now route through `train_cytoanvi(hierarchy_edges=...)` (see L-013).
**Consequences**: No result files were written before cancellation. Resubmitted as job 25108052.
**Status**: active

### D-008 — [2026-07-03] Distribute CytoANVI as a standalone package named `cytoanvi` (not an upstream scvi-tools PR)
**Source**: Publication-readiness review (session 13); maintainer decision.
**Context**: The packaging review found the fork still published under `name = "scvi-tools"` with scverse URLs/maintainers, which would collide with the upstream PyPI package. Two paths existed: upstream PR to scverse, or standalone release.
**Decision**: Release standalone under PyPI name `cytoanvi`. `pyproject.toml` identity (name, description, authors/maintainers, Source/Home-page URLs) now points to the fork (github.com/mdmanurung/scvi-tools, Mikhael Manurung). All self-referencing optional-dep extras rewritten `scvi-tools[...]` → `cytoanvi[...]`. Install docs/README updated accordingly. LICENSE now carries dual BSD-3 attribution (upstream scvi-tools dev team + CytoANVI contributors) to satisfy BSD-3 clause 1.
**Consequences**: The wheel still ships both `scvi` and `cytoanvi` packages. A new `[tool.hatch.build.targets.sdist]` exclude list + `.gitattributes` export-ignore keep `.scratch/`, `.living/`, `todo/`, logs, and PDFs out of published artifacts.
**Status**: active

### D-009 — [2026-07-03] B5 novelty headline metric is mean AUROC, not best AUROC
**Source**: Benchmark-rigor + methods reviews (session 13).
**Context**: B5 results led with `best_auroc` (max over ~47 held-out cell types ≈ 0.83) while `mean_auroc` (≈ 0.46, below chance) was buried. `best_auroc` is a maximum, not a summary statistic — reporting it as headline overstates the uncertainty module's utility.
**Decision**: `mean_auroc` (and `mean_auroc_fdr_sig`) is the PRIMARY reported field everywhere (task return dict, aggregation, manifest). `best_auroc` retained only as an explicitly-labeled secondary/alias (kept because `test_aggregate_results.py` asserts on it). Underlying computation unchanged — only presentation and labels.
**Consequences**: The user guide and manifests now report B5 honestly as near-chance on most T-cell subtypes; a genuine improvement requires either a better uncertainty formulation or an external novel-cell-type dataset.
**Status**: active

### D-010 — [2026-07-03] B3 cross-panel metric is inter-method concordance, labeled as NOT accuracy
**Source**: Benchmark-rigor + methods reviews (session 13).
**Context**: `task_b3_panel_divergent` reports `p2_inter_method_agreement_vs_knn` — agreement between CytoANVI predict() and CytoVI-kNN on panel-2, where both share the CytoVI encoder backbone and there is NO panel-2 ground truth. Two methods that are both wrong would score high.
**Decision**: Keep the metric (no independent labels available) but relabel it unmistakably as inter-method concordance, NOT ground-truth accuracy, in the metric docstring, task note, and ANALYSIS_MANIFEST. A definitive cross-panel claim requires independent manually-gated panel-2 labels or an architecturally-independent baseline (FlowSOM/XGBoost).
**Consequences**: B3 cannot be presented as validating cross-panel correctness until independent labels are obtained. Tracked as an open blocker.
**Status**: active
