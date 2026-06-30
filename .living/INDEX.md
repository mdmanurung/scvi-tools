# Living Repository Index

**Project**: scvi-tools — CytoANVI extension
**Branch**: feat/cytoanvi
**Last updated**: 2026-06-30

## Quick navigation

| File | Contents |
|------|----------|
| [decisions.md](decisions.md) | Architecture and design decisions (D-001…D-007, D-XXX) |
| [learnings.md](learnings.md) | Gotchas, bugs, edge cases (L-001…L-023) |
| [conventions.md](conventions.md) | Active conventions crystallized from learnings (C-001…C-003) |
| [conventions/bioinformatics/](conventions/bioinformatics/analysis-conventions.md) | Bioinformatics domain conventions (BIO-01…BIO-10) |
| [findings/FINDINGS_REGISTRY.md](findings/FINDINGS_REGISTRY.md) | Empirical results registry (F-001…F-011) |
| [log/LOG_REGISTRY.md](log/LOG_REGISTRY.md) | Session work log |

## <!-- BEGIN KNOWLEDGE SUMMARY -->

### Decisions by topic

| Decision | ID | ADR |
|----------|----|-----|
| M1+M2 hierarchy over GMM prior | D-001 | `docs/adr/0001` |
| Benchmark at max_epochs=1000, full cohorts | D-002 | — |
| Continual update: EWC + replay, follow paper not code | D-003 | `docs/adr/0002` |
| `encoder_marker_mask_` persisted on save/load | D-004 | — |
| HCE + scHPL: opt-in, matrix-gated, fail-fast | D-005 | `docs/adr/0003` |
| Top-level package; `scvi.external.cytoanvi` removed | D-006 | `docs/adr/0004` |
| Manifest-mode aggregation for publication | D-007 | PRD 2026-06-28 |
| B8 job cancel + resubmit after correctness fixes | D-XXX | — |

### By tag

| Tag | Entry IDs |
|-----|-----------|
| nan-layer, masking | L-001, C-001 |
| classifier, n_labels, edge-case | L-002, C-002 |
| fisher, ewc, continual-update | L-003, L-008, L-009, C-003, D-003 |
| surgery, save-load, encoder-mask | L-004, D-004 |
| scib, batch-correction | L-005 |
| eval-mode, train-mode, inference | L-007 |
| batch-size, nan-divergence, training | L-010 |
| aggregation, publication, manifest | L-011, D-007 |
| B8, hce, hierarchical, leaf-held | L-012, L-013, D-XXX, D-005 |
| lambda, ewc, continual | L-020, D-003, F-008 |
| figshare, slurm, environment | L-014 |
| nunez, annotation, leakage, knn, proxy-labels, b1 | L-022 |
| preprocessing, arcsinh, cofactor, file-location | L-023 |
| B5, novelty, auroc, bimodal | L-015, F-007, F-010 |
| transductive, leakage, evaluation | L-016 |
| convention, adr, documentation, drift | L-017 |
| exception-handling, baseline, silent-failure | L-018 |
| auroc, wilcoxon, se-formula, fdr | L-019 |
| lambda, ewc, continual-update, hyperparameter | L-020, D-003 |
| classifier, backbone, panel-specific, B3 | L-021, D-001 |
| hierarchy, scHPL | D-005 |
| top-level-package | D-006 |
| B1, macro-F1, label-transfer | F-001, F-002, F-003 |
| B2, scib, batch-mixing, bio-conservation | F-004, F-005 |
| B3, cross-panel, concordance | F-006 |
| B5, novelty, auroc | F-007, F-010 |
| B4, B6, ewc, lambda | F-008 |
| B8, hce, findings | F-009 |

### Heuristic clusters (≥2 related entries)

**Panel/masking correctness** (L-001, L-004, D-004, C-001): Multiple bugs from multi-panel encoding assumptions. Conventions C-001 + BIO-02 address.

**EWC/continual-update correctness** (L-003, L-008, L-009, C-003, D-003): Fisher subsampling, snapshot-from-reference, state not in state_dict, paper-over-code. Key: replay is NOT restorable from a loaded model alone.

**Classifier robustness** (L-002, D-001, C-002): `n_labels==0` edge case + GMM-off gotcha.

**B8 benchmark correctness** (L-012, L-013, D-XXX, D-005): Two interrelated bugs (leaf_held bias + HCE routing) that invalidated job 25107490.

**Benchmark validity / publication gate** (L-010, L-011, D-002, D-007, F-001…F-011): All current findings are pre-publication. batch_size root cause, manifest-mode discipline, and full-cohort resubmits are the active gate. B8 provisionally passes (F-011, Δ_hier +0.0862±0.0027, 3-seed).

**B1 Nuñez reversal** (F-003, L-022, todo Monitor-B1-reversal): CytoANVI Δ−0.013 on Nuñez r0.05 e1000 3-seed (confirmed, not just vignette). Interpretable: clean Nuñez labels + large homogeneous populations = strong kNN. Annotation was also transductively leaky (joint-Leiden, L-022); corrected via inductive kNN rerun (PID 2539851, ETA ~28h from 2026-06-30). F-003 status will update when inductive full-cohort result arrives.

**B5 bimodal novelty** (F-007, F-010, L-015): Phenotypically distinct types (Tfh, Treg CD69+) detectable as novel (AUROC≥0.70); similar subtypes (Treg CD69-/CD69+, Ttox EM series) at or below chance. Mean AUROC 0.467. Publication framing: novelty recall scales with phenotypic distance.

**Transductive evaluation footgun** (L-016, L-022): Three independent code paths had the same error — KNN imputation (A3), Nuñez annotation, B5 novelty detection. Pattern: any `fit()` call that includes test/query indices is transductive. Tripwire: assert held indices are absent from training set. L-022 documents the Nuñez-specific mechanism (joint-Leiden labels) and the inductive kNN fix.

**AUROC Wilcoxon SE normalization** (L-019): B5 SE formula was `sqrt(1/(12*n1*n2))` — missing `(n1+n2+1)` numerator. z-scores inflated ≈74× for typical group sizes; all prior B5 FDR calls invalid. Fixed in commit 6c99afe5.

## <!-- END KNOWLEDGE SUMMARY -->
