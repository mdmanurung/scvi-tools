# CytoANVI Publication Readiness — Ideation Index

**Session**: 2026-06-30 (refined 2026-06-30)
**Question**: What gaps, risks, and improvements need to be addressed to make the CytoANVI package and its benchmark suite publication-ready?
**Personas**: Peer Reviewer · Statistical Methods Critic · Cytometry Experimentalist · Reproducibility/Software Engineer
**Ideas per persona**: 3

> **Refinement pass complete.** All 4 persona files now contain concrete file paths, function signatures, implementation steps, and machine-checkable done criteria. See individual files for full specifications.

---

## All 12 Ideas

| # | Persona | Idea | Effort | Priority |
|---|---------|------|--------|----------|
| 1 | Peer Reviewer | Add FlowSOM/Phenograph baselines; stratify B1 claims by cohort type | High | **Critical** |
| 2 | Peer Reviewer | Reframe or fix B5 novelty detection — mean AUROC 0.467 is unpublishable as-is | Med–High | **Critical** |
| 3 | Peer Reviewer | Real case/control dataset for B4 continual update, or demote to supplement | Med–Low | High |
| 4 | Statistical Critic | B3 concordance is circular — needs ground-truth F1 against panel-2 expert labels | Medium | **Critical** |
| 5 | Statistical Critic | B5 "best AUROC" cherry-pick: max over 13 tests, mean sub-chance — apply FDR | Low | **Critical** |
| 6 | Statistical Critic | B1 information asymmetry (CytoANVI sees labels, CytoVI kNN does not) — add supervised XGBoost upper bound | Low–Med | High |
| 7 | Cytometry Expert | Arcsinh cofactor=5 is CyTOF-only; flow/spectral users get silent degenerate latent | Medium | **Critical** |
| 8 | Cytometry Expert | Continual update: drift=0.0 on pseudo-splits is not evidence; need real rLN vs FL/MCL split + fix save/load gap | High | **Critical** |
| 9 | Cytometry Expert | Novelty detection needs actionable threshold API + P@95S metric; conformal p-value optional | Medium | High |
| 10 | Repro Engineer | Archive `nunez_annotated.h5ad` to Figshare — its absence hard-blocks B3/B5/B6/B8/B9 | Low | **Critical** |
| 11 | Repro Engineer | Pin and archive the exact conda/pip environment (lock file + Singularity image) used for publication runs | Medium | **Critical** |
| 12 | Repro Engineer | Wire `@pytest.mark.optional` on hierarchy/mapqc tests so CI actually exercises optional-dep paths | Low | High |

---

## Grouped by priority

### Critical (6 ideas — must address before submission)

| ID | Idea | Effort | Persona |
|----|------|--------|---------|
| 1 | Add FlowSOM/Phenograph baselines; scope B1 claims to cohort type | High | Peer Reviewer |
| 2 | Fix/reframe B5 novelty detection — mean AUROC 0.467 is not a publishable claim | Med–High | Peer Reviewer |
| 4 | B3: replace circular concordance metric with ground-truth macro-F1 | Medium | Stats Critic |
| 5 | B5: apply FDR across 13 holdout tests; drop "best AUROC" gate | Low | Stats Critic |
| 7 | Arcsinh cofactor per-channel support for flow/spectral flow + UserWarning | Medium | Cytometry Expert |
| 8 | Continual update on real biological case/control split; fix EWC save/load | High | Cytometry Expert |
| 10 | Archive `nunez_annotated.h5ad` to Figshare; add to auto-download | Low | Repro Engineer |
| 11 | Environment lock file + Singularity image for benchmark reproducibility | Medium | Repro Engineer |

### High (4 ideas — major revision risk if not addressed)

| ID | Idea | Effort | Persona |
|----|------|--------|---------|
| 3 | Demote B4 continual update to supplement if no real dataset available | Low | Peer Reviewer |
| 6 | Add XGBoost supervised upper bound to B1; resolve Nuñez reversal mechanistically | Low–Med | Stats Critic |
| 9 | Add `get_uncertainty_threshold(specificity=0.95)` and P@95S metric | Medium | Cytometry Expert |
| 12 | Wire `@pytest.mark.optional` on hierarchy/mapqc CI paths | Low | Repro Engineer |

---

## Cross-cutting themes

Three themes surface independently across all four personas:

1. **B5 novelty detection (ideas 2, 5, 9)**: All three technical personas flag the same root problem from different angles — the mean AUROC, the multiple-testing inflation, and the missing threshold API. These are one integrated fix: reframe B5 as a precision-at-threshold benchmark with FDR correction, scope the claim to phenotypically distinct types, and add `get_uncertainty_threshold()`.

2. **Continual update credibility (ideas 3, 8)**: Two independent personas (peer reviewer, cytometry expert) flag that B4 drift=0.0 on pseudo-splits is not publishable evidence. The EWC save/load bug (L-009) makes this a software issue too. These converge on the same action: real rLN vs FL/MCL biological split + serialize `ContinualUpdate` state.

3. **B1 baseline comparison (ideas 1, 6)**: The peer reviewer flags missing community baselines (FlowSOM/Phenograph); the stats critic flags information asymmetry (CytoANVI sees labels, CytoVI-kNN does not) and suggests a supervised XGBoost upper bound. Both are needed: add XGBoost + FlowSOM to `baselines.py`.

---

## Highest-ROI quick wins (Low effort, High/Critical priority)

| ID | Action | Time estimate |
|----|--------|---------------|
| 5 | Add FDR correction to B5 AUROC, replace "best AUROC" gate with mean/FDR summary | ~2h |
| 10 | Upload `nunez_annotated.h5ad` to Figshare + wire auto-download | ~1h |
| 12 | Add `@pytest.mark.optional` to hierarchy/mapqc test files | ~30min |
| 11 | Export conda lock file (`conda list --export`) + write `REPRODUCE.md` | ~1h |

---

## Source files

- [01_peer-reviewer.md](01_peer-reviewer.md)
- [02_statistical-methods-critic.md](02_statistical-methods-critic.md)
- [03_cytometry-experimentalist.md](03_cytometry-experimentalist.md)
- [04_reproducibility-software-engineer.md](04_reproducibility-software-engineer.md)
