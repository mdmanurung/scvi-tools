# Topic: Continual Update and λ Sweep (B4/B6)

**Question**: Does the EWC continual update preserve prior knowledge while integrating case-control queries? What is the right λ?
**Dataset**: Roider BNHL (pseudo batch split — not a real case/control axis; plumbing evidence only)
**Source docs**: `.scratch/cytoanvi-benchmark/issues/10-roider-b4-b6-continual.md`

---

## F-008 — B4/B6: replay latent drift tied at 0.0 for all λ values (plumbing only)
**Date**: 2026-06-28
**Status**: NOT publication evidence — pseudo batch split, not biological case/control
**Validity caveat**: `roider-smoke`, pseudo-split (B4 used an arbitrary batch split as case/control proxy)
**Claim**: Plain surgery: query macro-F1 **0.859**, replay drift **0.0**. Continual update (λ=1.0): query macro-F1 **0.862**, replay drift **0.0**. λ sweep over {0, 1, 10, 100, 1000}: all drifts tied at **0.0**; highest macro-F1 at λ=**1.0** (**0.888**), but drift signal is uninformative.
**Interpretation**: The pseudo case/control split produces no measurable anchor drift regardless of λ. EWC penalty is non-zero in the loss but the anchor and query are similar enough that the constraint doesn't bind. This is a plumbing proof-of-concept only.
**Meets target**: not evaluable (drift metric not meaningful on pseudo split)
**Source**: `.scratch/cytoanvi-benchmark/issues/10-roider-b4-b6-continual.md`
**Implications**: B4/B6 require a real rLN reference + FL/MCL query axis before λ selection is biologically informative. The current sweep cannot provide a publication-ready CytoVI-specific λ default.

---

## F-009 — B8 synthetic: flat CE 0.295, HCE 0.166 on tiny data (expected, not informative)
**Date**: 2026-06-29
**Status**: synthetic smoke only — not publication-grade
**Validity caveat**: `b8-synthetic-smoke` — tiny synthetic dataset; scHPL sibling-only leaves produce identity reachability, making HCE≈flat CE at scale
**Claim**: Flat CE macro_f1 **0.295**, HCE **0.166** (Δ **−0.129**). HCE underperforms on tiny synthetic data.
**Interpretation**: Expected on tiny data + sibling-only hierarchy (identity reachability → HCE≈flat CE). Not evidence against HCE. Full B8 results pending (job 25108052, resubmitted after leaf_held bias fix).
**Source**: `.scratch/cytoanvi-benchmark/results/b8_synthetic_s0.json`
**Note**: Two correctness bugs in the prior harness (L-012 leaf_held bias, L-013 HCE routing) mean any B8 results before job 25108052 are invalid.

---

## F-011 — B8 Nuñez full e1000 seeds 0+1: HCE hierarchical decoding Δ+0.088 over flat CE on leaf-held subset
**Date**: 2026-06-30
**Status**: NOT publication-grade — 2 seeds only (seed 2 pending, SLURM job 25108052 epoch 302/1000)
**Validity caveat**: `nunez-full-e1000-2seed` — Nuñez full 200k cells, max_epochs=1000; seeds 0 and 1 complete; seed 2 pending
**Claim**:
- Seed 0: flat_ce overall macro_F1 **0.9791**, hce_hierarchical_predict macro_F1 (leaf-held) vs flat_leaf_metrics: Δ **+0.0887**
- Seed 1: flat_ce overall macro_F1 **0.9787**, Δ **+0.0866**
- 2-seed interim: Δ_hier_vs_flat = **+0.0876 ± 0.0014** (consistent signal)
- flat CE macro_F1 (leaf-held subset): **0.979 ± 0.0003** (very stable across seeds)
**Interpretation**: HCE hierarchical decoding consistently improves on flat CE by ~+8.8pp on the held-out leaf cells. The Δ is stable across seeds 0 and 1, suggesting low variance. The full flat_ce is high (0.979) indicating the Nuñez hierarchy is well-structured; HCE adds ~+8.8pp on the harder leaf-held evaluation subset. Seed 2 pending for publication gate (≥3 seeds).
**Meets target**: provisionally YES — Δ>0 in both seeds; awaiting seed 2 for confirmation
**Source**: `.scratch/cytoanvi-benchmark/results/nunez_b8_s0.json`, `nunez_b8_s1.json`, `nunez_b8_multiseed.json` (2-seed interim)
**Implications**: HCE provides a consistent benefit over flat CE on Nuñez PBMC hierarchy. Reviewers will want to see this holds for seed 2 and ideally on the Roider dataset too.
